"""Wav2Vec2 model implementation using Hugging Face Transformers.

This file supports the following pretrained models:
- facebook/mms-300m
- facebook/mms-1b

Usage:
    python -m src.model.wav2vec2.wav2vec2_model
NOTE(shikhar): The code here is inverse of the forward(encode) pattern followed elsewhere.
NOTE(shikhar): keeping all hs may also be wasteful for memory.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchaudio
from transformers import Wav2Vec2ForCTC, Wav2Vec2FeatureExtractor
from typing import Dict, List, Tuple, Any
import numpy as np
from src.utils import RankedLogger

log = RankedLogger(__name__, rank_zero_only=True)


def preprocess_inputs_wav2vec2(
    feature_extractor: Wav2Vec2FeatureExtractor,
    speech: List[torch.Tensor] | torch.Tensor,
    speech_lengths: List[int] | torch.Tensor,
    device: torch.device,
) -> Dict[str, torch.Tensor]:
    """Prepare batched input for Wav2Vec2 model."""
    if isinstance(speech, torch.Tensor):
        speech = speech if speech.ndim == 1 else list(speech)
    # convert to list of trimmed numpy arrays
    batch = [
        x.detach().cpu().float().numpy().squeeze()[:xl]
        for x, xl in zip(speech, speech_lengths)
    ]
    inputs = feature_extractor(
        batch,
        sampling_rate=feature_extractor.sampling_rate,
        return_tensors="pt",
        padding=True,
    )
    return {
        "input_values": inputs.input_values.to(device),
        "attention_mask": inputs.attention_mask.to(device),
    }


class Wav2Vec2Model(nn.Module):

    def __init__(
        self,
        hf_repo: str,
        output_vocabsz: int = None,
        blank_id: int = 0,
        weighted_sum: bool = False,
    ):
        """
        Args:
            hf_repo: one of the following pretrained models
                facebook/mms-300m
                facebook/mms-1b
            output_vocabsz: If set, creates a CTC head with this vocab size.
            blank_id: Blank token ID for CTC
            weighted_sum: Whether to use a weighted sum of encoder layers for CTC
        """
        super().__init__()
        self.feature_extractor = Wav2Vec2FeatureExtractor.from_pretrained(hf_repo)
        self.model = Wav2Vec2ForCTC.from_pretrained(hf_repo)
        self.model.train()
        self.model_stride = np.prod(self.model.config.conv_stride)
        self.encoder_dim = self.model.config.output_hidden_size
        self.vocab_size = self.model.config.vocab_size
        self.sampling_rate = self.feature_extractor.sampling_rate
        self.weighted_sum = weighted_sum
        if self.weighted_sum:
            self.n_layers = self.model.config.num_hidden_layers
            assert (
                self.n_layers is not None and self.n_layers > 0
            ), "Cannot infer number of encoder layers for weighted_sum"
            self.layer_weights = torch.nn.Parameter(torch.zeros(int(self.n_layers)))
        # pad is the blank token for w2v2
        self.blank_id = blank_id
        if output_vocabsz is not None:
            self.model.lm_head = nn.Linear(
                self.encoder_dim,
                output_vocabsz,
            )
            self.vocab_size = output_vocabsz

    @torch.no_grad()
    def points_by_frames(self) -> int:
        """Get the ratio of input points to output frames."""
        return 320

    def _calculate_stats(self, output, inputs):
        """Token-level accuracy."""
        logits = output.logits.detach()
        if "target" not in inputs:
            return {}
        target = inputs["target"]
        if logits.ndim != target.ndim or logits.size(1) != target.size(1):
            raise ValueError(
                f"Logits and target size mismatch: {logits.size()} vs {target.size()}"
            )
        preds = logits.argmax(dim=-1)  # (B, L)
        if "target_length" in inputs:
            B, L = target.shape
            lengths = inputs["target_length"]
            idxs = torch.arange(L, device=target.device)[None, :].expand(B, L)
            mask = idxs < lengths.unsqueeze(1)
            correct = (preds == target) & mask
            acc = correct.sum().float() / mask.sum().clamp_min(1)
        else:
            acc = (preds == target).float().mean()

        return {"acc": acc}

    def forward(self, inputs) -> Any:
        model_out = self.model(
            **inputs,
            output_hidden_states=True,
            return_dict=True,
        )
        if self.weighted_sum:
            hs_list = model_out.hidden_states  # 25!=nlayers=24, +input
            w = torch.softmax(self.layer_weights, dim=0).to(
                hs_list[0].device, hs_list[0].dtype
            )
            hs = torch.stack(hs_list[-self.n_layers :], dim=0)  # (L, B, T, D)
            model_out["embedding"] = (w.view(-1, 1, 1, 1) * hs).sum(0)
        else:
            model_out["embedding"] = model_out.hidden_states[-1]
        stats = self._calculate_stats(model_out, inputs)
        model_out["stats"] = stats
        return model_out

    def _extract_feats(self, speech, speech_lengths) -> torch.Tensor:
        """Frontend"""
        inputs = preprocess_inputs_wav2vec2(
            self.feature_extractor, speech, speech_lengths, device=self.model.device
        )
        return inputs

    def encode(self, speech, speech_lengths) -> Tuple[torch.Tensor, torch.Tensor]:
        """Frontend + Encoder"""
        inputs = self._extract_feats(speech, speech_lengths)
        model_out = self(inputs)
        encoder_out_lens = self.model._get_feat_extract_output_lengths(
            inputs["attention_mask"].sum(-1)
        )
        encoder_out = model_out["embedding"]
        return encoder_out, encoder_out_lens

    def ctc_logits(self, speech, speech_lengths) -> Tuple[torch.Tensor, torch.Tensor]:
        """Get CTC logits from encoder output"""
        inputs = self._extract_feats(speech, speech_lengths)
        model_out = self(inputs)
        logits = model_out.logits
        logit_lengths = self.model._get_feat_extract_output_lengths(
            inputs["attention_mask"].sum(-1)
        )
        return logits, logit_lengths

    def encode_with_interctc(
        self,
        speech,
        speech_lengths,
        interctc_layer_idx: list,
        ctc,
        conditioning_layer=None,
    ):
        """Layer-by-layer encoder forward with intermediate CTC and optional self-conditioning.

        Replicates HF Wav2Vec2EncoderStableLayerNorm.forward() manually so conditioning
        can be injected between specific layers.

        Args:
            speech: raw waveform tensor
            speech_lengths: lengths tensor
            interctc_layer_idx: list of 1-based layer indices at which to capture intermediates
            ctc: CTC module with a .softmax() method
            conditioning_layer: optional nn.Linear to project CTC softmax back to hidden dim

        Returns:
            ((hidden_states, intermediate_outs), encoder_out_lens) if intermediates collected,
            else (hidden_states, encoder_out_lens)
        """
        inputs = self._extract_feats(speech, speech_lengths)
        input_values = inputs["input_values"]
        attention_mask = inputs["attention_mask"]
        wav2vec2 = self.model.wav2vec2

        extract_features = wav2vec2.feature_extractor(input_values).transpose(1, 2)

        feat_attn_mask = None
        if attention_mask is not None:
            feat_attn_mask = wav2vec2._get_feature_vector_attention_mask(
                extract_features.shape[1], attention_mask
            )

        hidden_states, _ = wav2vec2.feature_projection(extract_features)
        hidden_states = wav2vec2._mask_hidden_states(
            hidden_states, mask_time_indices=None, attention_mask=feat_attn_mask
        )

        encoder = wav2vec2.encoder
        if feat_attn_mask is not None:
            expand_mask = feat_attn_mask.unsqueeze(-1).expand_as(hidden_states)
            hidden_states = hidden_states.masked_fill(~expand_mask.bool(), 0.0)
        hidden_states = hidden_states + encoder.pos_conv_embed(hidden_states)
        hidden_states = encoder.dropout(hidden_states)

        # Convert boolean [B, T] mask to 4D additive float mask [B, 1, T, T] for attention
        # layers, matching what HF's Wav2Vec2EncoderStableLayerNorm.forward() does.
        layer_attn_mask = None
        if feat_attn_mask is not None:
            layer_attn_mask = 1.0 - feat_attn_mask[:, None, None, :].to(dtype=hidden_states.dtype)
            layer_attn_mask = layer_attn_mask * torch.finfo(hidden_states.dtype).min
            layer_attn_mask = layer_attn_mask.expand(
                layer_attn_mask.shape[0], 1, feat_attn_mask.shape[-1], feat_attn_mask.shape[-1]
            )

        intermediate_outs = []
        for layer_idx, layer in enumerate(encoder.layers):
            hidden_states = layer(hidden_states, layer_attn_mask, output_attentions=False)[0]
            if layer_idx + 1 in interctc_layer_idx:
                intermediate_outs.append((layer_idx + 1, hidden_states))
                if conditioning_layer is not None:
                    ctc_out = ctc.softmax(hidden_states)
                    hidden_states = hidden_states + conditioning_layer(ctc_out)

        hidden_states = encoder.layer_norm(hidden_states)
        encoder_out_lens = self.model._get_feat_extract_output_lengths(
            attention_mask.sum(-1)
        )

        if intermediate_outs:
            return (hidden_states, intermediate_outs), encoder_out_lens
        return hidden_states, encoder_out_lens

    def encoder_output_size(self) -> int:
        """Get output dimension"""
        return self.encoder_dim

    @torch.no_grad()
    def forced_align(self, speech, speech_lengths, text, text_lengths, utt_id=None):
        """Calculate frame-wise alignment from CTC probabilities.
        Only an inference function that uses the ctc posteriors.

        Args:
            speech: (Batch, Length, ...)
            speech_lengths: (Batch,)
            text: (Batch, Length)
            text_lengths: (Batch,)
            utt_id: str, identifier for the utterance
        Returns:
            Tuple(tensor, tensor):
                - Label for each time step in the alignment path computed
                using forced alignment.
                - Log probability scores of the labels for each time
                step.
        """
        assert text_lengths.dim() == 1, text_lengths.shape
        # Check that batch_size is unified
        assert (
            speech.shape[0]
            == speech_lengths.shape[0]
            == text.shape[0]
            == text_lengths.shape[0]
        ), (
            speech.shape,
            speech_lengths.shape,
            text.shape,
            text_lengths.shape,
        )
        batch_size = speech.shape[0]
        assert batch_size == 1, "Forced alignment needs batch size 1."

        # -1 is used as padding index in collate fn
        text = text[:, : text_lengths.max()]  # for data-parallel
        logits, logit_lengths = self.ctc_logits(speech, speech_lengths)
        log_probs = F.log_softmax(logits, dim=-1)  # (B, Tmax, odim)
        assert log_probs.size(0) == 1, "Forced alignment needs batch size 1"
        assert not (text == self.blank_id).any(), "Target has blank tokens."
        if log_probs.shape[1] < text.shape[1]:
            log.error(
                f"Logits length {log_probs.shape} is shorter than "
                f"text length {text.shape}, for utt_id: {utt_id}"
            )
        align_label, align_prob = torchaudio.functional.forced_align(
            log_probs, text, logit_lengths, text_lengths, blank=self.blank_id
        )
        return align_label, align_prob

    def get_blank_id(self) -> int:
        """Get blank id for CTC"""
        return self.blank_id


if __name__ == "__main__":
    # python -m src.model.wav2vec2.wav2vec2_model
    model = Wav2Vec2Model("facebook/mms-300m", weighted_sum=True)
    dummy_speech = [
        torch.randn(16000),
        torch.randn(8000),
    ]  # Batch of 2 samples, 1 sec, 0.5 sec at 16kHz
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    encoder_out, encoder_out_lens = model.encode(dummy_speech, [16000, 8000])
    print(f"Encoder output shape: {encoder_out.shape}")
    print(f"Encoder output lengths: {encoder_out_lens}")
