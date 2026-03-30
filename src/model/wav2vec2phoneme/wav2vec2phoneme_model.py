"""Wav2Vec2Phoneme model implementation using Hugging Face Transformers.
Functionalities:
1. TODO(shikhar): Model fine-tuning with ctc loss

This file supports the following pretrained models:
- facebook/wav2vec2-lv-60-espeak-cv-ft
- facebook/wav2vec2-xlsr-53-espeak-cv-ft
- ctaguchi/wav2vec2-large-xlsr-japlmthufielta-ipa1000-ns

Note:
"facebook" models use phonemizer which needs espeak-ng
If you see an error like: "TypeError: Received a bool for argument tokenizer, but a PreTrainedTokenizerBase was expected"
Build espeak-ng following https://github.com/espeak-ng/espeak-ng/blob/master/docs/building.md
1. git clone https://github.com/espeak-ng/espeak-ng.git
2. cd espeak-ng
3. ./autogen.sh
4. ./configure --prefix=PREFIX (to store built files in PREFIX)
5. make
Then export the following paths:
export PHONEMIZER_ESPEAK_LIBRARY="path/to/cloned/espeak-ng/src/.libs/libespeak-ng.so.1.1.51"
export ESPEAK_DATA_PATH="path/to/cloned/espeak-ng/espeak-ng-data"
Both of these exports are necessary.

Usage:
    python -m src.model.wav2vec2phoneme.wav2vec2phoneme_model

"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchaudio
from transformers import Wav2Vec2Processor, Wav2Vec2ForCTC
from typing import Dict, List, Tuple, Any
import numpy as np
from src.utils import RankedLogger

log = RankedLogger(__name__, rank_zero_only=True)


def preprocess_inputs_wav2vec2(
    preprocessor: Wav2Vec2Processor,
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

    inputs = preprocessor(
        batch,
        sampling_rate=preprocessor.feature_extractor.sampling_rate,
        return_tensors="pt",
        padding=True,
    )
    return {
        "input_values": inputs.input_values.to(device),
        "attention_mask": inputs.attention_mask.to(device),
    }


class Wav2Vec2PhonemeModel(nn.Module):

    def __init__(self, hf_repo: str):
        """
        Args:
            hf_repo: one of the following pretrained models
                facebook/wav2vec2-lv-60-espeak-cv-ft
                facebook/wav2vec2-xlsr-53-espeak-cv-ft
                ctaguchi/wav2vec2-large-xlsr-japlmthufielta-ipa1000-ns
        """
        super().__init__()
        self.processor = Wav2Vec2Processor.from_pretrained(hf_repo)
        self.model = Wav2Vec2ForCTC.from_pretrained(hf_repo)
        self.model_stride = np.prod(self.model.config.conv_stride)
        self.encoder_dim = self.model.config.output_hidden_size
        self.vocab_size = self.model.config.vocab_size
        # Fixed!
        self.sampling_rate = self.processor.feature_extractor.sampling_rate
        # pad is the blank token for w2v2
        self.blank_id = self.model.config.pad_token_id

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
        """Forward pass compatible with PowsmModel interface"""
        model_out = self.model(
            **inputs,
            output_hidden_states=True,
            return_dict=True,
        )
        stats = self._calculate_stats(model_out, inputs)
        model_out["stats"] = stats
        return model_out

    def _extract_feats(self, speech, speech_lengths) -> torch.Tensor:
        """Frontend"""
        inputs = preprocess_inputs_wav2vec2(
            self.processor, speech, speech_lengths, device=self.model.device
        )
        return inputs

    def encode(self, speech, speech_lengths) -> Tuple[torch.Tensor, torch.Tensor]:
        """Frontend + Encoder"""
        inputs = self._extract_feats(speech, speech_lengths)
        model_out = self(inputs)
        encoder_out = model_out.hidden_states[-1]
        encoder_out_lens = self.model._get_feat_extract_output_lengths(
            inputs["attention_mask"].sum(-1)
        )
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
    # Example usage
    # model = Wav2Vec2PhonemeModel(
    #     "ctaguchi/wav2vec2-large-xlsr-japlmthufielta-ipa1000-ns"
    # )
    model = Wav2Vec2PhonemeModel("facebook/wav2vec2-lv-60-espeak-cv-ft")
    dummy_speech = [
        torch.randn(16000),
        torch.randn(8000),
    ]  # Batch of 2 samples, 1 sec, 0.5 sec at 16kHz
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    encoder_out, encoder_out_lens = model.encode(dummy_speech, [16000, 8000])
    print(f"Encoder output shape: {encoder_out.shape}")
    print(f"Encoder output lengths: {encoder_out_lens}")
