"""WavLM model wrapper for PhoneticXeus probing and forced alignment.

Provides the interface for PhoneticXeus probing and forced alignment:
- encode(speech, speech_lengths) -> (B, T, D), (B,)
- ctc_logits(speech, speech_lengths) -> (B, T, V), (B,)  [if output_vocabsz is set]
- forced_align(speech, speech_lengths, text, text_lengths) -> (align_label, align_prob)
- encoder_output_size(), points_by_frames(), get_blank_id(), sampling_rate

Notes:
    - This implementation requires `attention_mask` from the processor output.
      We explicitly request it with `return_attention_mask=True`. If the processor
      does not return it, a RuntimeError is raised (rather than silently guessing
      lengths), because forced alignment and downstream pooling depend on correct
      time lengths.
"""

from typing import Dict, List, Tuple, Optional, Union

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchaudio
from transformers import AutoProcessor, AutoFeatureExtractor, WavLMModel

from src.utils import RankedLogger

log = RankedLogger(__name__, rank_zero_only=True)


def preprocess_inputs_wavlm(
    processor: AutoProcessor,
    speech: Union[List[torch.Tensor], torch.Tensor],
    speech_lengths: Union[List[int], torch.Tensor],
    device: torch.device,
    sampling_rate: int = 16000,
) -> Dict[str, torch.Tensor]:
    """Prepare batched input for WavLM model.

    Args:
        processor: AutoProcessor instance.
        speech: (B, T) tensor or list of 1D tensors.
        speech_lengths: (B,) tensor or list of ints.
        device: target device.
        sampling_rate: audio sampling rate (default 16000).

    Returns:
        Dict with input_values and attention_mask tensors.

    Raises:
        RuntimeError: If the processor does not return attention_mask.
    """
    if isinstance(speech, torch.Tensor):
        speech = speech if speech.ndim == 1 else list(speech)

    # Convert to list of trimmed numpy arrays
    batch = [
        x.detach().cpu().float().numpy().squeeze()[: int(xl)]
        for x, xl in zip(speech, speech_lengths)
    ]

    try:
        inputs = processor(
            batch,
            sampling_rate=sampling_rate,
            return_tensors="pt",
            padding=True,
            return_attention_mask=True,
        )
    except TypeError as e:
        raise RuntimeError(
            "AutoProcessor did not accept `return_attention_mask=True`. "
            "WavLMEncoderModel requires an attention_mask to compute output lengths."
        ) from e

    attention_mask = getattr(inputs, "attention_mask", None)
    if attention_mask is None:
        raise RuntimeError(
            "AutoProcessor did not return `attention_mask` (even with "
            "return_attention_mask=True). WavLMEncoderModel requires attention_mask "
            "to compute output lengths."
        )
    return {
        "input_values": inputs.input_values.to(device),
        "attention_mask": attention_mask.to(device),
    }


class WavLMEncoderModel(nn.Module):
    """WavLM encoder wrapper with optional CTC head for forced alignment.

    Attributes:
        sampling_rate: Expected input sampling rate (16000 Hz).
        blank_id: Blank token ID for CTC (default 0, matching IPATokenizer).
    """

    def __init__(
        self,
        hf_repo: str,
        output_vocabsz: Optional[int] = None,
        blank_id: int = 0,
        freeze_encoder: bool = True,
        encoder_layer: int = -1,
        cache_dir: Optional[str] = None,
    ):
        """
        Args:
            hf_repo: HuggingFace model ID (e.g., "microsoft/wavlm-base").
            output_vocabsz: If set, creates a CTC head with this vocab size.
            blank_id: Blank token ID for CTC (default 0).
            freeze_encoder: Whether to freeze encoder weights (default True).
            encoder_layer: Which encoder layer to use (-1 = last, 0-indexed otherwise).
            cache_dir: Optional cache directory for HuggingFace model.
        """
        super().__init__()
        # NOTE: Many WavLM HF repos (e.g., microsoft/wavlm-base) do NOT ship a tokenizer/vocab.
        # AutoProcessor may still try to construct a tokenizer and crash with
        # `TypeError: expected str, bytes or os.PathLike object, not NoneType`.
        # PhoneticXeus uses WavLM as an *encoder feature extractor*; we don't need text tokenization.
        # Therefore, fall back to an audio-only feature extractor when AutoProcessor fails.
        self.feature_extractor = None
        try:
            self.processor = AutoProcessor.from_pretrained(hf_repo, cache_dir=cache_dir)
            self.feature_extractor = getattr(self.processor, "feature_extractor", None)
            if self.feature_extractor is None:
                raise RuntimeError(
                    "Loaded AutoProcessor does not expose `feature_extractor`."
                )
        except Exception as e:
            log.warning(
                "Failed to load AutoProcessor (likely due to missing tokenizer/vocab files). "
                f"Falling back to AutoFeatureExtractor. hf_repo={hf_repo}. Error: {e}"
            )
            try:
                self.processor = AutoFeatureExtractor.from_pretrained(
                    hf_repo, cache_dir=cache_dir
                )
            except Exception as e2:
                raise RuntimeError(
                    f"Failed to load both AutoProcessor and AutoFeatureExtractor from hf_repo={hf_repo}."
                ) from e2
            self.feature_extractor = self.processor

        self.model = WavLMModel.from_pretrained(hf_repo, cache_dir=cache_dir)
        self.encoder_layer = encoder_layer

        # WavLM config
        self.encoder_dim = self.model.config.hidden_size
        self.sampling_rate = int(
            getattr(self.feature_extractor, "sampling_rate", 16000)
        )
        self.blank_id = blank_id

        # Compute points_by_frames from conv_stride
        # WavLM uses convolutional feature extractor with specific strides
        # Default: [5, 2, 2, 2, 2, 2, 2] -> product = 320
        conv_stride = getattr(self.model.config, "conv_stride", [5, 2, 2, 2, 2, 2, 2])
        self._points_by_frames = int(np.prod(conv_stride))

        self.ctc_head: Optional[nn.Linear] = None
        if output_vocabsz is not None and output_vocabsz > 0:
            self.ctc_head = nn.Linear(self.encoder_dim, output_vocabsz)
            log.info(f"Created CTC head: {self.encoder_dim} -> {output_vocabsz}")

        # Freeze encoder if requested
        if freeze_encoder:
            self.model.eval()
            for param in self.model.parameters():
                param.requires_grad = False
            log.info("WavLM encoder frozen")

    @torch.no_grad()
    def points_by_frames(self) -> int:
        """Get the ratio of input samples to output frames.

        WavLM: conv_stride product (typically 320 samples at 16kHz = 20ms per frame).
        """
        return self._points_by_frames

    def encoder_output_size(self) -> int:
        """Get encoder hidden dimension."""
        return self.encoder_dim

    def get_blank_id(self) -> int:
        """Get blank token ID for CTC."""
        return self.blank_id

    def _extract_feats(
        self, speech: torch.Tensor, speech_lengths: torch.Tensor
    ) -> Dict[str, torch.Tensor]:
        """Extract input features using AutoProcessor.

        Args:
            speech: (B, T) waveform tensor.
            speech_lengths: (B,) lengths in samples.

        Returns:
            Dict with input_values and attention_mask tensors.
        """
        inputs = preprocess_inputs_wavlm(
            self.processor,
            speech,
            speech_lengths,
            device=self.model.device,
            sampling_rate=self.sampling_rate,
        )
        return inputs

    def encode(
        self, speech: torch.Tensor, speech_lengths: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Extract encoder hidden states.

        Args:
            speech: (B, T) waveform tensor.
            speech_lengths: (B,) lengths in samples.

        Returns:
            encoder_out: (B, T_enc, D) hidden states.
            encoder_out_lens: (B,) output lengths in frames.

        Raises:
            ValueError: If encoder_layer is invalid/out of range for the returned hidden_states.
        """
        inputs = self._extract_feats(speech, speech_lengths)

        # Forward through encoder
        outputs = self.model(
            input_values=inputs["input_values"],
            attention_mask=inputs["attention_mask"],
            output_hidden_states=True,
            return_dict=True,
        )

        # Select layer
        if self.encoder_layer == -1:
            encoder_out = outputs.last_hidden_state
        else:
            if self.encoder_layer < 0:
                raise ValueError(
                    f"encoder_layer must be -1 (last) or >= 0, but got {self.encoder_layer}"
                )
            hidden_states = getattr(outputs, "hidden_states", None)
            if hidden_states is None:
                raise ValueError(
                    "Model output did not include hidden_states (output_hidden_states=True was requested). "
                    "Cannot select encoder_layer."
                )
            num_hidden_states = len(hidden_states)
            if self.encoder_layer >= num_hidden_states:
                raise ValueError(
                    f"encoder_layer out of range: got {self.encoder_layer}, "
                    f"but hidden_states has length {num_hidden_states} (valid: 0..{num_hidden_states - 1})."
                )
            encoder_out = hidden_states[self.encoder_layer]

        # Compute output lengths using model's internal method
        encoder_out_lens = self.model._get_feat_extract_output_lengths(
            inputs["attention_mask"].sum(-1)
        )

        # WavLM/Wav2Vec2-style encoders return a time dimension determined by the
        # padded (max) input length in the batch. Therefore, encoder_out.size(1)
        # should match max(encoder_out_lens). If it doesn't, our attention_mask
        # or length computation is likely wrong, and we should not silently
        # "fix" it by slicing.
        t_out = int(encoder_out.size(1))
        t_expected = int(encoder_out_lens.max().item())
        if t_out != t_expected:
            log.warning(
                "Encoder time dimension mismatch: "
                f"encoder_out.size(1)={t_out}, max(encoder_out_lens)={t_expected}. "
                "This may indicate a bug in attention_mask/length computation."
            )
        assert t_out == t_expected, (
            f"Encoder time dimension mismatch: encoder_out.size(1)={t_out} "
            f"!= max(encoder_out_lens)={t_expected}"
        )

        return encoder_out, encoder_out_lens.to(encoder_out.device)

    def ctc_logits(
        self, speech: torch.Tensor, speech_lengths: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Get CTC logits from encoder output.

        Args:
            speech: (B, T) waveform tensor.
            speech_lengths: (B,) lengths in samples.

        Returns:
            logits: (B, T_enc, V) CTC logits.
            logit_lengths: (B,) output lengths.

        Raises:
            RuntimeError: If CTC head is not configured.
        """
        if self.ctc_head is None:
            raise RuntimeError(
                "CTC head not configured. Set output_vocabsz to enable ctc_logits."
            )

        encoder_out, encoder_out_lens = self.encode(speech, speech_lengths)
        logits = self.ctc_head(encoder_out)
        return logits, encoder_out_lens

    @torch.no_grad()
    def forced_align(
        self,
        speech: torch.Tensor,
        speech_lengths: torch.Tensor,
        text: torch.Tensor,
        text_lengths: torch.Tensor,
        utt_id: Optional[str] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Calculate frame-wise alignment from CTC probabilities.

        Args:
            speech: (Batch, Length) waveform tensor.
            speech_lengths: (Batch,) lengths in samples.
            text: (Batch, Length) tokenized text.
            text_lengths: (Batch,) text lengths.
            utt_id: Optional utterance identifier for logging.

        Returns:
            align_label: Frame-wise alignment labels.
            align_prob: Log probability scores for each frame.

        Raises:
            RuntimeError: If CTC head is not configured.
            AssertionError: If batch size is not 1.
        """
        if self.ctc_head is None:
            raise RuntimeError(
                "CTC head not configured. Set output_vocabsz to enable forced_align."
            )

        assert text_lengths.dim() == 1, text_lengths.shape
        # Check batch dimensions match
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

        # Trim text to actual length
        text = text[:, : text_lengths.max()]
        logits, logit_lengths = self.ctc_logits(speech, speech_lengths)
        log_probs = F.log_softmax(logits, dim=-1)

        assert log_probs.size(0) == 1, "Forced alignment needs batch size 1"
        assert not (text == self.blank_id).any(), "Target has blank tokens."

        if log_probs.shape[1] < text.shape[1]:
            log.error(
                f"Logits length {log_probs.shape} is shorter than "
                f"text length {text.shape}, for utt_id: {utt_id}"
            )

        # torchaudio forced_align expects input_lengths to match time dimension
        # Clamp to actual sequence length
        logit_lengths = torch.clamp(logit_lengths, max=log_probs.size(1))

        align_label, align_prob = torchaudio.functional.forced_align(
            log_probs, text, logit_lengths, text_lengths, blank=self.blank_id
        )
        return align_label, align_prob
