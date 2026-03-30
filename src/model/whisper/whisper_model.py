"""Whisper model wrapper for PhoneticXeus probing and forced alignment.

Provides the interface for PhoneticXeus probing and forced alignment:
- encode(speech, speech_lengths) -> (B, T, D), (B,)
- ctc_logits(speech, speech_lengths) -> (B, T, V), (B,)  [if output_vocabsz is set]
- forced_align(speech, speech_lengths, text, text_lengths) -> (align_label, align_prob)
- encoder_output_size(), points_by_frames(), get_blank_id(), sampling_rate

Usage:
    python -m src.model.whisper.whisper_model
"""

from typing import List, Tuple, Optional, Union

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchaudio
from transformers import WhisperProcessor, WhisperModel

from src.utils import RankedLogger

log = RankedLogger(__name__, rank_zero_only=True)


def preprocess_inputs_whisper(
    processor: WhisperProcessor,
    speech: Union[List[torch.Tensor], torch.Tensor],
    speech_lengths: Union[List[int], torch.Tensor],
    device: torch.device,
    sampling_rate: int = 16000,
) -> torch.Tensor:
    """Prepare batched input for Whisper model.

    Args:
        processor: WhisperProcessor instance.
        speech: (B, T) tensor or list of 1D tensors.
        speech_lengths: (B,) tensor or list of ints.
        device: target device.
        sampling_rate: audio sampling rate (default 16000).

    Returns:
        input_features: (B, n_mels, T_mel) tensor ready for Whisper encoder.
    """
    if isinstance(speech, torch.Tensor):
        speech = speech if speech.ndim == 1 else list(speech)

    # Convert to list of trimmed numpy arrays
    batch = [
        x.detach().cpu().float().numpy().squeeze()[: int(xl)]
        for x, xl in zip(speech, speech_lengths)
    ]

    inputs = processor(
        batch,
        sampling_rate=sampling_rate,
        return_tensors="pt",
        padding=True,
    )
    return inputs.input_features.to(device)


class WhisperEncoderModel(nn.Module):
    """Whisper encoder wrapper with optional CTC head for forced alignment.

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
            hf_repo: HuggingFace model ID (e.g., "openai/whisper-small").
            output_vocabsz: If set, creates a CTC head with this vocab size.
            blank_id: Blank token ID for CTC (default 0).
            freeze_encoder: Whether to freeze encoder weights (default True).
            encoder_layer: Which encoder layer to use (-1 = last, 0-indexed otherwise).
            cache_dir: Optional cache directory for HuggingFace model.
        """
        super().__init__()
        self.processor = WhisperProcessor.from_pretrained(hf_repo, cache_dir=cache_dir)
        self.model = WhisperModel.from_pretrained(hf_repo, cache_dir=cache_dir)
        self.encoder_layer = encoder_layer

        # Whisper config
        self.encoder_dim = self.model.config.d_model
        self.sampling_rate = 16000  # Whisper expects 16kHz
        self.blank_id = blank_id

        # Whisper mel-spectrogram: 80 mels, 10ms hop (160 samples at 16kHz)
        # Encoder conv subsampling: stride 2
        # => 20ms per encoder frame => 320 samples per frame
        self._points_by_frames = 320
        # HF Whisper encoder expects the mel time dimension to be exactly 2 * max_source_positions.
        # For standard Whisper configs: max_source_positions=1500 -> 3000 mel frames (30s).
        self._expected_mel_len = (
            int(getattr(self.model.config, "max_source_positions", 1500)) * 2
        )

        self.ctc_head: Optional[nn.Linear] = None
        if output_vocabsz is not None and output_vocabsz > 0:
            self.ctc_head = nn.Linear(self.encoder_dim, output_vocabsz)
            log.info(f"Created CTC head: {self.encoder_dim} -> {output_vocabsz}")

        # Freeze encoder if requested
        if freeze_encoder:
            self.model.encoder.eval()
            for param in self.model.encoder.parameters():
                param.requires_grad = False
            log.info("Whisper encoder frozen")

    @torch.no_grad()
    def points_by_frames(self) -> int:
        """Get the ratio of input samples to output frames.

        Whisper: 10ms mel hop * 2 (conv stride) = 20ms = 320 samples at 16kHz.
        """
        return self._points_by_frames

    def encoder_output_size(self) -> int:
        """Get encoder hidden dimension."""
        return self.encoder_dim

    def get_blank_id(self) -> int:
        """Get blank token ID for CTC."""
        return self.blank_id

    def _compute_encoder_lengths(self, speech_lengths: torch.Tensor) -> torch.Tensor:
        """Compute encoder output lengths from input speech lengths.

        Whisper processes up to 30s (480000 samples at 16kHz) and outputs
        1500 frames. For shorter inputs, output length scales proportionally.

        Args:
            speech_lengths: (B,) input lengths in samples.

        Returns:
            (B,) encoder output lengths in frames.
        """
        # Whisper max input: 30s = 480000 samples -> 1500 frames
        # Ratio: 480000 / 1500 = 320 samples per frame
        max_samples = 480000
        max_frames = 1500

        # Clamp to max length
        clamped = torch.clamp(speech_lengths, max=max_samples)
        # Compute output frames (ceil division to be safe)
        encoder_lens = torch.ceil(clamped.float() / self._points_by_frames).long()
        # Clamp to max frames
        encoder_lens = torch.clamp(encoder_lens, max=max_frames)
        return encoder_lens

    def _extract_feats(
        self, speech: torch.Tensor, speech_lengths: torch.Tensor
    ) -> torch.Tensor:
        """Extract mel-spectrogram features using WhisperProcessor.

        Args:
            speech: (B, T) waveform tensor.
            speech_lengths: (B,) lengths in samples.

        Returns:
            input_features: (B, n_mels, T_mel) tensor.
        """
        input_features = preprocess_inputs_whisper(
            self.processor,
            speech,
            speech_lengths,
            device=self.model.device,
            sampling_rate=self.sampling_rate,
        )
        # IMPORTANT: HF Whisper encoder enforces fixed mel length.
        # Pad/trim here so training/inference works for variable-length audio.
        cur_len = int(input_features.size(-1))
        if cur_len < self._expected_mel_len:
            input_features = F.pad(
                input_features, (0, self._expected_mel_len - cur_len)
            )
        elif cur_len > self._expected_mel_len:
            input_features = input_features[..., : self._expected_mel_len]
        return input_features

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
        """
        input_features = self._extract_feats(speech, speech_lengths)

        # Forward through encoder
        encoder_outputs = self.model.encoder(
            input_features,
            output_hidden_states=True,
            return_dict=True,
        )

        # Select layer
        if self.encoder_layer == -1:
            encoder_out = encoder_outputs.last_hidden_state
        else:
            encoder_out = encoder_outputs.hidden_states[self.encoder_layer]

        encoder_out_lens = self._compute_encoder_lengths(speech_lengths)
        # Ensure lengths don't exceed actual output
        encoder_out_lens = torch.clamp(encoder_out_lens, max=encoder_out.size(1))

        # IMPORTANT: Whisper encoder always returns a fixed number of frames (e.g., 1500),
        # but downstream pooling code (e.g., get_kv_pooling_mask) assumes the time dimension
        # equals max(valid_lengths) within the batch. Slice to keep shapes consistent.
        t_max = int(encoder_out_lens.max().item())
        t_max = max(1, min(t_max, int(encoder_out.size(1))))
        encoder_out = encoder_out[:, :t_max]
        encoder_out_lens = torch.clamp(encoder_out_lens, max=t_max)

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

        Uses torchaudio.functional.forced_align for CTC-based alignment.

        Args:
            speech: (1, T) waveform tensor (batch size must be 1).
            speech_lengths: (1,) length in samples.
            text: (1, L) target token IDs.
            text_lengths: (1,) target length.
            utt_id: Optional utterance ID for logging.

        Returns:
            align_label: (1, T_enc) frame-level labels.
            align_prob: (1, T_enc) frame-level log probabilities.

        Raises:
            RuntimeError: If CTC head is not configured.
            AssertionError: If batch size is not 1.
        """
        if self.ctc_head is None:
            raise RuntimeError(
                "CTC head not configured. Set output_vocabsz to enable forced_align."
            )

        assert text_lengths.dim() == 1, text_lengths.shape
        assert (
            speech.shape[0]
            == speech_lengths.shape[0]
            == text.shape[0]
            == text_lengths.shape[0]
        ), (speech.shape, speech_lengths.shape, text.shape, text_lengths.shape)
        batch_size = speech.shape[0]
        assert batch_size == 1, "Forced alignment needs batch size 1."

        # Trim text to valid length
        text = text[:, : text_lengths.max()]

        logits, logit_lengths = self.ctc_logits(speech, speech_lengths)
        # torchaudio.functional.forced_align requires input_lengths to match
        # log_probs.size(1) exactly. Whisper runs with fixed-length encoder output
        # (1500 frames) due to 30s padding, but our valid length can be shorter.
        # Trim to valid length to avoid "RuntimeError: input length mismatch".
        t_valid = int(logit_lengths.max().item())  # B=1 by design here
        t_valid = max(1, min(t_valid, int(logits.size(1))))
        logits = logits[:, :t_valid]
        logit_lengths = logit_lengths.new_full((logits.size(0),), t_valid)
        log_probs = F.log_softmax(logits, dim=-1)  # (1, T_valid, V)

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
