"""Builders for Whisper model in PhoneticXeus.

Usage:
    from src.model.whisper.builders import build_whisper_model

    model = build_whisper_model(
        hf_repo="openai/whisper-small",
        output_vocabsz=50,  # IPA vocab size for FA
    )
"""

from typing import Optional

from src.model.whisper.whisper_model import WhisperEncoderModel
from src.utils import RankedLogger

log = RankedLogger(__name__, rank_zero_only=True)


def build_whisper_model(
    hf_repo: str = "openai/whisper-small",
    output_vocabsz: Optional[int] = None,
    blank_id: int = 0,
    freeze_encoder: bool = False,
    encoder_layer: int = -1,
    cache_dir: Optional[str] = None,
) -> WhisperEncoderModel:
    """Build Whisper encoder model for PhoneticXeus.

    Args:
        hf_repo: HuggingFace model ID (e.g., "openai/whisper-small").
        output_vocabsz: If set, creates a CTC head with this vocab size.
            Required for forced alignment. Use IPATokenizer.vocab_size() for IPA.
        blank_id: Blank token ID for CTC (default 0, matching IPATokenizer).
        freeze_encoder: Whether to freeze encoder weights (default True).
        encoder_layer: Which encoder layer to use (-1 = last).
        cache_dir: Optional cache directory for HuggingFace model.

    Returns:
        WhisperEncoderModel instance.
    """
    model = WhisperEncoderModel(
        hf_repo=hf_repo,
        output_vocabsz=output_vocabsz,
        blank_id=blank_id,
        freeze_encoder=freeze_encoder,
        encoder_layer=encoder_layer,
        cache_dir=cache_dir,
    )
    log.info(f"Whisper model loaded from {hf_repo}")
    log.info(f"Encoder dim: {model.encoder_output_size()}")
    if output_vocabsz is not None:
        log.info(f"CTC head vocab size: {output_vocabsz}")
    return model
