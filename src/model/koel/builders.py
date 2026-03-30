from __future__ import annotations

from typing import List, Optional

from transformers import AutoModelForCTC, AutoProcessor

from src.model.koel.koel_inference import KoelInference
from src.utils import RankedLogger

log = RankedLogger(__name__, rank_zero_only=True)


def build_koel_inference(
    hf_repo: str = "KoelLabs/xlsr-english-01",
    device: str = "cpu",
    dtype: str = "float32",
    target_sampling_rate: int = 16000,
    cache_dir: Optional[str] = None,
    token: Optional[str] = None,
    ignored_tokens: Optional[List[str]] = None,
) -> KoelInference:
    """Build KoelLabs XLSR inference module."""
    processor = AutoProcessor.from_pretrained(
        hf_repo,
        cache_dir=cache_dir,
        token=token,
    )
    model = AutoModelForCTC.from_pretrained(
        hf_repo,
        cache_dir=cache_dir,
        token=token,
    )
    inference_module = KoelInference(
        model=model,
        processor=processor,
        device=device,
        dtype=dtype,
        target_sampling_rate=target_sampling_rate,
        ignored_tokens=ignored_tokens,
    )
    log.info("Koel inference module built from %s", hf_repo)
    return inference_module
