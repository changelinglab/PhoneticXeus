from __future__ import annotations

from typing import List, Optional

from transformers import Wav2Vec2Processor, WavLMForCTC

from src.model.huper.huper_inference import HuPERInference
from src.utils import RankedLogger

log = RankedLogger(__name__, rank_zero_only=True)


def build_huper_inference(
    hf_repo: str = "huper29/huper_recognizer",
    device: str = "cpu",
    dtype: str = "float32",
    target_sampling_rate: int = 16000,
    cache_dir: Optional[str] = None,
    ignored_tokens: Optional[List[str]] = None,
) -> HuPERInference:
    """Build HuPER inference module."""
    processor = Wav2Vec2Processor.from_pretrained(hf_repo, cache_dir=cache_dir)
    model = WavLMForCTC.from_pretrained(hf_repo, cache_dir=cache_dir)
    inference_module = HuPERInference(
        model=model,
        processor=processor,
        device=device,
        dtype=dtype,
        target_sampling_rate=target_sampling_rate,
        ignored_tokens=ignored_tokens,
    )
    log.info("HuPER inference module built from %s", hf_repo)
    return inference_module
