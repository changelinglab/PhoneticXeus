"""HuPER inference adapter compatible with distributed inference API."""

from __future__ import annotations

from typing import Any, Iterable, List, Optional, Union

import numpy as np
import torch
import torchaudio
from transformers import Wav2Vec2Processor, WavLMForCTC
from src.core.ipa_utils import arpabet_to_ipa
from src.utils import RankedLogger

log = RankedLogger(__name__, rank_zero_only=True)


class HuPERInference:
    """Greedy CTC inference wrapper for HuPER (WavLMForCTC)."""

    def __init__(
        self,
        model: WavLMForCTC,
        processor: Wav2Vec2Processor,
        device: str = "cpu",
        dtype: str = "float32",
        target_sampling_rate: int = 16000,
        ignored_tokens: Optional[Iterable[str]] = None,
    ) -> None:
        resolved_device = self._resolve_device(device)
        self.device = torch.device(resolved_device)
        self.dtype = self._resolve_dtype(dtype, self.device)

        self.model = model.to(device=self.device, dtype=self.dtype).eval()
        self.processor = processor
        self.target_sampling_rate = int(target_sampling_rate)
        self.blank_id = self._resolve_blank_id()

        tokenizer = getattr(self.processor, "tokenizer", None)
        self.special_tokens = set(getattr(tokenizer, "all_special_tokens", []) or [])
        self.ignored_tokens = set(
            ignored_tokens or {"<PAD>", "<UNK>", "<BOS>", "<EOS>", "|"}
        )

    @staticmethod
    def _resolve_device(device: str) -> str:
        if device == "auto":
            return "cuda" if torch.cuda.is_available() else "cpu"
        return device

    @staticmethod
    def _resolve_dtype(dtype: str, device: torch.device) -> torch.dtype:
        torch_dtype = getattr(torch, dtype, None)
        if torch_dtype is None:
            raise ValueError(f"Unsupported dtype: {dtype}")

        if device.type == "cpu" and torch_dtype in {torch.float16, torch.bfloat16}:
            log.warning(
                "dtype=%s is unsupported/slow on CPU; falling back to float32.", dtype
            )
            return torch.float32
        return torch_dtype

    def _resolve_blank_id(self) -> int:
        tokenizer = getattr(self.processor, "tokenizer", None)
        blank_id = getattr(tokenizer, "pad_token_id", None)
        if blank_id is None:
            blank_id = getattr(self.model.config, "pad_token_id", None)
        return int(blank_id) if blank_id is not None else 0

    @staticmethod
    def _as_waveform_1d(
        speech: Union[torch.Tensor, np.ndarray, List[float]],
    ) -> torch.Tensor:
        if isinstance(speech, np.ndarray):
            waveform = torch.from_numpy(speech)
        elif isinstance(speech, torch.Tensor):
            waveform = speech
        else:
            waveform = torch.tensor(speech, dtype=torch.float32)

        waveform = waveform.detach().cpu().to(torch.float32)
        if waveform.dim() == 2:
            if waveform.size(0) == 1:
                waveform = waveform.squeeze(0)
            else:
                waveform = waveform.mean(dim=0)
        elif waveform.dim() != 1:
            raise ValueError(
                f"Expected speech with shape (T,) or (C, T), got {tuple(waveform.shape)}"
            )

        if waveform.numel() == 0:
            raise ValueError("Received empty speech tensor.")
        return waveform.contiguous()

    def _resolve_input_sampling_rate(self, kwargs: dict[str, Any]) -> int:
        for key in ("sampling_rate", "sample_rate", "sr", "orig_sr"):
            val = kwargs.get(key)
            if val is None:
                continue
            if isinstance(val, torch.Tensor):
                val = val.item()
            return int(val)
        return self.target_sampling_rate

    @staticmethod
    def ctc_collapse(token_ids: List[int], blank_id: int) -> List[int]:
        collapsed: List[int] = []
        prev: Optional[int] = None
        for token_id in token_ids:
            if token_id == blank_id:
                prev = token_id
                continue
            if token_id == prev:
                continue
            collapsed.append(token_id)
            prev = token_id
        return collapsed

    def _id_to_token(self, token_id: int) -> str:
        token: Optional[str] = None
        id2label = getattr(self.model.config, "id2label", None)
        if isinstance(id2label, dict):
            token = id2label.get(token_id)

        if token is None:
            tokenizer = getattr(self.processor, "tokenizer", None)
            if tokenizer is not None:
                token = tokenizer.convert_ids_to_tokens(token_id)

        return str(token) if token is not None else str(token_id)

    def _looks_like_special_token(self, token: str) -> bool:
        if token in self.special_tokens:
            return True
        return token.startswith("<") and token.endswith(">")

    @torch.no_grad()
    def __call__(
        self,
        speech: Union[torch.Tensor, np.ndarray, List[float]],
        **kwargs,
    ) -> List[dict[str, str]]:
        waveform = self._as_waveform_1d(speech)
        input_sr = self._resolve_input_sampling_rate(kwargs)
        if input_sr != self.target_sampling_rate:
            waveform = torchaudio.functional.resample(
                waveform.unsqueeze(0),
                orig_freq=input_sr,
                new_freq=self.target_sampling_rate,
            ).squeeze(0)

        inputs = self.processor(
            waveform.numpy(),
            sampling_rate=self.target_sampling_rate,
            return_tensors="pt",
        )
        inputs = {k: v.to(self.device) for k, v in inputs.items()}

        logits = self.model(**inputs).logits
        pred_ids = torch.argmax(logits, dim=-1)[0].tolist()
        collapsed_ids = self.ctc_collapse(pred_ids, blank_id=self.blank_id)

        raw_tokens: List[str] = []
        clean_tokens: List[str] = []
        for token_id in collapsed_ids:
            token = self._id_to_token(token_id)
            raw_tokens.append(token)
            if token in self.ignored_tokens or self._looks_like_special_token(token):
                continue
            clean_tokens.append(token)

        clean_ipa_tokens = arpabet_to_ipa(clean_tokens)

        return [
            {
                "processed_transcript": "".join(clean_ipa_tokens).strip(),
                "predicted_transcript": "/".join(raw_tokens),
            }
        ]
