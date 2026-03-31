"""POWSM-CTC model wrapper around ESPnet2 (OWSM-CTC).

PhoneticXeus's probing/forced-alignment pipeline expects the following interface from net:
- encode(speech, speech_lengths) -> (B, T', D), (B,)
- ctc_logits(speech, speech_lengths) -> (B, T', V), (B,)
- forced_align(speech, speech_lengths, text, text_lengths) -> (labels, logprobs)  (B=1)
- points_by_frames(), sampling_rate, get_blank_id(), encoder_output_size()

OWSM-CTC (espnet/owsm_ctc_v4_1B) requires prefix (<lang>, <task>) and prompt (text_prev)
as mandatory inputs to the encoder. This wrapper generates default values when PhoneticXeus
does not provide them.

Core principle: **Use upstream ESPnet2 implementation as-is** and only add a thin wrapper
for PhoneticXeus's needs.
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional, Tuple

import numpy as np
import torch
from typeguard import typechecked

from src.core.utils import download_hf_snapshot
from src.model.powsm.builders_common import (
    POWSM_CTC_REL_BPE,
    POWSM_CTC_REL_CONFIG,
    POWSM_CTC_REL_CKPT,
    POWSM_CTC_REL_STATS,
    resolve_model_paths,
)
from src.model.powsm.powsm_ctc_inference import patch_espnet_config_paths
from src.utils import RankedLogger

log = RankedLogger(__name__, rank_zero_only=False)


def _parse_humanfriendly_int(x: object) -> int:
    """Parse values like 16000 / 16000.0 / "16k" into int."""
    if x is None:
        raise ValueError("Cannot parse None as int")
    if isinstance(x, (int, np.integer)):
        return int(x)
    if isinstance(x, float):
        return int(x)
    if isinstance(x, str):
        s = x.strip().lower()
        if s.endswith("k"):
            return int(float(s[:-1]) * 1000)
        return int(float(s))
    return int(x)


class PowsmCTCNet(torch.nn.Module):
    """Thin wrapper exposing PhoneticXeus-friendly methods around ESPnetS2TCTCModel."""

    @typechecked
    def __init__(
        self,
        *,
        model: torch.nn.Module,
        training_args: object,
        default_lang_sym: str = "<unk>",
        default_task_sym: str = "<pr>",
    ):
        super().__init__()
        self.model = model
        self.training_args = training_args

        token_list = getattr(self.model, "token_list", None)
        if token_list is None:
            raise RuntimeError(
                "ESPnet model is missing token_list; cannot build wrapper."
            )
        self.token_list = list(token_list)
        self.token2id = {t: i for i, t in enumerate(self.token_list)}

        self.default_lang_sym = default_lang_sym
        self.default_task_sym = default_task_sym

        # Sampling rate (for forced-alignment time conversion)
        frontend_conf = getattr(self.training_args, "frontend_conf", {}) or {}
        fs = frontend_conf.get("fs", 16000)
        self.sampling_rate = _parse_humanfriendly_int(fs)

        # needed for upsampled forced alignment
        self.ctc = getattr(self.model, "ctc", None)
        self.ignore_id = getattr(self.model, "ignore_id", -1)

    def encoder_output_size(self) -> int:
        return int(self.model.encoder.output_size())

    @torch.no_grad()
    def points_by_frames(self) -> int:
        """points_per_frame = hop_length * subsample_factor (ESPnet OWSM-CTC convention)."""
        frontend_conf = getattr(self.training_args, "frontend_conf", {}) or {}
        encoder_conf = getattr(self.training_args, "encoder_conf", {}) or {}
        hop_length = _parse_humanfriendly_int(frontend_conf.get("hop_length", 160))
        input_layer = str(encoder_conf.get("input_layer", "conv2d"))
        subsample = {
            "conv2d1": 1,
            "conv2d2": 2,
            "conv2d": 4,
            "conv2d6": 6,
            "conv2d8": 8,
        }.get(input_layer, 4)
        return int(hop_length * subsample)

    def get_blank_id(self) -> int:
        return int(getattr(self.model, "blank_id"))

    def _make_defaults(
        self,
        batch_size: int,
        device: torch.device,
        *,
        lang_sym: Optional[str] = None,
        task_sym: Optional[str] = None,
    ):
        lang_sym = lang_sym or self.default_lang_sym
        task_sym = task_sym or self.default_task_sym
        if lang_sym not in self.token2id:
            raise KeyError(f"lang_sym {lang_sym!r} not in token_list")
        if task_sym not in self.token2id:
            raise KeyError(f"task_sym {task_sym!r} not in token_list")

        na_id = int(getattr(self.model, "na"))
        lang_id = int(self.token2id[lang_sym])
        task_id = int(self.token2id[task_sym])

        text_prev = torch.full((batch_size, 1), na_id, dtype=torch.long, device=device)
        text_prev_lengths = torch.full(
            (batch_size,), 1, dtype=torch.long, device=device
        )

        prefix = torch.tensor(
            [[lang_id, task_id]], dtype=torch.long, device=device
        ).repeat(batch_size, 1)
        prefix_lengths = torch.full((batch_size,), 2, dtype=torch.long, device=device)
        return text_prev, text_prev_lengths, prefix, prefix_lengths

    def encode(
        self,
        speech: torch.Tensor,
        speech_lengths: torch.Tensor,
        *,
        lang_sym: Optional[str] = None,
        task_sym: Optional[str] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Return encoder hidden states (final layer only) and lengths."""
        bsz = int(speech.size(0))
        device = speech.device
        text_prev, text_prev_lengths, prefix, prefix_lengths = self._make_defaults(
            bsz, device, lang_sym=lang_sym, task_sym=task_sym
        )
        enc, enc_lens = self.model.encode(
            speech,
            speech_lengths,
            text_prev,
            text_prev_lengths,
            prefix,
            prefix_lengths,
        )
        if isinstance(enc, tuple):
            enc = enc[0]
        return enc, enc_lens

    def ctc_logits(
        self, speech: torch.Tensor, speech_lengths: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        enc, enc_lens = self.encode(speech, speech_lengths)
        logits = self.model.ctc.ctc_lo(enc)
        return logits, enc_lens

    @torch.no_grad()
    def forced_align(self, speech, speech_lengths, text, text_lengths, utt_id=None):
        """CTC forced alignment for batch size 1."""
        assert speech.size(0) == 1, "Forced alignment needs batch size 1."
        assert text_lengths.dim() == 1, text_lengths.shape
        text = text[:, : int(text_lengths.max().item())]
        # Avoid -1 in valid region (torchaudio doesn't accept negative labels)
        eos_id = int(getattr(self.model, "eos"))
        text = torch.where(text < 0, torch.tensor(eos_id, device=text.device), text)

        enc, enc_lens = self.encode(speech, speech_lengths)
        align_label, align_prob = self.model.ctc.forced_align(
            enc, enc_lens, text, text_lengths, blank_idx=self.get_blank_id()
        )
        return align_label, align_prob


def build_powsm_ctc(
    *,
    work_dir: str,
    hf_repo: Optional[str] = None,
    force: bool = False,
    config_file: Optional[str] = None,
    model_file: Optional[str] = None,
    stats_file: Optional[str] = None,
    bpemodel: Optional[str] = None,
    default_lang_sym: str = "<unk>",
    default_task_sym: str = "<pr>",
) -> PowsmCTCNet:
    """Build a PhoneticXeus-ready POWSM-CTC net by wrapping ESPnet2's OWSM-CTC model."""

    # Optional: download HF snapshot into work_dir.
    if (
        hf_repo
        and (config_file is None)
        and (model_file is None)
        and (stats_file is None)
    ):
        root = Path(work_dir)
        required_rel = [
            POWSM_CTC_REL_CONFIG,
            POWSM_CTC_REL_CKPT,
            POWSM_CTC_REL_STATS,
            POWSM_CTC_REL_BPE,
        ]
        missing = [str(root / r) for r in required_rel if not (root / r).exists()]
        if force or missing:
            # If a local snapshot exists but is incomplete, we must allow network
            # download to fetch missing large files (e.g., the .pth checkpoint).
            download_hf_snapshot(
                repo_id=hf_repo,
                work_dir=work_dir,
                force_download=True if missing else force,
            )
            missing = [str(root / r) for r in required_rel if not (root / r).exists()]
        if missing:
            raise RuntimeError(
                "HuggingFace snapshot is incomplete under work_dir. "
                f"work_dir={work_dir}, hf_repo={hf_repo}, missing={missing}. "
                "Try: (1) rerun with force=true, or (2) delete the cache dir and retry."
            )

    cfg_path, mdl_path, stats_path = resolve_model_paths(
        work_dir=work_dir,
        hf_repo=None,  # snapshot is handled above; keep this pure/fail-fast
        force_download=force,
        config_file=config_file,
        model_file=model_file,
        stats_file=stats_file,
        rel_config=POWSM_CTC_REL_CONFIG,
        rel_ckpt=POWSM_CTC_REL_CKPT,
        rel_stats=POWSM_CTC_REL_STATS,
    )
    root = Path(work_dir)
    bpe_path = bpemodel or str(root / POWSM_CTC_REL_BPE)
    if not Path(bpe_path).exists():
        log.warning(
            f"BPE model not found at {bpe_path}; continuing (model build itself does not require it)."
        )

    patched_cfg = str(Path(work_dir) / ".phoneticxeus" / "powsm_ctc_patched_config.yaml")
    patched_cfg = patch_espnet_config_paths(
        original_config_path=cfg_path,
        stats_file=stats_path,
        bpemodel=bpe_path,
        output_path=patched_cfg,
    )

    from src.espnet_import.minimal_s2t_inference_ctc import _build_model_from_file

    # Keep builder consistent with other PhoneticXeus nets: load on CPU by default.
    model, train_args = _build_model_from_file(patched_cfg, mdl_path, device="cpu")
    return PowsmCTCNet(
        model=model,
        training_args=train_args,
        default_lang_sym=default_lang_sym,
        default_task_sym=default_task_sym,
    )
