"""POWSM-CTC model wrapper around ESPnet2 (OWSM-CTC).

PhoneticXeus's probing/forced-alignment pipeline expects the following interface:
- encode(speech, speech_lengths) -> (B, T', D), (B,)
- ctc_logits(speech, speech_lengths) -> (B, T', V), (B,)
- forced_align(speech, speech_lengths, text, text_lengths) -> (labels, logprobs)
- points_by_frames(), sampling_rate, get_blank_id(), encoder_output_size()
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import List, Optional, Tuple, Union

import numpy as np
import torch
import yaml
from typeguard import typechecked

from src.core.utils import download_hf_snapshot
from src.espnet_import.embedding import PositionalEncoding
from src.espnet_import.nets_utils import make_pad_mask
from src.model.powsm.builders_common import (
    POWSM_CTC_REL_BPE,
    POWSM_CTC_REL_CONFIG,
    POWSM_CTC_REL_CKPT,
    POWSM_CTC_REL_STATS,
    build_ctc,
    build_frontend,
    build_normalize,
    build_specaug,
    load_token_list,
    patch_espnet_config_paths,
    resolve_model_paths,
)
from src.model.powsm.e_branchformer_ctc import EBranchformerCTCEncoder
from src.model.powsm.prompt_encoder import PromptEncoder
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


class PowsmCTCModel(torch.nn.Module):
    """Intermediate-CTC model (inference subset only)."""

    @typechecked
    def __init__(
        self,
        vocab_size: int,
        token_list: Union[Tuple[str, ...], List[str]],
        frontend,
        specaug,
        normalize,
        encoder,
        prompt_encoder,
        ctc,
        interctc_weight: float = 0.0,
        ignore_id: int = -1,
        sym_space: str = "<space>",
        sym_blank: str = "<blank>",
        sym_sos: str = "<sos>",
        sym_eos: str = "<eos>",
        sym_sop: str = "<sop>",
        sym_na: str = "<na>",
        **_,
    ):
        super().__init__()
        self.blank_id = token_list.index(sym_blank)
        self.sos = token_list.index(sym_sos)
        self.eos = token_list.index(sym_eos)
        self.sop = token_list.index(sym_sop)
        self.na = token_list.index(sym_na)
        self.vocab_size = vocab_size
        self.ignore_id = ignore_id
        self.interctc_weight = interctc_weight
        self.token_list = list(token_list)

        self.frontend = frontend
        self.specaug = specaug
        self.normalize = normalize
        self.encoder = encoder
        self.prompt_encoder = prompt_encoder
        self.ctc = ctc

        prompt_size = self.prompt_encoder.output_size()
        self.embed = torch.nn.Embedding(vocab_size, prompt_size)
        self.pos_enc = PositionalEncoding(prompt_size, 0.0)

        enc_size = self.encoder.output_size()
        if enc_size != prompt_size:
            self.embed_proj = torch.nn.Linear(prompt_size, enc_size)
            self.prompt_proj = torch.nn.Linear(prompt_size, enc_size)
        else:
            self.embed_proj = torch.nn.Identity()
            self.prompt_proj = torch.nn.Identity()

        if not hasattr(self.encoder, "interctc_use_conditioning"):
            self.encoder.interctc_use_conditioning = False
        if self.encoder.interctc_use_conditioning:
            self.encoder.conditioning_layer = torch.nn.Linear(
                vocab_size, enc_size
            )

        self.is_encoder_whisper = "Whisper" in type(self.encoder).__name__

    def encode(
        self,
        speech: torch.Tensor,
        speech_lengths: torch.Tensor,
        text_prev: torch.Tensor,
        text_prev_lengths: torch.Tensor,
        prefix: torch.Tensor,
        prefix_lengths: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        text_prev[text_prev == -1] = self.eos
        memory, memory_lengths, _ = self.prompt_encoder(
            self.pos_enc(self.embed(text_prev)), text_prev_lengths
        )
        memory_mask = (
            (~make_pad_mask(memory_lengths)[:, None, :]).to(memory.device)
        )

        feats, feats_lengths = self._extract_feats(speech, speech_lengths)
        if self.normalize is not None:
            feats, feats_lengths = self.normalize(feats, feats_lengths)

        encoder_out, encoder_out_lens, _ = self.encoder(
            feats,
            feats_lengths,
            ctc=self.ctc,
            prefix_embeds=self.embed_proj(self.embed(prefix)),
            memory=self.prompt_proj(memory),
            memory_mask=memory_mask,
        )
        return encoder_out, encoder_out_lens

    def _extract_feats(
        self, speech: torch.Tensor, speech_lengths: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        speech = speech[:, : int(speech_lengths.max().item())]
        if self.frontend is not None:
            return self.frontend(speech, speech_lengths)
        return speech, speech_lengths


class PowsmCTCNet(torch.nn.Module):
    """Thin wrapper exposing PhoneticXeus-friendly methods around PowsmCTCModel."""

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
        """points_per_frame = hop_length * subsample_factor."""
        frontend_conf = getattr(self.training_args, "frontend_conf", {}) or {}
        encoder_conf = getattr(self.training_args, "encoder_conf", {}) or {}
        hop_length = _parse_humanfriendly_int(frontend_conf.get("hop_length", 160))
        input_layer = str(encoder_conf.get("input_layer", "conv2d"))
        subsample = {
            "conv2d1": 1, "conv2d2": 2, "conv2d": 4, "conv2d6": 6, "conv2d8": 8,
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

        text_prev = torch.full(
            (batch_size, 1), na_id, dtype=torch.long, device=device
        )
        text_prev_lengths = torch.full(
            (batch_size,), 1, dtype=torch.long, device=device
        )
        prefix = torch.tensor(
            [[lang_id, task_id]], dtype=torch.long, device=device
        ).repeat(batch_size, 1)
        prefix_lengths = torch.full(
            (batch_size,), 2, dtype=torch.long, device=device
        )
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
            speech, speech_lengths,
            text_prev, text_prev_lengths,
            prefix, prefix_lengths,
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
        eos_id = int(getattr(self.model, "eos"))
        text = torch.where(
            text < 0, torch.tensor(eos_id, device=text.device), text
        )

        enc, enc_lens = self.encode(speech, speech_lengths)
        align_label, align_prob = self.model.ctc.forced_align(
            enc, enc_lens, text, text_lengths, blank_idx=self.get_blank_id()
        )
        return align_label, align_prob


def _build_powsm_ctc_model(args: argparse.Namespace) -> PowsmCTCModel:
    """Build PowsmCTCModel from config args."""
    token_list = load_token_list(args.token_list)
    vocab_size = len(token_list)

    frontend, input_size = build_frontend(args)
    specaug = build_specaug(args) if getattr(args, "specaug", None) else None
    normalize = build_normalize(args, args.normalize_conf["stats_file"])

    encoder = EBranchformerCTCEncoder(input_size=input_size, **args.encoder_conf)
    prompt_encoder = PromptEncoder(
        input_size=args.promptencoder_conf["output_size"],
        **args.promptencoder_conf,
    )
    ctc = build_ctc(vocab_size, encoder.output_size(), getattr(args, "ctc_conf", {}))

    return PowsmCTCModel(
        vocab_size=vocab_size,
        token_list=token_list,
        frontend=frontend,
        specaug=specaug,
        normalize=normalize,
        encoder=encoder,
        prompt_encoder=prompt_encoder,
        ctc=ctc,
        **getattr(args, "model_conf", {}),
    )


def build_powsm_ctc_from_files(
    config_file: str,
    model_file: str,
    device: str = "cpu",
    default_lang_sym: str = "<unk>",
    default_task_sym: str = "<pr>",
) -> PowsmCTCNet:
    """Build PowsmCTCNet from a (patched) config file and model checkpoint.

    Analogous to build_powsm_from_files in powsm_model.py.

    Args:
        config_file: Path to YAML config (with absolute stats/bpe paths).
        model_file: Path to .pth checkpoint.
        device: Device to load onto.
        default_lang_sym: Fallback language symbol for PowsmCTCNet.
        default_task_sym: Fallback task symbol for PowsmCTCNet.

    Returns:
        PowsmCTCNet wrapping the loaded PowsmCTCModel.
    """
    with open(config_file, "r", encoding="utf-8") as f:
        args = argparse.Namespace(**yaml.safe_load(f))

    model = _build_powsm_ctc_model(args)
    model.to(device)

    state = torch.load(model_file, map_location=device, weights_only=False)
    if "module" in state:
        state = state["module"]
    model.load_state_dict(state, strict=False)
    log.info(f"Loaded POWSM-CTC checkpoint: {model_file}")

    return PowsmCTCNet(
        model=model,
        training_args=args,
        default_lang_sym=default_lang_sym,
        default_task_sym=default_task_sym,
    )


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
    if (
        hf_repo
        and (config_file is None)
        and (model_file is None)
        and (stats_file is None)
    ):
        root = Path(work_dir)
        required_rel = [
            POWSM_CTC_REL_CONFIG, POWSM_CTC_REL_CKPT,
            POWSM_CTC_REL_STATS, POWSM_CTC_REL_BPE,
        ]
        missing = [str(root / r) for r in required_rel if not (root / r).exists()]
        if force or missing:
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
        hf_repo=None,
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
            f"BPE model not found at {bpe_path}; "
            "continuing (model build itself does not require it)."
        )

    patched_cfg = str(
        Path(work_dir) / ".phoneticxeus" / "powsm_ctc_patched_config.yaml"
    )
    patched_cfg = patch_espnet_config_paths(
        original_config_path=cfg_path,
        stats_file=stats_path,
        bpemodel=bpe_path,
        output_path=patched_cfg,
    )

    return build_powsm_ctc_from_files(
        patched_cfg, mdl_path,
        default_lang_sym=default_lang_sym,
        default_task_sym=default_task_sym,
    )
