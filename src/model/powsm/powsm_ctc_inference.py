"""POWSM-CTC Inference (wrapping ESPnet).

This module intentionally keeps PhoneticXeus's Hydra/distributed inference contract,
but delegates actual greedy CTC decoding (short-form + long-form buffered) to ESPnet.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import numpy as np
import torch
import yaml
from typeguard import typechecked

from src.core.utils import download_hf_snapshot
from src.model.powsm.builders_common import (
    POWSM_CTC_REL_BPE,
    POWSM_CTC_REL_CONFIG,
    POWSM_CTC_REL_CKPT,
    POWSM_CTC_REL_STATS,
    resolve_model_paths,
)
from src.utils import RankedLogger

log = RankedLogger(__name__, rank_zero_only=True)


def _write_text(path: Union[str, Path], text: str) -> None:
    """Write text to disk (best-effort atomic write).

    Multiple distributed inference workers may create the same patched config file
    concurrently. Writing to a temp file and renaming prevents readers from seeing
    a partially-written YAML.
    """
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    tmp = p.parent / f".{p.name}.tmp.{os.getpid()}"
    tmp.write_text(text, encoding="utf-8")
    os.replace(str(tmp), str(p))


def patch_espnet_config_paths(
    *,
    original_config_path: Union[str, Path],
    stats_file: Union[str, Path],
    bpemodel: Union[str, Path],
    output_path: Union[str, Path],
) -> str:
    """Patch ESPnet config.yaml to use absolute paths for stats_file and bpemodel.

    ESPnet's inference builders rebuild modules from config.yaml and expect paths
    like `normalize_conf.stats_file` and `bpemodel` to be valid from the current CWD.
    In PhoneticXeus, assets live under `work_dir`, so we patch them to absolute paths.
    """
    original_config_path = str(original_config_path)
    stats_file = str(stats_file)
    bpemodel = str(bpemodel)
    output_path = str(output_path)

    with open(original_config_path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    # bpemodel: top-level key in ESPnet2 S2T configs
    cfg["bpemodel"] = bpemodel

    # normalize_conf.stats_file: nested key used by GlobalMVN
    normalize_conf = cfg.get("normalize_conf") or {}
    if not isinstance(normalize_conf, dict):
        normalize_conf = {}
    normalize_conf["stats_file"] = stats_file
    cfg["normalize_conf"] = normalize_conf

    _write_text(output_path, yaml.safe_dump(cfg, sort_keys=False, allow_unicode=True))
    return output_path


class PowsmCTCInference:
    """PhoneticXeus wrapper around ESPnet's Speech2TextGreedySearch (CTC greedy)."""

    @typechecked
    def __init__(
        self,
        *,
        backend: Any,
        device: str = "cpu",
        dtype: str = "float32",
        lang_sym: str = "<unk>",
        task_sym: str = "<pr>",
        max_segment_length: float = 30.0,
        context_len_in_secs: float = 2.0,
        long_form_batch_size: int = 1,
        use_prompt_encoder: bool = True,
    ):
        self.backend = backend
        self.device = device
        self.dtype = dtype
        self.lang_sym = lang_sym
        self.task_sym = task_sym
        self.max_segment_length = float(max_segment_length)
        self.context_len_in_secs = float(context_len_in_secs)
        self.long_form_batch_size = int(long_form_batch_size)
        self.use_prompt_encoder = bool(use_prompt_encoder)

        # These are set by ESPnet and used for long/short split.
        self.sample_rate = getattr(self.backend, "sample_rate", 16000)
        self.preprocessor_conf = (
            getattr(
                getattr(self.backend, "s2t_train_args", None), "preprocessor_conf", {}
            )
            or {}
        )

        log.info(f"Decoding device={device}, dtype={dtype}")

    @torch.no_grad()
    @typechecked
    def __call__(
        self,
        speech: Union[torch.Tensor, np.ndarray],
        *args,
        text_prev: Optional[Union[torch.Tensor, np.ndarray, str, List]] = None,
        lang_sym: Optional[str] = None,
        task_sym: Optional[str] = None,
        **kwargs,
    ) -> List[Dict[str, Any]]:
        # Ignore extra keys from dataset items (distributed inference compatibility)
        _ = args, kwargs

        lang_sym = lang_sym if lang_sym is not None else self.lang_sym
        if lang_sym not in self.backend.converter.token2id:
            lang_sym = "<unk>"  # Hack to allow inference with unseen langs (doreco)
        task_sym = task_sym if task_sym is not None else self.task_sym

        # Normalize speech to 1-D array (ESPnet accepts np.ndarray or torch.Tensor)
        if isinstance(speech, torch.Tensor):
            speech_1d = speech.detach()
            if speech_1d.dim() > 1:
                speech_1d = speech_1d.squeeze()
            speech_1d = speech_1d.cpu().numpy()
        else:
            speech_1d = np.asarray(speech)
            if speech_1d.ndim != 1:
                speech_1d = np.squeeze(speech_1d)

        # Prompt handling: ESPnet Speech2TextGreedySearch expects
        #   - str: will be tokenized
        #   - torch.Tensor / np.ndarray: will be converted via `.tolist()`
        # Passing a Python list is unsafe because ESPnet calls `.tolist()` on non-str inputs.
        if (not self.use_prompt_encoder) or text_prev is None:
            text_prev_arg: Union[str, torch.Tensor, np.ndarray] = "<na>"
        elif isinstance(text_prev, str):
            text_prev_arg = text_prev
        elif isinstance(text_prev, torch.Tensor):
            t = text_prev.detach()
            if t.dim() > 1:
                t = t.squeeze()
            text_prev_arg = t.to(dtype=torch.long)
        elif isinstance(text_prev, np.ndarray):
            arr = np.asarray(text_prev).squeeze()
            text_prev_arg = arr.astype(np.int64, copy=False)
        else:
            # Best-effort conversion for list/tuple/etc (assumed to be token ids)
            try:
                arr = np.asarray(text_prev).squeeze()
                text_prev_arg = arr.astype(np.int64, copy=False)
            except Exception:
                text_prev_arg = "<na>"

        buffer_len_in_secs = float(
            self.preprocessor_conf.get("speech_length", self.max_segment_length)
        )
        buffer_len = (
            int(round(buffer_len_in_secs * float(self.sample_rate)))
            if buffer_len_in_secs > 0
            else 0
        )
        duration_seconds = float(len(speech_1d)) / float(self.sample_rate)

        try:
            if duration_seconds > buffer_len_in_secs:
                text_nospecial = self.backend.decode_long_batched_buffered(
                    speech_1d,
                    batch_size=max(1, int(self.long_form_batch_size)),
                    context_len_in_secs=float(self.context_len_in_secs),
                    lang_sym=lang_sym,
                    task_sym=task_sym,
                )
                token_int: List[int] = []
            else:
                # OWSM-CTC short-form convention: pad/trim to the fixed buffer length (typically 30s).
                if buffer_len > 0:
                    if len(speech_1d) > buffer_len:
                        speech_1d = speech_1d[:buffer_len]
                    elif len(speech_1d) < buffer_len:
                        speech_1d = np.pad(speech_1d, (0, buffer_len - len(speech_1d)))

                results = self.backend(
                    speech_1d,
                    text_prev=text_prev_arg,
                    lang_sym=lang_sym,
                    task_sym=task_sym,
                )
                # ESPnet returns ListOfHypothesis: [(text, token, token_int, text_nospecial, hyp)]
                if not results:
                    text_nospecial = ""
                    token_int = []
                else:
                    _, _, token_int, text_nospecial, _ = results[0]
                    token_int = [int(t) for t in (token_int or [])]
                    text_nospecial = text_nospecial or ""
        except KeyError as e:
            missing = e.args[0] if e.args else "<unknown>"
            raise KeyError(
                "Failed to map conditioning token to an id. "
                f"missing={missing!r}, lang_sym={lang_sym!r}, task_sym={task_sym!r}. "
                "Make sure the model token_list includes these symbols. "
            ) from e

        processed = (
            text_nospecial.split(">")[-1].replace("/", "") if text_nospecial else ""
        )
        return [
            {
                "predicted_transcript": text_nospecial,
                "processed_transcript": processed,
                # "token_int": token_int,
            }
        ]


def build_powsm_ctc_inference(
    work_dir: str = "./powsm_ctc_model",
    hf_repo: Optional[str] = None,
    force_download: bool = False,
    config_file: Optional[str] = None,
    model_file: Optional[str] = None,
    bpemodel: Optional[str] = None,
    stats_file: Optional[str] = None,
    device: str = "cpu",
    dtype: str = "float32",
    # Long-form settings
    max_segment_length: float = 30.0,
    context_len_in_secs: float = 2.0,
    long_form_batch_size: int = 1,
    use_prompt_encoder: bool = True,
) -> PowsmCTCInference:
    """Build PowsmCTCInference from HuggingFace repo or local files.

    Args:
        work_dir: Directory to store downloaded files.
        hf_repo: HuggingFace repository ID.
        force_download: Force re-download from HF repo.
        config_file: Optional path to config file.
        model_file: Optional path to model checkpoint.
        bpemodel: Optional path to BPE model.
        stats_file: Optional path to stats file.
        device: Device to run on ("cpu" or "cuda").
        dtype: Data type ("float32", "float16").
        max_segment_length: Max segment length for long-form decoding (seconds).
        context_len_in_secs: Symmetric context length (seconds) used by ESPnet buffered decoding.
        long_form_batch_size: Number of long-form buffers to decode in parallel.
        use_prompt_encoder: Whether to use prompt encoder.

    Returns:
        PowsmCTCInference object for CTC decoding.
    """
    # Download snapshot (no cross-process locking). If a multi-process race causes an
    # incomplete snapshot, we fail-fast with a clear error below.
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
        # Avoid calling snapshot_download(local_files_only=True) from every worker once
        # the snapshot is already present (it uses file locks and can stall).
        if force_download or missing:
            download_hf_snapshot(
                repo_id=hf_repo,
                work_dir=work_dir,
                force_download=True if missing else force_download,
            )
            missing = [str(root / r) for r in required_rel if not (root / r).exists()]
        if missing:
            raise RuntimeError(
                "HuggingFace snapshot is incomplete under work_dir. "
                "This can happen if multiple processes download into the same local_dir concurrently. "
                f"work_dir={work_dir}, hf_repo={hf_repo}, missing={missing}. "
                "Try: (1) pre-download once before spawning workers, or (2) run with num_workers=1, "
                "or (3) use a per-worker work_dir."
            )

    # Resolve files under work_dir (optionally download HF snapshot first)
    cfg_path, mdl_path, stats_path = resolve_model_paths(
        work_dir=work_dir,
        hf_repo=None,  # already downloaded above; keep this function pure/fail-fast
        force_download=force_download,
        config_file=config_file,
        model_file=model_file,
        stats_file=stats_file,
        rel_config=POWSM_CTC_REL_CONFIG,
        rel_ckpt=POWSM_CTC_REL_CKPT,
        rel_stats=POWSM_CTC_REL_STATS,
    )
    root = Path(work_dir)

    # Resolve BPE model path
    bpe_path = bpemodel or str(root / POWSM_CTC_REL_BPE)
    if not Path(bpe_path).exists():
        raise FileNotFoundError(
            f"BPE model not found: {bpe_path}. "
            "Check HF repo layout or pass bpemodel explicitly."
        )

    # Create patched config under work_dir (absolute stats/bpe paths)
    patched_cfg = str(Path(work_dir) / ".phoneticxeus" / "powsm_ctc_patched_config.yaml")
    patched_cfg = patch_espnet_config_paths(
        original_config_path=cfg_path,
        stats_file=stats_path,
        bpemodel=bpe_path,
        output_path=patched_cfg,
    )

    # Lazy import to keep module import cheaper/clearer failure mode
    from src.espnet_import.minimal_s2t_inference_ctc import Speech2TextGreedySearch

    # NOTE: The ESPnet builder path uses torch.load() internally. On some systems
    # (e.g., strict RSS limits), loading a 1B .pth checkpoint can be killed due
    # to peak memory. We temporarily force `mmap=True` to reduce peak usage.
    # This is a scoped monkeypatch and does not affect the global program after init.
    _orig_torch_load = torch.load

    def _torch_load_mmap(*args, **kwargs):
        # mmap=True helps reduce peak RSS on CPU, but can be problematic/slow when
        # map_location is CUDA. Only enable it for non-CUDA loads unless explicitly set.
        added_mmap = False
        if kwargs.get("mmap", None) is None:
            map_location = kwargs.get("map_location", None)
            is_cuda = False
            if isinstance(map_location, str) and map_location.startswith("cuda"):
                is_cuda = True
            elif isinstance(map_location, torch.device) and map_location.type == "cuda":
                is_cuda = True
            if not is_cuda:
                kwargs["mmap"] = True
                added_mmap = True
        try:
            return _orig_torch_load(*args, **kwargs)
        except TypeError:
            # Older torch builds may not support mmap kwarg; fall back safely.
            if added_mmap:
                kwargs.pop("mmap", None)
                return _orig_torch_load(*args, **kwargs)
            raise

    torch.load = _torch_load_mmap  # type: ignore[assignment]
    try:
        backend = Speech2TextGreedySearch(
            s2t_train_config=patched_cfg,
            s2t_model_file=mdl_path,
            device=device,
            dtype=dtype,
            # Match ESPnet default (caller may override per-call via inference args)
            lang_sym="<unk>",
            task_sym="<pr>",
        )
    finally:
        torch.load = _orig_torch_load  # type: ignore[assignment]

    return PowsmCTCInference(
        backend=backend,
        device=device,
        dtype=dtype,
        max_segment_length=max_segment_length,
        context_len_in_secs=context_len_in_secs,
        long_form_batch_size=long_form_batch_size,
        use_prompt_encoder=use_prompt_encoder,
    )
