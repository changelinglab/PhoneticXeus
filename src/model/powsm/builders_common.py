"""Common builder utilities for POWSM models.

This module provides shared utilities for building POWSM (hybrid) and POWSM-CTC models:
- Token list loading
- Hugging Face snapshot download
- Frontend/SpecAug/Normalize/CTC module creation
"""

from pathlib import Path
from typing import List, Optional, Tuple, Union
import yaml
import argparse

import torch

from src.core.utils import download_hf_snapshot
from src.model.powsm.ctc import CTC
from src.model.powsm.frontend import DefaultFrontend, GlobalMVN
from src.model.powsm.specaug import SpecAug
from src.utils import RankedLogger

log = RankedLogger(__name__, rank_zero_only=False)


def load_token_list(
    token_list_source: Union[str, List[str], Tuple[str, ...]],
) -> List[str]:
    """Load token list from file path or return as-is if already a list.

    Args:
        token_list_source: Either a path to token list file or a list/tuple of tokens.

    Returns:
        List of tokens.

    Raises:
        RuntimeError: If token_list_source is not str, list, or tuple.
    """
    if isinstance(token_list_source, str):
        with open(token_list_source, encoding="utf-8") as f:
            token_list = [line.rstrip() for line in f]
        return list(token_list)
    elif isinstance(token_list_source, (tuple, list)):
        return list(token_list_source)
    else:
        raise RuntimeError("token_list must be str or list")


def load_config(config_file: str) -> argparse.Namespace:
    """Load configuration from YAML file.

    Args:
        config_file: Path to YAML configuration file.

    Returns:
        argparse.Namespace with configuration parameters.
    """
    with open(config_file, "r", encoding="utf-8") as f:
        args = yaml.safe_load(f)
    return argparse.Namespace(**args)


def build_frontend(args: argparse.Namespace) -> Tuple[torch.nn.Module, int]:
    """Build frontend module for audio feature extraction.

    Args:
        args: Configuration namespace containing frontend settings.

    Returns:
        Tuple of (frontend module, input_size for encoder).

    Raises:
        AssertionError: If frontend type is not supported.
    """
    assert args.input_size is None, "Set frontend in the powsm config."
    assert args.frontend == "default", "Only default frontend is supported!"
    frontend = DefaultFrontend(**args.frontend_conf)
    input_size = frontend.output_size()
    return frontend, input_size


def build_specaug(args: argparse.Namespace) -> torch.nn.Module:
    """Build SpecAugment module for data augmentation.

    Args:
        args: Configuration namespace containing specaug settings.

    Returns:
        SpecAug module.

    Raises:
        AssertionError: If specaug type is not supported.
    """
    assert args.specaug == "specaug", "Only SpecAug is supported!"
    return SpecAug(**args.specaug_conf)


def build_normalize(args: argparse.Namespace, stats_file: str) -> torch.nn.Module:
    """Build normalization module.

    Args:
        args: Configuration namespace containing normalize settings.
        stats_file: Path to statistics file for GlobalMVN.

    Returns:
        GlobalMVN module.

    Raises:
        AssertionError: If normalize type is not supported.
    """
    assert args.normalize == "global_mvn", "Only GlobalMVN is supported!"
    base_conf = getattr(args, "normalize_conf", {}) or {}
    normalize_conf = dict(base_conf)
    normalize_conf["stats_file"] = stats_file
    args.normalize_conf = normalize_conf
    return GlobalMVN(**normalize_conf)


def build_ctc(
    vocab_size: int,
    encoder_output_size: int,
    ctc_conf: Optional[dict] = None,
) -> CTC:
    """Build CTC module.

    Args:
        vocab_size: Size of vocabulary (output dimension).
        encoder_output_size: Encoder output dimension.
        ctc_conf: Optional CTC configuration dict.

    Returns:
        CTC module.
    """
    ctc_conf = ctc_conf or {}
    return CTC(
        odim=vocab_size,
        encoder_output_size=encoder_output_size,
        **ctc_conf,
    )


def resolve_model_paths(
    work_dir: str,
    hf_repo: Optional[str] = None,
    force_download: bool = False,
    config_file: Optional[str] = None,
    model_file: Optional[str] = None,
    stats_file: Optional[str] = None,
    rel_config: str = "",
    rel_ckpt: str = "",
    rel_stats: str = "",
) -> Tuple[str, str, str]:
    """Resolve model file paths, downloading from HF if needed.

    Args:
        work_dir: Working directory for downloaded files.
        hf_repo: Hugging Face repository ID (e.g., "espnet/powsm").
        force_download: Force re-download from HF repo.
        config_file: Optional explicit config file path.
        model_file: Optional explicit model file path.
        stats_file: Optional explicit stats file path.
        rel_config: Relative path to config in HF repo.
        rel_ckpt: Relative path to checkpoint in HF repo.
        rel_stats: Relative path to stats in HF repo.

    Returns:
        Tuple of (config_path, model_path, stats_path).

    Raises:
        AssertionError: If any required file is not found.
    """
    root = Path(work_dir)
    cfg = config_file or str(root / rel_config)
    mdl = model_file or str(root / rel_ckpt)
    stats = stats_file or str(root / rel_stats)

    if hf_repo:
        needs_download = (
            force_download
            or (not Path(cfg).exists())
            or (not Path(mdl).exists())
            or (not Path(stats).exists())
        )
        if needs_download:
            download_hf_snapshot(
                repo_id=hf_repo,
                force_download=force_download,
                work_dir=work_dir,
            )

    # Assert files exist
    assert Path(cfg).exists(), f"Config file not found: {cfg}"
    assert Path(mdl).exists(), f"Model file not found: {mdl}"
    assert Path(stats).exists(), f"Stats file not found: {stats}"

    return cfg, mdl, stats


# Relative paths from hf repo structure (espnet style)
# TODO(shikhar): Convert to patterns and match patterns within downloaded files.

# POWSM (hybrid) specific paths
POWSM_REL_CONFIG = "exp/s2t_train_s2t_ebf_conv2d_size768_e9_d9_piecewise_lr5e-4_warmup60k_flashattn_raw_bpe40000/config.yaml"
POWSM_REL_CKPT = "exp/s2t_train_s2t_ebf_conv2d_size768_e9_d9_piecewise_lr5e-4_warmup60k_flashattn_raw_bpe40000/valid.acc.ave_5best.till45epoch.pth"
POWSM_REL_STATS = "exp/s2t_stats_raw_bpe40000/train/feats_stats.npz"
POWSM_REL_BPE = "data/token_list/bpe_unigram40000/bpe.model"


# POWSM-CTC specific paths (to be updated when HF repo is available)
# Default: ESPnet OWSM-CTC v4 1B (`espnet/owsm_ctc_v4_1B`) layout
POWSM_CTC_REL_CONFIG = "exp/temp/config.yaml"
POWSM_CTC_REL_CKPT = "exp/temp/valid.total_count.ave.till70epoch.pth"
POWSM_CTC_REL_STATS = "exp/s2t_stats_raw_bpe40000/train/feats_stats.npz"
POWSM_CTC_REL_BPE = "data/token_list/bpe_unigram40000/bpe.model"
