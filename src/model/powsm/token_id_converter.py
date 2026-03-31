import argparse
from pathlib import Path
from typing import Dict, Iterable, List, Union, Optional

from huggingface_hub import snapshot_download
import numpy as np
from typeguard import typechecked
import yaml

from src.utils import RankedLogger

log = RankedLogger(__name__, rank_zero_only=True)


class TokenIDConverter:
    @typechecked
    def __init__(
        self,
        token_list: Union[Path, str, Iterable[str]],
        unk_symbol: str = "<unk>",
    ):

        if isinstance(token_list, (Path, str)):
            token_list = Path(token_list)
            self.token_list_repr = str(token_list)
            self.token_list: List[str] = []

            with token_list.open("r", encoding="utf-8") as f:
                for idx, line in enumerate(f):
                    line = line[0] + line[1:].rstrip()
                    self.token_list.append(line)

        else:
            self.token_list: List[str] = list(token_list)
            self.token_list_repr = ""
            for i, t in enumerate(self.token_list):
                if i == 3:
                    break
                self.token_list_repr += f"{t}, "
            self.token_list_repr += f"... (NVocab={(len(self.token_list))})"

        self.token2id: Dict[str, int] = {}
        for i, t in enumerate(self.token_list):
            if t in self.token2id:
                raise RuntimeError(f'Symbol "{t}" is duplicated')
            self.token2id[t] = i

        self.unk_symbol = unk_symbol
        if self.unk_symbol not in self.token2id:
            raise RuntimeError(
                f"Unknown symbol '{unk_symbol}' doesn't exist in the token_list"
            )
        self.unk_id = self.token2id[self.unk_symbol]

    def get_num_vocabulary_size(self) -> int:
        return len(self.token_list)

    def ids2tokens(self, integers: Union[np.ndarray, Iterable[int]]) -> List[str]:
        if isinstance(integers, np.ndarray) and integers.ndim != 1:
            raise ValueError(f"Must be 1 dim ndarray, but got {integers.ndim}")
        return [self.token_list[i] for i in integers]

    def tokens2ids(self, tokens: Iterable[str]) -> List[int]:
        # hack for powsm tokenizer to work with non / delimited tokens.
        tokens = [
            f"/{t}/" if not (t.startswith("/") and t.endswith("/")) else t
            for t in tokens
        ]
        return [self.token2id.get(i, self.unk_id) for i in tokens]


def build_powsm_tokenizer_from_files(
    config_file: str,
) -> TokenIDConverter:
    with open(config_file, "r", encoding="utf-8") as f:
        args = yaml.safe_load(f)
    args = argparse.Namespace(**args)
    if isinstance(args.token_list, str):
        with open(args.token_list, encoding="utf-8") as f:
            token_list = [line.rstrip() for line in f]
        args.token_list = list(token_list)
    elif isinstance(args.token_list, (tuple, list)):
        token_list = list(args.token_list)
    else:
        raise RuntimeError("token_list must be str or list")

    vocab_size = len(token_list)
    log.info(f"Vocabulary size: {vocab_size}")
    return TokenIDConverter(token_list=token_list)


def build_powsm_tokenizer(
    *,
    work_dir: str,
    hf_repo: Optional[str] = "espnet/powsm",
    force: bool = False,
    config_file: Optional[str] = None,
):
    """Build Powsm tokenizer from local files or huggingface repo.
    Args:
        work_dir: Directory to store downloaded files from hf repo.
        hf_repo: Huggingface repo name. If None, load from local files.
        force: Whether to force re-download from hf repo.
        config_file: Path to config file. If None, use default path in hf repo.
          Takes precedence over hf_repo.
    Returns: TokenIDConverter instance
    """
    # Relative paths from hf repo structure (espnet style)
    REL_CONFIG = "exp/s2t_train_s2t_ebf_conv2d_size768_e9_d9_piecewise_lr5e-4_warmup60k_flashattn_raw_bpe40000/config.yaml"

    if hf_repo:
        snapshot_download(
            repo_id=hf_repo,
            force_download=force,
            local_dir=work_dir,
            local_dir_use_symlinks=False,  # materialize files under work_dir
        )

    root = Path(work_dir)
    cfg = config_file or str(root / REL_CONFIG)
    # assert file exists
    assert Path(cfg).exists(), f"Config file not found: {cfg}"
    return build_powsm_tokenizer_from_files(cfg)
