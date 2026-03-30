"""Average multiple Lightning checkpoints by their state_dicts.

Usage:
    python scripts/average_checkpoints.py \
        exp/my_run/checkpoint-10000.ckpt \
        exp/my_run/checkpoint-20000.ckpt \
        exp/my_run/checkpoint-30000.ckpt \
        [--out exp/my_run/checkpoint-avg-s10000+s20000+s30000.ckpt]

The output checkpoint name encodes the averaged steps. The optimizer, scheduler,
and loop states are dropped (they are not meaningful for an averaged checkpoint).
"""

import argparse
import re
from pathlib import Path

import torch


# Keys that carry training-state; meaningless after averaging.
_DROP_KEYS = {"optimizer_states", "lr_schedulers", "loops"}


def _parse_step(path: Path) -> str:
    """Extract the step tag from a checkpoint filename.

    For 'checkpoint-12345.ckpt' returns 's12345'.
    For 'last.ckpt' returns 'slast'.
    Falls back to the stem for unrecognised patterns.
    """
    m = re.search(r"checkpoint-(\d+)", path.stem)
    if m:
        return f"s{m.group(1)}"
    return path.stem


def average_checkpoints(paths: list[Path]) -> dict:
    """Load checkpoints and return one with an averaged state_dict."""
    if not paths:
        raise ValueError("No checkpoint paths provided.")

    # --- average state_dicts ---
    first = torch.load(paths[0], map_location="cpu")
    avg_state: dict = {k: v.clone().float() for k, v in first["state_dict"].items()}

    for ckpt_path in paths[1:]:
        ckpt = torch.load(ckpt_path, map_location="cpu")
        for k, v in ckpt["state_dict"].items():
            avg_state[k] += v.float()

    n = len(paths)
    for k in avg_state:
        avg_state[k] /= n

    # --- build result from the last checkpoint, drop training-state keys ---
    last_ckpt = torch.load(paths[-1], map_location="cpu")
    result = {k: v for k, v in last_ckpt.items() if k not in _DROP_KEYS}
    result["state_dict"] = avg_state
    return result


def _default_out(paths: list[Path]) -> Path:
    tags = "+".join(_parse_step(p) for p in paths)
    return paths[0].parent / f"checkpoint-avg-{tags}.ckpt"


def main():
    parser = argparse.ArgumentParser(description="Average Lightning checkpoints.")
    parser.add_argument("checkpoints", nargs="+", type=Path, help="Checkpoint files to average.")
    parser.add_argument(
        "--out",
        type=Path,
        default=None,
        help="Output path. Default: <parent>/checkpoint-avg-s<step1>+s<step2>+....ckpt",
    )
    args = parser.parse_args()

    paths = [Path(p) for p in args.checkpoints]
    for p in paths:
        if not p.exists():
            raise FileNotFoundError(p)

    out = args.out or _default_out(paths)

    print(f"Averaging {len(paths)} checkpoint(s):")
    for p in paths:
        print(f"  {p}")

    result = average_checkpoints(paths)

    out.parent.mkdir(parents=True, exist_ok=True)
    torch.save(result, out)
    print(f"Saved averaged checkpoint → {out}")


if __name__ == "__main__":
    main()
