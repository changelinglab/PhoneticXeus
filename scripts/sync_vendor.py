"""Regenerate the vendored inference package for the Space and the HF model repo.

Usage:
    python scripts/sync_vendor.py
"""

import re
import shutil
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PKG = "pxeus"
DESTS = [ROOT / "space", ROOT / "hub"]
OVERRIDES = ROOT / "scripts" / "vendor_overrides"
FILES = [
    "core/utils.py",
    "espnet_import/attention.py",
    "espnet_import/cgmlp.py",
    "espnet_import/embedding.py",
    "espnet_import/fastformer.py",
    "espnet_import/label_smoothing_loss.py",
    "espnet_import/layer_norm.py",
    "espnet_import/nets_utils.py",
    "espnet_import/positionwise_feed_forward.py",
    "espnet_import/repeat.py",
    "espnet_import/subsampling.py",
    "model/powsm/ctc.py",
    "model/powsm/e_branchformer.py",
    "model/powsm/specaug.py",
    "model/powsm/utils.py",
    "model/xeusphoneme/builders.py",
    "model/xeusphoneme/cnn_frontend.py",
    "model/xeusphoneme/linear_layer.py",
    "model/xeusphoneme/resources/ipa_vocab.json",
    "model/xeusphoneme/resources/xeus_config.yaml",
    "model/xeusphoneme/xeuspr_inference.py",
    "model/xeusphoneme/xeuspr_model.py",
    "recipe/phone_recognition/greedy_ctc_strategy.py",
    "utils/__init__.py",
    "utils/pylogger.py",
]


def main() -> None:
    """Copy the allowlisted files into each destination under the PKG name."""
    for dest in DESTS:
        out = dest / PKG
        shutil.rmtree(out, ignore_errors=True)
        for rel in FILES:
            source = OVERRIDES / rel
            if not source.exists():
                source = ROOT / "src" / rel
            target = out / rel
            target.parent.mkdir(parents=True, exist_ok=True)
            if rel.endswith(".py"):
                text = re.sub(r"\bsrc\.", f"{PKG}.", source.read_text())
                target.write_text(text)
            else:
                shutil.copy(source, target)
        for pkg_dir in [out, *(p for p in out.rglob("*") if p.is_dir())]:
            (pkg_dir / "__init__.py").touch()
        print(f"Synced {len(FILES)} files to {out}")


if __name__ == "__main__":
    main()
