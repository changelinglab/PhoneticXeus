# CLAUDE.md

## Project Overview

**PhoneticXeus** is a multilingual phone recognition model using self-conditioned CTC on the XEUS speech encoder. This repo contains the training recipe and evaluation code, built on PyTorch Lightning + Hydra.

## Environment Setup

Requires a GPU node and an active environment.

```bash
make install            # auto-detects x86 vs aarch64
source .venv/bin/activate
```

Set `IPAPACK_DATA_ROOT` to the root of your Kaldi-style data directory before training or inference.

## Sanity Check

```bash
python -m src.recipe.phone_recognition.model_module
```

## Coding Style

Prefer simplicity. Extract helpers only for non-trivial logic. Guard clause first. Minimal lines.

Follow the [Google Python Style Guide](https://google.github.io/styleguide/pyguide.html). Key rules: 80-char lines, Google docstrings, type annotations on public functions, no relative imports, no `staticmethod`.

Full style reference: `docs/style.md`

## Architecture

### Config: `configs/main.yaml` composed with experiment overrides in `configs/experiment/{train,inference}/`

### Flow: `src/main.py` -> `src/core/task.py` -> Lightning Trainer

### Models (`src/model/`): XeusPR, PowSM, Wav2Vec2, WavLM, Whisper, Zipformer -- all swappable via config

### Data (`src/data/`): Kaldi-style ark/scp loaders. Training splits in `configs/data/ipapack_index.yaml`, eval sets in `configs/data/prism_pr_evalsets.yaml`

### Metrics (`src/metrics/phone_recognition.py`):
- `PhoneRecognitionEvaluator(normalize_ipa=True)`
- `evaluator.evaluate(utt_data)` -> `(PhoneRecognitionSummary, instance_metrics)`
- Summary fields: `PFER`, `FED`, `PER`, `SUB`, `INS`, `DEL`, `N`, `phones`
- Per-utterance metrics: `pfer`, `fed`, `per`

### Distributed Inference (`src/core/distributed_inference.py`):
Triggered by `distributed_predict: True` in experiment config. Two-level parallelism: SLURM array jobs x `num_workers` per job. Outputs JSONL shards.

## Environment Variables

| Variable | Purpose |
|---|---|
| `IPAPACK_DATA_ROOT` | Root for Kaldi-style data |
| `PHONEMIZER_ESPEAK_LIBRARY` | Path to `libespeak-ng.so` |
| `ESPEAK_DATA_PATH` | espeak-ng data directory |
| `FLASH_ATTN_EGG` | Pre-built flash_attn_3 egg (aarch64 only) |
