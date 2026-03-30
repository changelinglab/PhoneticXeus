______________________________________________________________________

<div align="center">

<a href="https://pytorch.org/get-started/locally/"><img alt="PyTorch" src="https://img.shields.io/badge/PyTorch-ee4c2c?logo=pytorch&logoColor=white"></a>
<a href="https://pytorchlightning.ai/"><img alt="Lightning" src="https://img.shields.io/badge/-Lightning-792ee5?logo=pytorchlightning&logoColor=white"></a>
<a href="https://hydra.cc/"><img alt="Config: Hydra" src="https://img.shields.io/badge/Config-Hydra-89b8cd"></a><br>

</div>

# PhoneticXeus

Code and training recipe for PhoneticXeus, a multilingual phone recognition model using self-conditioned CTC on the XEUS speech encoder.

## Setup

```bash
git clone git@github.com:changelinglab/PhoneticXeus.git
cd PhoneticXeus

# install (auto-detects x86_64 vs aarch64)
make install

# activate environment (once per session)
source .venv/bin/activate
```

### Environment Variables

Set these before training or inference:

```bash
export IPAPACK_DATA_ROOT=/path/to/ipapack/data   # root directory for Kaldi-style data
export PHONEMIZER_ESPEAK_LIBRARY=/path/to/libespeak-ng.so  # needed for wav2vec2-phoneme models
export ESPEAK_DATA_PATH=/path/to/espeak-ng-data
```

## Pre-trained Model

The pre-trained PhoneticXeus checkpoint is available on HuggingFace: [changelinglab/PhoneticXeus](https://huggingface.co/changelinglab/PhoneticXeus)

```python
from huggingface_hub import hf_hub_download

ckpt_path = hf_hub_download("changelinglab/PhoneticXeus", "checkpoint-22000.ckpt")
```

### Quick inference

```python
import torch, torchaudio
from src.model.xeusphoneme.builders import build_xeus_pr_inference

inference = build_xeus_pr_inference(
    work_dir="exp/cache/xeus",
    checkpoint=ckpt_path,
    vocab_file="src/model/xeusphoneme/resources/ipa_vocab.json",
    hf_repo="espnet/xeus",
    device="cuda" if torch.cuda.is_available() else "cpu",
)

waveform, sr = torchaudio.load("audio.wav")
if sr != 16000:
    waveform = torchaudio.functional.resample(waveform, sr, 16000)

results = inference(waveform.squeeze(0))
print(results[0]["processed_transcript"])
```

## Data Setup

Training and evaluation datasets use Kaldi-style `wav.scp` / `text` files. Dataset paths are configured in:

- **Training data**: `configs/data/ipapack_index.yaml` -- defines train/dev splits
- **Evaluation data**: `configs/data/prism_pr_evalsets.yaml` -- defines eval datasets (DoReCo, GMU Accent, TIMIT, Buckeye, VoxAngeles, TUSOM, FLEURS, etc.)

All paths are relative to `IPAPACK_DATA_ROOT`. Prepare data with the IPAPack pipeline, then point the env var to the output directory.

Pre-trained model weights are downloaded automatically from HuggingFace (e.g., `espnet/xeus`, `espnet/powsm`) on first use.

## Training

```bash
# single GPU
python src/main.py experiment=train/ipapack_xeuspr trainer=gpu

# multi-GPU (DDP)
python src/main.py experiment=train/ipapack_xeuspr trainer=ddp

# SLURM
sbatch scripts/daixpr.batch experiment=train/ipapack_xeuspr run_folder=my_run
```

Override any parameter from the command line:

```bash
python src/main.py experiment=train/ipapack_xeuspr \
    trainer.max_steps=50000 data.batch_size=32 model.optimizer.lr=3e-5
```

Available training configs are in `configs/experiment/train/`.

## Inference

Run inference on any evaluation dataset:

```bash
# single dataset
python src/main.py experiment=inference/powsmpreval data.dataset_name=doreco

# distributed (SLURM array)
sbatch --array=0-3 scripts/daixpr_inference.batch \
    experiment=inference/powsmpreval data.dataset_name=doreco
```

Results are written as JSONL shards: `<out_file>.<task_id>.jsonl`.

Available inference configs are in `configs/experiment/inference/`.

## Evaluation

Evaluate predictions from distributed inference shards using a glob pattern:

```bash
# evaluate all shards at once
python -m src.metrics.phone_recognition \
    --prediction_file "exp/runs/my_run/transcription.*.jsonl" \
    --output_file results.csv \
    --evaluation_name my_model \
    --gt_field target --key_field utt_id

# or a single file (JSON or JSONL)
python -m src.metrics.phone_recognition \
    --prediction_file exp/runs/my_run/transcription.0.jsonl \
    --output_file results.csv \
    --evaluation_name my_model \
    --gt_field target --key_field utt_id
```

Metrics: PER (Phone Error Rate), PFER (Phone Feature Error Rate), FED (Feature Edit Distance), SUB/INS/DEL rates.

## More Documentation

- **[Running Inference](docs/running_inference.md)** -- distributed inference guide
- **[Contributing Guide](CONTRIBUTING.md)** -- project structure and workflow
