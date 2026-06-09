______________________________________________________________________

<div align="center">

<a href="https://arxiv.org/abs/2603.29042"><img alt="arXiv" src="https://img.shields.io/badge/arXiv-2603.29042-b31b1b?logo=arxiv&logoColor=white"></a>
<a href="https://pytorch.org/get-started/locally/"><img alt="PyTorch" src="https://img.shields.io/badge/PyTorch-ee4c2c?logo=pytorch&logoColor=white"></a>
<a href="https://pytorchlightning.ai/"><img alt="Lightning" src="https://img.shields.io/badge/-Lightning-792ee5?logo=pytorchlightning&logoColor=white"></a>
<a href="https://hydra.cc/"><img alt="Config: Hydra" src="https://img.shields.io/badge/Config-Hydra-89b8cd"></a><br>

</div>

# PhoneticXeus

> 🎉 **PhoneticXeus will be presented at Interspeech 2026!**

Code and training recipe for PhoneticXeus, a multilingual phone recognition model using self-conditioned CTC on the XEUS speech encoder. It transcribes speech in 70+ languages into IPA (International Phonetic Alphabet) phones.

- 🤗 **Model**: [changelinglab/PhoneticXeus](https://huggingface.co/changelinglab/PhoneticXeus)
- 🕹️ **Demo**: [changelinglab/PhoneticXeus (Space)](https://huggingface.co/spaces/changelinglab/PhoneticXeus)
- 📄 **Paper**: [arXiv 2603.29042](https://arxiv.org/abs/2603.29042)

## Quick Inference

The fastest way to use PhoneticXeus is the 🤗 Transformers `AutoModel` interface,
**no need to clone this repo**. It downloads the weights, vocab, and code from
[changelinglab/PhoneticXeus](https://huggingface.co/changelinglab/PhoneticXeus).

```python
import torch, torchaudio
from transformers import AutoModel

model = AutoModel.from_pretrained(
    "changelinglab/PhoneticXeus", trust_remote_code=True
).eval()

# Load audio and convert to 16 kHz mono (required).
waveform, sr = torchaudio.load("audio.wav")
if waveform.dim() == 2:
    waveform = waveform.mean(dim=0)          # downmix to mono
if sr != 16000:
    waveform = torchaudio.functional.resample(waveform, sr, 16000)

results = model.transcribe(waveform, sampling_rate=16000)
print(results[0]["processed_transcript"])    # joined IPA, e.g. "ðɪsɪzɐtɛst"
print(results[0]["predicted_transcript"])     # slash-separated phones, e.g. "ð/ɪ/s/..."
```

`transcribe()` returns one dict per utterance with:

| Key | Description |
|---|---|
| `processed_transcript` | IPA phones joined into a string (special tokens removed) |
| `predicted_transcript` | raw slash-separated phone sequence |

For frame-level CTC logits (custom decoding, alignment, confidence):

```python
logits = model(input_values=waveform.unsqueeze(0)).logits  # (batch, frames, vocab)
```

## Use from source

Useful for training or to modify decoding.

### Setup

> **Audio input must be mono, 16 kHz.** Resample before calling the model.

```bash
git clone git@github.com:changelinglab/PhoneticXeus.git
cd PhoneticXeus

# install (auto-detects x86_64 vs aarch64)
make install

# activate environment (once per session)
source .venv/bin/activate
```

#### Environment Variables

Set these before training or inference:

```bash
export IPAPACK_DATA_ROOT=/path/to/ipapack/data   # root directory for Kaldi-style data
export PHONEMIZER_ESPEAK_LIBRARY=/path/to/libespeak-ng.so  # needed for wav2vec2-phoneme models
export ESPEAK_DATA_PATH=/path/to/espeak-ng-data
```

### Programmatic inference

```python
import torch, torchaudio
from huggingface_hub import hf_hub_download
from src.model.xeusphoneme.builders import build_xeus_pr_inference

ckpt_path = hf_hub_download("changelinglab/PhoneticXeus", "phoneticxeus_state_dict.pt")

inference = build_xeus_pr_inference(
    work_dir="exp/cache/xeus",
    checkpoint=ckpt_path,
    config_file="src/model/xeusphoneme/resources/xeus_config.yaml",
    vocab_file="src/model/xeusphoneme/resources/ipa_vocab.json",
    device="cuda" if torch.cuda.is_available() else "cpu",
    interctc_use_conditioning=True,
)

waveform, sr = torchaudio.load("audio.wav")
if waveform.dim() == 2:
    waveform = waveform.mean(dim=0)
if sr != 16000:
    waveform = torchaudio.functional.resample(waveform, sr, 16000)

results = inference(waveform)
print(results[0]["processed_transcript"])
```

## Use Cases

PhoneticXeus produces language-agnostic IPA, which makes it useful well beyond plain
transcription. The examples below assume `model` from the [Quick Inference](#quick-inference)
snippet and a 16 kHz mono `waveform`.

### Pronunciation scoring

Compare a learner's pronunciation against a reference (canonical) IPA transcription
using the built-in phone-level metrics.

```python
from src.metrics.phone_recognition import PhoneRecognitionEvaluator

hyp = model.transcribe(waveform, sampling_rate=16000)[0]["processed_transcript"]
reference = "ðɪsɪzɐtɛst"   # canonical IPA for the target phrase

evaluator = PhoneRecognitionEvaluator(normalize_ipa=True)
summary, per_utt = evaluator.evaluate(
    {"utt0": {"prediction": hyp, "transcription": reference}}
)

print(f"PER  {summary.PER:.1f}%")    # phone error rate (lower = closer to target)
print(f"PFER {summary.PFER:.1f}%")   # phone-feature error rate (partial credit)
print(per_utt["utt0"])               # {"pfer", "fed", "per", "fer"}
```

`PER` counts whole-phone errors; `PFER`/`FED` use articulatory features so that a
near-miss (e.g. `s`-vs-`z`) is penalized less than an unrelated substitution.

### TTS evaluation

Score a TTS system by transcribing its synthesized audio and comparing to the
intended phones (phonetic intelligibility). Pass a batch of utterances at once.

```python
# waveforms: list of 16 kHz mono tensors; refs: intended IPA per utterance
test_data = {}
for i, (wav, ref) in enumerate(zip(waveforms, refs)):
    hyp = model.transcribe(wav, sampling_rate=16000)[0]["processed_transcript"]
    test_data[f"utt{i}"] = {"prediction": hyp, "transcription": ref}

summary, _ = PhoneRecognitionEvaluator(normalize_ipa=True).evaluate(test_data)
print(f"Corpus PER {summary.PER:.1f}% over {summary.N} utterances")
```

For large-scale evaluation over Kaldi-style datasets, use the distributed
inference + evaluation pipeline described in [Inference](#inference) and
[Evaluation](#evaluation) below.

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

## Citation

```bibtex
@misc{pxeus26,
      title={An Empirical Recipe for Universal Phone Recognition},
      author={Shikhar Bharadwaj and Chin-Jou Li and Kwanghee Choi and Eunjung Yeo and William Chen and Shinji Watanabe and David R. Mortensen},
      year={2026},
      eprint={2603.29042},
      archivePrefix={arXiv},
      primaryClass={cs.CL},
      url={https://arxiv.org/abs/2603.29042},
}
```

## More Documentation

- **[Running Inference](docs/running_inference.md)** -- distributed inference guide
- **[Contributing Guide](CONTRIBUTING.md)** -- project structure and workflow
