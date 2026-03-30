# Running Phone Recognition Inference

## Quick Start

Use the `powsmpreval` experiment config with a dataset override:

```bash
python src/main.py experiment=inference/powsmpreval data.dataset_name=doreco
```

Or use a dataset-specific config directly:

```bash
python src/main.py experiment=inference/vaani_powsmpr
```

## Distributed Inference

Set `distributed_predict: True` in the experiment config (already set in
`powsmpreval`). The system uses two levels of parallelism:

1. **SLURM array jobs** -- dataset is sharded across `SLURM_ARRAY_TASK_COUNT`
   tasks.
2. **`num_workers` per job** -- each SLURM task spawns this many processes,
   each pinned to a GPU.

Submit with:

```bash
sbatch --array=0-3 scripts/daixpr_inference.batch \
    experiment=inference/powsmpreval data.dataset_name=doreco
```

Results are written as per-job JSONL shards:
`<out_file_base>.<SLURM_TASK_ID>.jsonl`. Merge after all tasks complete.

## Output Format

One JSON object per line:

```json
{"<idx>": {"pred": "<transcript>", "passthrough": {"utt_id": "...", "phones": [...]}}}
```

## DataModule Contract

The datamodule must implement `predict_dataloader()` returning a dataset whose
`__getitem__` yields a dict containing a `speech` key (raw waveform tensor).
Any key in `inference_call_args` can be overridden per-sample by including it
in the dataset item dict.

## See Also

- `src/core/distributed_inference.py` -- implementation details
- `configs/experiment/inference/` -- all inference configs
