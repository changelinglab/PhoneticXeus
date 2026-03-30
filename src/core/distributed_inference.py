""" Distributed inference script.
scripts/run.sh --recipe inference \
    --cluster babel --sbatch_args "--array=0-7 -p preempt --gres=gpu:1 -c 13 -t 48:00:00 --mem=40G" \
    --model powsm,ctag,lv60,xlsr53 \
    --data uaspeech --extra_args "inference.num_workers=11 run_folder=8jobARR"
"""

from dataclasses import fields, is_dataclass
import math
import os
import hydra
import torch
from functools import partial
import multiprocessing as mp
from tqdm import tqdm
import json
import traceback
from lightning_utilities.core.rank_zero import rank_zero_only
from omegaconf import OmegaConf


def _init_worker():
    proc = mp.current_process()
    rank_zero_only.rank = (proc._identity[0] - 1) if proc._identity else 0


def work_chunk_(
    args,
    dataset_cfg,
    inference_config,
    inference_call_args=None,
    passthrough_keys=None,
):
    """Worker function to run inference on a chunk of data."""
    try:
        slurm_task_id, worker_id, idxs, device = args
        print(
            f"SLURM_TASK_ID={slurm_task_id}, Worker {worker_id} "
            f"processing {len(idxs)} items on device {device}.",
            flush=True,
        )
        if inference_config.get("cache_path", None):
            # NOTE: be sure to read all cache files if resuming!
            base, ext = os.path.splitext(inference_config["cache_path"])
            inference_config["cache_path"] = f"{base}.{slurm_task_id}.{worker_id}{ext}"
        inference_obj = hydra.utils.instantiate(inference_config, device=device)
        out = []
        dataset = get_dataset_from_cfg(dataset_cfg)
        for i in tqdm(idxs, desc="Processing", leave=False):
            it = dataset[i]
            # keys from dataset override those in inference_call_args
            call_args = {**(inference_call_args or {}), **it}
            pred = inference_obj(**call_args)
            # keys from dataset that must be passthroughly passed to
            # output to be written
            out.append(
                (i, pred, {k: it[k] for k in (passthrough_keys or []) if k in it})
            )
        return out
    except Exception as e:
        tb = traceback.format_exc()
        print(f"[WORKER_ERROR_{worker_id}]: {e}\n{tb}", flush=True)
        return [("__error__", {"chunk": worker_id, "err": str(e), "traceback": tb}, {})]


def default_encoder(o):
    if is_dataclass(o):
        return {f.name: getattr(o, f.name) for f in fields(o)}
    return str(o)


def get_dataset_from_cfg(dataset_cfg):
    datamodule = hydra.utils.instantiate(dataset_cfg)
    datamodule.prepare_data()  # take care: MUST be cached
    datamodule.setup(stage="predict")
    dataset = datamodule.predict_dataloader().dataset
    return dataset


def run_distributed_inference_(
    dataset_cfg,
    inference_config,
    inference_call_args=None,
    num_workers: int = 1,
    out_file=None,
    passthrough_keys=None,
    limit_samples: int = None,
):
    """Splits dataset and runs inference in parallel workers.

    Args:
        dataset_cfg: Dataset configuration to instantiate the dataset object
        inference_config: config for inference object to be instantiated in each worker
        inference_call_args: additional args to be passed to inference __call__ method
        num_workers: number of parallel workers
        out_file: output file to save results
        passthrough_keys: list of keys in dataset item to be written directly to
            output without processing
        limit_samples: if set, limit the number of samples to process (useful for testing)
    """

    dataset_cfg = OmegaConf.to_container(dataset_cfg, resolve=True)
    inference_config = OmegaConf.to_container(inference_config, resolve=True)
    if inference_call_args is not None:
        inference_call_args = OmegaConf.to_container(inference_call_args, resolve=True)

    # fail fast
    assert out_file, "Please provide an out_file to save results."
    SLURM_TASK_ID = int(os.environ.get("SLURM_ARRAY_TASK_ID", 0))
    SLURM_NUM_TASKS = int(os.environ.get("SLURM_ARRAY_TASK_COUNT", 1))

    ######## MAKE FILE TO WRITE TO ########
    base, ext = os.path.splitext(out_file)
    ext = "jsonl"  # switch to jsonl
    out_file = f"{base}.{SLURM_TASK_ID}.{ext}"
    print(f"OUTPUT_FILE: {out_file}", flush=True)
    os.makedirs(os.path.dirname(out_file), exist_ok=True)
    open(out_file, "w").close()  # !!overwrite!!
    ############DEVICE DISTRIBUTION#########
    device = inference_config.get("device", "auto")
    if device != "cpu":  # cuda or auto
        n_gpus = torch.cuda.device_count() if torch.cuda.is_available() else 0
        devices = [f"cuda:{i}" for i in range(n_gpus)] if n_gpus > 0 else ["cpu"]
    else:
        devices = ["cpu"]
    #######################################
    dataset = get_dataset_from_cfg(dataset_cfg)
    ######## SOME MATHS TO DISTRIBUTE WORK ########
    N = len(dataset)
    if limit_samples is not None and limit_samples > 0:
        N = min(N, limit_samples)
        print(
            f"Limiting inference to {N} samples (out of {len(dataset)} total).",
            flush=True,
        )
    N_shard = math.ceil(N / SLURM_NUM_TASKS)  # items per slurm worker
    shard_start = SLURM_TASK_ID * N_shard
    shard_end = min(shard_start + N_shard, N)  # exclusive
    print(
        f"Running inference on {N_shard} utterances on"
        f" device={device} with {num_workers} workers.",
        flush=True,
    )
    shard_size = shard_end - shard_start
    if shard_size <= 0:
        print(
            f"Task ID {SLURM_TASK_ID} has no work to do (shard size {shard_size} <= 0). Exiting.",
            flush=True,
        )
        return
    chunk_sz = math.ceil(shard_size / num_workers)
    chunks = [
        range(
            shard_start + i * chunk_sz, min(shard_start + (i + 1) * chunk_sz, shard_end)
        )
        for i in range(num_workers)
        if shard_start + i * chunk_sz < shard_end
    ]
    #############################################
    worker = partial(
        work_chunk_,
        dataset_cfg=dataset_cfg,
        inference_config=inference_config,
        inference_call_args=inference_call_args,
        passthrough_keys=passthrough_keys,
    )
    print("Total chunks to process:", len(chunks), flush=True)
    worker_args = [
        (SLURM_TASK_ID, i, chunk, devices[i % len(devices)])
        for i, chunk in enumerate(chunks)
    ]

    with mp.get_context("spawn").Pool(num_workers, initializer=_init_worker) as pool:
        chunk_idx = 0
        for chunk_results in pool.imap_unordered(worker, worker_args):
            with open(out_file, "a") as f:
                for i, pred, passthrough in chunk_results:
                    record_payload = {i: {"pred": pred, "passthrough": passthrough}}
                    json_payload = json.dumps(
                        record_payload,
                        default=default_encoder,
                        ensure_ascii=False,
                    )
                    f.write(json_payload + "\n")
                f.flush()
                print(f"Dumped chunk {chunk_idx} results to file.", flush=True)
                chunk_idx += 1
    print(
        f"Finished distributed inference. Final output saved to {out_file}.", flush=True
    )
