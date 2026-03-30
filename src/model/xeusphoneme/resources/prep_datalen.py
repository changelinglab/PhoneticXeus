# Helper script to compute and save the lengths of audio files listed in a wav.scp file.
import os

import kaldiio
import numpy as np
import torch
import torchaudio
import yaml
from concurrent.futures import ProcessPoolExecutor
from tqdm import tqdm

# --- Configuration ---
CFG = "configs/data/ipapack_index.yaml"
DATASET = "pr_fixed"
BASE = os.environ.get("IPAPACK_DATA_ROOT", "")
OUT_NPY = "lengths_pr.npy"
CHUNK_SIZE = 10000  # Number of lines sent to each worker at once


def process_chunk(lines):
    """Processes a batch of lines and returns a list of (key, duration) pairs."""
    chunk_results = []
    for line in lines:
        parts = line.strip().split()
        if len(parts) < 2 or not parts[0].endswith("_pr"):
            continue

        key, path = parts[0], parts[1]
        if not path.startswith("/work"):
            path = f"{BASE}/{path}"

        try:
            if ".ark" in path:
                # kaldiio.load_mat returns (rate, numpy_array)
                sr, wav = kaldiio.load_mat(path)
                dur = wav.shape[-1] / sr
            else:
                # torchaudio.info is much faster than loading the full file
                info = torchaudio.info(path)
                dur = info.num_frames / info.sample_rate
            chunk_results.append((key, dur))
        except Exception:
            continue
    return chunk_results


def get_chunk_generator(file_path, chunk_size):
    """A generator that yields chunks of lines from a file without loading it all into RAM."""
    with open(file_path, "r") as f:
        chunk = []
        for line in f:
            chunk.append(line)
            if len(chunk) == chunk_size:
                yield chunk
                chunk = []
        if chunk:
            yield chunk


if __name__ == "__main__":
    # 1. Load Config and Prep Paths
    with open(CFG, "r") as f:
        cfg = yaml.safe_load(f)

    wavscp = (
        cfg["datasets"][DATASET].get("wav_scp")
        or cfg["datasets"][DATASET]["train"]["wav_scp"]
    )

    # 2. Estimate total lines for the progress bar (wc -l is faster than readlines)
    print(f"Counting lines in {wavscp}...")
    total_lines = sum(1 for _ in open(wavscp))
    print(f"Found {total_lines} lines.")

    workers = int(os.environ.get("SLURM_CPUS_PER_TASK", os.cpu_count()))
    print(f"Using {workers} workers with chunk size {CHUNK_SIZE}.")

    results = {}

    # 3. Process in Parallel
    with ProcessPoolExecutor(max_workers=workers) as executor:
        # We wrap the generator in a future submission
        chunks = get_chunk_generator(wavscp, CHUNK_SIZE)
        # Submit all chunks
        futures = [executor.submit(process_chunk, c) for c in chunks]
        print(f"Submitted {len(futures)} chunks for processing.")

        # Collect results as they finish
        for future in tqdm(futures, desc="Processing Chunks", unit="chunk"):
            batch_res = future.result()
            for key, dur in batch_res:
                results[key] = dur

    # 4. Save and Report
    print(f"Saving {len(results)} entries to {OUT_NPY}...")
    np.save(OUT_NPY, results)

    durs = list(results.values())
    if durs:
        print(
            f"Mean: {np.mean(durs):.2f}s, Max: {np.max(durs):.2f}s, Total: {sum(durs)/3600:.2f}h"
        )
    else:
        print("No valid entries found.")
