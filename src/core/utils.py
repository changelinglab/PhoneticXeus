import os
import torch
import torchaudio
from tqdm import tqdm
from huggingface_hub import snapshot_download
from huggingface_hub.utils import LocalEntryNotFoundError
import logging


def download_hf_snapshot(
    repo_id: str,
    work_dir: str,
    force_download: bool = False,
    **kwargs,
) -> str:
    """
    Download a snapshot from Hugging Face Hub to `work_dir`, but skip network interaction if
    the snapshot already exists locally.

    Args:
        repo_id: e.g. "facebook/whisper-large"
        work_dir: path to local directory where to store snapshot
        force_download: if True, enforce re-download if remote snapshot differs
        **kwargs: other snapshot_download arguments (token, repo_type, allow_patterns, etc.)

    Returns:
        The path to the local snapshot folder (i.e. work_dir)
    """
    # If the caller explicitly requested a re-download, skip local-only mode.
    if force_download:
        logging.info(f"Force-downloading snapshot for {repo_id} into {work_dir}...")
        path = snapshot_download(
            repo_id=repo_id,
            local_dir=work_dir,
            force_download=True,
            local_files_only=False,
            **kwargs,
        )
        logging.info(f"Downloaded snapshot for {repo_id} to {path}")
        return path

    # First try local-only (no network)
    try:
        path = snapshot_download(
            repo_id=repo_id,
            local_dir=work_dir,
            local_files_only=True,
            **kwargs,
        )
        logging.info(f"Using existing local snapshot for {repo_id} at {path}")
        return path
    except LocalEntryNotFoundError:
        # Local snapshot doesn't exist — allow network and download.
        logging.info(f"No local snapshot found for {repo_id}. Downloading now...")
        path = snapshot_download(
            repo_id=repo_id,
            local_dir=work_dir,
            local_files_only=False,
            **kwargs,
        )
        logging.info(f"Downloaded snapshot for {repo_id} to {path}")
        return path


def resample_dataset(
    metadata_df,
    path_key,
    src_data_dir,
    tgt_data_dir,
    tgt_sr,
    force_resample=False,
):
    """Resample all audio in `metadata_df[path_key]` from src_sr -> tgt_sr into tgt_data_dir on GPU, if available."""
    if os.path.exists(tgt_data_dir) and not force_resample:
        logging.info(f"Found existing resampled data at {tgt_data_dir}, skipping.")
        return

    os.makedirs(tgt_data_dir, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    resamplers = {}
    logging.info(f"Resampling audio to {tgt_sr} Hz into {tgt_data_dir}...")

    cnt = 0
    for _, row in tqdm(
        metadata_df.iterrows(), total=len(metadata_df), desc="Resampling"
    ):
        rel_path = row[path_key]
        src = os.path.join(src_data_dir, rel_path)
        dst = os.path.join(tgt_data_dir, rel_path)
        os.makedirs(os.path.dirname(dst), exist_ok=True)

        if os.path.exists(dst) and not force_resample:
            continue

        wav, src_sr = torchaudio.load(src)
        if src_sr not in resamplers:
            # cache resampler
            resamplers[src_sr] = torchaudio.transforms.Resample(
                orig_freq=src_sr, new_freq=tgt_sr
            ).to(device)
        wav = wav.to(device)
        with torch.no_grad():
            if src_sr == tgt_sr:
                wav_resampled = wav.cpu()
            else:
                wav_resampled = resamplers[src_sr](wav).cpu()
                cnt += 1
        torchaudio.save(dst, wav_resampled, tgt_sr)
    logging.info(f"Resampled {cnt} files to {tgt_data_dir}.")
