import logging
from huggingface_hub import snapshot_download
from huggingface_hub.utils import LocalEntryNotFoundError


def download_hf_snapshot(
    repo_id: str,
    work_dir: str,
    force_download: bool = False,
    **kwargs,
) -> str:
    """Download a snapshot from Hugging Face Hub to `work_dir`.

    Args:
        repo_id: e.g. "espnet/xeus"
        work_dir: path to local directory where to store snapshot
        force_download: if True, enforce re-download
        **kwargs: other snapshot_download arguments

    Returns:
        The path to the local snapshot folder
    """
    if force_download:
        logging.info(
            f"Force-downloading snapshot for {repo_id} into {work_dir}..."
        )
        path = snapshot_download(
            repo_id=repo_id,
            local_dir=work_dir,
            force_download=True,
            local_files_only=False,
            **kwargs,
        )
        logging.info(f"Downloaded snapshot for {repo_id} to {path}")
        return path

    try:
        path = snapshot_download(
            repo_id=repo_id,
            local_dir=work_dir,
            local_files_only=True,
            **kwargs,
        )
        logging.info(
            f"Using existing local snapshot for {repo_id} at {path}"
        )
        return path
    except LocalEntryNotFoundError:
        logging.info(
            f"No local snapshot found for {repo_id}. Downloading now..."
        )
        path = snapshot_download(
            repo_id=repo_id,
            local_dir=work_dir,
            local_files_only=False,
            **kwargs,
        )
        logging.info(f"Downloaded snapshot for {repo_id} to {path}")
        return path
