import pyarrow.parquet as pq  # before torch
from typing import Any, Dict, Optional, Tuple
import os

# Limit the number of threads to prevent hangup with multiprocessing
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
import hydra
import lightning as L
import rootutils
from omegaconf import DictConfig

rootutils.setup_root(__file__, indicator=".project-root", pythonpath=True)

# ------------------------------------------------------------------------------------ #
# the setup_root above is equivalent to:
# - adding project root dir to PYTHONPATH
#       (so you don't need to force user to install project as a package)
#       (necessary before importing any local modules e.g. `from src import utils`)
# - setting up PROJECT_ROOT environment variable
#       (which is used as a base for paths in "configs/paths/default.yaml")
#       (this way all filepaths are the same no matter where you run the code)
# - loading environment variables from ".env" in root dir
# ------------------------------------------------------------------------------------ #

from src.core.task import Task
from src.utils import (
    RankedLogger,
    extras,
    get_metric_value,
    task_wrapper,
)

log = RankedLogger(__name__, rank_zero_only=True)


@task_wrapper
def run_task(cfg: DictConfig) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    """
    :param cfg: A DictConfig configuration composed by Hydra.
    :return: A tuple with metrics and dict with all instantiated objects.
    """
    # set seed for random number generators in pytorch, numpy and python.random
    if cfg.get("seed"):
        L.seed_everything(cfg.seed, workers=True)
    task = Task(cfg)

    if cfg.get("distributed_predict", False):
        if cfg.get("train", False) or cfg.get("test", False):
            log.warning(
                "Distributed inference cannot be combined with training or testing. "
                "Please set 'train' and 'test' to False in the configuration."
            )
        task.run_distributed_inference()
        return None, None  # signature consistency

    # run normal training/testing/prediction with lightning
    metric_dict, object_dict = task.run_experiment()
    return metric_dict, object_dict


@hydra.main(version_base="1.3", config_path="../configs", config_name="main.yaml")
def main(cfg: DictConfig) -> Optional[float]:
    """Main entry point for training.

    :param cfg: DictConfig configuration composed by Hydra.
    :return: Optional[float] with optimized metric value.
    """
    # apply extra utilities
    # (e.g. ask for tags if none are provided in cfg, print cfg tree, etc.)
    extras(cfg)

    # run task
    metric_dict, _ = run_task(cfg)
    # safely retrieve metric value for hydra-based hyperparameter optimization
    metric_value = get_metric_value(
        metric_dict=metric_dict, metric_name=cfg.get("optimized_metric")
    )
    return metric_value


if __name__ == "__main__":
    main()
