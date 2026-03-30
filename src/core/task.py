"""Main task class that controls execution flow for all stages."""

from typing import Any, Dict, List, Tuple

import json
import hydra
import os
from lightning import Callback, LightningDataModule, LightningModule, Trainer
from lightning.pytorch.loggers import Logger
from omegaconf import DictConfig
import torch
from src.core.distributed_inference import run_distributed_inference_

from src.utils import (
    RankedLogger,
    instantiate_callbacks,
    instantiate_loggers,
    log_hyperparameters,
)

log = RankedLogger(__name__, rank_zero_only=True)


class Task:
    def __init__(self, cfg: DictConfig) -> None:
        self.task_cfg = cfg
        self.name = cfg.get("task_name", "Task")

    def _load_partial_ckpt_for_training(
        self, model: LightningModule, ckpt_path: str
    ) -> None:
        if not ckpt_path:
            return
        log.info(f"Loading (non-strict) checkpoint weights from: {ckpt_path}")
        ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
        state_dict = ckpt.get(
            "state_dict", ckpt
        )  # handle raw state_dict checkpoints too
        load_info = model.load_state_dict(state_dict, strict=False)
        if load_info:
            log.info(f"Missing keys when loading checkpoint: {load_info.missing_keys}")
            log.info(
                f"Unexpected keys when loading checkpoint: {load_info.unexpected_keys}"
            )

    def train(
        self, trainer: Trainer, model: LightningModule, datamodule: LightningDataModule
    ) -> Tuple[Dict[str, Any], str]:
        log.info("Starting training!")
        if self.task_cfg.get("ckpt_path") is not None:
            self._load_partial_ckpt_for_training(model, self.task_cfg.ckpt_path)
        trainer.fit(
            model=model,
            datamodule=datamodule,
        )
        ckpt_cb = getattr(trainer, "checkpoint_callback", None)
        ckpt_path = getattr(ckpt_cb, "best_model_path", "") if ckpt_cb else ""
        return dict(trainer.callback_metrics), ckpt_path

    def test(
        self,
        trainer: Trainer,
        model: LightningModule,
        datamodule: LightningDataModule,
        ckpt_path: str,
    ) -> Dict[str, Any]:
        if self.task_cfg.get("ckpt_path") is not None:
            ckpt_path = self.task_cfg.ckpt_path
        # if not ckpt_path:
        #     log.error("Testing ckpt not provided!")
        # else:
        log.info(f"Ckpt path: {ckpt_path}")
        trainer.test(
            model=model,
            datamodule=datamodule,
            ckpt_path=ckpt_path or None,
            weights_only=False,
        )
        return dict(trainer.callback_metrics)

    def predict(
        self,
        trainer: Trainer,
        model: LightningModule,
        datamodule: LightningDataModule,
        ckpt_path: str,
    ):
        if self.task_cfg.get("ckpt_path") is not None:
            ckpt_path = self.task_cfg.ckpt_path
        log.info("Starting prediction!")
        return trainer.predict(
            model=model,
            datamodule=datamodule,
            ckpt_path=ckpt_path or None,
            weights_only=False,
        )

    def run_distributed_inference(self):
        """Wraps the utility function for distributed inference."""
        log.info("Starting distributed prediction!")
        run_distributed_inference_(
            dataset_cfg=self.task_cfg.data,
            inference_config=self.task_cfg.inference.inference_runner,
            inference_call_args=self.task_cfg.inference.get("inference_call_args"),
            num_workers=self.task_cfg.inference.num_workers,
            out_file=self.task_cfg.inference.out_file,
            passthrough_keys=self.task_cfg.inference.get("passthrough_keys"),
            limit_samples=self.task_cfg.inference.get("limit_samples"),
        )

    def run_experiment(self) -> Tuple[Dict[str, Any], Dict[str, Any]]:
        log.info(f"Instantiating datamodule <{self.task_cfg.data._target_}>")
        datamodule: LightningDataModule = hydra.utils.instantiate(self.task_cfg.data)

        log.info("Instantiating loggers...")
        logger: List[Logger] = instantiate_loggers(self.task_cfg.get("logger"))

        log.info(f"Instantiating model <{self.task_cfg.model._target_}>")
        model: LightningModule = hydra.utils.instantiate(self.task_cfg.model)

        log.info("Instantiating callbacks...")
        callbacks: List[Callback] = instantiate_callbacks(
            self.task_cfg.get("callbacks")
        )

        log.info(f"Instantiating trainer <{self.task_cfg.trainer._target_}>")
        trainer: Trainer = hydra.utils.instantiate(
            self.task_cfg.trainer, callbacks=callbacks, logger=logger
        )

        object_dict = {
            "cfg": self.task_cfg,
            "datamodule": datamodule,
            "model": model,
            "callbacks": callbacks,
            "logger": logger,
            "trainer": trainer,
        }

        if logger:
            log.info("logging hyperparameters!")
            log_hyperparameters(object_dict)

        metrics: Dict[str, Any] = {}
        ckpt_path = ""

        if self.task_cfg.get("train"):
            train_metrics, ckpt_path = self.train(trainer, model, datamodule)
            metrics.update(train_metrics)

        if self.task_cfg.get("test"):
            test_metrics = self.test(trainer, model, datamodule, ckpt_path)
            metrics.update(test_metrics)

        if (
            self.task_cfg.get("predict", False)
            or self.task_cfg.get("pred_file", None) is not None
        ):
            preds = self.predict(trainer, model, datamodule, ckpt_path)
            object_dict["predictions"] = preds
            # Write predictions
            pred_file = self.task_cfg.get("pred_file", None)
            if pred_file is not None:
                os.makedirs(os.path.dirname(pred_file), exist_ok=True)
                json.dump(preds, open(pred_file, "w", encoding="utf-8"), indent=2)
                log.info(f"Wrote predictions to {pred_file}")

        return metrics, object_dict
