"""Module for phone recognition.

Usage:
    python -m src.recipe.phone_recognition.model_module
"""

from typing import Any, Dict, List, Optional
import torch
import torch.nn as nn
from torch.optim.lr_scheduler import LambdaLR
from lightning import LightningModule
from torchmetrics import MinMetric, MeanMetric
from lightning.pytorch.utilities import grad_norm


def get_w2v2ph_schedule(
    optimizer,
    num_training_steps: int,
    encoder_unfreeze_step: int = 0,
    warmup_fraction: float = 0.1,
    constant_fraction: float = 0.4,
):
    """
    Implements the schedule:
    1. Warmup (0-10%): Linear increase from 0 to 1.
    2. Constant (10-50%): Constant at 1.
    3. Decay (50-100%): Linear decay from 1 to 0.

    Additionally, if encoder_unfreeze_step > 0, the encoder parameters are
    frozen until that step is reached.
    """
    assert 0.0 <= warmup_fraction <= 1.0, "warmup_fraction must be in [0.0, 1.0]"
    assert 0.0 <= constant_fraction <= 1.0, "constant_fraction must be in [0.0, 1.0]"
    assert (
        warmup_fraction + constant_fraction <= 1.0
    ), "Sum of warmup_fraction and constant_fraction must be less than or equal to 1.0"

    def three_piece_factor(current_step: int):
        warmup_steps = int(warmup_fraction * num_training_steps)
        # The constant phase lasts for 40% of updates, so it ends at 10% + 40% = 50%
        constant_end_step = int(
            (warmup_fraction + constant_fraction) * num_training_steps
        )

        if current_step < warmup_steps:
            # Phase 1: Linear Warmup
            return float(current_step) / float(max(1, warmup_steps))
        elif current_step < constant_end_step:
            # Phase 2: Constant
            return 1.0
        else:
            # Phase 3: Linear Decay
            decay_steps = num_training_steps - constant_end_step
            progress = current_step - constant_end_step
            return max(0.0, 1.0 - (progress / float(max(1, decay_steps))))

    def encoder_lambda(current_step: int):
        # encoder layers
        if current_step < encoder_unfreeze_step:
            return 0.0
        else:
            return three_piece_factor(current_step)

    def head_lambda(current_step: int):
        # ctc head
        return three_piece_factor(current_step)

    return LambdaLR(optimizer, lr_lambda=[head_lambda, encoder_lambda])


class PhoneRecognitionModel(LightningModule):
    def __init__(
        self,
        net: nn.Module,
        optimizer: torch.optim.Optimizer,
        scheduler: Optional[torch.optim.lr_scheduler._LRScheduler] = None,
        inference_strategy: Optional[Any] = None,
        dev_splits: Optional[List[str]] = None,
    ) -> None:
        super().__init__()
        self.save_hyperparameters(logger=False, ignore=["net", "inference_strategy"])
        self.net = net
        self.inference_strategy = inference_strategy
        self.blank_id: Optional[int] = getattr(self.net, "blank_id", None)
        self.dev_splits = dev_splits or []
        self.losses = nn.ModuleDict(
            {
                s: MeanMetric()
                for s in ["trainloss", "testloss"]
                + [f"{x}loss" for x in self.dev_splits]
            }
        )
        self.cers = nn.ModuleDict({s: MeanMetric() for s in self.dev_splits})
        self.val_loss_best = MinMetric()

    def forward(self, batch: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        return self.net(
            speech=batch["speech"],
            speech_lengths=batch["speech_length"],
            text=batch["text"],
            text_lengths=batch["text_length"],
            lang_sym=batch.get("lang_sym"),
            accent_sym=batch.get("accent_sym"),
            asr_text_tokens=batch.get("asr_text_tokens"),
            asr_text_length=batch.get("asr_text_length"),
        )

    def on_save_checkpoint(self, checkpoint: dict) -> None:
        net_config = getattr(self.net, "_net_config", None)
        if net_config is not None:
            checkpoint["net_config"] = net_config

    def on_before_optimizer_step(self, optimizer) -> None:
        self.log_dict(grad_norm(self, norm_type=2))

    def on_train_start(self) -> None:
        for m in self.losses.values():
            m.reset()
        for m in self.cers.values():
            m.reset()
        self.val_loss_best.reset()

    def _run_stage(self, batch, split: str, log_on_step: bool):
        """Run forward pass and log metrics for a split."""
        out = self(batch)
        self.losses[f"{split}loss"](out["loss"].detach())
        self.log(
            f"{split}/loss",
            self.losses[f"{split}loss"],
            on_step=log_on_step,
            on_epoch=True,
            prog_bar=True,
        )

        stats = out.get("stats", {})
        if split in self.cers and stats.get("cer_ctc") is not None:
            self.cers[split](stats["cer_ctc"])
            self.log(
                f"{split}/cer",
                self.cers[split],
                on_step=log_on_step,
                on_epoch=True,
                prog_bar=True,
            )
        for k, v in stats.items():
            if k == "cer_ctc" and split in self.cers:
                continue
            self.log(
                f"{split}/{k}", v, on_step=log_on_step, on_epoch=True, prog_bar=False
            )
        return out

    def training_step(self, batch, batch_idx, dataloader_idx: int = 0) -> torch.Tensor:
        return self._run_stage(batch, "train", log_on_step=True)["loss"]

    def validation_step(self, batch, batch_idx, dataloader_idx: int = 0) -> None:
        self._run_stage(batch, self.dev_splits[dataloader_idx], log_on_step=False)

    def on_validation_epoch_end(self) -> None:
        loss = self.losses[f"{self.dev_splits[0]}loss"].compute()
        self.val_loss_best(loss)
        self.log(
            "val/loss_best", self.val_loss_best.compute(), sync_dist=True, prog_bar=True
        )
        self.log("val/loss", loss, sync_dist=True, prog_bar=False)

    def test_step(self, batch, batch_idx) -> None:
        self._run_stage(batch, "test", log_on_step=False)

    def predict_step(self, batch, batch_idx, dataloader_idx=None):
        if self.inference_strategy is None:
            raise RuntimeError("Inference engine not provided.")
        speech = batch["speech"]
        speech_lengths = batch["speech_length"]
        results = self.inference_strategy(
            model=self.net, speech=speech, speech_lengths=speech_lengths
        )
        return results

    def set_inference_strategy(self, inference_strategy_cls: Any) -> None:
        self.inference_strategy = inference_strategy_cls(
            self.net.token_list, self.blank_id
        )

    def configure_optimizers(self) -> Dict[str, Any]:
        trainable_params = self.net.get_trainable_parameters()
        # must have two keys - head and encoder
        # for ctc and encoder respectively
        optimizer = self.hparams.optimizer(
            params=[
                {"params": trainable_params["head"], "name": "head"},
                {"params": trainable_params["encoder"], "name": "encoder"},
            ]
        )
        scheduler_cls = self.hparams.scheduler

        if scheduler_cls is not None:
            scheduler = scheduler_cls(optimizer=optimizer)
            return {
                "optimizer": optimizer,
                "lr_scheduler": {
                    "scheduler": scheduler,
                    # only for ReduceLROnPlateau
                    "monitor": "val/loss",
                    "interval": "step",
                    "frequency": 1,
                },
            }

        return {"optimizer": optimizer}


if __name__ == "__main__":
    # python -m src.recipe.phone_recognition.model_module
    from src.data.kaldi_pretraining_dataset import build_kaldi_pretraining_datamodule
    from src.model.wav2vec2.builders import build_wav2vec2pr

    net = build_wav2vec2pr(
        hf_repo="facebook/mms-300m",
        vocab_file="src/model/xeusphoneme/resources/ipa_vocab.json",
        freeze_frontend=True,
    )
    net.get_trainable_parameters()
    print("Built Wav2Vec2 model ")

    datamodule = build_kaldi_pretraining_datamodule(
        train_splits=["dev_1k"],  # "train_accentmix_multi",
        dev_splits=[
            "dev_1k",
            "dev_gmuaccent",
            "dev_buckeye",
            "dev_epadb",
            "dev_speechoceanotth",
            "dev_l2arctic",
        ],
        predict_split="predict",
        dataset_config_path="configs/data/ipapack_index.yaml",
        batch_size=1,
        num_workers=1,
        vocab_file="src/model/xeusphoneme/resources/ipa_vocab.json",
        limit_samples=2,
    )
    datamodule.setup()
    batch = next(iter(datamodule.train_dataloader()))
    print(batch)
    model = PhoneRecognitionModel(
        net=net,
        optimizer=torch.optim.AdamW,
        scheduler=None,
        dev_splits=[
            "dev_1k",
            "dev_gmuaccent",
            "dev_buckeye",
            "dev_epadb",
            "dev_speechoceanotth",
            "dev_l2arctic",
        ],
    )
    out = model.training_step(*batch)
    print(out)
