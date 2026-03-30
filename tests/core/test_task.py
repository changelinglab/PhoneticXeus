"""Tests for Task class."""

import pytest
from omegaconf import DictConfig, OmegaConf
from lightning import LightningDataModule, LightningModule, Trainer
from unittest.mock import Mock, MagicMock

from src.core.task import Task


class DummyDataModule(LightningDataModule):
    """Dummy datamodule for testing."""
    
    def setup(self, stage=None):
        pass
    
    def train_dataloader(self):
        return None
    
    def val_dataloader(self):
        return None
    
    def test_dataloader(self):
        return None
    
    def predict_dataloader(self):
        return None


class DummyModel(LightningModule):
    """Dummy model for testing."""
    
    def forward(self, x):
        return x


def test_task_initialization():
    """Test Task class initialization."""
    cfg = OmegaConf.create({
        "task_name": "test_task",
        "data": {"_target_": "tests.core.test_task.DummyDataModule"},
        "model": {"_target_": "tests.core.test_task.DummyModel"},
    })
    
    task = Task(cfg)
    
    assert task.task_cfg == cfg
    assert task.name == "test_task"


def test_task_initialization_default_name():
    """Test Task class initialization with default name."""
    cfg = OmegaConf.create({
        "data": {"_target_": "tests.core.test_task.DummyDataModule"},
        "model": {"_target_": "tests.core.test_task.DummyModel"},
    })
    
    task = Task(cfg)
    
    assert task.name == "Task"  # Default name


def test_task_train_method():
    """Test Task.train method."""
    cfg = OmegaConf.create({
        "ckpt_path": None,
    })
    task = Task(cfg)
    
    # Create mock trainer and model
    trainer = Mock(spec=Trainer)
    trainer.fit = Mock()
    trainer.callback_metrics = {"train/loss": 0.5}
    trainer.checkpoint_callback = None
    
    model = Mock(spec=LightningModule)
    datamodule = Mock(spec=LightningDataModule)
    
    metrics, ckpt_path = task.train(trainer, model, datamodule)
    
    trainer.fit.assert_called_once()
    assert metrics == {"train/loss": 0.5}
    assert ckpt_path == ""


def test_task_test_method():
    """Test Task.test method."""
    cfg = OmegaConf.create({
        "ckpt_path": None,
    })
    task = Task(cfg)
    
    trainer = Mock(spec=Trainer)
    trainer.test = Mock()
    trainer.callback_metrics = {"test/acc": 0.9}
    
    model = Mock(spec=LightningModule)
    datamodule = Mock(spec=LightningDataModule)
    
    metrics = task.test(trainer, model, datamodule, "path/to/ckpt.ckpt")
    
    trainer.test.assert_called_once()
    assert metrics == {"test/acc": 0.9}


def test_task_test_method_with_ckpt_path_in_cfg():
    """Test Task.test method with ckpt_path in config."""
    cfg = OmegaConf.create({
        "ckpt_path": "config_ckpt.ckpt",
    })
    task = Task(cfg)
    
    trainer = Mock(spec=Trainer)
    trainer.test = Mock()
    trainer.callback_metrics = {}
    
    model = Mock(spec=LightningModule)
    datamodule = Mock(spec=LightningDataModule)
    
    task.test(trainer, model, datamodule, "provided_ckpt.ckpt")
    
    # Should use ckpt_path from config, not the parameter
    call_args = trainer.test.call_args
    assert call_args[1]["ckpt_path"] == "config_ckpt.ckpt"


def test_task_predict_method():
    """Test Task.predict method."""
    cfg = OmegaConf.create({
        "ckpt_path": None,
    })
    task = Task(cfg)
    
    trainer = Mock(spec=Trainer)
    trainer.predict = Mock(return_value=["pred1", "pred2"])
    
    model = Mock(spec=LightningModule)
    datamodule = Mock(spec=LightningDataModule)
    
    predictions = task.predict(trainer, model, datamodule, "path/to/ckpt.ckpt")
    
    trainer.predict.assert_called_once()
    assert predictions == ["pred1", "pred2"]


def test_task_run_distributed_inference(monkeypatch):
    """Test Task.run_distributed_inference method."""
    cfg = OmegaConf.create({
        "data": {
            "_target_": "tests.core.test_task.DummyDataModule"
        },
        "inference": {
            "inference_runner": {"_target_": "dummy", "device": "cpu"},
            "num_workers": 1,
            "out_file": "/tmp/test.json",
            "passthrough_keys": [],
        }
    })
    task = Task(cfg)
    
    # Mock the datamodule and distributed inference
    mock_datamodule = Mock()
    mock_dataloader = Mock()
    mock_dataset = Mock()
    mock_dataset.__len__ = Mock(return_value=5)
    mock_dataloader.dataset = mock_dataset
    mock_datamodule.predict_dataloader = Mock(return_value=mock_dataloader)
    
    mock_run_distributed = Mock()
    monkeypatch.setattr("src.core.task.run_distributed_inference_", mock_run_distributed)
    monkeypatch.setattr("hydra.utils.instantiate", lambda x: mock_datamodule)
    
    task.run_distributed_inference()
    
    mock_run_distributed.assert_called_once()

