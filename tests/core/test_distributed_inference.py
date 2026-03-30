"""Tests for distributed inference utilities."""

import json
import os
from dataclasses import dataclass
from unittest.mock import MagicMock

import pytest
import torch
import multiprocessing

from src.core.distributed_inference import (
    default_encoder,
    run_distributed_inference_,
    get_dataset_from_cfg,
)

# -------------------------------------------------------------------------
# Dummy Classes
# -------------------------------------------------------------------------


@dataclass
class DummyData:
    value: int
    name: str


class DummyDataset:
    def __init__(self, size=10):
        self.size = size
        self.data = [
            {"speech": torch.randn(10), "key": f"item_{i}"} for i in range(size)
        ]

    def __len__(self):
        return self.size

    def __getitem__(self, idx):
        return self.data[idx]


class DummyDataModule:
    def __init__(self, size=10):
        self.dataset = DummyDataset(size=size)

    def prepare_data(self):
        pass

    def setup(self, stage=None):
        pass

    def predict_dataloader(self):
        loader = MagicMock()
        loader.dataset = self.dataset
        return loader


class DummyInference:
    def __init__(self, device="cpu"):
        self.device = device

    def __call__(self, speech, **kwargs):
        return f"pred_len_{speech.shape[0]}"


# -------------------------------------------------------------------------
# Synchronous Pool Mock
# -------------------------------------------------------------------------


class MockPool:
    """Mocks multiprocessing.Pool to run code synchronously in the main thread."""

    def __init__(self, processes=None, initializer=None):
        if initializer:
            initializer()

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        pass

    def imap_unordered(self, func, iterable):
        # Run synchronously
        for item in iterable:
            yield func(item)


class MockContext:
    """Mocks multiprocessing.get_context."""

    def Pool(self, processes=None, initializer=None):
        return MockPool(processes, initializer)


# -------------------------------------------------------------------------
# Fixtures
# -------------------------------------------------------------------------


@pytest.fixture
def mock_env(monkeypatch):
    """
    1. Mocks hydra.utils.instantiate to return dummy objects.
    2. Mocks multiprocessing.get_context to avoid spawning real processes.
    """

    # 1. Mock Hydra
    def mock_instantiate(config, **kwargs):
        target = config.get("_target_")
        if target == "dummy_dataset":
            return DummyDataModule(size=config.get("size", 10))
        if target == "dummy_inference":
            return DummyInference(device=kwargs.get("device", "cpu"))
        return MagicMock()

    monkeypatch.setattr("hydra.utils.instantiate", mock_instantiate)

    # 2. Mock Multiprocessing to run synchronously
    # This ensures the worker code sees the 'mock_instantiate' above.
    def mock_get_context(method):
        return MockContext()

    monkeypatch.setattr("multiprocessing.get_context", mock_get_context)


# -------------------------------------------------------------------------
# Tests
# -------------------------------------------------------------------------


def test_default_encoder_with_dataclass():
    obj = DummyData(value=42, name="test")
    assert default_encoder(obj) == {"value": 42, "name": "test"}


def test_default_encoder_with_object():
    class TestObj:
        def __str__(self):
            return "TestObjString"

    assert default_encoder(TestObj()) == "TestObjString"


def test_get_dataset_from_cfg(mock_env):
    cfg = {"_target_": "dummy_dataset", "size": 5}
    dataset = get_dataset_from_cfg(cfg)
    assert isinstance(dataset, DummyDataset)
    assert len(dataset) == 5


@pytest.mark.slow
def test_run_distributed_inference_flow(tmp_path, mock_env, monkeypatch):
    """Test full flow with 0 error records."""
    monkeypatch.setenv("SLURM_ARRAY_TASK_ID", "0")
    monkeypatch.setenv("SLURM_ARRAY_TASK_COUNT", "1")

    out_base = tmp_path / "inference_out"
    expected_out_file = tmp_path / "inference_out.0.jsonl"

    dataset_cfg = {"_target_": "dummy_dataset", "size": 6}
    inference_cfg = {"_target_": "dummy_inference", "device": "cpu"}

    run_distributed_inference_(
        dataset_cfg=dataset_cfg,
        inference_config=inference_cfg,
        num_workers=2,
        out_file=str(out_base),
        passthrough_keys=["key"],
    )

    assert expected_out_file.exists()

    with open(expected_out_file, "r") as f:
        lines = [json.loads(line) for line in f]

    assert len(lines) == 6

    indices_found = []
    for entry in lines:
        # Check for error records
        if "__error__" in entry:
            pytest.fail(f"Found error record in output: {entry}")

        idx_str = list(entry.keys())[0]
        indices_found.append(int(idx_str))
        data = entry[idx_str]

        assert "pred" in data
        assert data["passthrough"]["key"] == f"item_{idx_str}"

    assert sorted(indices_found) == [0, 1, 2, 3, 4, 5]


def test_slurm_sharding_logic(tmp_path, mock_env, monkeypatch):
    """Test that Task 1/2 gets the second half of data."""
    monkeypatch.setenv("SLURM_ARRAY_TASK_ID", "1")
    monkeypatch.setenv("SLURM_ARRAY_TASK_COUNT", "2")

    out_base = tmp_path / "inference_part"
    expected_out_file = tmp_path / "inference_part.1.jsonl"

    dataset_cfg = {"_target_": "dummy_dataset", "size": 10}
    inference_cfg = {"_target_": "dummy_inference", "device": "cpu"}

    run_distributed_inference_(
        dataset_cfg=dataset_cfg,
        inference_config=inference_cfg,
        num_workers=1,
        out_file=str(out_base),
    )

    with open(expected_out_file, "r") as f:
        lines = [json.loads(line) for line in f]

    indices_found = [int(list(x.keys())[0]) for x in lines]
    assert sorted(indices_found) == [5, 6, 7, 8, 9]


def test_limit_samples(tmp_path, mock_env, monkeypatch):
    """Test limit_samples cuts off processing."""
    monkeypatch.setenv("SLURM_ARRAY_TASK_ID", "0")
    monkeypatch.setenv("SLURM_ARRAY_TASK_COUNT", "1")

    out_base = tmp_path / "inference_limit"
    expected_out_file = tmp_path / "inference_limit.0.jsonl"

    dataset_cfg = {"_target_": "dummy_dataset", "size": 100}
    inference_cfg = {"_target_": "dummy_inference", "device": "cpu"}

    run_distributed_inference_(
        dataset_cfg=dataset_cfg,
        inference_config=inference_cfg,
        num_workers=1,
        out_file=str(out_base),
        limit_samples=5,
    )

    with open(expected_out_file, "r") as f:
        lines = f.readlines()

    assert len(lines) == 5


def test_missing_out_file_assertion():
    with pytest.raises(AssertionError, match="Please provide an out_file"):
        run_distributed_inference_(dataset_cfg={}, inference_config={}, out_file=None)
