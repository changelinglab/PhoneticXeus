"""Tests for utility functions."""

import pytest
from omegaconf import DictConfig, OmegaConf

from src.utils.utils import extras, get_metric_value, task_wrapper


def test_extras_no_config():
    """Test extras function with no extras config."""
    cfg = OmegaConf.create({"paths": {"output_dir": "/tmp"}})
    # Should not raise
    extras(cfg)


def test_extras_ignore_warnings(capsys):
    """Test extras function with ignore_warnings."""
    cfg = OmegaConf.create({
        "extras": {"ignore_warnings": True},
        "paths": {"output_dir": "/tmp"}
    })
    extras(cfg)
    # Should have logged about disabling warnings
    captured = capsys.readouterr()
    # Note: actual logging output depends on logger configuration


def test_extras_enforce_tags(monkeypatch):
    """Test extras function with enforce_tags."""
    # Mock input to avoid interactive prompt
    monkeypatch.setattr("builtins.input", lambda _: "test_tag")
    
    cfg = OmegaConf.create({
        "extras": {"enforce_tags": True},
        "tags": [],
        "paths": {"output_dir": "/tmp"}
    })
    # Should not raise (though it may prompt for input)
    extras(cfg)


def test_extras_print_config():
    """Test extras function with print_config."""
    cfg = OmegaConf.create({
        "extras": {"print_config": True},
        "paths": {"output_dir": "/tmp"}
    })
    # Should not raise
    extras(cfg)


def test_get_metric_value_with_valid_metric():
    """Test get_metric_value with valid metric name."""
    import torch
    
    metric_dict = {
        "val/acc": torch.tensor(0.95),
        "train/loss": torch.tensor(0.5),
    }
    
    value = get_metric_value(metric_dict, "val/acc")
    assert value == pytest.approx(0.95)


def test_get_metric_value_with_none():
    """Test get_metric_value with None metric name."""
    metric_dict = {"val/acc": 0.95}
    
    value = get_metric_value(metric_dict, None)
    assert value is None


def test_get_metric_value_with_missing_metric():
    """Test get_metric_value with missing metric name."""
    metric_dict = {"val/acc": 0.95}
    
    with pytest.raises(Exception, match="Metric value not found"):
        get_metric_value(metric_dict, "missing/metric")


def test_task_wrapper_success():
    """Test task_wrapper with successful task."""
    @task_wrapper
    def dummy_task(cfg):
        return {"metric": 1.0}, {"obj": "test"}
    
    cfg = OmegaConf.create({"paths": {"output_dir": "/tmp"}})
    metrics, objects = dummy_task(cfg)
    
    assert metrics == {"metric": 1.0}
    assert objects == {"obj": "test"}


def test_task_wrapper_exception():
    """Test task_wrapper with exception."""
    @task_wrapper
    def failing_task(cfg):
        raise ValueError("Test error")
    
    cfg = OmegaConf.create({"paths": {"output_dir": "/tmp"}})
    
    with pytest.raises(ValueError, match="Test error"):
        failing_task(cfg)

