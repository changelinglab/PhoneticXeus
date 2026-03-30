import numpy as np
import torch

from src.model.powsm import powsm_inference as powsm_inf


def test_scorefilter_blocks_timestamps_when_notimestamps_present():
    score_filter = powsm_inf.ScoreFilter(
        notimestamps=99,
        first_time=4,
        last_time=6,
        sos=1,
        eos=2,
        vocab_size=10,
    )
    prefix = torch.tensor([0, 1, 99], dtype=torch.long)

    scores, _ = score_filter.score(prefix, None, torch.zeros(1))

    timestamp_scores = scores[4:7]
    assert torch.all(torch.isinf(timestamp_scores))
    assert torch.all(timestamp_scores == -np.inf)


def test_scorefilter_blocks_eos_when_timestamps_unpaired():
    score_filter = powsm_inf.ScoreFilter(
        notimestamps=99,
        first_time=3,
        last_time=5,
        sos=0,
        eos=1,
        vocab_size=8,
    )
    prefix = torch.tensor([7, 3, 4, 5], dtype=torch.long)

    scores, _ = score_filter.score(prefix, None, torch.zeros(1))

    assert scores[1] == -np.inf
    assert torch.all(scores[3:6] == -np.inf)


def test_build_powsm_inference_uses_dependencies(monkeypatch, tmp_path):
    called = {}

    class DummyInference:
        def __init__(self, **kwargs):
            called["inference_kwargs"] = kwargs

    class DummyTokenizer:
        def __init__(self, model_path):
            called["tokenizer_path"] = model_path

    def fake_build_powsm(**kwargs):
        called["build_powsm_kwargs"] = kwargs
        return object()

    monkeypatch.setattr(powsm_inf, "PowsmInference", DummyInference)
    monkeypatch.setattr(powsm_inf, "SentencepiecesTokenizer", DummyTokenizer)
    monkeypatch.setattr(powsm_inf, "build_powsm", fake_build_powsm)

    inference = powsm_inf.build_powsm_inference(
        work_dir=str(tmp_path),
        hf_repo="espnet/test",
        force_download=True,
        config_file="cfg",
        model_file="mdl",
        bpemodel="bpe.model",
        stats_file="stats",
        device="cuda",
        dtype="float16",
        beam_size=4,
        ctc_weight=0.7,
        penalty=0.1,
        nbest=2,
        normalize_length=True,
        maxlenratio=1.0,
        minlenratio=0.5,
    )

    assert isinstance(inference, DummyInference)
    assert called["build_powsm_kwargs"] == {
        "work_dir": str(tmp_path),
        "hf_repo": "espnet/test",
        "force": True,
        "config_file": "cfg",
        "model_file": "mdl",
        "stats_file": "stats",
    }
    assert called["tokenizer_path"] == "bpe.model"
    assert called["inference_kwargs"]["beam_size"] == 4
    assert called["inference_kwargs"]["ctc_weight"] == 0.7
