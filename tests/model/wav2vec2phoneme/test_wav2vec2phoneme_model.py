# test_wav2vec2_phoneme_model.py

import types
from typing import List

import numpy as np
import pytest
import torch
import torch.nn as nn

import src.model.wav2vec2phoneme.wav2vec2phoneme_model as w2v2ph


# -------------------------
# Dummy HF-like components
# -------------------------


class DummyFeatureExtractor:
    def __init__(self, sampling_rate: int = 16000):
        self.sampling_rate = sampling_rate


class DummyProcessor:
    """
    Minimal stand-in for Wav2Vec2Processor.
    """

    def __init__(self):
        self.feature_extractor = DummyFeatureExtractor()

    @classmethod
    def from_pretrained(cls, hf_repo: str):
        return cls()

    def __call__(
        self,
        batch: List[np.ndarray],
        sampling_rate: int,
        return_tensors: str,
        padding: bool,
    ):
        # Simple zero-padded batching
        assert return_tensors == "pt"
        max_len = max(x.shape[0] for x in batch)
        padded = []
        attention = []
        for x in batch:
            pad_len = max_len - x.shape[0]
            padded.append(np.pad(x, (0, pad_len), mode="constant"))
            attention.append(
                np.concatenate(
                    [np.ones_like(x, dtype=np.int64), np.zeros(pad_len, dtype=np.int64)]
                )
            )

        input_values = torch.tensor(np.stack(padded), dtype=torch.float32)
        attention_mask = torch.tensor(np.stack(attention), dtype=torch.long)

        out = types.SimpleNamespace()
        out.input_values = input_values
        out.attention_mask = attention_mask
        return out


class DummyConfig:
    def __init__(self):
        # conv_stride product gives overall downsampling factor
        self.conv_stride = [2, 2, 2, 2]
        self.output_hidden_size = 16
        self.vocab_size = 32
        self.pad_token_id = 0


class DummyModel(nn.Module):
    """
    Minimal stand-in for Wav2Vec2ForCTC.
    """

    def __init__(self):
        super().__init__()
        self.config = DummyConfig()
        # just to have a device attribute
        self.device = torch.device("cpu")

    @classmethod
    def from_pretrained(cls, hf_repo: str):
        return cls()

    def forward(
        self,
        input_values=None,
        attention_mask=None,
        output_hidden_states=False,
        return_dict=True,
    ):
        batch_size, t_in = input_values.shape
        # pretend encoder downsamples by factor 2
        t_out = t_in // 2
        hidden_size = self.config.output_hidden_size
        vocab_size = self.config.vocab_size

        # simple deterministic outputs
        logits = torch.zeros(batch_size, t_out, vocab_size)
        last_hidden = torch.zeros(batch_size, t_out, hidden_size)

        hidden_states = [last_hidden]  # we only care about the last one
        out = types.SimpleNamespace()
        out.logits = logits
        out.hidden_states = hidden_states
        return out

    def _get_feat_extract_output_lengths(
        self, attention_lengths: torch.Tensor
    ) -> torch.Tensor:
        # mirror the same "factor 2" downsampling
        return attention_lengths // 2


# -------------------------
# Helper to build model with mocks
# -------------------------


@pytest.fixture
def patched_model(monkeypatch):
    # Patch HF classes used in the module to our dummy ones
    monkeypatch.setattr(w2v2ph, "Wav2Vec2Processor", DummyProcessor)
    monkeypatch.setattr(w2v2ph, "Wav2Vec2ForCTC", DummyModel)

    model = w2v2ph.Wav2Vec2PhonemeModel("dummy/repo")
    return model


# -------------------------
# Tests for preprocess_inputs_wav2vec2
# -------------------------


def test_preprocess_inputs_shape_and_keys():
    preprocessor = DummyProcessor()
    # two waveforms with different lengths
    speech = [torch.randn(16000), torch.randn(8000)]
    speech_lengths = torch.tensor([16000, 8000])
    device = torch.device("cpu")

    out = w2v2ph.preprocess_inputs_wav2vec2(
        preprocessor=preprocessor,
        speech=speech,
        speech_lengths=speech_lengths,
        device=device,
    )

    assert set(out.keys()) == {"input_values", "attention_mask"}
    input_values = out["input_values"]
    attention_mask = out["attention_mask"]

    assert input_values.shape[0] == 2
    assert attention_mask.shape[0] == 2
    assert input_values.dtype == torch.float32
    assert attention_mask.dtype == torch.long


# -------------------------
# Tests for encode / ctc_logits
# -------------------------


def test_encode_returns_hidden_states_and_lengths(patched_model):
    model = patched_model
    speech = [torch.randn(16000), torch.randn(8000)]
    speech_lengths = torch.tensor([16000, 8000])

    enc_out, enc_lens = model.encode(speech, speech_lengths)

    assert enc_out.ndim == 3  # (B, T, D)
    B, T, D = enc_out.shape
    assert B == 2
    assert D == model.encoder_output_size()

    assert enc_lens.shape == (2,)
    # lengths should be positive and less than or equal to T
    assert (enc_lens > 0).all()
    assert (enc_lens <= T).all()


def test_ctc_logits_returns_logits_and_lengths(patched_model):
    model = patched_model
    speech = [torch.randn(16000), torch.randn(8000)]
    speech_lengths = torch.tensor([16000, 8000])

    logits, logit_lens = model.ctc_logits(speech, speech_lengths)

    assert logits.ndim == 3  # (B, T, V)
    B, T, V = logits.shape
    assert B == 2
    assert V == model.vocab_size

    assert logit_lens.shape == (2,)
    assert (logit_lens > 0).all()
    assert (logit_lens <= T).all()


# -------------------------
# Test points_by_frames caching behavior
# -------------------------


def test_points_by_frames_cached(monkeypatch, patched_model):
    model = patched_model

    # Counter to see how many times encode is called
    call_count = {"n": 0}

    def fake_encode(speech, speech_lengths):
        call_count["n"] += 1
        # pretend 16000 input -> 100 frames
        # points_by_frames = 16000 // 100 = 160
        dummy_enc = torch.zeros(1, 100, model.encoder_output_size())
        dummy_lens = torch.tensor([100])
        return dummy_enc, dummy_lens

    # patch the *instance* method encode
    monkeypatch.setattr(model, "encode", fake_encode)

    r1 = model.points_by_frames()
    r2 = model.points_by_frames()

    assert r1 == 160
    assert r2 == 160
    # encode should be called only once due to caching
    assert call_count["n"] == 1


# -------------------------
# Test forced_align wiring
# -------------------------


def test_forced_align_calls_torchaudio(monkeypatch, patched_model):
    model = patched_model

    # fake forced_align to inspect args and return dummy outputs
    captured = {}

    def fake_forced_align(log_probs, text, logit_lengths, text_lengths, blank):
        captured["log_probs_shape"] = tuple(log_probs.shape)
        captured["text_shape"] = tuple(text.shape)
        captured["logit_lengths"] = logit_lengths.clone()
        captured["text_lengths"] = text_lengths.clone()
        captured["blank"] = blank

        B, T, V = log_probs.shape
        align_label = torch.zeros(B, T, dtype=torch.long)
        align_prob = torch.zeros(B, T, dtype=log_probs.dtype)
        return align_label, align_prob

    monkeypatch.setattr(w2v2ph.torchaudio.functional, "forced_align", fake_forced_align)

    # batch size 1 as required by forced_align
    speech = torch.randn(1, 16000)
    speech_lengths = torch.tensor([16000])
    text = torch.ones(1, 5, dtype=torch.long)  # no blank id (0) inside
    text_lengths = torch.tensor([5])

    align_label, align_prob = model.forced_align(
        speech, speech_lengths, text, text_lengths
    )

    # Check returned shapes
    assert align_label.shape[0] == 1
    assert align_prob.shape[0] == 1
    assert align_label.shape == align_prob.shape

    # Check that our fake was called with sensible arguments
    assert "log_probs_shape" in captured
    B, T, V = captured["log_probs_shape"]
    assert B == 1
    assert captured["text_shape"] == (1, 5)
    assert captured["blank"] == model.blank_id
