"""Tests for Wav2Vec2PRModel with interCTC + self-conditioning.

Self-conditioning feeds each intermediate CTC distribution back into the encoder
as an additive residual (Nozaki & Komatsu, 2021), enabled via
`interctc_use_conditioning=True` in the model / builder.

Tests cover:
  - conditioning_layer is created with the right shape when flag is True
  - conditioning_layer is None when flag is False
  - interctc_layer_idx=[] leaves conditioning_layer None even if flag is True
  - encode() returns a plain tensor when interctc_layer_idx is empty
  - encode() returns (final, [(idx, hs), ...]) tuple when interctc_layer_idx is set
  - encode() tuple contains entries for exactly the requested layer indices
  - forward() skips interCTC path when interctc_weight=0.0
  - forward() logs loss_interctc_layer{N} for each intermediate layer
  - forward() blends losses: (1-w)*loss_final + w*mean(loss_inter)
  - forward() loss is finite with self-conditioning
  - backward() produces gradients for conditioning_layer.weight
  - backward() produces gradients for ctc_lo.weight
  - ctc_logits() returns a plain tensor even when encoder returns a tuple
  - baseline (no interctc): encode() returns plain tensor, ctc_logits() unchanged
  - GPU: forward+backward on CUDA with conditioning
  - GPU+real model: build_wav2vec2pr with mms-300m, run ctc_logits (integration)
"""

import pathlib

import pytest
import torch
import torch.nn as nn

from src.model.powsm.ctc import CTC
from src.model.wav2vec2.wav2vec2pr_model import Wav2Vec2PRModel

# ---------------------------------------------------------------------------
# Shared constants
# ---------------------------------------------------------------------------

VOCAB = ["<blank>", "<space>", "a", "b", "c", "d", "e", "f"]
BLANK_ID = 0
ENC_DIM = 64
T_FRAMES = 10       # fixed output frames returned by the mock encoder
T_SAMPLES = 16000   # raw waveform length used in make_batch

MMS_REPO = "facebook/mms-300m"
IPA_VOCAB = "src/model/xeusphoneme/resources/ipa_vocab.json"

_ipa_vocab_exists = pathlib.Path(IPA_VOCAB).exists()


# ---------------------------------------------------------------------------
# MockWav2Vec2Encoder — mimics Wav2Vec2Model's encode / encode_with_interctc
# ---------------------------------------------------------------------------


class MockWav2Vec2Encoder(nn.Module):
    """Lightweight stand-in for Wav2Vec2Model.

    - encode()               → (tensor[B, T, D], lens)
    - encode_with_interctc() → ((tensor, [(idx, hs), ...]), lens)
                               or (tensor, lens) when interctc_layer_idx is empty
    """

    def __init__(self, enc_dim: int = ENC_DIM):
        super().__init__()
        self._enc_dim = enc_dim
        # Learnable projection so backward tests get real gradients
        self.proj = nn.Linear(enc_dim, enc_dim)

    def _make_out(self, B: int, device: torch.device) -> torch.Tensor:
        """Return a [B, T_FRAMES, enc_dim] tensor connected to self.proj."""
        x = torch.randn(B, T_FRAMES, self._enc_dim, device=device)
        return self.proj(x)

    def encode(self, speech, speech_lengths):
        B = speech.shape[0]
        out = self._make_out(B, speech.device)
        lens = torch.full((B,), T_FRAMES, dtype=torch.long, device=speech.device)
        return out, lens

    def encode_with_interctc(
        self, speech, speech_lengths, interctc_layer_idx, ctc, conditioning_layer=None
    ):
        B = speech.shape[0]
        device = speech.device
        hidden_states = self._make_out(B, device)
        lens = torch.full((B,), T_FRAMES, dtype=torch.long, device=device)

        intermediate_outs = []
        for layer_idx in sorted(interctc_layer_idx):
            hs = self._make_out(B, device)
            intermediate_outs.append((layer_idx, hs))
            if conditioning_layer is not None:
                ctc_out = ctc.softmax(hs)
                hidden_states = hidden_states + conditioning_layer(ctc_out)

        if intermediate_outs:
            return (hidden_states, intermediate_outs), lens
        return hidden_states, lens

    def encoder_output_size(self) -> int:
        return self._enc_dim


# ---------------------------------------------------------------------------
# Factories
# ---------------------------------------------------------------------------


def make_ctc(vocab=VOCAB, enc_dim=ENC_DIM):
    return CTC(odim=len(vocab), encoder_output_size=enc_dim)


def make_model(
    interctc_weight: float = 0.3,
    interctc_layer_idx=None,
    interctc_use_conditioning: bool = False,
    device: str = "cpu",
) -> Wav2Vec2PRModel:
    """Build a Wav2Vec2PRModel backed by MockWav2Vec2Encoder."""
    if interctc_layer_idx is None:
        interctc_layer_idx = [8, 16]
    encoder = MockWav2Vec2Encoder(enc_dim=ENC_DIM)
    ctc = make_ctc()
    model = Wav2Vec2PRModel(
        encoder=encoder,
        ctc=ctc,
        token_list=VOCAB,
        ignore_id=-1,
        sym_blank="<blank>",
        freeze_frontend=False,
        interctc_weight=interctc_weight,
        interctc_layer_idx=interctc_layer_idx,
        interctc_use_conditioning=interctc_use_conditioning,
    )
    return model.to(device)


def make_batch(B: int = 4, device: str = "cpu"):
    """Create a minimal (speech, text) batch with raw waveform inputs."""
    speech = torch.randn(B, T_SAMPLES, device=device)
    speech_lengths = torch.full((B,), T_SAMPLES, dtype=torch.long, device=device)
    max_text_len = 4
    text = torch.randint(1, len(VOCAB), (B, max_text_len), device=device)
    text_lengths = torch.full((B,), max_text_len, dtype=torch.long, device=device)
    return speech, speech_lengths, text, text_lengths


# ---------------------------------------------------------------------------
# 1. Structural: conditioning_layer creation
# ---------------------------------------------------------------------------


def test_conditioning_layer_created_when_flag_true():
    """conditioning_layer is Linear(vocab_size → enc_dim) when flag is True."""
    model = make_model(interctc_use_conditioning=True)
    cl = model.conditioning_layer
    assert cl is not None, "conditioning_layer should be set when interctc_use_conditioning=True"
    assert isinstance(cl, nn.Linear), f"Expected nn.Linear, got {type(cl)}"
    assert cl.in_features == len(VOCAB), (
        f"in_features should be vocab_size={len(VOCAB)}, got {cl.in_features}"
    )
    assert cl.out_features == ENC_DIM, (
        f"out_features should be enc_dim={ENC_DIM}, got {cl.out_features}"
    )


def test_no_conditioning_layer_by_default():
    """conditioning_layer is None when interctc_use_conditioning=False."""
    model = make_model(interctc_use_conditioning=False)
    assert model.conditioning_layer is None


def test_no_conditioning_layer_when_no_interctc_layers():
    """conditioning_layer is None when interctc_layer_idx=[] even if flag is True."""
    model = make_model(interctc_layer_idx=[], interctc_use_conditioning=True)
    assert model.conditioning_layer is None


# ---------------------------------------------------------------------------
# 2. encode() return shape
# ---------------------------------------------------------------------------


def test_encode_no_interctc_returns_tensor():
    """With interctc_layer_idx=[], encode() returns a plain tensor."""
    model = make_model(interctc_layer_idx=[])
    model.eval()
    speech, sl, _, _ = make_batch()
    with torch.no_grad():
        out, lens = model.encode(speech, sl)
    assert isinstance(out, torch.Tensor), f"Expected plain tensor, got {type(out)}"
    assert out.shape == (speech.shape[0], T_FRAMES, ENC_DIM)
    assert lens.shape == (speech.shape[0],)


def test_encode_with_interctc_returns_tuple():
    """With interctc_layer_idx=[8,16], encode() returns (final, [(8,hs),(16,hs)])."""
    model = make_model(interctc_layer_idx=[8, 16])
    model.eval()
    speech, sl, _, _ = make_batch()
    with torch.no_grad():
        out, lens = model.encode(speech, sl)
    assert isinstance(out, tuple), f"Expected tuple when interctc_layer_idx is set, got {type(out)}"
    final, intermediates = out
    assert isinstance(final, torch.Tensor)
    assert isinstance(intermediates, list) and len(intermediates) == 2
    assert [idx for idx, _ in intermediates] == [8, 16]


def test_encode_intermediate_shapes():
    """Each intermediate hidden state has the same shape as the final output."""
    model = make_model(interctc_layer_idx=[8, 16])
    model.eval()
    speech, sl, _, _ = make_batch()
    with torch.no_grad():
        (final, intermediates), lens = model.encode(speech, sl)
    for idx, hs in intermediates:
        assert hs.shape == final.shape, (
            f"Layer {idx}: expected {final.shape}, got {hs.shape}"
        )


# ---------------------------------------------------------------------------
# 3. forward() loss blending and stats
# ---------------------------------------------------------------------------


def test_forward_no_interctc_weight_skips_path():
    """interctc_weight=0.0: no loss_interctc_layerN keys in stats."""
    model = make_model(interctc_weight=0.0, interctc_layer_idx=[8, 16])
    model.train()
    speech, sl, text, tl = make_batch()
    out = model(speech, sl, text, tl)
    interctc_keys = [k for k in out["stats"] if k.startswith("loss_interctc")]
    assert interctc_keys == [], f"Unexpected interCTC keys: {interctc_keys}"


def test_forward_interctc_stats_keys_present():
    """forward() logs loss_interctc_layer8 and loss_interctc_layer16."""
    model = make_model(interctc_weight=0.3, interctc_layer_idx=[8, 16])
    model.train()
    speech, sl, text, tl = make_batch()
    out = model(speech, sl, text, tl)
    assert "loss_interctc_layer8" in out["stats"], (
        f"Missing loss_interctc_layer8. Got: {list(out['stats'].keys())}"
    )
    assert "loss_interctc_layer16" in out["stats"], (
        f"Missing loss_interctc_layer16. Got: {list(out['stats'].keys())}"
    )


def test_forward_loss_finite():
    """Loss is finite (no NaN / Inf) with self-conditioning active."""
    model = make_model(interctc_weight=0.3, interctc_use_conditioning=True)
    model.train()
    speech, sl, text, tl = make_batch(B=2)
    out = model(speech, sl, text, tl)
    assert not torch.isnan(out["loss"]), "Loss is NaN with self-conditioning"
    assert not torch.isinf(out["loss"]), "Loss is Inf with self-conditioning"


def test_forward_interctc_loss_blended():
    """Blended loss = (1-w)*loss_final + w*mean(loss_inter), within 1e-4."""
    w = 0.4
    model = make_model(interctc_weight=w, interctc_layer_idx=[8, 16])
    model.train()
    speech, sl, text, tl = make_batch(B=2)

    # Capture CTC call order via a thin wrapper
    captured = {}
    original_ctc = model.ctc
    call_count = [0]

    class CaptureCTC(nn.Module):
        def forward(self, *args, **kwargs):
            loss = original_ctc(*args, **kwargs)
            captured[call_count[0]] = loss
            call_count[0] += 1
            return loss

        def argmax(self, x):
            return original_ctc.argmax(x)

        def ctc_lo(self, x):
            return original_ctc.ctc_lo(x)

        def softmax(self, x):
            return original_ctc.softmax(x)

    model.ctc = CaptureCTC()
    out = model(speech, sl, text, tl)

    # call 0 = final layer, calls 1 and 2 = intermediate layers 8 and 16
    assert len(captured) == 3, f"Expected 3 CTC calls (1 final + 2 inter), got {len(captured)}"
    loss_final = captured[0]
    loss_inter_mean = (captured[1] + captured[2]) / 2
    expected = (1 - w) * loss_final + w * loss_inter_mean
    assert abs(out["loss"].item() - expected.item()) < 1e-4, (
        f"Loss blending mismatch: got {out['loss'].item():.6f}, expected {expected.item():.6f}"
    )


# ---------------------------------------------------------------------------
# 4. Gradient flow
# ---------------------------------------------------------------------------


def test_backward_flows_to_conditioning_layer():
    """backward() produces nonzero gradients on conditioning_layer.weight."""
    model = make_model(interctc_weight=0.3, interctc_use_conditioning=True)
    model.train()
    speech, sl, text, tl = make_batch()
    out = model(speech, sl, text, tl)
    out["loss"].backward()
    grad = model.conditioning_layer.weight.grad
    assert grad is not None, "No grad on conditioning_layer.weight"
    assert grad.abs().sum() > 0, "conditioning_layer.weight grad is zero"


def test_backward_flows_to_ctc_head():
    """backward() produces nonzero gradients on ctc_lo.weight."""
    model = make_model(interctc_weight=0.3, interctc_use_conditioning=True)
    model.train()
    speech, sl, text, tl = make_batch()
    out = model(speech, sl, text, tl)
    out["loss"].backward()
    grad = model.ctc.ctc_lo.weight.grad
    assert grad is not None, "No grad on ctc_lo.weight"
    assert grad.abs().sum() > 0, "ctc_lo.weight grad is zero"


def test_backward_no_interctc_baseline():
    """Baseline (no interctc): backward still works and ctc_lo gets gradients."""
    model = make_model(interctc_weight=0.0, interctc_layer_idx=[])
    model.train()
    speech, sl, text, tl = make_batch()
    out = model(speech, sl, text, tl)
    out["loss"].backward()
    grad = model.ctc.ctc_lo.weight.grad
    assert grad is not None and grad.abs().sum() > 0


# ---------------------------------------------------------------------------
# 5. ctc_logits() safety
# ---------------------------------------------------------------------------


def test_ctc_logits_returns_tensor_with_interctc():
    """ctc_logits() returns a plain tensor even when encoder returns a tuple."""
    model = make_model(interctc_weight=0.3, interctc_layer_idx=[8, 16])
    model.eval()
    speech, sl, _, _ = make_batch()
    with torch.no_grad():
        logits, lens = model.ctc_logits(speech, sl)
    assert isinstance(logits, torch.Tensor), (
        f"ctc_logits() should return a Tensor, got {type(logits)}"
    )
    assert logits.shape[-1] == len(VOCAB), (
        f"Last dim should be vocab_size={len(VOCAB)}, got {logits.shape[-1]}"
    )
    assert not torch.isnan(logits).any()


def test_ctc_logits_baseline_no_interctc():
    """ctc_logits() baseline: no interctc still returns correct shape."""
    model = make_model(interctc_weight=0.0, interctc_layer_idx=[])
    model.eval()
    speech, sl, _, _ = make_batch()
    with torch.no_grad():
        logits, lens = model.ctc_logits(speech, sl)
    assert isinstance(logits, torch.Tensor)
    assert logits.shape[-1] == len(VOCAB)


# ---------------------------------------------------------------------------
# 6. GPU tests
# ---------------------------------------------------------------------------


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
def test_forward_backward_with_conditioning_on_gpu():
    """Self-conditioning forward+backward on CUDA is finite and produces gradients."""
    device = "cuda"
    model = make_model(interctc_weight=0.3, interctc_use_conditioning=True, device=device)
    model.train()
    speech, sl, text, tl = make_batch(B=4, device=device)
    out = model(speech, sl, text, tl)
    assert not torch.isnan(out["loss"]), "Loss is NaN on GPU"
    assert not torch.isinf(out["loss"]), "Loss is Inf on GPU"
    out["loss"].backward()
    assert model.conditioning_layer.weight.grad is not None
    assert model.conditioning_layer.weight.grad.abs().sum() > 0
    assert model.ctc.ctc_lo.weight.grad is not None


# ---------------------------------------------------------------------------
# 7. Integration tests: real mms-300m model (require GPU + IPA vocab)
# ---------------------------------------------------------------------------


@pytest.mark.skipif(
    not torch.cuda.is_available() or not _ipa_vocab_exists,
    reason="CUDA or IPA vocab not available",
)
def test_real_model_ctc_logits_on_gpu():
    """build_wav2vec2pr() with mms-300m + interctc: ctc_logits() shape is correct."""
    from src.model.wav2vec2.builders import build_wav2vec2pr

    model = build_wav2vec2pr(
        hf_repo=MMS_REPO,
        vocab_file=IPA_VOCAB,
        interctc_weight=0.3,
        interctc_layer_idx=[8, 16],
        interctc_use_conditioning=True,
    ).cuda()
    model.eval()

    B, T = 2, 16000
    speech = torch.randn(B, T, device="cuda")
    speech_lengths = torch.tensor([T, T // 2], device="cuda")

    with torch.no_grad():
        logits, lens = model.ctc_logits(speech, speech_lengths)

    assert isinstance(logits, torch.Tensor)
    assert logits.ndim == 3
    assert not torch.isnan(logits).any(), "NaN in logits from real model"


@pytest.mark.skipif(
    not torch.cuda.is_available() or not _ipa_vocab_exists,
    reason="CUDA or IPA vocab not available",
)
def test_real_model_encode_returns_tuple_on_gpu():
    """build_wav2vec2pr() with interctc: encode() returns a tuple with intermediates."""
    from src.model.wav2vec2.builders import build_wav2vec2pr

    model = build_wav2vec2pr(
        hf_repo=MMS_REPO,
        vocab_file=IPA_VOCAB,
        interctc_weight=0.3,
        interctc_layer_idx=[8, 16],
        interctc_use_conditioning=True,
    ).cuda()
    model.eval()

    B, T = 1, 16000
    speech = torch.randn(B, T, device="cuda")
    speech_lengths = torch.tensor([T], device="cuda")

    with torch.no_grad():
        encoder_out, lens = model.encode(speech, speech_lengths)

    assert isinstance(encoder_out, tuple), "Expected tuple from encode() with interctc"
    final, intermediates = encoder_out
    assert [idx for idx, _ in intermediates] == [8, 16]
    assert not torch.isnan(final).any()


@pytest.mark.skipif(
    not torch.cuda.is_available() or not _ipa_vocab_exists,
    reason="CUDA or IPA vocab not available",
)
def test_real_model_baseline_unchanged_on_gpu():
    """build_wav2vec2pr() without interctc: encode() returns a plain tensor."""
    from src.model.wav2vec2.builders import build_wav2vec2pr

    model = build_wav2vec2pr(hf_repo=MMS_REPO, vocab_file=IPA_VOCAB).cuda()
    model.eval()

    B, T = 1, 16000
    speech = torch.randn(B, T, device="cuda")
    speech_lengths = torch.tensor([T], device="cuda")

    with torch.no_grad():
        encoder_out, lens = model.encode(speech, speech_lengths)

    assert isinstance(encoder_out, torch.Tensor), (
        "Baseline encode() should return a plain tensor"
    )


# ---------------------------------------------------------------------------
# __main__ runner (same pattern as other test files in this project)
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import traceback

    tests = [
        # Structural
        ("conditioning_layer created", test_conditioning_layer_created_when_flag_true),
        ("no conditioning_layer by default", test_no_conditioning_layer_by_default),
        ("no conditioning_layer with empty layer_idx", test_no_conditioning_layer_when_no_interctc_layers),
        # encode() shape
        ("encode() no interctc → tensor", test_encode_no_interctc_returns_tensor),
        ("encode() with interctc → tuple", test_encode_with_interctc_returns_tuple),
        ("encode() intermediate shapes match", test_encode_intermediate_shapes),
        # forward()
        ("forward() weight=0 skips path", test_forward_no_interctc_weight_skips_path),
        ("forward() stats keys present", test_forward_interctc_stats_keys_present),
        ("forward() loss finite", test_forward_loss_finite),
        ("forward() loss blending formula", test_forward_interctc_loss_blended),
        # Gradient flow
        ("backward() → conditioning_layer grad", test_backward_flows_to_conditioning_layer),
        ("backward() → ctc_lo grad", test_backward_flows_to_ctc_head),
        ("backward() baseline no interctc", test_backward_no_interctc_baseline),
        # ctc_logits()
        ("ctc_logits() strips tuple", test_ctc_logits_returns_tensor_with_interctc),
        ("ctc_logits() baseline", test_ctc_logits_baseline_no_interctc),
    ]

    if torch.cuda.is_available():
        tests += [
            ("GPU: forward+backward with conditioning", test_forward_backward_with_conditioning_on_gpu),
        ]
        if _ipa_vocab_exists:
            tests += [
                ("GPU+real: ctc_logits shape", test_real_model_ctc_logits_on_gpu),
                ("GPU+real: encode() returns tuple", test_real_model_encode_returns_tuple_on_gpu),
                ("GPU+real: baseline unchanged", test_real_model_baseline_unchanged_on_gpu),
            ]
    else:
        print("[SKIP] GPU tests: CUDA not available\n")

    if not _ipa_vocab_exists:
        print(f"[SKIP] Integration tests: {IPA_VOCAB} not found\n")

    passed = failed = 0
    for name, fn in tests:
        print(f"\n{'='*60}")
        print(f"TEST: {name}")
        print(f"{'='*60}")
        try:
            fn()
            print("  PASSED")
            passed += 1
        except Exception as e:
            print(f"  FAILED: {e}")
            traceback.print_exc()
            failed += 1

    print(f"\n{'='*60}")
    print(f"RESULTS: {passed} passed, {failed} failed out of {passed + failed}")
    print(f"{'='*60}")
    if failed:
        raise SystemExit(1)
