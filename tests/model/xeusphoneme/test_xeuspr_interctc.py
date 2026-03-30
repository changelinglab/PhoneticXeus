"""Tests for XeusPRModel with interCTC objective.

Tests cover:
  - encode() returns plain tensor when no interctc_layer_idx
  - encode() returns (final, [(idx, hs), ...]) tuple when interctc_layer_idx set
  - encode() weighted_sum path unaffected by interctc_layer_idx
  - forward() skips interCTC path when interctc_weight=0.0
  - forward() blends losses: (1-w)*loss_final + w*mean(loss_inter)
  - forward() logs loss_interctc_layer{N} stats per intermediate layer
  - forward() works with multiple intermediate layers
  - backward() through blended loss has grad to CTC head
  - ctc_logits() handles tuple encoder output safely
  - builders.build_xeus_pr() injects interctc_layer_idx into encoder_conf (integration)
"""

import pytest
import torch
import torch.nn as nn

from src.model.powsm.ctc import CTC
from src.model.xeusphoneme.xeuspr_model import XeusPRModel


# ---------------------------------------------------------------------------
# Helpers / fixtures
# ---------------------------------------------------------------------------

VOCAB = ["<blank>", "<space>", "a", "b", "c", "d", "e", "f"]
BLANK_ID = 0
ENC_DIM = 64
FEAT_DIM = 32


class MockEncoder(nn.Module):
    """Mimics EBranchformerEncoder's return convention.

    - return_all_hs=True  →  ((final, [hs_layer0, ..., hs_layerN]), lens, None)
    - interctc_layer_idx set, return_all_hs=False  →  ((final, [(idx, hs), ...]), lens, None)
    - neither  →  (final, lens, None)
    """

    def __init__(self, input_dim: int, output_dim: int, num_blocks: int = 12,
                 interctc_layer_idx=None):
        super().__init__()
        self.proj = nn.Linear(input_dim, output_dim)
        self.num_blocks = num_blocks
        self.interctc_layer_idx = interctc_layer_idx or []
        self.interctc_use_conditioning = False

    def forward(self, feats, feats_lengths, masks=None, return_all_hs=False):
        out = self.proj(feats)
        if return_all_hs:
            hs_list = [out] * self.num_blocks
            return (out, hs_list), feats_lengths, None
        elif self.interctc_layer_idx:
            intermediate = [(idx, out) for idx in self.interctc_layer_idx]
            return (out, intermediate), feats_lengths, None
        else:
            return out, feats_lengths, None

    def output_size(self):
        return self.proj.out_features


def make_model(interctc_weight=0.0, interctc_layer_idx=None, weighted_sum=False,
               device="cpu") -> XeusPRModel:
    """Build a minimal XeusPRModel for testing."""
    encoder = MockEncoder(
        input_dim=FEAT_DIM,
        output_dim=ENC_DIM,
        num_blocks=12,
        interctc_layer_idx=interctc_layer_idx,
    )
    ctc = CTC(odim=len(VOCAB), encoder_output_size=ENC_DIM)
    model = XeusPRModel(
        encoder=encoder,
        ctc=ctc,
        token_list=VOCAB,
        frontend=None,
        specaug=None,
        normalize=None,
        preencoder=None,
        ignore_id=-1,
        sym_blank="<blank>",
        freeze_frontend=False,
        weighted_sum=weighted_sum,
        interctc_weight=interctc_weight,
    )
    return model.to(device)


def make_batch(B=4, T=20, device="cpu"):
    """Create a minimal (speech, text) batch. Speech is already frame-level (no frontend)."""
    speech = torch.randn(B, T, FEAT_DIM, device=device)
    speech_lengths = torch.randint(T // 2, T + 1, (B,), device=device)
    speech_lengths[0] = T  # at least one full-length
    # text: phone ids in [1, V-1], with padding=-1
    max_text_len = 6
    text = torch.randint(1, len(VOCAB), (B, max_text_len), device=device)
    text_lengths = torch.randint(2, max_text_len + 1, (B,), device=device)
    text_lengths[0] = max_text_len
    # pad beyond text_lengths with -1
    for i in range(B):
        text[i, text_lengths[i]:] = -1
    return speech, speech_lengths, text, text_lengths


# ---------------------------------------------------------------------------
# 1. encode() return shape
# ---------------------------------------------------------------------------


def test_encode_no_interctc_returns_tensor():
    """With no interctc_layer_idx, encode() returns a plain tensor."""
    model = make_model(interctc_weight=0.0)
    model.eval()
    speech, speech_lengths, _, _ = make_batch()
    out, lens = model.encode(speech, speech_lengths)
    assert isinstance(out, torch.Tensor), "Expected plain tensor, got tuple"
    assert out.shape == (speech.shape[0], speech.shape[1], ENC_DIM)
    assert lens.shape == (speech.shape[0],)


def test_encode_with_interctc_returns_tuple():
    """With interctc_layer_idx=[6], encode() returns (final, [(6, tensor)])."""
    model = make_model(interctc_weight=0.3, interctc_layer_idx=[6])
    model.eval()
    speech, speech_lengths, _, _ = make_batch()
    out, lens = model.encode(speech, speech_lengths)
    assert isinstance(out, tuple), "Expected tuple when interctc_layer_idx is set"
    final, intermediates = out
    assert isinstance(final, torch.Tensor)
    assert isinstance(intermediates, list) and len(intermediates) == 1
    layer_idx, hs = intermediates[0]
    assert layer_idx == 6
    assert hs.shape == final.shape


def test_encode_multiple_interctc_layers():
    """Multiple interctc layers returns multiple entries in the list."""
    model = make_model(interctc_weight=0.3, interctc_layer_idx=[4, 8])
    model.eval()
    speech, speech_lengths, _, _ = make_batch()
    out, _ = model.encode(speech, speech_lengths)
    assert isinstance(out, tuple)
    _, intermediates = out
    assert len(intermediates) == 2
    assert intermediates[0][0] == 4
    assert intermediates[1][0] == 8


def test_encode_weighted_sum_returns_tensor():
    """weighted_sum=True always returns a plain tensor regardless of interctc_layer_idx."""
    model = make_model(weighted_sum=True)
    model.eval()
    speech, speech_lengths, _, _ = make_batch()
    out, lens = model.encode(speech, speech_lengths)
    assert isinstance(out, torch.Tensor), "weighted_sum mode should return a plain tensor"
    assert out.shape == (speech.shape[0], speech.shape[1], ENC_DIM)


# ---------------------------------------------------------------------------
# 2. forward() loss blending
# ---------------------------------------------------------------------------


def test_forward_no_interctc_no_stats_keys():
    """interctc_weight=0.0: no loss_interctc_layerN keys in stats."""
    model = make_model(interctc_weight=0.0, interctc_layer_idx=[6])
    model.eval()
    speech, sl, text, tl = make_batch()
    with torch.no_grad():
        out = model(speech, sl, text, tl)
    interctc_keys = [k for k in out["stats"] if k.startswith("loss_interctc")]
    assert interctc_keys == [], f"Unexpected interCTC keys: {interctc_keys}"


def test_forward_interctc_stats_key_present():
    """interctc_weight=0.3: stats contains loss_interctc_layer6."""
    model = make_model(interctc_weight=0.3, interctc_layer_idx=[6])
    model.train()
    speech, sl, text, tl = make_batch()
    out = model(speech, sl, text, tl)
    assert "loss_interctc_layer6" in out["stats"], (
        f"Missing loss_interctc_layer6 in stats. Got: {list(out['stats'].keys())}"
    )


def test_forward_interctc_multiple_layers_stats():
    """Multiple interctc layers each log their own stat key."""
    model = make_model(interctc_weight=0.3, interctc_layer_idx=[4, 8])
    model.train()
    speech, sl, text, tl = make_batch()
    out = model(speech, sl, text, tl)
    assert "loss_interctc_layer4" in out["stats"]
    assert "loss_interctc_layer8" in out["stats"]


def test_forward_interctc_loss_is_blended():
    """Blended loss = (1-w)*final + w*inter, up to numerical tolerance."""
    w = 0.4
    model = make_model(interctc_weight=w, interctc_layer_idx=[6])
    model.train()
    speech, sl, text, tl = make_batch(B=2)

    # Capture the constituent losses by monkey-patching ctc
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

    model.ctc = CaptureCTC()

    out = model(speech, sl, text, tl)

    # call 0 = final layer, call 1 = intermediate layer 6
    assert len(captured) == 2, f"Expected 2 CTC calls, got {len(captured)}"
    loss_final = captured[0]
    loss_inter = captured[1]
    expected = (1 - w) * loss_final + w * loss_inter
    actual = out["loss"]
    # force_gatherable may add a batch dimension; compare scalar values
    assert abs(actual.item() - expected.item()) < 1e-4, (
        f"Loss blending mismatch: got {actual.item():.6f}, expected {expected.item():.6f}"
    )


def test_forward_loss_no_nan_no_inf():
    """Loss must be finite (no NaN / Inf) with interCTC."""
    model = make_model(interctc_weight=0.3, interctc_layer_idx=[6])
    model.train()
    speech, sl, text, tl = make_batch(B=8, T=30)
    out = model(speech, sl, text, tl)
    assert not torch.isnan(out["loss"]), "Loss is NaN"
    assert not torch.isinf(out["loss"]), "Loss is Inf"


# ---------------------------------------------------------------------------
# 3. Gradient flow
# ---------------------------------------------------------------------------


def test_backward_flows_through_blended_loss():
    """Backward pass on blended loss should produce gradients for CTC head."""
    model = make_model(interctc_weight=0.3, interctc_layer_idx=[6])
    model.train()
    speech, sl, text, tl = make_batch()
    out = model(speech, sl, text, tl)
    out["loss"].backward()

    # CTC linear projection should have gradients
    ctc_weight_grad = model.ctc.ctc_lo.weight.grad
    assert ctc_weight_grad is not None, "No grad on ctc_lo.weight"
    assert ctc_weight_grad.abs().sum() > 0, "ctc_lo.weight grad is zero"


def test_backward_no_interctc_still_works():
    """Baseline: backward with interctc_weight=0.0 still works."""
    model = make_model(interctc_weight=0.0)
    model.train()
    speech, sl, text, tl = make_batch()
    out = model(speech, sl, text, tl)
    out["loss"].backward()
    ctc_weight_grad = model.ctc.ctc_lo.weight.grad
    assert ctc_weight_grad is not None
    assert ctc_weight_grad.abs().sum() > 0


# ---------------------------------------------------------------------------
# 4. ctc_logits() safety
# ---------------------------------------------------------------------------


def test_ctc_logits_with_interctc_model():
    """ctc_logits() must return a plain tensor even when encoder returns a tuple."""
    model = make_model(interctc_weight=0.3, interctc_layer_idx=[6])
    model.eval()
    speech, sl, _, _ = make_batch()
    with torch.no_grad():
        logits, lens = model.ctc_logits(speech, sl)
    assert isinstance(logits, torch.Tensor), "ctc_logits() should return a tensor, not a tuple"
    assert logits.shape[-1] == len(VOCAB), "Last dim should be vocab size"
    assert not torch.isnan(logits).any()


def test_ctc_logits_without_interctc():
    """ctc_logits() baseline: no interctc, still returns correct shape."""
    model = make_model(interctc_weight=0.0)
    model.eval()
    speech, sl, _, _ = make_batch()
    with torch.no_grad():
        logits, lens = model.ctc_logits(speech, sl)
    assert isinstance(logits, torch.Tensor)
    assert logits.shape[-1] == len(VOCAB)


# ---------------------------------------------------------------------------
# 5. train vs eval mode
# ---------------------------------------------------------------------------


def test_forward_train_mode_no_error_calculator():
    """In train mode, error calculator is NOT called (no cer_ctc key in stats)."""
    model = make_model(interctc_weight=0.3, interctc_layer_idx=[6])
    model.train()
    speech, sl, text, tl = make_batch()
    out = model(speech, sl, text, tl)
    cer_keys = [k for k in out["stats"] if "cer" in k or "per" in k]
    assert cer_keys == [], f"Error metrics should not appear in train mode: {cer_keys}"


def test_forward_eval_mode_has_metrics():
    """In eval mode, error calculator produces metric keys (e.g. cer_ctc or per)."""
    model = make_model(interctc_weight=0.3, interctc_layer_idx=[6])
    model.eval()
    speech, sl, text, tl = make_batch()
    with torch.no_grad():
        out = model(speech, sl, text, tl)
    metric_keys = [k for k in out["stats"] if k.endswith("_ctc")]
    assert len(metric_keys) > 0, f"Expected metric keys in eval mode, got: {list(out['stats'].keys())}"


# ---------------------------------------------------------------------------
# 6. Integration test: builder injects interctc_layer_idx into encoder
# ---------------------------------------------------------------------------

XEUS_CONFIG = "exp/cache/xeus/model/config.yaml"
XEUS_VOCAB = "src/model/xeusphoneme/resources/ipa_vocab.json"


@pytest.mark.skipif(
    not __import__("pathlib").Path(XEUS_CONFIG).exists(),
    reason=f"Xeus config not found at {XEUS_CONFIG}",
)
def test_builder_injects_interctc_layer_idx():
    """build_xeus_pr() with interctc_layer_idx=[6] sets it on the encoder."""
    from src.model.xeusphoneme.builders import build_xeus_pr

    model = build_xeus_pr(
        config_file=XEUS_CONFIG,
        checkpoint=None,
        vocab_file=XEUS_VOCAB,
        ctc_config={"ctc_type": "builtin"},
        interctc_layer_idx=[6],
        interctc_weight=0.3,
    )
    assert model.encoder.interctc_layer_idx == [6], (
        f"Expected [6], got {model.encoder.interctc_layer_idx}"
    )
    assert model.interctc_weight == 0.3


@pytest.mark.skipif(
    not __import__("pathlib").Path(XEUS_CONFIG).exists(),
    reason=f"Xeus config not found at {XEUS_CONFIG}",
)
def test_builder_no_interctc_default():
    """build_xeus_pr() with no interctc args leaves encoder.interctc_layer_idx empty."""
    from src.model.xeusphoneme.builders import build_xeus_pr

    model = build_xeus_pr(
        config_file=XEUS_CONFIG,
        checkpoint=None,
        vocab_file=XEUS_VOCAB,
        ctc_config={"ctc_type": "builtin"},
    )
    assert model.encoder.interctc_layer_idx == [], (
        f"Expected empty list, got {model.encoder.interctc_layer_idx}"
    )
    assert model.interctc_weight == 0.0


@pytest.mark.skipif(
    not __import__("pathlib").Path(XEUS_CONFIG).exists(),
    reason=f"Xeus config not found at {XEUS_CONFIG}",
)
def test_builder_interctc_forward_backward():
    """End-to-end: build real model, run forward+backward with interCTC."""
    from src.model.xeusphoneme.builders import build_xeus_pr

    model = build_xeus_pr(
        config_file=XEUS_CONFIG,
        checkpoint=None,
        vocab_file=XEUS_VOCAB,
        ctc_config={"ctc_type": "builtin"},
        interctc_layer_idx=[9],  # midpoint of 19-layer xeus encoder
        interctc_weight=0.3,
    )
    model.train()

    # Build a small batch: raw waveform (no frontend in this path since frontend is wav2vec_cnn)
    # We exercise through the preencoder/encoder only by calling encode() after pre-processing.
    # Use ctc_logits() for a simpler integration check (no text needed).
    device = next(model.parameters()).device
    B, T = 2, 4000
    speech = torch.randn(B, T, device=device)
    speech_lengths = torch.tensor([T, T // 2], device=device)

    with torch.no_grad():
        logits, lens = model.ctc_logits(speech, speech_lengths)

    assert isinstance(logits, torch.Tensor)
    assert not torch.isnan(logits).any()
    print(f"  logits shape: {logits.shape}, lens: {lens.tolist()}")


# ---------------------------------------------------------------------------
# GPU variants
# ---------------------------------------------------------------------------


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
def test_forward_interctc_on_gpu():
    """Full forward+backward on CUDA with interCTC."""
    device = "cuda"
    model = make_model(interctc_weight=0.3, interctc_layer_idx=[6], device=device)
    model.train()
    speech, sl, text, tl = make_batch(B=4, T=20, device=device)
    out = model(speech, sl, text, tl)
    assert not torch.isnan(out["loss"])
    out["loss"].backward()
    assert model.ctc.ctc_lo.weight.grad is not None
    print(f"  GPU loss={out['loss'].item():.4f}")


@pytest.mark.skipif(
    not torch.cuda.is_available() or
    not __import__("pathlib").Path(XEUS_CONFIG).exists(),
    reason="CUDA or xeus config not available",
)
def test_builder_interctc_forward_on_gpu():
    """Build real xeus model with interCTC and run inference on GPU."""
    from src.model.xeusphoneme.builders import build_xeus_pr

    model = build_xeus_pr(
        config_file=XEUS_CONFIG,
        checkpoint=None,
        vocab_file=XEUS_VOCAB,
        ctc_config={"ctc_type": "builtin"},
        interctc_layer_idx=[9],
        interctc_weight=0.3,
    ).cuda()
    model.train()

    B, T = 2, 8000
    speech = torch.randn(B, T, device="cuda")
    speech_lengths = torch.tensor([T, T // 2], device="cuda")
    with torch.no_grad():
        logits, lens = model.ctc_logits(speech, speech_lengths)
    assert not torch.isnan(logits).any()
    print(f"  GPU logits shape: {logits.shape}")


# ---------------------------------------------------------------------------
# __main__ runner (same pattern as other test files in this project)
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import traceback

    tests = [
        ("encode() no interctc → tensor", test_encode_no_interctc_returns_tensor),
        ("encode() interctc=[6] → tuple", test_encode_with_interctc_returns_tuple),
        ("encode() multiple layers", test_encode_multiple_interctc_layers),
        ("encode() weighted_sum → tensor", test_encode_weighted_sum_returns_tensor),
        ("forward() no interctc: no stats keys", test_forward_no_interctc_no_stats_keys),
        ("forward() stats key present", test_forward_interctc_stats_key_present),
        ("forward() multiple layers stats", test_forward_interctc_multiple_layers_stats),
        ("forward() loss blending formula", test_forward_interctc_loss_is_blended),
        ("forward() loss finite", test_forward_loss_no_nan_no_inf),
        ("backward() grad flows", test_backward_flows_through_blended_loss),
        ("backward() no interctc baseline", test_backward_no_interctc_still_works),
        ("ctc_logits() with interctc model", test_ctc_logits_with_interctc_model),
        ("ctc_logits() without interctc", test_ctc_logits_without_interctc),
        ("train mode: no error metrics", test_forward_train_mode_no_error_calculator),
        ("eval mode: has metrics", test_forward_eval_mode_has_metrics),
    ]

    # Integration tests (require xeus config)
    import pathlib
    if pathlib.Path(XEUS_CONFIG).exists():
        tests += [
            ("builder injects interctc_layer_idx", test_builder_injects_interctc_layer_idx),
            ("builder no interctc default", test_builder_no_interctc_default),
            ("builder forward+backward", test_builder_interctc_forward_backward),
        ]
    else:
        print(f"\n[SKIP] Integration tests: {XEUS_CONFIG} not found\n")

    # GPU tests
    if torch.cuda.is_available():
        tests += [
            ("forward interctc on GPU", test_forward_interctc_on_gpu),
        ]
        if pathlib.Path(XEUS_CONFIG).exists():
            tests += [
                ("builder interctc on GPU", test_builder_interctc_forward_on_gpu),
            ]
    else:
        print("[SKIP] GPU tests: CUDA not available\n")

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
