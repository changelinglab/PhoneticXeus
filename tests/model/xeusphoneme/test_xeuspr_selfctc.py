"""Tests for XeusPRModel self-conditioned CTC.

Self-conditioning feeds each intermediate CTC output distribution back into the
encoder as an additive residual (Nozaki & Komatsu, 2021). This is enabled via
`interctc_use_conditioning=True` in the model / builder.

Tests cover:
  - conditioning_layer is created with the right shape when flag is True
  - encoder.interctc_use_conditioning is set to True
  - default (False): no conditioning_layer, flag stays False
  - encoder.forward() receives ctc= kwarg in the non-weighted_sum path
  - forward() + backward() with conditioning MockEncoder is finite and has grads
  - forward() + backward() with conditioning MockEncoder logs interctc stats
  - builder (build_xeus_pr) sets conditioning flag on real encoder (integration)
  - builder (build_xeus_pr_from_hf) propagates interctc_use_conditioning (integration)
  - GPU: forward+backward with conditioning MockEncoder on CUDA
  - GPU+real model: build_xeus_pr with real encoder, run forward on GPU (integration)
"""

import pathlib
import traceback

import pytest
import torch
import torch.nn as nn

from src.model.powsm.ctc import CTC
from src.model.xeusphoneme.xeuspr_model import XeusPRModel

# ---------------------------------------------------------------------------
# Shared constants
# ---------------------------------------------------------------------------

VOCAB = ["<blank>", "<space>", "a", "b", "c", "d", "e", "f"]
BLANK_ID = 0
ENC_DIM = 64
FEAT_DIM = 32

XEUS_CONFIG = "exp/cache/xeus/model/config.yaml"
XEUS_VOCAB = "src/model/xeusphoneme/resources/ipa_vocab.json"

_xeus_config_exists = pathlib.Path(XEUS_CONFIG).exists()


# ---------------------------------------------------------------------------
# MockEncoder that supports self-conditioning and records ctc kwarg
# ---------------------------------------------------------------------------


class MockConditioningEncoder(nn.Module):
    """Minimal encoder that mimics EBranchformerEncoder's conditioning contract.

    When interctc_use_conditioning=True and ctc is passed in forward(), it applies
    conditioning_layer(ctc.ctc_lo(out)) to the output just like the real encoder.
    Also records whether ctc was passed so tests can verify the kwarg propagation.
    """

    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        num_blocks: int = 12,
        interctc_layer_idx=None,
        interctc_use_conditioning: bool = False,
    ):
        super().__init__()
        self.proj = nn.Linear(input_dim, output_dim)
        self.num_blocks = num_blocks
        self.interctc_layer_idx = interctc_layer_idx or []
        self.interctc_use_conditioning = interctc_use_conditioning
        self.conditioning_layer: nn.Linear | None = None  # set externally if conditioning
        # Track whether ctc kwarg was received
        self.last_ctc_arg = None

    def forward(self, feats, feats_lengths, masks=None, return_all_hs=False, ctc=None):
        self.last_ctc_arg = ctc
        out = self.proj(feats)

        # Apply conditioning if enabled (mirrors e_branchformer.py logic)
        if self.interctc_use_conditioning and ctc is not None and self.conditioning_layer is not None:
            logits = ctc.ctc_lo(out)  # (B, T, V)
            probs = logits.softmax(dim=-1)
            out = out + self.conditioning_layer(probs)

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


# ---------------------------------------------------------------------------
# Factories
# ---------------------------------------------------------------------------


def make_ctc(vocab=VOCAB, enc_dim=ENC_DIM):
    return CTC(odim=len(vocab), encoder_output_size=enc_dim)


def make_model(
    interctc_weight: float = 0.3,
    interctc_layer_idx=None,
    interctc_use_conditioning: bool = False,
    weighted_sum: bool = False,
    device: str = "cpu",
) -> XeusPRModel:
    if interctc_layer_idx is None:
        interctc_layer_idx = [6]
    encoder = MockConditioningEncoder(
        input_dim=FEAT_DIM,
        output_dim=ENC_DIM,
        num_blocks=12,
        interctc_layer_idx=interctc_layer_idx,
        interctc_use_conditioning=interctc_use_conditioning,
    )
    ctc = make_ctc()
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
        interctc_use_conditioning=interctc_use_conditioning,
    )
    return model.to(device)


def make_batch(B: int = 4, T: int = 20, device: str = "cpu"):
    speech = torch.randn(B, T, FEAT_DIM, device=device)
    speech_lengths = torch.randint(T // 2, T + 1, (B,), device=device)
    speech_lengths[0] = T
    max_text_len = 6
    text = torch.randint(1, len(VOCAB), (B, max_text_len), device=device)
    text_lengths = torch.randint(2, max_text_len + 1, (B,), device=device)
    text_lengths[0] = max_text_len
    for i in range(B):
        text[i, text_lengths[i] :] = -1
    return speech, speech_lengths, text, text_lengths


# ---------------------------------------------------------------------------
# 1. Structural: conditioning_layer creation
# ---------------------------------------------------------------------------


def test_conditioning_layer_created_when_flag_true():
    """encoder.conditioning_layer is a Linear(vocab_size -> enc_dim) when flag is True."""
    model = make_model(interctc_use_conditioning=True)
    cl = model.encoder.conditioning_layer
    assert cl is not None, "conditioning_layer should be set when interctc_use_conditioning=True"
    assert isinstance(cl, nn.Linear), f"Expected nn.Linear, got {type(cl)}"
    assert cl.in_features == len(VOCAB), (
        f"conditioning_layer input should be vocab size {len(VOCAB)}, got {cl.in_features}"
    )
    assert cl.out_features == ENC_DIM, (
        f"conditioning_layer output should be enc_dim {ENC_DIM}, got {cl.out_features}"
    )


def test_conditioning_flag_set_on_encoder():
    """encoder.interctc_use_conditioning is True when the model is built with the flag."""
    model = make_model(interctc_use_conditioning=True)
    assert model.encoder.interctc_use_conditioning is True


def test_no_conditioning_layer_by_default():
    """By default, encoder.conditioning_layer is None and flag is False."""
    model = make_model(interctc_use_conditioning=False)
    # conditioning_layer should be None (as initialized in MockConditioningEncoder)
    assert model.encoder.conditioning_layer is None
    assert model.encoder.interctc_use_conditioning is False


# ---------------------------------------------------------------------------
# 2. ctc kwarg propagation in encode()
# ---------------------------------------------------------------------------


def test_encode_passes_ctc_to_encoder():
    """encode() passes ctc=self.ctc to the encoder in the non-weighted_sum path."""
    model = make_model(interctc_use_conditioning=True)
    model.eval()
    speech, sl, _, _ = make_batch()
    with torch.no_grad():
        model.encode(speech, sl)
    assert model.encoder.last_ctc_arg is model.ctc, (
        "encode() must pass ctc=self.ctc to encoder.forward()"
    )


def test_encode_passes_ctc_even_without_conditioning():
    """encode() always passes ctc= (safe when interctc_use_conditioning=False)."""
    model = make_model(interctc_use_conditioning=False)
    model.eval()
    speech, sl, _, _ = make_batch()
    with torch.no_grad():
        model.encode(speech, sl)
    assert model.encoder.last_ctc_arg is model.ctc


def test_encode_weighted_sum_does_not_receive_ctc():
    """weighted_sum path uses return_all_hs=True and does not call the conditioning branch."""
    model = make_model(weighted_sum=True, interctc_use_conditioning=False)
    model.eval()
    speech, sl, _, _ = make_batch()
    with torch.no_grad():
        model.encode(speech, sl)
    # weighted_sum path passes return_all_hs=True but not ctc= explicitly;
    # MockEncoder will record None for last_ctc_arg in that branch
    # (the weighted_sum path in xeuspr_model.py does not pass ctc=)
    assert model.encoder.last_ctc_arg is None


# ---------------------------------------------------------------------------
# 3. Forward / backward with conditioning
# ---------------------------------------------------------------------------


def test_forward_with_conditioning_is_finite():
    """forward() with self-conditioning returns a finite loss."""
    model = make_model(interctc_use_conditioning=True)
    model.train()
    speech, sl, text, tl = make_batch()
    out = model(speech, sl, text, tl)
    assert not torch.isnan(out["loss"]), "Loss is NaN with self-conditioning"
    assert not torch.isinf(out["loss"]), "Loss is Inf with self-conditioning"


def test_forward_conditioning_logs_interctc_stats():
    """With interctc_weight>0 and conditioning, loss_interctc_layerN appears in stats."""
    model = make_model(interctc_weight=0.3, interctc_layer_idx=[6], interctc_use_conditioning=True)
    model.train()
    speech, sl, text, tl = make_batch()
    out = model(speech, sl, text, tl)
    assert "loss_interctc_layer6" in out["stats"], (
        f"Missing loss_interctc_layer6. Got: {list(out['stats'].keys())}"
    )


def test_backward_flows_through_conditioning_layer():
    """backward() produces gradients for conditioning_layer weights."""
    model = make_model(interctc_use_conditioning=True)
    model.train()
    speech, sl, text, tl = make_batch()
    out = model(speech, sl, text, tl)
    out["loss"].backward()
    cl_grad = model.encoder.conditioning_layer.weight.grad
    assert cl_grad is not None, "No grad on conditioning_layer.weight"
    assert cl_grad.abs().sum() > 0, "conditioning_layer.weight grad is zero"


def test_backward_flows_through_ctc_head_with_conditioning():
    """With self-conditioning, CTC head still receives gradients."""
    model = make_model(interctc_use_conditioning=True)
    model.train()
    speech, sl, text, tl = make_batch()
    out = model(speech, sl, text, tl)
    out["loss"].backward()
    ctc_grad = model.ctc.ctc_lo.weight.grad
    assert ctc_grad is not None, "No grad on ctc_lo.weight"
    assert ctc_grad.abs().sum() > 0, "ctc_lo.weight grad is zero"


def test_conditioning_vs_no_conditioning_losses_differ():
    """With identical weights, self-conditioning changes the loss value."""
    torch.manual_seed(0)
    model_base = make_model(interctc_use_conditioning=False)
    torch.manual_seed(0)
    model_cond = make_model(interctc_use_conditioning=True)

    # Copy weights from base to conditioning model so only the conditioning path differs
    model_cond.load_state_dict(model_base.state_dict(), strict=False)

    model_base.eval()
    model_cond.eval()
    speech, sl, text, tl = make_batch(B=2, T=15)

    with torch.no_grad():
        loss_base = model_base(speech, sl, text, tl)["loss"].item()
        loss_cond = model_cond(speech, sl, text, tl)["loss"].item()

    assert loss_base != loss_cond, (
        "Self-conditioning should change the loss value vs no-conditioning "
        f"(base={loss_base:.4f}, cond={loss_cond:.4f})"
    )


# ---------------------------------------------------------------------------
# 4. Integration tests: real builder (require xeus config on disk)
# ---------------------------------------------------------------------------


@pytest.mark.skipif(not _xeus_config_exists, reason=f"Xeus config not found at {XEUS_CONFIG}")
def test_builder_sets_conditioning_layer_on_real_encoder():
    """build_xeus_pr() with interctc_use_conditioning=True creates conditioning_layer."""
    from src.model.xeusphoneme.builders import build_xeus_pr

    model = build_xeus_pr(
        config_file=XEUS_CONFIG,
        checkpoint=None,
        vocab_file=XEUS_VOCAB,
        ctc_config={"ctc_type": "builtin"},
        interctc_layer_idx=[9],
        interctc_weight=0.3,
        interctc_use_conditioning=True,
    )
    assert model.encoder.conditioning_layer is not None, (
        "conditioning_layer should be set on real encoder"
    )
    assert model.encoder.interctc_use_conditioning is True
    # Shape check: in_features = vocab size (from ipa_vocab.json), out_features = encoder output size
    assert model.encoder.conditioning_layer.out_features == model.encoder.output_size()


@pytest.mark.skipif(not _xeus_config_exists, reason=f"Xeus config not found at {XEUS_CONFIG}")
def test_builder_no_conditioning_default_on_real_encoder():
    """build_xeus_pr() without the flag leaves conditioning_layer as None."""
    from src.model.xeusphoneme.builders import build_xeus_pr

    model = build_xeus_pr(
        config_file=XEUS_CONFIG,
        checkpoint=None,
        vocab_file=XEUS_VOCAB,
        ctc_config={"ctc_type": "builtin"},
    )
    assert model.encoder.conditioning_layer is None
    assert model.encoder.interctc_use_conditioning is False


@pytest.mark.skipif(not _xeus_config_exists, reason=f"Xeus config not found at {XEUS_CONFIG}")
def test_builder_from_hf_propagates_conditioning():
    """build_xeus_pr_from_hf() propagates interctc_use_conditioning to the model."""
    from src.model.xeusphoneme.builders import build_xeus_pr_from_hf

    model = build_xeus_pr_from_hf(
        work_dir="exp/cache/xeus",
        load_ckpt=False,
        vocab_file=XEUS_VOCAB,
        ctc_config={"ctc_type": "builtin"},
        interctc_layer_idx=[9],
        interctc_weight=0.3,
        interctc_use_conditioning=True,
    )
    assert model.encoder.conditioning_layer is not None
    assert model.encoder.interctc_use_conditioning is True


# ---------------------------------------------------------------------------
# 5. GPU tests
# ---------------------------------------------------------------------------


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
def test_forward_backward_with_conditioning_on_gpu():
    """Self-conditioning forward+backward on CUDA is finite and produces gradients."""
    device = "cuda"
    model = make_model(interctc_use_conditioning=True, device=device)
    model.train()
    speech, sl, text, tl = make_batch(B=4, T=20, device=device)
    out = model(speech, sl, text, tl)
    assert not torch.isnan(out["loss"]), "Loss is NaN on GPU"
    assert not torch.isinf(out["loss"]), "Loss is Inf on GPU"
    out["loss"].backward()
    cl_grad = model.encoder.conditioning_layer.weight.grad
    assert cl_grad is not None and cl_grad.abs().sum() > 0
    ctc_grad = model.ctc.ctc_lo.weight.grad
    assert ctc_grad is not None and ctc_grad.abs().sum() > 0
    print(f"  GPU loss={out['loss'].item():.4f}")


@pytest.mark.skipif(
    not torch.cuda.is_available() or not _xeus_config_exists,
    reason="CUDA or xeus config not available",
)
def test_real_model_with_conditioning_on_gpu():
    """Build real xeus model with self-conditioning and run a forward pass on GPU."""
    from src.model.xeusphoneme.builders import build_xeus_pr

    model = build_xeus_pr(
        config_file=XEUS_CONFIG,
        checkpoint=None,
        vocab_file=XEUS_VOCAB,
        ctc_config={"ctc_type": "builtin"},
        interctc_layer_idx=[9],
        interctc_weight=0.3,
        interctc_use_conditioning=True,
    ).cuda()
    model.eval()

    B, T = 2, 8000
    speech = torch.randn(B, T, device="cuda")
    speech_lengths = torch.tensor([T, T // 2], device="cuda")

    with torch.no_grad():
        logits, lens = model.ctc_logits(speech, speech_lengths)

    assert isinstance(logits, torch.Tensor)
    assert not torch.isnan(logits).any(), "NaN in logits on GPU with real model"
    print(f"  GPU logits shape: {logits.shape}, lens: {lens.tolist()}")


# ---------------------------------------------------------------------------
# __main__ runner (mirrors project convention from test_xeuspr_interctc.py)
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    tests = [
        # Structural
        ("conditioning_layer created", test_conditioning_layer_created_when_flag_true),
        ("conditioning flag set on encoder", test_conditioning_flag_set_on_encoder),
        ("no conditioning layer by default", test_no_conditioning_layer_by_default),
        # ctc kwarg propagation
        ("encode() passes ctc= to encoder", test_encode_passes_ctc_to_encoder),
        ("encode() passes ctc= without conditioning", test_encode_passes_ctc_even_without_conditioning),
        ("weighted_sum path: ctc not passed", test_encode_weighted_sum_does_not_receive_ctc),
        # Forward / backward
        ("forward() with conditioning: finite loss", test_forward_with_conditioning_is_finite),
        ("forward() with conditioning: interctc stats", test_forward_conditioning_logs_interctc_stats),
        ("backward(): grad to conditioning_layer", test_backward_flows_through_conditioning_layer),
        ("backward(): grad to ctc head", test_backward_flows_through_ctc_head_with_conditioning),
        ("conditioning changes loss value", test_conditioning_vs_no_conditioning_losses_differ),
    ]

    if _xeus_config_exists:
        tests += [
            ("builder sets conditioning_layer (real)", test_builder_sets_conditioning_layer_on_real_encoder),
            ("builder no conditioning default (real)", test_builder_no_conditioning_default_on_real_encoder),
            ("build_xeus_pr_from_hf propagates flag", test_builder_from_hf_propagates_conditioning),
        ]
    else:
        print(f"\n[SKIP] Integration tests: {XEUS_CONFIG} not found\n")

    if torch.cuda.is_available():
        tests += [
            ("GPU: forward+backward with conditioning", test_forward_backward_with_conditioning_on_gpu),
        ]
        if _xeus_config_exists:
            tests += [
                ("GPU+real model: conditioning forward", test_real_model_with_conditioning_on_gpu),
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
    if failed:
        raise SystemExit(1)
