"""Tests for XeusPRModel joint CTC+Attention training.

Joint training combines a CTC loss with an attention-decoder cross-entropy loss:
    L = ctc_weight * L_ctc + (1 - ctc_weight) * L_att

Setting ctc_weight=1.0 (default) keeps existing CTC-only behaviour.
The TransformerDecoder is only built when decoder_config is passed to the builder.

Tests cover:
  - Default (decoder=None, ctc_weight=1.0): no decoder, no attention path
  - With decoder: model.decoder is not None, criterion_att exists, sos/eos set
  - _calc_att_loss() returns finite loss and acc_att in [0, 1]
  - forward() with ctc_weight=0.3: loss_att and acc_att appear in stats
  - forward() with ctc_weight=1.0: no loss_att in stats
  - backward(): grads flow to decoder and criterion_att
  - builder (build_xeus_pr) creates a TransformerDecoder when decoder_config is given
  - GPU: forward+backward with decoder on CUDA
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

# Vocabulary must contain <sos> (id=1) and <eos> (id=2) to match real vocabs
VOCAB = ["<blank>", "<sos>", "<eos>", "<unk>", "a", "b", "c", "d", "e", "f"]
BLANK_ID = 0
SOS_ID = 1
EOS_ID = 2
ENC_DIM = 64
FEAT_DIM = 32

XEUS_CONFIG = "exp/cache/xeus/model/config.yaml"
XEUS_VOCAB = "src/model/xeusphoneme/resources/ipa_vocab.json"

_xeus_config_exists = pathlib.Path(XEUS_CONFIG).exists()


# ---------------------------------------------------------------------------
# Minimal mock encoder
# ---------------------------------------------------------------------------


class MockEncoder(nn.Module):
    def __init__(self, input_dim: int, output_dim: int, num_blocks: int = 12):
        super().__init__()
        self.proj = nn.Linear(input_dim, output_dim)
        self.num_blocks = num_blocks
        self.interctc_layer_idx = []
        self.interctc_use_conditioning = False
        self.conditioning_layer = None

    def forward(self, feats, feats_lengths, masks=None, return_all_hs=False, ctc=None):
        out = self.proj(feats)
        return out, feats_lengths, None

    def output_size(self):
        return self.proj.out_features


# ---------------------------------------------------------------------------
# Minimal mock decoder that mimics TransformerDecoder's forward signature
# ---------------------------------------------------------------------------


class MockDecoder(nn.Module):
    """Forward: (enc_out, enc_lens, ys_in_pad, ys_in_lens) -> (logits, None)."""

    def __init__(self, vocab_size: int, encoder_output_size: int):
        super().__init__()
        self.embed = nn.Embedding(vocab_size, encoder_output_size)
        self.out_proj = nn.Linear(encoder_output_size, vocab_size)
        # Expose a sub-module so tests can check grad propagation
        self.decoders = nn.ModuleList(
            [nn.Linear(encoder_output_size, encoder_output_size)]
        )

    def forward(self, enc_out, enc_lens, ys_in_pad, ys_in_lens):
        emb = self.embed(ys_in_pad)  # (B, L, D)
        # Blend with a single cross-attention-like step
        ctx = enc_out.mean(dim=1, keepdim=True)  # (B, 1, D)
        h = emb + ctx
        h = self.decoders[0](h)
        logits = self.out_proj(h)  # (B, L, V)
        return logits, None


# ---------------------------------------------------------------------------
# Factories
# ---------------------------------------------------------------------------


def make_ctc(vocab=VOCAB, enc_dim=ENC_DIM):
    return CTC(odim=len(vocab), encoder_output_size=enc_dim)


def make_model(
    ctc_weight: float = 1.0,
    with_decoder: bool = False,
    device: str = "cpu",
) -> XeusPRModel:
    encoder = MockEncoder(input_dim=FEAT_DIM, output_dim=ENC_DIM)
    ctc = make_ctc()
    decoder = (
        MockDecoder(vocab_size=len(VOCAB), encoder_output_size=ENC_DIM)
        if with_decoder
        else None
    )
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
        sym_sos="<sos>",
        sym_eos="<eos>",
        freeze_frontend=False,
        ctc_weight=ctc_weight,
        decoder=decoder,
    )
    return model.to(device)


def make_batch(B: int = 4, T: int = 20, device: str = "cpu"):
    speech = torch.randn(B, T, FEAT_DIM, device=device)
    speech_lengths = torch.randint(T // 2, T + 1, (B,), device=device)
    speech_lengths[0] = T
    max_text_len = 5
    # Use token ids >= 4 (skip special tokens) for actual phones
    text = torch.randint(4, len(VOCAB), (B, max_text_len), device=device)
    text_lengths = torch.randint(2, max_text_len + 1, (B,), device=device)
    text_lengths[0] = max_text_len
    for i in range(B):
        text[i, text_lengths[i] :] = -1
    return speech, speech_lengths, text, text_lengths


# ---------------------------------------------------------------------------
# 1. Structural: default (CTC-only) behaviour
# ---------------------------------------------------------------------------


def test_default_no_decoder():
    """With default params, model.decoder is None and ctc_weight is 1.0."""
    model = make_model(ctc_weight=1.0, with_decoder=False)
    assert model.decoder is None, "decoder should be None by default"
    assert model.ctc_weight == 1.0


def test_default_no_criterion_att():
    """Without decoder, criterion_att should not exist."""
    model = make_model(ctc_weight=1.0, with_decoder=False)
    assert not hasattr(
        model, "criterion_att"
    ), "criterion_att should not exist without decoder"


def test_with_decoder_sets_attributes():
    """With decoder, model.decoder, criterion_att, sos, eos are all set."""
    model = make_model(ctc_weight=0.3, with_decoder=True)
    assert model.decoder is not None, "decoder should be set"
    assert hasattr(model, "criterion_att"), "criterion_att should exist"
    assert model.sos == SOS_ID, f"sos should be {SOS_ID}, got {model.sos}"
    assert model.eos == EOS_ID, f"eos should be {EOS_ID}, got {model.eos}"


# ---------------------------------------------------------------------------
# 2. _calc_att_loss: correctness
# ---------------------------------------------------------------------------


def test_calc_att_loss_finite():
    """_calc_att_loss() returns a finite scalar loss."""
    model = make_model(ctc_weight=0.3, with_decoder=True)
    model.train()
    _, _, text, text_lengths = make_batch()
    enc_out = torch.randn(4, 20, ENC_DIM)
    enc_lens = torch.full((4,), 20)
    loss_att, acc_att = model._calc_att_loss(enc_out, enc_lens, text, text_lengths)
    assert loss_att.dim() == 0, "loss_att should be a scalar"
    assert torch.isfinite(loss_att), f"loss_att is not finite: {loss_att.item()}"


def test_calc_att_loss_acc_in_range():
    """acc_att returned by _calc_att_loss is in [0, 1]."""
    model = make_model(ctc_weight=0.3, with_decoder=True)
    model.eval()
    _, _, text, text_lengths = make_batch()
    enc_out = torch.randn(4, 20, ENC_DIM)
    enc_lens = torch.full((4,), 20)
    with torch.no_grad():
        _, acc_att = model._calc_att_loss(enc_out, enc_lens, text, text_lengths)
    assert 0.0 <= acc_att <= 1.0, f"acc_att out of range: {acc_att}"


# ---------------------------------------------------------------------------
# 3. forward(): attention stats presence
# ---------------------------------------------------------------------------


def test_forward_joint_has_loss_att_in_stats():
    """forward() with ctc_weight=0.3 and decoder adds loss_att/acc_att to stats."""
    model = make_model(ctc_weight=0.3, with_decoder=True)
    model.train()
    speech, sl, text, tl = make_batch()
    out = model(speech, sl, text, tl)
    assert (
        "loss_att" in out["stats"]
    ), f"loss_att missing from stats: {list(out['stats'].keys())}"
    assert (
        "acc_att" in out["stats"]
    ), f"acc_att missing from stats: {list(out['stats'].keys())}"


def test_forward_joint_loss_is_finite():
    """Joint loss is finite with ctc_weight=0.3."""
    model = make_model(ctc_weight=0.3, with_decoder=True)
    model.train()
    speech, sl, text, tl = make_batch()
    out = model(speech, sl, text, tl)
    assert torch.isfinite(
        out["loss"]
    ), f"Joint loss is not finite: {out['loss'].item()}"


def test_forward_ctc_only_no_loss_att():
    """forward() with ctc_weight=1.0 does NOT call the decoder; no loss_att in stats."""
    model = make_model(ctc_weight=1.0, with_decoder=True)
    model.train()
    speech, sl, text, tl = make_batch()
    out = model(speech, sl, text, tl)
    assert (
        "loss_att" not in out["stats"]
    ), f"loss_att should not appear with ctc_weight=1.0. Stats: {list(out['stats'].keys())}"


def test_forward_ctc_only_default_no_loss_att():
    """Default model (no decoder) also has no loss_att in stats."""
    model = make_model(ctc_weight=1.0, with_decoder=False)
    model.train()
    speech, sl, text, tl = make_batch()
    out = model(speech, sl, text, tl)
    assert "loss_att" not in out["stats"]


# ---------------------------------------------------------------------------
# 4. Backward: gradient propagation
# ---------------------------------------------------------------------------


def test_backward_grads_to_decoder():
    """backward() flows gradients to decoder parameters."""
    model = make_model(ctc_weight=0.3, with_decoder=True)
    model.train()
    speech, sl, text, tl = make_batch()
    out = model(speech, sl, text, tl)
    out["loss"].backward()
    dec_grad = model.decoder.decoders[0].weight.grad
    assert dec_grad is not None, "No grad on decoder.decoders[0].weight"
    assert dec_grad.abs().sum() > 0, "decoder.decoders[0].weight grad is zero"


def test_backward_grads_to_criterion_att():
    """backward() flows gradients to criterion_att (label smoothing has no params,
    but loss_att should still flow back through criterion_att to decoder embed)."""
    model = make_model(ctc_weight=0.3, with_decoder=True)
    model.train()
    speech, sl, text, tl = make_batch()
    out = model(speech, sl, text, tl)
    out["loss"].backward()
    # Verify embed gets grad (flows through criterion_att.forward -> decoder.out_proj -> embed)
    embed_grad = model.decoder.embed.weight.grad
    assert embed_grad is not None, "No grad on decoder.embed.weight"
    assert embed_grad.abs().sum() > 0, "decoder.embed.weight grad is zero"


def test_backward_grads_to_ctc_head():
    """CTC head still gets gradients in joint mode."""
    model = make_model(ctc_weight=0.3, with_decoder=True)
    model.train()
    speech, sl, text, tl = make_batch()
    out = model(speech, sl, text, tl)
    out["loss"].backward()
    ctc_grad = model.ctc.ctc_lo.weight.grad
    assert ctc_grad is not None, "No grad on ctc_lo.weight"
    assert ctc_grad.abs().sum() > 0, "ctc_lo.weight grad is zero"


# ---------------------------------------------------------------------------
# 5. get_trainable_parameters: decoder in "head" group
# ---------------------------------------------------------------------------


def test_trainable_params_decoder_in_head():
    """get_trainable_parameters() puts decoder params in 'head' group."""
    model = make_model(ctc_weight=0.3, with_decoder=True)
    groups = model.get_trainable_parameters()
    head_params = set(id(p) for p in groups["head"])
    # Check that decoder.embed.weight is in head
    dec_embed_id = id(model.decoder.embed.weight)
    assert (
        dec_embed_id in head_params
    ), "decoder.embed.weight should be in 'head' param group"


# ---------------------------------------------------------------------------
# 6. Integration tests: real builder (require xeus config on disk)
# ---------------------------------------------------------------------------


@pytest.mark.skipif(
    not _xeus_config_exists, reason=f"Xeus config not found at {XEUS_CONFIG}"
)
def test_builder_creates_transformer_decoder():
    """build_xeus_pr() with decoder_config creates a TransformerDecoder."""
    from src.model.powsm.transformer_decoder import TransformerDecoder
    from src.model.xeusphoneme.builders import build_xeus_pr

    model = build_xeus_pr(
        config_file=XEUS_CONFIG,
        checkpoint=None,
        vocab_file=XEUS_VOCAB,
        ctc_config={"ctc_type": "builtin"},
        ctc_weight=0.3,
        decoder_config={
            "attention_heads": 4,
            "linear_units": 512,
            "num_blocks": 2,
            "dropout_rate": 0.1,
            "positional_dropout_rate": 0.1,
            "input_layer": "embed",
            "use_output_layer": True,
            "normalize_before": True,
        },
    )
    assert isinstance(
        model.decoder, TransformerDecoder
    ), f"Expected TransformerDecoder, got {type(model.decoder)}"
    assert model.ctc_weight == 0.3
    assert hasattr(model, "criterion_att")


@pytest.mark.skipif(
    not _xeus_config_exists, reason=f"Xeus config not found at {XEUS_CONFIG}"
)
def test_builder_no_decoder_by_default():
    """build_xeus_pr() without decoder_config leaves model.decoder as None."""
    from src.model.xeusphoneme.builders import build_xeus_pr

    model = build_xeus_pr(
        config_file=XEUS_CONFIG,
        checkpoint=None,
        vocab_file=XEUS_VOCAB,
        ctc_config={"ctc_type": "builtin"},
    )
    assert model.decoder is None


# ---------------------------------------------------------------------------
# 7. GPU tests
# ---------------------------------------------------------------------------


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
def test_forward_backward_joint_on_gpu():
    """Joint CTC+Attention forward+backward on CUDA is finite and produces gradients."""
    device = "cuda"
    model = make_model(ctc_weight=0.3, with_decoder=True, device=device)
    model.train()
    speech, sl, text, tl = make_batch(B=4, T=20, device=device)
    out = model(speech, sl, text, tl)
    assert torch.isfinite(out["loss"]), f"Loss not finite on GPU: {out['loss'].item()}"
    assert "loss_att" in out["stats"]
    out["loss"].backward()
    dec_grad = model.decoder.decoders[0].weight.grad
    assert dec_grad is not None and dec_grad.abs().sum() > 0
    print(
        f"  GPU joint loss={out['loss'].item():.4f}, acc_att={out['stats']['acc_att'].item():.4f}"
    )


# ---------------------------------------------------------------------------
# __main__ runner
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    tests = [
        # Structural
        ("default: no decoder", test_default_no_decoder),
        ("default: no criterion_att", test_default_no_criterion_att),
        ("with_decoder: attributes set", test_with_decoder_sets_attributes),
        # _calc_att_loss
        ("_calc_att_loss: finite loss", test_calc_att_loss_finite),
        ("_calc_att_loss: acc in [0,1]", test_calc_att_loss_acc_in_range),
        # forward
        ("forward joint: loss_att in stats", test_forward_joint_has_loss_att_in_stats),
        ("forward joint: finite loss", test_forward_joint_loss_is_finite),
        ("forward ctc_weight=1.0: no loss_att", test_forward_ctc_only_no_loss_att),
        ("forward default: no loss_att", test_forward_ctc_only_default_no_loss_att),
        # backward
        ("backward: grads to decoder", test_backward_grads_to_decoder),
        (
            "backward: grads to embed via criterion_att",
            test_backward_grads_to_criterion_att,
        ),
        ("backward: grads to ctc head", test_backward_grads_to_ctc_head),
        # trainable params
        ("trainable params: decoder in head", test_trainable_params_decoder_in_head),
    ]

    if _xeus_config_exists:
        tests += [
            (
                "builder creates TransformerDecoder",
                test_builder_creates_transformer_decoder,
            ),
            ("builder: no decoder by default", test_builder_no_decoder_by_default),
        ]
    else:
        print(f"\n[SKIP] Integration tests: {XEUS_CONFIG} not found\n")

    if torch.cuda.is_available():
        tests += [
            ("GPU: joint forward+backward", test_forward_backward_joint_on_gpu),
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
