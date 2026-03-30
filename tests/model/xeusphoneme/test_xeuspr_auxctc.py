"""Tests for aux-CTC (orthographic vocabulary) in XeusPRModel.

Covers:
  1. TextTokenizer — CharTokenizer and build_text_tokenizer factory
  2. collate_fn — asr_text_tokens padding when aux_tokenizer is present
  3. XeusPRModel — backward-compat phone path (no ctc_aux)
  4. XeusPRModel — ortho path: conditioning layer sized from aux vocab
  5. XeusPRModel — ortho path: interctc loss uses asr_text_tokens, not phone text
  6. XeusPRModel — ortho path with missing asr_text_tokens: no interctc loss
  7. XeusPRModel — ctc_aux.* in 'head' param group
  8. XeusPRModel — backward grads through ctc_aux
  9. builders.py — build_xeus_pr with ctc_aux_config (skip if config absent)
 10. builders.py — build_xeus_pr without ctc_aux_config stays phone-only (skip if absent)
 11. GPU forward+backward in ortho mode
"""

import json
import pathlib
import tempfile
import traceback

import pytest
import torch
import torch.nn as nn

from src.model.powsm.ctc import CTC
from src.model.xeusphoneme.xeuspr_model import XeusPRModel
from src.data.text_tokenizer import CharTokenizer, build_text_tokenizer

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

PHONE_VOCAB = ["<blank>", "<sos>", "<eos>", "<unk>", "a", "b", "c", "d", "e", "f"]
AUX_VOCAB = {"<blank>": 0, "<unk>": 1, "<space>": 2, "h": 3, "e": 4, "l": 5, "o": 6, "w": 7, "r": 8, "d": 9}
ENC_DIM = 64
FEAT_DIM = 32
AUX_VOCAB_SIZE = len(AUX_VOCAB)  # 10
PHONE_VOCAB_SIZE = len(PHONE_VOCAB)  # 10

XEUS_CONFIG = "exp/cache/xeus/model/config.yaml"
XEUS_VOCAB = "src/model/xeusphoneme/resources/ipa_vocab.json"
_xeus_config_exists = pathlib.Path(XEUS_CONFIG).exists()

# ---------------------------------------------------------------------------
# Mock modules
# ---------------------------------------------------------------------------


class MockEncoder(nn.Module):
    """Encoder without interctc support — mirrors the existing test mock."""

    def __init__(self, input_dim=FEAT_DIM, output_dim=ENC_DIM, num_blocks=12):
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


class InterCTCMockEncoder(nn.Module):
    """Encoder that returns one intermediate output — used to test interctc paths."""

    def __init__(self, input_dim=FEAT_DIM, output_dim=ENC_DIM, interctc_layer_idx=None):
        super().__init__()
        self.proj = nn.Linear(input_dim, output_dim)
        self.num_blocks = 12
        self.interctc_layer_idx = interctc_layer_idx or [4]
        self.interctc_use_conditioning = False
        self.conditioning_layer = None

    def forward(self, feats, feats_lengths, masks=None, return_all_hs=False, ctc=None):
        out = self.proj(feats)
        # Simulate intermediate out at every configured layer
        intermediate_outs = [(idx, out.detach().clone()) for idx in self.interctc_layer_idx]
        # Apply conditioning if configured
        if self.interctc_use_conditioning and ctc is not None and self.conditioning_layer is not None:
            ctc_out = ctc.softmax(out)
            out = out + self.conditioning_layer(ctc_out)
        return (out, intermediate_outs), feats_lengths, None

    def output_size(self):
        return self.proj.out_features


# ---------------------------------------------------------------------------
# Factories
# ---------------------------------------------------------------------------


def make_phone_ctc(enc_dim=ENC_DIM):
    return CTC(odim=PHONE_VOCAB_SIZE, encoder_output_size=enc_dim)


def make_aux_ctc(enc_dim=ENC_DIM):
    return CTC(odim=AUX_VOCAB_SIZE, encoder_output_size=enc_dim)


def make_model(
    interctc_ctc_type="phone",
    with_aux_ctc=False,
    with_interctc=False,
    device="cpu",
) -> XeusPRModel:
    encoder_cls = InterCTCMockEncoder if with_interctc else MockEncoder
    encoder = encoder_cls()
    ctc = make_phone_ctc()
    ctc_aux = make_aux_ctc() if with_aux_ctc else None
    model = XeusPRModel(
        encoder=encoder,
        ctc=ctc,
        token_list=PHONE_VOCAB,
        frontend=None,
        specaug=None,
        normalize=None,
        preencoder=None,
        ignore_id=-1,
        sym_blank="<blank>",
        sym_sos="<sos>",
        sym_eos="<eos>",
        freeze_frontend=False,
        interctc_weight=0.3 if with_interctc else 0.0,
        interctc_use_conditioning=False,
        interctc_ctc_type=interctc_ctc_type,
        ctc_aux=ctc_aux,
    )
    return model.to(device)


def make_phone_batch(B=2, T=20, device="cpu"):
    speech = torch.randn(B, T, FEAT_DIM, device=device)
    speech_lengths = torch.full((B,), T, dtype=torch.long, device=device)
    max_text_len = 5
    text = torch.randint(4, PHONE_VOCAB_SIZE, (B, max_text_len), device=device)
    text_lengths = torch.full((B,), max_text_len, dtype=torch.long, device=device)
    return speech, speech_lengths, text, text_lengths


def make_aux_text(B=2, T_asr=6, device="cpu"):
    asr_text = torch.randint(0, AUX_VOCAB_SIZE, (B, T_asr), device=device)
    asr_text_length = torch.full((B,), T_asr, dtype=torch.long, device=device)
    return asr_text, asr_text_length


# ---------------------------------------------------------------------------
# 1. TextTokenizer tests
# ---------------------------------------------------------------------------


def test_char_tokenizer_basic():
    """CharTokenizer maps chars correctly from a JSON vocab."""
    with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
        json.dump(AUX_VOCAB, f)
        tmp_path = f.name
    tok = CharTokenizer(tmp_path)
    assert tok.vocab_size == AUX_VOCAB_SIZE
    ids = tok.tokenize("hello")
    assert ids == [AUX_VOCAB["h"], AUX_VOCAB["e"], AUX_VOCAB["l"], AUX_VOCAB["l"], AUX_VOCAB["o"]]


def test_char_tokenizer_space():
    """CharTokenizer converts spaces to <space> token."""
    with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
        json.dump(AUX_VOCAB, f)
        tmp_path = f.name
    tok = CharTokenizer(tmp_path)
    ids = tok.tokenize("h e")
    assert ids[1] == AUX_VOCAB["<space>"], f"Expected <space>={AUX_VOCAB['<space>']}, got {ids[1]}"


def test_char_tokenizer_unk():
    """CharTokenizer maps unknown characters to <unk>."""
    with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
        json.dump(AUX_VOCAB, f)
        tmp_path = f.name
    tok = CharTokenizer(tmp_path)
    ids = tok.tokenize("z")  # 'z' not in AUX_VOCAB
    assert ids == [AUX_VOCAB["<unk>"]]


def test_build_text_tokenizer_char():
    """build_text_tokenizer with tokenizer_type='char' returns CharTokenizer."""
    with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
        json.dump(AUX_VOCAB, f)
        tmp_path = f.name
    tok = build_text_tokenizer(tmp_path, tokenizer_type="char")
    assert isinstance(tok, CharTokenizer)
    assert tok.vocab_size == AUX_VOCAB_SIZE


def test_build_text_tokenizer_bad_type():
    """build_text_tokenizer raises ValueError for unknown type."""
    with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
        json.dump(AUX_VOCAB, f)
        tmp_path = f.name
    with pytest.raises(ValueError, match="Unknown tokenizer_type"):
        build_text_tokenizer(tmp_path, tokenizer_type="bpe_unknown")


# ---------------------------------------------------------------------------
# 2. collate_fn — asr_text_tokens padding
# ---------------------------------------------------------------------------


def test_collate_fn_pads_asr_text_tokens():
    """collate_fn produces asr_text_tokens and asr_text_length when items have them."""
    import torch
    from src.data.kaldi_pretraining_dataset import KaldiPretrainingDataModule

    # Build a minimal KaldiPretrainingDataModule just to call its collate_fn
    # (we bypass __init__ by calling collate_fn directly as a regular function)
    batch = [
        {
            "key": "a",
            "speech": torch.zeros(100),
            "speech_length": 100,
            "text": "foo",
            "asr_text": "foo bar",
            "asr_text_tokens": [3, 4, 5],          # length 3
            "wavpath": "",
            "lang_sym": "eng",
            "accent_sym": "<unk>",
            "text_tokens": None,
        },
        {
            "key": "b",
            "speech": torch.zeros(80),
            "speech_length": 80,
            "text": "baz",
            "asr_text": "baz",
            "asr_text_tokens": [6, 7],              # length 2 — shorter
            "wavpath": "",
            "lang_sym": "eng",
            "accent_sym": "<unk>",
            "text_tokens": None,
        },
    ]

    dm = KaldiPretrainingDataModule.__new__(KaldiPretrainingDataModule)
    dm.ignore_id = -1
    dm.batch_size = 2
    dm.max_duration_sec = 20
    dm.sampling_rate = 16000

    result = dm.collate_fn(batch)

    assert "asr_text_tokens" in result, "asr_text_tokens missing from collate output"
    assert "asr_text_length" in result, "asr_text_length missing from collate output"
    assert result["asr_text_tokens"].shape == (2, 3), (
        f"Expected (2, 3), got {result['asr_text_tokens'].shape}"
    )
    assert result["asr_text_length"].tolist() == [3, 2]
    # Padding with ignore_id=-1
    assert result["asr_text_tokens"][1, 2].item() == -1, "Shorter sequence should be padded with -1"


def test_collate_fn_no_asr_tokens_skipped():
    """collate_fn does not add asr_text_tokens key if no items have it."""
    from src.data.kaldi_pretraining_dataset import KaldiPretrainingDataModule

    batch = [
        {
            "key": "a",
            "speech": torch.zeros(100),
            "speech_length": 100,
            "text": "foo",
            "asr_text": None,
            "asr_text_tokens": None,
            "wavpath": "",
            "lang_sym": "eng",
            "accent_sym": "<unk>",
            "text_tokens": None,
        },
    ]

    dm = KaldiPretrainingDataModule.__new__(KaldiPretrainingDataModule)
    dm.ignore_id = -1
    dm.batch_size = 1
    dm.max_duration_sec = 20
    dm.sampling_rate = 16000

    result = dm.collate_fn(batch)
    assert "asr_text_tokens" not in result, "asr_text_tokens should not appear when no items have it"
    assert "asr_text_length" not in result


# ---------------------------------------------------------------------------
# 3. XeusPRModel — backward-compat phone path (no ctc_aux)
# ---------------------------------------------------------------------------


def test_phone_path_no_ctc_aux_default():
    """Default model has ctc_aux=None and interctc_ctc_type='phone'."""
    model = make_model()
    assert model.ctc_aux is None
    assert model.interctc_ctc_type == "phone"


def test_phone_path_forward_finite():
    """Phone-only path (no ctc_aux) produces finite loss."""
    model = make_model()
    model.train()
    speech, sl, text, tl = make_phone_batch()
    out = model(speech, sl, text, tl)
    assert torch.isfinite(out["loss"]), f"Loss not finite: {out['loss'].item()}"


def test_phone_interctc_path_no_asr_tokens():
    """Phone interctc (no ctc_aux): interctc loss uses phone text, still works without asr_text_tokens."""
    model = make_model(interctc_ctc_type="phone", with_interctc=True)
    model.train()
    speech, sl, text, tl = make_phone_batch()
    # No asr_text_tokens passed — should still produce interctc loss from phone targets
    out = model(speech, sl, text, tl)
    assert torch.isfinite(out["loss"])
    assert any(k.startswith("loss_interctc_layer") for k in out["stats"]), (
        f"Expected loss_interctc_layer* in stats, got: {list(out['stats'].keys())}"
    )


def test_phone_interctc_backward():
    """Phone interctc: grads flow through ctc head."""
    model = make_model(interctc_ctc_type="phone", with_interctc=True)
    model.train()
    speech, sl, text, tl = make_phone_batch()
    out = model(speech, sl, text, tl)
    out["loss"].backward()
    assert model.ctc.ctc_lo.weight.grad is not None
    assert model.ctc.ctc_lo.weight.grad.abs().sum() > 0


# ---------------------------------------------------------------------------
# 4. XeusPRModel — ortho conditioning layer size
# ---------------------------------------------------------------------------


def test_ortho_conditioning_layer_size():
    """When interctc_ctc_type='ortho', conditioning_layer uses aux vocab size."""
    encoder = InterCTCMockEncoder()
    ctc = make_phone_ctc()
    ctc_aux = make_aux_ctc()
    model = XeusPRModel(
        encoder=encoder,
        ctc=ctc,
        token_list=PHONE_VOCAB,
        ignore_id=-1,
        sym_blank="<blank>",
        sym_sos="<sos>",
        sym_eos="<eos>",
        interctc_weight=0.3,
        interctc_use_conditioning=True,
        interctc_ctc_type="ortho",
        ctc_aux=ctc_aux,
    )
    cond = model.encoder.conditioning_layer
    assert cond is not None, "conditioning_layer should be set"
    assert cond.in_features == AUX_VOCAB_SIZE, (
        f"conditioning_layer input should be aux vocab size {AUX_VOCAB_SIZE}, "
        f"got {cond.in_features}"
    )
    assert cond.out_features == ENC_DIM


def test_phone_conditioning_layer_size():
    """When interctc_ctc_type='phone', conditioning_layer uses phone vocab size."""
    encoder = InterCTCMockEncoder()
    ctc = make_phone_ctc()
    model = XeusPRModel(
        encoder=encoder,
        ctc=ctc,
        token_list=PHONE_VOCAB,
        ignore_id=-1,
        sym_blank="<blank>",
        sym_sos="<sos>",
        sym_eos="<eos>",
        interctc_weight=0.3,
        interctc_use_conditioning=True,
        interctc_ctc_type="phone",
        ctc_aux=None,
    )
    cond = model.encoder.conditioning_layer
    assert cond is not None
    assert cond.in_features == PHONE_VOCAB_SIZE, (
        f"Expected {PHONE_VOCAB_SIZE}, got {cond.in_features}"
    )


# ---------------------------------------------------------------------------
# 5. XeusPRModel — ortho path: interctc loss uses asr_text_tokens
# ---------------------------------------------------------------------------


def test_ortho_interctc_loss_finite():
    """Ortho interctc with asr_text_tokens produces finite loss."""
    model = make_model(interctc_ctc_type="ortho", with_aux_ctc=True, with_interctc=True)
    model.train()
    speech, sl, text, tl = make_phone_batch()
    asr_text, asr_text_length = make_aux_text()
    out = model(speech, sl, text, tl, asr_text_tokens=asr_text, asr_text_length=asr_text_length)
    assert torch.isfinite(out["loss"]), f"Loss not finite: {out['loss'].item()}"
    assert any(k.startswith("loss_interctc_layer") for k in out["stats"]), (
        f"loss_interctc_layer* missing from stats: {list(out['stats'].keys())}"
    )


def test_ortho_interctc_uses_aux_ctc_head():
    """In ortho mode, interctc loss is computed via ctc_aux (different vocab), not self.ctc."""
    model = make_model(interctc_ctc_type="ortho", with_aux_ctc=True, with_interctc=True)
    model.train()
    speech, sl, text, tl = make_phone_batch()
    asr_text, asr_text_length = make_aux_text()

    # Patch ctc_aux.forward to record it was called
    called = {"aux": False, "phone": False}
    orig_aux = model.ctc_aux.forward
    orig_phone = model.ctc.forward

    def patched_aux(*args, **kwargs):
        called["aux"] = True
        return orig_aux(*args, **kwargs)

    def patched_phone(*args, **kwargs):
        called["phone"] = True
        return orig_phone(*args, **kwargs)

    model.ctc_aux.forward = patched_aux
    model.ctc.forward = patched_phone

    model(speech, sl, text, tl, asr_text_tokens=asr_text, asr_text_length=asr_text_length)

    # Restore
    model.ctc_aux.forward = orig_aux
    model.ctc.forward = orig_phone

    # ctc_aux should have been called for interctc; phone ctc for final loss
    assert called["aux"], "ctc_aux was not called in ortho interctc mode"
    assert called["phone"], "self.ctc (final loss) was not called"


def test_ortho_backward_grads_through_ctc_aux():
    """Backward pass flows gradients through ctc_aux.ctc_lo."""
    model = make_model(interctc_ctc_type="ortho", with_aux_ctc=True, with_interctc=True)
    model.train()
    speech, sl, text, tl = make_phone_batch()
    asr_text, asr_text_length = make_aux_text()
    out = model(speech, sl, text, tl, asr_text_tokens=asr_text, asr_text_length=asr_text_length)
    out["loss"].backward()
    grad = model.ctc_aux.ctc_lo.weight.grad
    assert grad is not None, "No grad on ctc_aux.ctc_lo.weight"
    assert grad.abs().sum() > 0, "ctc_aux.ctc_lo.weight grad is zero"


# ---------------------------------------------------------------------------
# 6. Ortho path with missing asr_text_tokens: no interctc loss
# ---------------------------------------------------------------------------


def test_ortho_no_interctc_loss_when_missing_asr_tokens():
    """When ortho is set but asr_text_tokens not provided, no interctc loss is added."""
    model = make_model(interctc_ctc_type="ortho", with_aux_ctc=True, with_interctc=True)
    model.train()
    speech, sl, text, tl = make_phone_batch()
    out = model(speech, sl, text, tl)  # no asr_text_tokens
    assert torch.isfinite(out["loss"])
    assert not any(k.startswith("loss_interctc_layer") for k in out["stats"]), (
        "interctc loss should be skipped when asr_text_tokens is missing"
    )


# ---------------------------------------------------------------------------
# 7. get_trainable_parameters: ctc_aux in 'head' group
# ---------------------------------------------------------------------------


def test_ctc_aux_in_head_param_group():
    """ctc_aux.* parameters land in the 'head' group."""
    model = make_model(interctc_ctc_type="ortho", with_aux_ctc=True)
    groups = model.get_trainable_parameters()
    head_ids = {id(p) for p in groups["head"]}
    aux_ctc_lo_id = id(model.ctc_aux.ctc_lo.weight)
    assert aux_ctc_lo_id in head_ids, "ctc_aux.ctc_lo.weight should be in 'head' param group"


def test_phone_ctc_still_in_head_with_aux():
    """self.ctc.* parameters remain in 'head' group even when ctc_aux is present."""
    model = make_model(interctc_ctc_type="ortho", with_aux_ctc=True)
    groups = model.get_trainable_parameters()
    head_ids = {id(p) for p in groups["head"]}
    phone_ctc_id = id(model.ctc.ctc_lo.weight)
    assert phone_ctc_id in head_ids, "ctc.ctc_lo.weight should still be in 'head' param group"


# ---------------------------------------------------------------------------
# 8. encode() routes CTC correctly
# ---------------------------------------------------------------------------


def test_encode_uses_aux_ctc_in_ortho_mode():
    """encode() passes ctc_aux (not self.ctc) to the encoder in ortho mode."""
    model = make_model(interctc_ctc_type="ortho", with_aux_ctc=True, with_interctc=True)
    model.eval()

    received_ctc = {}

    orig_forward = model.encoder.forward

    def capturing_forward(feats, feats_lengths, masks=None, return_all_hs=False, ctc=None):
        received_ctc["ctc"] = ctc
        return orig_forward(feats, feats_lengths, masks=masks, return_all_hs=return_all_hs, ctc=ctc)

    model.encoder.forward = capturing_forward

    speech, sl, _, _ = make_phone_batch()
    with torch.no_grad():
        model.encode(speech, sl)

    model.encoder.forward = orig_forward

    assert received_ctc.get("ctc") is model.ctc_aux, (
        "encode() should pass ctc_aux to encoder in ortho mode"
    )


def test_encode_uses_phone_ctc_in_phone_mode():
    """encode() passes self.ctc (not ctc_aux) to the encoder in phone mode."""
    model = make_model(interctc_ctc_type="phone", with_interctc=True)
    model.eval()

    received_ctc = {}
    orig_forward = model.encoder.forward

    def capturing_forward(feats, feats_lengths, masks=None, return_all_hs=False, ctc=None):
        received_ctc["ctc"] = ctc
        return orig_forward(feats, feats_lengths, masks=masks, return_all_hs=return_all_hs, ctc=ctc)

    model.encoder.forward = capturing_forward
    speech, sl, _, _ = make_phone_batch()
    with torch.no_grad():
        model.encode(speech, sl)

    model.encoder.forward = orig_forward
    assert received_ctc.get("ctc") is model.ctc, (
        "encode() should pass self.ctc to encoder in phone mode"
    )


# ---------------------------------------------------------------------------
# 9 & 10. Integration with real builder (skip if config absent)
# ---------------------------------------------------------------------------


def _build_tiny_sp_model(tmp_dir: str) -> str:
    """Train a tiny SentencePiece unigram model and return the .model path."""
    import sentencepiece as spm
    import os

    text_file = os.path.join(tmp_dir, "train.txt")
    model_prefix = os.path.join(tmp_dir, "tiny")
    with open(text_file, "w") as f:
        for line in ["hello world", "foo bar baz", "the quick brown fox", "abc def ghi"]:
            f.write(line + "\n")
    spm.SentencePieceTrainer.train(
        input=text_file,
        model_prefix=model_prefix,
        vocab_size=29,
        model_type="unigram",
        pad_id=0, unk_id=1, bos_id=2, eos_id=3,
        character_coverage=1.0,
    )
    return model_prefix + ".model"


@pytest.mark.skipif(not _xeus_config_exists, reason=f"Xeus config not found at {XEUS_CONFIG}")
def test_builder_with_ctc_aux_config():
    """build_xeus_pr() with ctc_aux_config creates ctc_aux on the model."""
    from src.model.xeusphoneme.builders import build_xeus_pr

    with tempfile.TemporaryDirectory() as tmp_dir:
        sp_model_path = _build_tiny_sp_model(tmp_dir)
        import sentencepiece as spm
        sp = spm.SentencePieceProcessor()
        sp.load(sp_model_path)
        expected_vocab_size = sp.get_piece_size()

        model = build_xeus_pr(
            config_file=XEUS_CONFIG,
            checkpoint=None,
            vocab_file=XEUS_VOCAB,
            ctc_config={"ctc_type": "builtin"},
            interctc_layer_idx=[4, 8],
            interctc_weight=0.3,
            interctc_ctc_type="ortho",
            ctc_aux_config={"vocab_file": sp_model_path, "dropout_rate": 0.0},
        )
    assert model.ctc_aux is not None, "ctc_aux should be built"
    assert model.ctc_aux.ctc_lo.out_features == expected_vocab_size
    assert model.interctc_ctc_type == "ortho"
    assert "ctc_aux_config" in model._net_config


@pytest.mark.skipif(not _xeus_config_exists, reason=f"Xeus config not found at {XEUS_CONFIG}")
def test_builder_default_no_ctc_aux():
    """build_xeus_pr() without ctc_aux_config leaves ctc_aux=None (backward compat)."""
    from src.model.xeusphoneme.builders import build_xeus_pr

    model = build_xeus_pr(
        config_file=XEUS_CONFIG,
        checkpoint=None,
        vocab_file=XEUS_VOCAB,
        ctc_config={"ctc_type": "builtin"},
    )
    assert model.ctc_aux is None
    assert model.interctc_ctc_type == "phone"


# ---------------------------------------------------------------------------
# 11. GPU
# ---------------------------------------------------------------------------


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
def test_ortho_forward_backward_gpu():
    """Ortho interctc forward+backward on CUDA is finite with grads."""
    device = "cuda"
    model = make_model(
        interctc_ctc_type="ortho", with_aux_ctc=True, with_interctc=True, device=device
    )
    model.train()
    speech, sl, text, tl = make_phone_batch(device=device)
    asr_text, asr_text_length = make_aux_text(device=device)
    out = model(speech, sl, text, tl, asr_text_tokens=asr_text, asr_text_length=asr_text_length)
    assert torch.isfinite(out["loss"])
    out["loss"].backward()
    assert model.ctc_aux.ctc_lo.weight.grad is not None


# ---------------------------------------------------------------------------
# __main__ runner (mirrors test_xeuspr_joint.py style)
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    tests = [
        # 1. TextTokenizer
        ("CharTokenizer: basic tokenization", test_char_tokenizer_basic),
        ("CharTokenizer: space → <space>", test_char_tokenizer_space),
        ("CharTokenizer: unknown → <unk>", test_char_tokenizer_unk),
        ("build_text_tokenizer: char factory", test_build_text_tokenizer_char),
        ("build_text_tokenizer: bad type raises", test_build_text_tokenizer_bad_type),
        # 2. collate_fn
        ("collate_fn: pads asr_text_tokens", test_collate_fn_pads_asr_text_tokens),
        ("collate_fn: no tokens → key absent", test_collate_fn_no_asr_tokens_skipped),
        # 3. Phone path (backward compat)
        ("phone path: no ctc_aux by default", test_phone_path_no_ctc_aux_default),
        ("phone path: forward finite", test_phone_path_forward_finite),
        ("phone interctc: no asr tokens → phone targets", test_phone_interctc_path_no_asr_tokens),
        ("phone interctc: backward grads", test_phone_interctc_backward),
        # 4. Conditioning layer size
        ("ortho: conditioning layer uses aux vocab size", test_ortho_conditioning_layer_size),
        ("phone: conditioning layer uses phone vocab size", test_phone_conditioning_layer_size),
        # 5. Ortho interctc
        ("ortho interctc: finite loss", test_ortho_interctc_loss_finite),
        ("ortho interctc: uses ctc_aux head", test_ortho_interctc_uses_aux_ctc_head),
        ("ortho interctc: backward grads through ctc_aux", test_ortho_backward_grads_through_ctc_aux),
        # 6. Missing asr tokens
        ("ortho: no interctc when asr_tokens missing", test_ortho_no_interctc_loss_when_missing_asr_tokens),
        # 7. Param groups
        ("param groups: ctc_aux in head", test_ctc_aux_in_head_param_group),
        ("param groups: phone ctc still in head", test_phone_ctc_still_in_head_with_aux),
        # 8. encode() routing
        ("encode: uses ctc_aux in ortho mode", test_encode_uses_aux_ctc_in_ortho_mode),
        ("encode: uses self.ctc in phone mode", test_encode_uses_phone_ctc_in_phone_mode),
    ]

    if _xeus_config_exists:
        tests += [
            ("builder: ctc_aux built from config", test_builder_with_ctc_aux_config),
            ("builder: no ctc_aux by default", test_builder_default_no_ctc_aux),
        ]
    else:
        print(f"\n[SKIP] Builder integration tests: {XEUS_CONFIG} not found\n")

    if torch.cuda.is_available():
        tests += [("GPU: ortho forward+backward", test_ortho_forward_backward_gpu)]
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
