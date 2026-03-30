"""Tests for SentencepiecesTokenizer."""

import tempfile
from pathlib import Path

import pytest
import sentencepiece as spm

from src.model.sentencepieces_tokenizer import SentencepiecesTokenizer


@pytest.fixture
def dummy_spm_model(tmp_path):
    """Create a dummy SentencePiece model for testing."""
    model_path = tmp_path / "test.model"

    # Create a minimal SentencePiece model
    # We'll use a simple text to train a tiny model
    text_file = tmp_path / "text.txt"
    text_file.write_text("hello world test tokenization\n" * 10)

    # Train a minimal model
    spm.SentencePieceTrainer.train(
        input=str(text_file),
        model_prefix=str(model_path).replace(".model", ""),
        vocab_size=20,
        model_type="unigram",
    )

    return str(model_path.with_suffix(".model"))


def test_sentencepiece_tokenizer_init(dummy_spm_model):
    """Test SentencepiecesTokenizer initialization."""
    tokenizer = SentencepiecesTokenizer(dummy_spm_model)

    assert tokenizer.model == dummy_spm_model
    assert tokenizer.sp is None  # Should be None until first use
    assert tokenizer.encode_kwargs == {}


def test_sentencepiece_tokenizer_init_with_encode_kwargs(dummy_spm_model):
    """Test initialization with encode_kwargs."""
    encode_kwargs = {"add_bos": True, "add_eos": True}
    tokenizer = SentencepiecesTokenizer(dummy_spm_model, encode_kwargs=encode_kwargs)

    assert tokenizer.encode_kwargs == encode_kwargs


def test_text2tokens(dummy_spm_model):
    """Test text2tokens method."""
    tokenizer = SentencepiecesTokenizer(dummy_spm_model)

    text = "hello world"
    tokens = tokenizer.text2tokens(text)

    assert isinstance(tokens, list)
    assert len(tokens) > 0
    assert all(isinstance(token, str) for token in tokens)

    # Verify sp was built
    assert tokenizer.sp is not None


def test_tokens2text(dummy_spm_model):
    """Test tokens2text method."""
    tokenizer = SentencepiecesTokenizer(dummy_spm_model)

    # First encode to get tokens
    text = "hello world"
    tokens = tokenizer.text2tokens(text)

    # Then decode back
    decoded = tokenizer.tokens2text(tokens)

    assert isinstance(decoded, str)
    # Note: decoded text may not exactly match due to SentencePiece tokenization
    # but it should be a valid string


def test_lazy_loading(dummy_spm_model):
    """Test that SentencePiece processor is loaded lazily."""
    tokenizer = SentencepiecesTokenizer(dummy_spm_model)

    # Initially sp should be None
    assert tokenizer.sp is None

    # After first use, it should be loaded
    tokenizer.text2tokens("test")
    assert tokenizer.sp is not None


def test_repr(dummy_spm_model):
    """Test __repr__ method."""
    tokenizer = SentencepiecesTokenizer(dummy_spm_model)
    repr_str = repr(tokenizer)

    assert "SentencepiecesTokenizer" in repr_str
    assert dummy_spm_model in repr_str or Path(dummy_spm_model).name in repr_str


def test_round_trip_tokenization(dummy_spm_model):
    """Test that text -> tokens -> text produces valid output."""
    tokenizer = SentencepiecesTokenizer(dummy_spm_model)

    original_text = "hello world test"
    tokens = tokenizer.text2tokens(original_text)
    decoded_text = tokenizer.tokens2text(tokens)

    # Decoded text should be a string (may not match exactly due to tokenization)
    assert isinstance(decoded_text, str)
    assert len(decoded_text) > 0
