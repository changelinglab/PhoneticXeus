"""Tests for IPATokenizer."""

import pytest
from src.core.ipa_utils import IPATokenizer, IPA_TO_ARPABET


def test_ipatokenizer_initialization():
    """Test IPATokenizer initialization."""
    tokenizer = IPATokenizer()
    
    assert tokenizer.blank_id == 0
    assert tokenizer.blank_token == "<blank>"
    assert tokenizer.unk_token == "<unk>"
    assert tokenizer.unk_id == len(IPA_TO_ARPABET) + 1  # blank + phones + unk
    assert "<blank>" in tokenizer.phone2id
    assert "<unk>" in tokenizer.phone2id


def test_tokens2ids():
    """Test converting IPA tokens to IDs."""
    tokenizer = IPATokenizer()
    
    # Test known tokens
    tokens = ["ɑ", "æ", "b"]
    ids = tokenizer.tokens2ids(tokens)
    assert len(ids) == 3
    assert all(isinstance(id_val, int) for id_val in ids)
    assert ids[0] == tokenizer.phone2id["ɑ"]
    assert ids[1] == tokenizer.phone2id["æ"]
    assert ids[2] == tokenizer.phone2id["b"]
    
    # Test unknown token
    tokens_with_unk = ["ɑ", "unknown_phone", "b"]
    ids_with_unk = tokenizer.tokens2ids(tokens_with_unk)
    assert ids_with_unk[1] == tokenizer.unk_id


def test_ids2tokens():
    """Test converting IDs to IPA tokens."""
    tokenizer = IPATokenizer()
    
    # Test known IDs
    ids = [tokenizer.phone2id["ɑ"], tokenizer.phone2id["æ"], tokenizer.phone2id["b"]]
    tokens = tokenizer.ids2tokens(ids)
    assert tokens == ["ɑ", "æ", "b"]
    
    # Test unknown ID
    unknown_id = 99999
    tokens_with_unk = tokenizer.ids2tokens([tokenizer.phone2id["ɑ"], unknown_id])
    assert tokens_with_unk[0] == "ɑ"
    assert tokens_with_unk[1] == "<unk>"


def test_vocab_size():
    """Test vocab_size static method."""
    vocab_size = IPATokenizer.vocab_size()
    assert vocab_size == len(IPA_TO_ARPABET) + 2  # +2 for blank and unk
    assert vocab_size > 0


def test_round_trip_conversion():
    """Test that tokens -> ids -> tokens preserves known tokens."""
    tokenizer = IPATokenizer()
    
    test_tokens = ["ɑ", "æ", "b", "d", "ɛ"]
    ids = tokenizer.tokens2ids(test_tokens)
    recovered_tokens = tokenizer.ids2tokens(ids)
    assert recovered_tokens == test_tokens


def test_blank_and_unk_tokens():
    """Test that blank and unk tokens are handled correctly."""
    tokenizer = IPATokenizer()
    
    # Test blank token
    blank_id = tokenizer.tokens2ids(["<blank>"])[0]
    assert blank_id == tokenizer.blank_id
    
    # Test unk token
    unk_id = tokenizer.tokens2ids(["<unk>"])[0]
    assert unk_id == tokenizer.unk_id
    
    # Test that blank and unk are in vocab
    assert "<blank>" in tokenizer.phone2id
    assert "<unk>" in tokenizer.phone2id

