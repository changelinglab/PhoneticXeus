import inspect
from typing import List

from src.model.wav2vec2phoneme import tokenizer as w2v2ph_tokenizer


def test_tokens2ids_signature():
    sig = inspect.signature(w2v2ph_tokenizer.Wav2Vec2PhonemeTokenizer.tokens2ids)
    params = list(sig.parameters.values())

    assert len(params) == 2
    self_param, tokens_param = params

    assert self_param.name == "self"
    assert self_param.kind == inspect.Parameter.POSITIONAL_OR_KEYWORD

    assert tokens_param.name == "tokens"
    assert tokens_param.kind == inspect.Parameter.POSITIONAL_OR_KEYWORD
    assert tokens_param.annotation == List[str]
    assert tokens_param.default is inspect._empty

    assert sig.return_annotation == List[int]


def test_ids2tokens_signature():
    sig = inspect.signature(w2v2ph_tokenizer.Wav2Vec2PhonemeTokenizer.ids2tokens)
    params = list(sig.parameters.values())

    assert len(params) == 2
    self_param, ids_param = params

    assert self_param.name == "self"
    assert self_param.kind == inspect.Parameter.POSITIONAL_OR_KEYWORD

    assert ids_param.name == "ids"
    assert ids_param.kind == inspect.Parameter.POSITIONAL_OR_KEYWORD
    assert ids_param.annotation == List[int]
    assert ids_param.default is inspect._empty

    assert sig.return_annotation == List[str]
