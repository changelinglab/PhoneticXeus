"""Tokenizers for orthographic ASR text (auxiliary CTC supervision)."""

import json
from typing import List


class CharTokenizer:
    """Character-level tokenizer. Vocab: JSON {token: id} with <blank>, <unk>, <space>."""

    def __init__(self, vocab_file: str):
        with open(vocab_file) as f:
            self._vocab = json.load(f)
        self._unk_id = self._vocab.get("<unk>", 1)
        self._space_id = self._vocab.get("<space>", 2)

    def tokenize(self, text: str) -> List[int]:
        """Tokenize text to char IDs; spaces become the <space> token."""
        ids = []
        for ch in text:
            if ch == " ":
                ids.append(self._space_id)
            else:
                ids.append(self._vocab.get(ch, self._unk_id))
        return ids

    @property
    def vocab_size(self) -> int:
        return len(self._vocab)


class SentencePieceTokenizer:
    """SentencePiece tokenizer wrapper."""

    def __init__(self, model_file: str):
        import sentencepiece as spm

        self._sp = spm.SentencePieceProcessor()
        self._sp.load(model_file)

    def tokenize(self, text: str) -> List[int]:
        return self._sp.encode(text, out_type=int)

    @property
    def vocab_size(self) -> int:
        return self._sp.get_piece_size()


def build_text_tokenizer(vocab_file: str, tokenizer_type: str = "sentencepiece"):
    """Return a CharTokenizer or SentencePieceTokenizer.

    Args:
        vocab_file: Path to vocab JSON (char) or .model file (sentencepiece).
        tokenizer_type: 'char' or 'sentencepiece'.
    """
    if tokenizer_type == "char":
        return CharTokenizer(vocab_file)
    elif tokenizer_type == "sentencepiece":
        return SentencePieceTokenizer(vocab_file)
    else:
        raise ValueError(
            f"Unknown tokenizer_type: {tokenizer_type!r}. Use 'char' or 'sentencepiece'."
        )
