from pathlib import Path
from typing import Dict, Iterable, List, Union

import sentencepiece as spm
from typeguard import typechecked

from espnet2.text.abs_tokenizer import AbsTokenizer


class SentencepiecesTokenizer(AbsTokenizer):
    @typechecked
    def __init__(self, model: Union[Path, str], encode_kwargs: Dict = dict()):
        self.model = str(model)
        # NOTE(kamo):
        # Don't build SentencePieceProcessor in __init__()
        # because it's not picklable and it may cause following error,
        # "TypeError: can't pickle SwigPyObject objects",
        # when giving it as argument of "multiprocessing.Process()".
        self.sp = None
        self.encode_kwargs = encode_kwargs

    def __repr__(self):
        return f'{self.__class__.__name__}(model="{self.model}")'

    def _build_sentence_piece_processor(self):
        # Build SentencePieceProcessor lazily.
        if self.sp is None:
            self.sp = spm.SentencePieceProcessor()
            self.sp.load(self.model)

    def text2tokens(self, line: str) -> List[str]:
        self._build_sentence_piece_processor()
        return self.sp.EncodeAsPieces(line, **self.encode_kwargs)

    def tokens2text(self, tokens: Iterable[str]) -> str:
        self._build_sentence_piece_processor()
        return self.sp.DecodePieces(list(tokens))

    def ids2tokens(self, ids: Iterable[int]) -> List[str]:
        self._build_sentence_piece_processor()
        return self.sp.IdToPiece(list(ids))

    def tokens2ids(self, tokens: Iterable[str]) -> List[int]:
        self._build_sentence_piece_processor()
        return self.sp.PieceToId(list(tokens))

    def ids2text(self, ids: Iterable[int]) -> str:
        self._build_sentence_piece_processor()
        return self.sp.DecodeIds(list(ids))

if __name__ == "__main__":
    # python -m src.model.sentencepieces_tokenizer
    tokenizer = SentencepiecesTokenizer(model='src/model/zipa/resources/unigram_127.model')
    print(tokenizer)
    text = "This is a test."
    tokens = tokenizer.text2tokens(text)
    print(tokens)
    ids = tokenizer.tokens2ids(tokens)
    print(ids)
    print(tokenizer.ids2tokens(ids))
    print(tokenizer.ids2text(ids))
    print('===')
    print('All tokens')
    vocab={}
    for i in range(tokenizer.sp.GetPieceSize()):
        piece = tokenizer.sp.IdToPiece(i)
        vocab[piece] = i
    print(vocab)