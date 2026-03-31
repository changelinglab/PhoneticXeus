from abc import ABC, abstractmethod
from typing import Iterable, List

# from espnet2/text/abs_tokenizer.py


class AbsTokenizer(ABC):
    @abstractmethod
    def text2tokens(self, line: str) -> List[str]:
        raise NotImplementedError

    @abstractmethod
    def tokens2text(self, tokens: Iterable[str]) -> str:
        raise NotImplementedError
