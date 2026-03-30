"""Wav2Vec2Phoneme model implementation using Hugging Face Transformers.
Functionalities:
1. TODO(shikhar): Model fine-tuning with ctc loss
2. Encoder output extraction via encode() method.
3. TODO(shikhar): CTC-Decoding via decode() method.
4. TODO(shikhar): Forced alignment via forced_align() method.

This file supports the following pretrained models:
- facebook/wav2vec2-lv-60-espeak-cv-ft
- facebook/wav2vec2-xlsr-53-espeak-cv-ft
- ctaguchi/wav2vec2-large-xlsr-japlmthufielta-ipa1000-ns

"facebook" models use phonemizer which needs espeak-ng
Build espeak-ng following https://github.com/espeak-ng/espeak-ng/blob/master/docs/building.md
Then export the following paths:
export PHONEMIZER_ESPEAK_LIBRARY="${PHONEMIZER_ESPEAK_LIBRARY}"
export ESPEAK_DATA_PATH="${ESPEAK_DATA_PATH}"
Here the prefix is the path passed to ./configure --prefix=/usr during build.
Both of these exports are necessary.
"""

from typing import List
from transformers import Wav2Vec2Processor


class Wav2Vec2PhonemeTokenizer:
    """Wav2Vec2Phoneme tokenizer using Hugging Face Transformers.
    Wrapper to have consistency with espnet.
    """

    def __init__(self, hf_repo: str):
        """
        Args:
            hf_repo: one of the following pretrained models
                facebook/wav2vec2-lv-60-espeak-cv-ft
                facebook/wav2vec2-xlsr-53-espeak-cv-ft
                ctaguchi/wav2vec2-large-xlsr-japlmthufielta-ipa1000-ns
        """
        super().__init__()
        self.tokenizer = Wav2Vec2Processor.from_pretrained(hf_repo).tokenizer
        self.unk_symbol = self.tokenizer.unk_token
        self.pad_symbol = self.tokenizer.pad_token

    def tokens2ids(self, tokens: List[str]) -> List[int]:
        """Convert list of tokens to list of IDs."""
        return self.tokenizer.convert_tokens_to_ids(tokens)

    def ids2tokens(self, ids: List[int]) -> List[str]:
        """Convert list of IDs to list of tokens."""
        return self.tokenizer.convert_ids_to_tokens(ids)


if __name__ == "__main__":
    # Example usage
    tokenizer = Wav2Vec2PhonemeTokenizer(
        "ctaguchi/wav2vec2-large-xlsr-japlmthufielta-ipa1000-ns"
    )
    sample_tokens = ["a", "b", "k", "tS", "sil"]
    token_ids = tokenizer.tokens2ids(sample_tokens)
    print("Tokens:", sample_tokens)
    print("Token IDs:", token_ids)
    recovered_tokens = tokenizer.ids2tokens(token_ids)
    print("Recovered Tokens:", recovered_tokens)
