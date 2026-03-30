# https://en.wikipedia.org/wiki/ARPABET

from typing import List
import panphon
import panphon.distance

IPA_TO_ARPABET = {
    "aʊ": "AW",
    "aɪ": "AY",
    "eɪ": "EY",
    "oʊ": "OW",
    "ɔɪ": "OY",  # until this points not included in powsm vocab
    "t͡ʃ": "CH",
    "d͡ʒ": "JH",
    "ɑ": "AA",
    "æ": "AE",
    "ʌ": "AH",  # unstressed “uh”; schwa is AX
    "ɔ": "AO",
    "ə˞": "AXR",  # r-colored schwa (alternate)
    "ɜ˞": "ER",   # stressed r-colored vowel (alternate)
    "b": "B",
    "d": "D",
    "ð": "DH",
    "ɛ": "EH",
    "f": "F",
    "ɡ": "G",
    "h": "HH",
    "ɪ": "IH",
    "i": "IY",
    "k": "K",
    "l": "L",
    "m": "M",
    "n": "N",
    "ŋ": "NG",
    "p": "P",
    "ɹ": "R",
    "s": "S",
    "ʃ": "SH",
    "t": "T",
    "θ": "TH",
    "ʊ": "UH",
    "u": "UW",
    "v": "V",
    "w": "W",
    "j": "Y",
    "z": "Z",
    "ʒ": "ZH",
    "ə": "AX",
    "ɨ": "IX",
    "l̩": "EL",  # syllabic consonants
    "m̩": "EM",
    "n̩": "EN",
    "ɾ̃": "NX",
    "ɾ": "DX",
    "ʔ": "Q",
    "ʉ": "UX",  # d<u>de
    # -------------------------------
    # TIMIT-SPECIFIC EXTRA SYMBOLS
    # -------------------------------
    "ə̥": "AX-H",  # ax-h (voiceless schwa)
    # Closure symbols (no IPA exact equivalent, best approximation)
    "b̚": "BCL",
    "d̚": "DCL",
    "ɡ̚": "GCL",
    "k̚": "KCL",
    "p̚": "PCL",
    "t̚": "TCL",
    "ŋ̍": "ENG",
    "ʔ̞": "EPI",  #  Glottal or epenthetic event
    "ɦ": "HV",  # voiced /h/
    "ʍ": "WH",
}
ARPABET_TO_IPA = {v.lower(): k for k, v in IPA_TO_ARPABET.items()}
panphon_segmenter = panphon.distance.Distance().fm.ipa_segs


def arpabet_to_ipa(phones: List[str]) -> List[str]:
    """Map a sequence of ARPABET symbols to IPA. Unknown symbols use lowercase as fallback (same as timit/buckeye)."""
    mapped_ipa = [ARPABET_TO_IPA.get(p.lower(), p.lower()) for p in phones]
    single_ipa = [p for phone in mapped_ipa for p in panphon_segmenter(phone)]
    return single_ipa


class IPATokenizer:
    """Tokenizer mapping IPA phones to indices."""

    def __init__(self):
        self.blank_id = 0
        self.blank_token = "<blank>"
        self.unk_token = "<unk>"
        VOCAB = [self.blank_token] + sorted(IPA_TO_ARPABET.keys()) + [self.unk_token]
        self.phone2id = {phone: idx for idx, phone in enumerate(VOCAB)}
        self.id2phone = {idx: phone for phone, idx in self.phone2id.items()}
        self.unk_id = self.phone2id[self.unk_token]

    def tokens2ids(self, tokens):
        return [self.phone2id.get(token, self.unk_id) for token in tokens]

    def ids2tokens(self, ids):
        return [self.id2phone.get(idx, self.unk_token) for idx in ids]

    @staticmethod
    def vocab_size():
        return len(IPA_TO_ARPABET) + 2
