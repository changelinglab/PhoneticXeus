from src.model.wav2vec2phoneme.tokenizer import Wav2Vec2PhonemeTokenizer
from src.model.wav2vec2phoneme.wav2vec2phoneme_model import Wav2Vec2PhonemeModel
from src.model.wav2vec2phoneme.wav2vec2phoneme_inference import Wav2Vec2PhonemeInference
from src.utils import RankedLogger

log = RankedLogger(__name__, rank_zero_only=True)


def build_wav2vec2phoneme_tokenizer(
    hf_repo: str = "ctaguchi/wav2vec2-large-xlsr-japlmthufielta-ipa1000-ns",
):
    """Build Wav2Vec2Phoneme tokenizer

    Args:
        hf_repo: HuggingFace repository ID

    Returns:
        Wav2Vec2Phoneme tokenizer
    """
    tokenizer = Wav2Vec2PhonemeTokenizer(hf_repo=hf_repo)
    log.info(f"Wav2Vec2Phoneme tokenizer loaded from {hf_repo}")
    return tokenizer


def build_wav2vec2phoneme_model(
    hf_repo: str = "ctaguchi/wav2vec2-large-xlsr-japlmthufielta-ipa1000-ns",
):
    """Build Wav2Vec2Phoneme model

    Args:
        hf_repo: HuggingFace repository ID

    Returns:
        Wav2Vec2Phoneme model
    """
    model = Wav2Vec2PhonemeModel(hf_repo=hf_repo)
    log.info(f"Wav2Vec2Phoneme model loaded from {hf_repo}")
    log.info(f"Model vocab size: {model.vocab_size}")
    return model


def build_wav2vec2phoneme_inference(
    hf_repo: str = "ctaguchi/wav2vec2-large-xlsr-japlmthufielta-ipa1000-ns",
    device: str = "cpu",
):
    """Build Wav2Vec2Phoneme inference module

    Returns:
        Wav2Vec2Phoneme inference module
    """
    model = build_wav2vec2phoneme_model(hf_repo=hf_repo)
    tokenizer = build_wav2vec2phoneme_tokenizer(hf_repo=hf_repo)
    inference_module = Wav2Vec2PhonemeInference(model, tokenizer, device=device)
    log.info("Wav2Vec2Phoneme inference module built")
    return inference_module
