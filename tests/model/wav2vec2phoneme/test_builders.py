from src.model.wav2vec2phoneme import builders as w2v2ph_builders


def test_build_tokenizer_uses_default_repo(monkeypatch):
    created = {}

    class DummyTokenizer:
        def __init__(self, hf_repo):
            created["hf_repo"] = hf_repo

    monkeypatch.setattr(
        w2v2ph_builders, "Wav2Vec2PhonemeTokenizer", DummyTokenizer
    )

    tokenizer = w2v2ph_builders.build_wav2vec2phoneme_tokenizer()

    assert isinstance(tokenizer, DummyTokenizer)
    assert (
        created["hf_repo"]
        == "ctaguchi/wav2vec2-large-xlsr-japlmthufielta-ipa1000-ns"
    )


def test_build_tokenizer_passes_custom_repo(monkeypatch):
    created = {}

    class DummyTokenizer:
        def __init__(self, hf_repo):
            created["hf_repo"] = hf_repo

    monkeypatch.setattr(
        w2v2ph_builders, "Wav2Vec2PhonemeTokenizer", DummyTokenizer
    )

    custom_repo = "custom/tokenizer"
    tokenizer = w2v2ph_builders.build_wav2vec2phoneme_tokenizer(
        hf_repo=custom_repo
    )

    assert isinstance(tokenizer, DummyTokenizer)
    assert created["hf_repo"] == custom_repo


def test_build_model_uses_default_repo(monkeypatch):
    created = {}

    class DummyModel:
        def __init__(self, hf_repo):
            created["hf_repo"] = hf_repo
            self.vocab_size = 128

    monkeypatch.setattr(
        w2v2ph_builders, "Wav2Vec2PhonemeModel", DummyModel
    )

    model = w2v2ph_builders.build_wav2vec2phoneme_model()

    assert isinstance(model, DummyModel)
    assert (
        created["hf_repo"]
        == "ctaguchi/wav2vec2-large-xlsr-japlmthufielta-ipa1000-ns"
    )


def test_build_model_passes_custom_repo(monkeypatch):
    created = {}

    class DummyModel:
        def __init__(self, hf_repo):
            created["hf_repo"] = hf_repo
            self.vocab_size = 256

    monkeypatch.setattr(
        w2v2ph_builders, "Wav2Vec2PhonemeModel", DummyModel
    )

    custom_repo = "custom/model"
    model = w2v2ph_builders.build_wav2vec2phoneme_model(hf_repo=custom_repo)

    assert isinstance(model, DummyModel)
    assert created["hf_repo"] == custom_repo
