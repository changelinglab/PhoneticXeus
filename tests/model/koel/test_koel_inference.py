from types import SimpleNamespace

import torch

from src.model.koel import builders as koel_builders
from src.model.koel.koel_inference import KoelInference


class DummyTokenizer:
    def __init__(self, id2token: dict[int, str]):
        self.pad_token_id = 0
        self.id2token = id2token
        self.all_special_tokens = ["<PAD>", "<UNK>", "<BOS>", "<EOS>", "|"]

    def convert_ids_to_tokens(self, token_id: int) -> str:
        return self.id2token.get(token_id, str(token_id))


class DummyProcessor:
    def __init__(self, tokenizer: DummyTokenizer):
        self.tokenizer = tokenizer

    def __call__(self, waveform, sampling_rate: int, return_tensors: str):
        assert sampling_rate == 16000
        assert return_tensors == "pt"
        values = torch.tensor(waveform, dtype=torch.float32).unsqueeze(0)
        return {
            "input_values": values,
            "attention_mask": torch.ones_like(values, dtype=torch.long),
        }


class DummyModel(torch.nn.Module):
    def __init__(self, pred_ids: list[int], id2label: dict[int, str]):
        super().__init__()
        self.config = SimpleNamespace(id2label=id2label, pad_token_id=0)
        vocab_size = max(id2label.keys()) + 1
        logits = torch.zeros(1, len(pred_ids), vocab_size, dtype=torch.float32)
        for t, idx in enumerate(pred_ids):
            logits[0, t, idx] = 10.0
        self.register_buffer("_logits", logits)

    def forward(self, **kwargs):
        return SimpleNamespace(logits=self._logits.to(kwargs["input_values"].device))


def test_koel_inference_ctc_collapse_and_token_filtering():
    pred_ids = [0, 1, 1, 0, 2, 2, 4, 5, 0, 3, 3, 0]
    id2label = {0: "<PAD>", 1: "a", 2: "b", 3: "<UNK>", 4: "|", 5: "c"}
    processor = DummyProcessor(DummyTokenizer(id2label))
    model = DummyModel(pred_ids=pred_ids, id2label=id2label)

    inference = KoelInference(model=model, processor=processor, device="cpu")
    output = inference(speech=torch.randn(400), sr=16000)

    assert output == [
        {
            "processed_transcript": "abc",
            "predicted_transcript": "a/b/|/c/<UNK>",
        }
    ]


def test_koel_inference_resamples_when_sr_mismatch(monkeypatch):
    called = {}

    def fake_resample(waveform: torch.Tensor, orig_freq: int, new_freq: int):
        called["orig_freq"] = orig_freq
        called["new_freq"] = new_freq
        return waveform

    monkeypatch.setattr(
        "src.model.koel.koel_inference.torchaudio.functional.resample",
        fake_resample,
    )

    id2label = {0: "<PAD>", 1: "a"}
    processor = DummyProcessor(DummyTokenizer(id2label))
    model = DummyModel(pred_ids=[0, 1], id2label=id2label)
    inference = KoelInference(model=model, processor=processor, device="cpu")

    inference(speech=torch.randn(120), sr=8000)

    assert called == {"orig_freq": 8000, "new_freq": 16000}


def test_build_koel_inference_uses_dependencies(monkeypatch):
    called = {}

    class DummyProcessorFactory:
        @staticmethod
        def from_pretrained(
            hf_repo: str,
            cache_dir: str | None = None,
            token: str | None = None,
        ):
            called["processor"] = {
                "hf_repo": hf_repo,
                "cache_dir": cache_dir,
                "token": token,
            }
            return "processor_obj"

    class DummyModelFactory:
        @staticmethod
        def from_pretrained(
            hf_repo: str,
            cache_dir: str | None = None,
            token: str | None = None,
        ):
            called["model"] = {"hf_repo": hf_repo, "cache_dir": cache_dir, "token": token}
            return "model_obj"

    class DummyInference:
        def __init__(
            self,
            model,
            processor,
            device,
            dtype,
            target_sampling_rate,
            ignored_tokens,
        ):
            called["inference"] = {
                "model": model,
                "processor": processor,
                "device": device,
                "dtype": dtype,
                "target_sampling_rate": target_sampling_rate,
                "ignored_tokens": ignored_tokens,
            }

    monkeypatch.setattr(koel_builders, "AutoProcessor", DummyProcessorFactory)
    monkeypatch.setattr(koel_builders, "AutoModelForCTC", DummyModelFactory)
    monkeypatch.setattr(koel_builders, "KoelInference", DummyInference)

    inference = koel_builders.build_koel_inference(
        hf_repo="KoelLabs/xlsr-english-01",
        device="cuda",
        dtype="float16",
        target_sampling_rate=22050,
        cache_dir="/tmp/hf_cache",
        token="hf_abc123",
        ignored_tokens=["<PAD>", "|"],
    )

    assert isinstance(inference, DummyInference)
    assert called["processor"] == {
        "hf_repo": "KoelLabs/xlsr-english-01",
        "cache_dir": "/tmp/hf_cache",
        "token": "hf_abc123",
    }
    assert called["model"] == {
        "hf_repo": "KoelLabs/xlsr-english-01",
        "cache_dir": "/tmp/hf_cache",
        "token": "hf_abc123",
    }
    assert called["inference"] == {
        "model": "model_obj",
        "processor": "processor_obj",
        "device": "cuda",
        "dtype": "float16",
        "target_sampling_rate": 22050,
        "ignored_tokens": ["<PAD>", "|"],
    }
