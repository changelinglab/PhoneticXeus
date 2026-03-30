# Compatible with distributed inference api, uses greedy ctc inference strategy
# python -m src.model.wav2vec2.wav2vec2_inference
import torch
import numpy as np
from typing import Union, List, Dict, Any

from src.recipe.phone_recognition.greedy_ctc_strategy import GreedyCTCInference


class Wav2Vec2Inference:
    """Greedy inference for Wav2vecPR."""

    def __init__(
        self,
        model: torch.nn.Module,
        device: str = "cpu",
        dtype: str = "float32",
    ):
        self.device = device
        self.dtype = getattr(torch, dtype)
        self.model = model.to(device=self.device, dtype=self.dtype).eval()

        self.token_list = model.token_list
        self.blank_id = model.get_blank_id()
        self.ignore_id = getattr(model, "ignore_id", -1)
        self.inference_strategy = GreedyCTCInference(
            token_list=self.token_list, blank_id=self.blank_id
        )

    @torch.no_grad()
    def __call__(
        self, speech: Union[torch.Tensor, np.ndarray], **kwargs
    ) -> List[Dict[str, Any]]:
        """
        Perform greedy inference.
        Args:
            speech: Input speech of shape (nsamples,) or (batch, nsamples)
        Returns:
            List of results matching Powsm API
        """
        # 1. Prepare Input
        if isinstance(speech, np.ndarray):
            speech = torch.from_numpy(speech)

        if speech.dim() == 1:
            speech = speech.unsqueeze(0)

        speech = speech.to(device=self.device, dtype=self.dtype)
        speech_lengths = torch.full(
            (speech.size(0),), speech.size(1), device=self.device, dtype=torch.long
        )
        results = self.inference_strategy(
            model=self.model,
            speech=speech,
            speech_lengths=speech_lengths,
            **kwargs,
        )
        return results


if __name__ == "__main__":
    # python -m src.model.wav2vec2.wav2vec2_inference
    from src.model.wav2vec2.builders import build_wav2vec2pr

    net = build_wav2vec2pr(
        hf_repo="facebook/mms-300m",
        vocab_file="src/model/xeusphoneme/resources/ipa_vocab.json",
    )
    inference_engine = Wav2Vec2Inference(model=net, device="cpu", dtype="float32")
    dummy_input = np.random.randn(16000 * 5).astype(np.float32)
    results = inference_engine(speech=dummy_input)
    print("Inference results: ", results)
