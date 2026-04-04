# Compatible with distributed inference api, uses greedy ctc inference strategy
# python -m src.model.xeusphoneme.xeuspr_inference
import torch
import numpy as np
import torch.nn.functional as F
from typing import Union, List, Dict, Any, Optional

from src.recipe.phone_recognition.greedy_ctc_strategy import GreedyCTCInference


class XeusPRInference:
    """Greedy inference for Xeus Phoneme Recognition model."""

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
    from src.model.xeusphoneme.builders import build_xeus_pr_inference

    # Example usage
    ckpt_path = "path/to/checkpoints/last.ckpt"
    work_dir = "path/to/exp/cache/xeus"
    vocab_file = "src/model/xeusphoneme/resources/ipa_vocab.json"
    device = "cpu" if not torch.cuda.is_available() else "cuda:0"
    inference_obj = build_xeus_pr_inference(
        work_dir=work_dir,
        checkpoint=ckpt_path,
        vocab_file=vocab_file,
        hf_repo="espnet/xeus",
        config_file=None,
        device=device,
        force_download=False,
    )
    import torchaudio

    speechpath = "path/to/test_audio.wav"
    speech = torchaudio.load(speechpath)[0].squeeze(0)
    # speech = speech[: 16000 * 40]  # 10 seconds of audio
    # dummy_speech = np.random.randn(16000 * 5).astype(np.float32)  # 5 seconds of audio
    results = inference_obj(speech=speech)
    print(results)
