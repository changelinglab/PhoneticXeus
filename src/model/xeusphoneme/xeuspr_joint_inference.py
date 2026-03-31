"""Autoregressive joint CTC+Attention inference for XeusPRModel.

Usage:
    python -m src.model.xeusphoneme.xeuspr_joint_inference
"""

from typing import Any, Dict, List, Optional, Union

import numpy as np
import torch

from src.espnet_import.minimal_ctcattn_beamsearch import (
    BatchBeamSearch,
    CTCPrefixScorer,
    LengthBonus,
)

from src.utils import RankedLogger
from src.model.xeusphoneme.builders import build_xeus_pr_from_hf

log = RankedLogger(__name__, rank_zero_only=True)


class XeusPRJointInference:
    """Joint CTC+Attention beam-search inference for XeusPRModel.

    Requires the model to have been trained with a TransformerDecoder
    (i.e. ``model.decoder is not None``).
    """

    def __init__(
        self,
        model: torch.nn.Module,
        device: str = "cpu",
        dtype: str = "float32",
        beam_size: int = 5,
        ctc_weight: float = 0.3,
        penalty: float = 0.0,
        nbest: int = 1,
        maxlenratio: float = 0.0,
        minlenratio: float = 0.0,
        normalize_length: bool = False,
    ):
        assert model.decoder is not None, (
            "XeusPRJointInference requires a model with a TransformerDecoder. "
            "Ensure the checkpoint was trained with joint CTC+Attention."
        )

        self.torch_dtype = getattr(torch, dtype)
        self.device = device
        self.model = model.to(device=device, dtype=self.torch_dtype).eval()
        self.nbest = nbest
        self.maxlenratio = maxlenratio
        self.minlenratio = minlenratio

        ctc_scorer = CTCPrefixScorer(ctc=model.ctc, eos=model.eos)
        scorers = {
            "decoder": model.decoder,
            "ctc": ctc_scorer,
            "length_bonus": LengthBonus(len(model.token_list)),
        }
        weights = {
            "decoder": 1.0 - ctc_weight,
            "ctc": ctc_weight,
            "length_bonus": penalty,
        }

        self.beam_search = (
            BatchBeamSearch(
                beam_size=beam_size,
                weights=weights,
                scorers=scorers,
                sos=model.sos,
                eos=model.eos,
                vocab_size=len(model.token_list),
                token_list=model.token_list,
                pre_beam_score_key=None if ctc_weight == 1.0 else "full",
                normalize_length=normalize_length,
            )
            .to(device=device, dtype=self.torch_dtype)
            .eval()
        )

        log.info(
            f"XeusPRJointInference: beam_size={beam_size}, ctc_weight={ctc_weight}"
        )
        log.info(f"Decoding device={device}, dtype={dtype}")

    @torch.no_grad()
    def __call__(
        self, speech: Union[torch.Tensor, np.ndarray], **kwargs
    ) -> List[Dict[str, Any]]:
        """Perform joint CTC+Attention beam search on a single utterance.

        Args:
            speech: Waveform of shape ``(nsamples,)`` or ``(1, nsamples)``.

        Returns:
            List of result dicts (length == nbest), each with keys
            ``predicted_transcript`` and ``processed_transcript``.
        """
        if isinstance(speech, np.ndarray):
            speech = torch.from_numpy(speech)

        if speech.dim() == 1:
            speech = speech.unsqueeze(0)  # (1, nsamples)

        speech = speech.to(device=self.device, dtype=self.torch_dtype)
        speech_lengths = torch.full(
            (speech.size(0),), speech.size(1), device=self.device, dtype=torch.long
        )

        enc, enc_olens = self.model.encode(speech, speech_lengths)

        # Strip intermediate outputs when interctc is active
        if isinstance(enc, tuple):
            enc = enc[0]

        # Trim to actual length for single-utterance beam search
        enc_single = enc[0, : enc_olens[0]]  # (T, D)

        nbest_hyps = self.beam_search(
            x=enc_single,
            maxlenratio=self.maxlenratio,
            minlenratio=self.minlenratio,
        )
        nbest_hyps = nbest_hyps[: self.nbest]

        results = []
        for hyp in nbest_hyps:
            token_int = (
                hyp.yseq.tolist() if not isinstance(hyp.yseq, list) else hyp.yseq
            )

            # Remove everything up to and including SOS
            if self.model.sos in token_int:
                token_int = token_int[token_int.index(self.model.sos) + 1 :]

            # Remove EOS and everything after it
            if self.model.eos in token_int:
                token_int = token_int[: token_int.index(self.model.eos)]

            # Remove blank tokens
            token_int = [t for t in token_int if t != self.model.blank_id]

            tokens = [self.model.token_list[i] for i in token_int]

            predicted_transcript = "/".join(tokens)
            processed_transcript = "".join(
                t for t in tokens if not (t.startswith("<") and t.endswith(">"))
            )

            results.append(
                {
                    "predicted_transcript": predicted_transcript,
                    "processed_transcript": processed_transcript,
                }
            )

        return results


def build_xeus_pr_joint_inference(
    work_dir: str,
    checkpoint: str,
    vocab_file: str,
    device: str = "cpu",
    hf_repo: Optional[str] = None,
    force_download: bool = False,
    config_file: Optional[str] = None,
    dtype: str = "float32",
    # net config — all optional, auto-loaded from checkpoint if not provided
    ctc_config: Optional[dict] = None,
    weighted_sum: Optional[bool] = None,
    interctc_layer_idx: Optional[list] = None,
    interctc_weight: Optional[float] = None,
    interctc_use_conditioning: Optional[bool] = None,
    decoder_config: Optional[dict] = None,
    ctc_weight_model: Optional[float] = None,
    # beam-search params
    ctc_weight: float = 0.3,
    beam_size: int = 5,
    penalty: float = 0.0,
    nbest: int = 1,
    maxlenratio: float = 0.0,
    minlenratio: float = 0.0,
    normalize_length: bool = False,
) -> XeusPRJointInference:
    """Build XeusPRJointInference from a joint-trained checkpoint.

    Net architecture params (``decoder_config``, ``ctc_config``, ``interctc_*``,
    ``weighted_sum``) are optional: if not provided they are recovered automatically
    from the ``net_config`` key embedded in the Lightning checkpoint by
    ``PhoneRecognitionModel.on_save_checkpoint``.

    Args:
        work_dir: Directory containing (or to download) the Xeus base model.
        checkpoint: Path to the joint CTC+Attention Lightning checkpoint.
        vocab_file: Path to the IPA vocabulary JSON.
        device: Torch device string (``"cpu"``, ``"cuda"``, …).
        hf_repo: HuggingFace repo ID for the Xeus base model.
        force_download: Re-download the base model even if cached.
        config_file: Optional explicit path to the Xeus config YAML.
        dtype: Floating-point precision (``"float32"``, ``"float16"``, …).
        ctc_config: CTC head config. Auto-loaded from checkpoint if ``None``.
        weighted_sum: Use weighted layer sum. Auto-loaded from checkpoint if ``None``.
        interctc_layer_idx: InterCTC layer indices. Auto-loaded from checkpoint if ``None``.
        interctc_weight: InterCTC loss weight. Auto-loaded from checkpoint if ``None``.
        interctc_use_conditioning: InterCTC conditioning flag. Auto-loaded if ``None``.
        decoder_config: Decoder architecture config. Auto-loaded from checkpoint if ``None``.
        ctc_weight_model: Training-time CTC loss weight (stored in checkpoint).
            Auto-loaded from checkpoint if ``None``. Distinct from ``ctc_weight``.
        ctc_weight: CTC fraction used in beam-search scoring (0 = pure attention,
            1 = pure CTC).  Inference-time only.
        beam_size: Number of active beams.
        penalty: Length-bonus weight.
        nbest: Number of hypotheses to return.
        maxlenratio: Maximum output-length ratio (0 = no constraint).
        minlenratio: Minimum output-length ratio (0 = no constraint).
        normalize_length: Normalize hypothesis scores by sequence length.

    Returns:
        Ready-to-call :class:`XeusPRJointInference` object.
    """
    # Recover net config from checkpoint as fallbacks for any unspecified param.
    raw_ckpt = torch.load(checkpoint, map_location="cpu", weights_only=False)
    saved = raw_ckpt.get("net_config", {})
    if saved:
        log.info(f"Recovered net_config from checkpoint: {list(saved.keys())}")

    ctc_config = ctc_config if ctc_config is not None else saved.get("ctc_config")
    weighted_sum = (
        weighted_sum if weighted_sum is not None else saved.get("weighted_sum", False)
    )
    interctc_layer_idx = (
        interctc_layer_idx
        if interctc_layer_idx is not None
        else saved.get("interctc_layer_idx")
    )
    interctc_weight = (
        interctc_weight
        if interctc_weight is not None
        else saved.get("interctc_weight", 0.0)
    )
    interctc_use_conditioning = (
        interctc_use_conditioning
        if interctc_use_conditioning is not None
        else saved.get("interctc_use_conditioning", False)
    )
    decoder_config = (
        decoder_config if decoder_config is not None else saved.get("decoder_config")
    )
    ctc_weight_model = (
        ctc_weight_model
        if ctc_weight_model is not None
        else saved.get("ctc_weight", 1.0)
    )

    model = build_xeus_pr_from_hf(
        work_dir=work_dir,
        hf_repo=hf_repo,
        force=force_download,
        config_file=config_file,
        checkpoint=checkpoint,
        vocab_file=vocab_file,
        ctc_config=ctc_config,
        weighted_sum=weighted_sum,
        interctc_layer_idx=interctc_layer_idx,
        interctc_weight=interctc_weight,
        interctc_use_conditioning=interctc_use_conditioning,
        decoder_config=decoder_config,
        ctc_weight=ctc_weight_model,
    )

    return XeusPRJointInference(
        model=model,
        device=device,
        dtype=dtype,
        beam_size=beam_size,
        ctc_weight=ctc_weight,
        penalty=penalty,
        nbest=nbest,
        maxlenratio=maxlenratio,
        minlenratio=minlenratio,
        normalize_length=normalize_length,
    )


if __name__ == "__main__":
    ckpt_path = "path/to/checkpoints/last.ckpt"
    work_dir = "path/to/exp/cache/xeus"
    vocab_file = (
        "src/model/xeusphoneme/resources/ipa_vocab.json"
    )
    device = "cpu" if not torch.cuda.is_available() else "cuda:0"
    # decoder_config is auto-recovered from checkpoint's net_config
    inference_obj = build_xeus_pr_joint_inference(
        work_dir=work_dir,
        checkpoint=ckpt_path,
        vocab_file=vocab_file,
        hf_repo="espnet/xeus",
        device=device,
        beam_size=5,
        ctc_weight=0.3,
    )
    speech = torch.randn(16000)  # 1s dummy audio
    results = inference_obj(speech=speech)
    print(results)
