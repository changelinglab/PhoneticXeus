"""PhoneticXeus model wrapper for `AutoModel.from_pretrained(..., trust_remote_code=True)`.

The recipe code (ESPnet-style building blocks + the XeusPR model) is vendored in
this repo under ``pxeus/`` and wired with absolute ``pxeus.`` imports. The HF dynamic
module loader only auto-fetches files reachable via *relative* imports, so it will
not pull the ``pxeus`` package on its own. To keep that recipe code unmodified, this
module bootstraps it at load time: it locates the repo snapshot, puts it on
``sys.path``, and imports the builder via ``importlib`` (a string import, so the HF
``check_imports`` scan never sees a bare ``import pxeus`` and never demands a pip
package named ``pxeus``).

Weights are loaded the idiomatic way: ``__init__`` builds the architecture with
random weights and HF's ``from_pretrained`` fills them from ``model.safetensors``
(keys prefixed with ``model.``).
"""

import importlib
import os
import sys

import torch
from huggingface_hub import snapshot_download
from transformers import PreTrainedModel
from transformers.modeling_outputs import ModelOutput

from .configuration_phoneticxeus import PhoneticXeusConfig


def _resolve_repo_dir(name_or_path: str) -> str:
    """Return a local directory containing the vendored ``pxeus/`` tree."""
    if name_or_path and os.path.isdir(name_or_path):
        return name_or_path
    # Fetch only the code + resources; weights come through from_pretrained.
    return snapshot_download(
        name_or_path or "changelinglab/PhoneticXeus",
        allow_patterns=["pxeus/**"],
    )


class PhoneticXeusModel(PreTrainedModel):
    """Self-conditioned CTC phone recognizer on the XEUS encoder."""

    config_class = PhoneticXeusConfig

    def __init__(self, config: PhoneticXeusConfig):
        super().__init__(config)
        repo_dir = _resolve_repo_dir(getattr(config, "_name_or_path", "") or "")
        if repo_dir not in sys.path:
            sys.path.insert(0, repo_dir)

        builders = importlib.import_module("pxeus.model.xeusphoneme.builders")
        self._inference_cls = importlib.import_module(
            "pxeus.model.xeusphoneme.xeuspr_inference"
        ).XeusPRInference

        # Build architecture with random weights; from_pretrained loads the real
        # ones into ``self.model`` afterwards (state_dict keys -> "model.<...>").
        self.model = builders.build_xeus_pr_from_hf(
            work_dir=repo_dir,
            hf_repo=None,
            config_file=os.path.join(repo_dir, config.config_yaml),
            vocab_file=os.path.join(repo_dir, config.vocab_file),
            load_ckpt=False,
            interctc_layer_idx=config.interctc_layer_idx,
            interctc_use_conditioning=config.interctc_use_conditioning,
        )
        self._inference = None

    def _encode_logits(self, speech, speech_lengths):
        encoder_out, _ = self.model.encode(speech, speech_lengths)
        if isinstance(encoder_out, tuple):
            encoder_out = encoder_out[0]
        return self.model.ctc.ctc_lo(encoder_out)

    def forward(self, input_values, attention_mask=None, **kwargs):
        """Return frame-level CTC logits of shape ``(batch, frames, vocab)``."""
        if input_values.dim() == 1:
            input_values = input_values.unsqueeze(0)
        if attention_mask is not None:
            lengths = attention_mask.sum(-1).long()
        else:
            lengths = torch.full(
                (input_values.size(0),),
                input_values.size(1),
                dtype=torch.long,
                device=input_values.device,
            )
        logits = self._encode_logits(input_values.to(self.dtype), lengths)
        return ModelOutput(logits=logits)

    @torch.no_grad()
    def transcribe(self, speech, sampling_rate: int = 16000):
        """Greedy-CTC transcription.

        Args:
            speech: waveform tensor/array, shape ``(samples,)`` or ``(batch, samples)``.
            sampling_rate: must be 16000 (the model's expected rate).

        Returns:
            List of dicts with ``processed_transcript`` and ``predicted_transcript``.
        """
        if sampling_rate != self.config.sampling_rate:
            raise ValueError(
                f"Expected {self.config.sampling_rate} Hz audio, got {sampling_rate}. "
                "Resample before calling transcribe()."
            )
        if self._inference is None:
            self._inference = self._inference_cls(
                self.model, device=str(self.device), dtype="float32"
            )
        return self._inference(speech)
