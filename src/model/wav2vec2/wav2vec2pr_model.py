"""Wav2Vec2 Phone Recognition Model."""

from typing import List, Optional, Tuple, Union

import torch
from src.model.powsm.utils import force_gatherable
from src.recipe.phone_recognition.error_calculator import ErrorCalculator

from src.model.powsm.ctc import CTC
from src.model.wav2vec2.wav2vec2_model import Wav2Vec2Model


class Wav2Vec2PRModel(torch.nn.Module):
    """CTC model for phone recognition using Wav2Vec2 encoder."""

    def __init__(
        self,
        encoder: Wav2Vec2Model,
        ctc: CTC,
        token_list: Union[Tuple, list],
        ignore_id: int = -1,
        sym_blank: str = "<blank>",
        freeze_frontend: bool = True,
        interctc_weight: float = 0.0,
        interctc_layer_idx: Optional[List[int]] = None,
        interctc_use_conditioning: bool = False,
        **kwargs,
    ):
        super().__init__()
        self.encoder = encoder
        self.ctc = ctc
        self.token_list = list(token_list)
        self.ignore_id = ignore_id
        assert sym_blank in token_list, "Blank symbol must be in token list."
        self.blank_id = token_list.index(sym_blank)
        self.freeze_frontend = freeze_frontend
        self.error_calculator = ErrorCalculator(
            token_list,
            blank_id=self.blank_id,
            sym_space=kwargs.get("sym_space", "<space>"),
            ignore_id=ignore_id,
            log_phone_metrics=True,
        )
        self.interctc_weight = interctc_weight
        self.interctc_layer_idx = interctc_layer_idx or []
        self.conditioning_layer = None
        if self.interctc_layer_idx and interctc_use_conditioning:
            self.conditioning_layer = torch.nn.Linear(
                len(token_list), encoder.encoder_output_size()
            )

    def forward(self, speech, speech_lengths, text, text_lengths, **kwargs):
        encoder_out, encoder_out_lens = self.encode(speech, speech_lengths)

        intermediate_outs = None
        if isinstance(encoder_out, tuple):
            intermediate_outs = encoder_out[1]
            encoder_out = encoder_out[0]

        loss_ctc, stats = self._calc_ctc_loss(
            encoder_out, encoder_out_lens, text, text_lengths
        )

        if self.interctc_weight > 0.0 and intermediate_outs:
            loss_interctc = 0.0
            ys_pad_clean = torch.where(text == -1, self.ignore_id, text)[:, : text_lengths.max()]
            for layer_idx, intermediate_out in intermediate_outs:
                loss_ic = self.ctc(intermediate_out, encoder_out_lens, ys_pad_clean, text_lengths)
                loss_interctc = loss_interctc + loss_ic
                stats[f"loss_interctc_layer{layer_idx}"] = loss_ic.detach()
            loss_interctc = loss_interctc / len(intermediate_outs)
            loss_ctc = (
                (1 - self.interctc_weight) * loss_ctc + self.interctc_weight * loss_interctc
            )

        loss, stats, weight = force_gatherable(
            (loss_ctc, stats, speech.shape[0]), loss_ctc.device
        )
        return {"loss": loss, "stats": stats, "weight": weight}

    def encode(
        self, speech: torch.Tensor, speech_lengths: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        if self.interctc_layer_idx:
            return self.encoder.encode_with_interctc(
                speech,
                speech_lengths,
                interctc_layer_idx=self.interctc_layer_idx,
                ctc=self.ctc,
                conditioning_layer=self.conditioning_layer,
            )
        return self.encoder.encode(speech, speech_lengths)

    def _calc_ctc_loss(self, encoder_out, encoder_out_lens, ys_pad, ys_pad_lens):
        ys_pad = torch.where(ys_pad == -1, self.ignore_id, ys_pad)
        ys_pad = ys_pad[:, : ys_pad_lens.max()]
        loss_ctc = self.ctc(encoder_out, encoder_out_lens, ys_pad, ys_pad_lens)
        stats = {}
        if not self.training:
            with torch.no_grad():
                ys_hat = self.ctc.argmax(encoder_out).data
                metrics = self.error_calculator(
                    ys_hat.cpu(), ys_pad.cpu(), ys_pad_lens.cpu()
                )
                for k, v in metrics.items():
                    stats[k + "_ctc"] = v
        return loss_ctc, stats

    def ctc_logits(
        self, speech: torch.Tensor, speech_lengths: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        encoder_out, encoder_out_lens = self.encode(speech, speech_lengths)
        if isinstance(encoder_out, tuple):
            encoder_out = encoder_out[0]
        return self.ctc.ctc_lo(encoder_out), encoder_out_lens

    def encoder_output_size(self) -> int:
        return self.encoder.encoder_output_size()

    def get_blank_id(self) -> int:
        return self.blank_id

    def get_frontend(self):
        return self.encoder.model.wav2vec2.feature_extractor

    def get_trainable_parameters(self):
        trainable_params = {"head": [], "encoder": []}
        for n, p in self.named_parameters():
            if n.startswith("ctc"):
                trainable_params["head"].append(p)
            elif n.startswith("encoder.model.wav2vec2.encoder"):
                trainable_params["encoder"].append(p)
            elif n.startswith("encoder.model.wav2vec2.feature"):
                # feature_extractor and feature_projection
                if self.freeze_frontend:
                    p.requires_grad = False
                else:
                    trainable_params["encoder"].append(p)
            else:
                # freeze other parts:
                p.requires_grad = False
        return trainable_params
