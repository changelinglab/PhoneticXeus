#!/usr/bin/env python3
"""Xeus Phoneme Recognition Model.
# -*- coding: utf-8 -*-

# Copyright 2025 William Chen. Adapted from ESPnet.
#  Apache 2.0  (http://www.apache.org/licenses/LICENSE-2.0)

Usage:
    python -m src.model.xeusphoneme.xeuspr_model \
        --work_dir path/to/cache/xeus
"""
from typing import Any, Dict, Optional, Tuple, Union
import argparse

import torch
import torch.nn.functional as F
import torchaudio
from src.model.powsm.utils import force_gatherable
from src.espnet_import.nets_utils import make_pad_mask, pad_list, th_accuracy
from src.espnet_import.label_smoothing_loss import LabelSmoothingLoss

# from src.espnet_import.error_calculator import ErrorCalculator
from src.recipe.phone_recognition.error_calculator import ErrorCalculator

from src.model.powsm.ctc import CTC
from src.utils import RankedLogger

log = RankedLogger(__name__, rank_zero_only=False)


class XeusPRModel(torch.nn.Module):
    """Encoder-only CTC model for phone recognition using Xeus pretrained weights."""

    def __init__(
        self,
        encoder: Any,
        ctc: CTC,
        token_list: Union[Tuple, list],
        frontend: Optional[Any] = None,
        specaug: Optional[Any] = None,
        normalize: Optional[Any] = None,
        preencoder: Optional[Any] = None,
        ignore_id: int = -1,
        sym_blank: str = "<blank>",
        freeze_frontend: bool = True,
        weighted_sum: bool = False,
        interctc_weight: float = 0.0,
        interctc_use_conditioning: bool = False,
        interctc_ctc_type: str = "phone",
        ctc_aux: Optional[Any] = None,
        decoder: Optional[Any] = None,
        ctc_weight: float = 1.0,
        lsm_weight: float = 0.0,
        sym_sos: str = "<sos>",
        sym_eos: str = "<eos>",
        **kwargs,
    ):
        super().__init__()
        self.frontend = frontend
        self.specaug = specaug
        self.normalize = normalize
        self.preencoder = preencoder
        self.encoder = encoder
        self.ctc = ctc
        self.ctc_aux = ctc_aux
        self.interctc_ctc_type = interctc_ctc_type
        if interctc_use_conditioning:
            vocab_size_cond = (
                ctc_aux.ctc_lo.out_features
                if interctc_ctc_type == "ortho" and ctc_aux is not None
                else len(token_list)
            )
            self.encoder.conditioning_layer = torch.nn.Linear(
                vocab_size_cond, encoder.output_size()
            )
            self.encoder.interctc_use_conditioning = True
        self.token_list = list(token_list)
        self.ignore_id = ignore_id
        self.blank_id = token_list.index(sym_blank) if sym_blank in token_list else 0
        sym_space = kwargs.get("sym_space", "<space>")
        self.freeze_frontend = freeze_frontend
        # self.error_calculator = ErrorCalculator(
        #     token_list, sym_space, sym_blank, report_cer=True, report_wer=False
        # )
        self.error_calculator = ErrorCalculator(
            token_list,
            blank_id=self.blank_id,
            sym_space=sym_space,
            ignore_id=ignore_id,
            log_phone_metrics=True,
        )

        self.decoder = decoder
        self.ctc_weight = ctc_weight
        if decoder is not None:
            self.sos = token_list.index(sym_sos)
            self.eos = token_list.index(sym_eos)
            self.criterion_att = LabelSmoothingLoss(
                size=len(token_list),
                padding_idx=ignore_id,
                smoothing=lsm_weight,
                normalize_length=False,
            )

        self.weighted_sum = weighted_sum
        if self.weighted_sum:
            n_layers = encoder.num_blocks
            assert (
                n_layers is not None and n_layers > 0
            ), "Cannot infer number of encoder layers for weighted_sum"
            self.layer_weights = torch.nn.Parameter(torch.zeros(int(n_layers)))
        self.interctc_weight = interctc_weight
        self.sampling_rate = 16000

    def points_by_frames(self) -> int:
        """Samples per encoder frame (CNN downsampling factor)."""
        return self.frontend.downsampling_factor

    @torch.no_grad()
    def forced_align(self, speech, speech_lengths, text, text_lengths, utt_id=None):
        """CTC forced alignment via torchaudio.functional.forced_align (batch size 1)."""
        assert speech.shape[0] == 1, "forced_align requires batch size 1"
        text = text[:, : text_lengths.max()]
        logits, logit_lengths = self.ctc_logits(speech, speech_lengths)
        log_probs = F.log_softmax(logits, dim=-1)
        align_label, align_prob = torchaudio.functional.forced_align(
            log_probs, text, logit_lengths, text_lengths, blank=self.blank_id
        )
        return align_label, align_prob

    def collect_feats(
        self, speech: torch.Tensor, speech_lengths: torch.Tensor, **kwargs
    ) -> Dict[str, torch.Tensor]:
        """Extract features for stats collection."""
        feats, feats_lengths = self._extract_feats(speech, speech_lengths)
        return {"feats": feats, "feats_lengths": feats_lengths}

    def forward(self, speech, speech_lengths, text, text_lengths, **kwargs):
        encoder_out, encoder_out_lens = self.encode(speech, speech_lengths)

        intermediate_outs = None
        if isinstance(encoder_out, tuple):
            intermediate_outs = encoder_out[1]
            encoder_out = encoder_out[0]

        loss_ctc, stats = self._calc_ctc_loss(
            encoder_out, encoder_out_lens, text, text_lengths, **kwargs
        )

        if self.interctc_weight > 0.0 and intermediate_outs:
            if self.interctc_ctc_type == "ortho" and self.ctc_aux is not None:
                ctc_inter = self.ctc_aux
                ys_inter = kwargs.get("asr_text_tokens")
                ys_inter_lens = kwargs.get("asr_text_length")
            else:
                ctc_inter = self.ctc
                ys_inter = torch.where(text == -1, self.ignore_id, text)[
                    :, : text_lengths.max()
                ]
                ys_inter_lens = text_lengths

            if ys_inter is not None and ys_inter_lens is not None:
                loss_interctc = 0.0
                for layer_idx, intermediate_out in intermediate_outs:
                    loss_ic = ctc_inter(
                        intermediate_out,
                        encoder_out_lens,
                        ys_inter,
                        ys_inter_lens,
                    )
                    loss_interctc = loss_interctc + loss_ic
                    stats[f"loss_interctc_layer{layer_idx}"] = loss_ic.detach()
                loss_interctc = loss_interctc / len(intermediate_outs)
                loss_ctc = (
                    1 - self.interctc_weight
                ) * loss_ctc + self.interctc_weight * loss_interctc

        # Attention branch
        if self.ctc_weight < 1.0 and self.decoder is not None:
            loss_att, acc_att = self._calc_att_loss(
                encoder_out, encoder_out_lens, text, text_lengths
            )
            stats["loss_att"] = loss_att.detach()
            stats["acc_att"] = acc_att
            loss = self.ctc_weight * loss_ctc + (1 - self.ctc_weight) * loss_att
        else:
            loss = loss_ctc

        loss, stats, weight = force_gatherable(
            (loss, stats, speech.shape[0]), loss.device
        )
        return {"loss": loss, "stats": stats, "weight": weight}

    def _extract_feats(
        self, speech: torch.Tensor, speech_lengths: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Extract features using frontend."""
        speech = speech[:, : speech_lengths.max()]
        return (
            self.frontend(speech, speech_lengths)
            if self.frontend
            else (speech, speech_lengths)
        )

    def _apply_preprocessing(
        self, speech: torch.Tensor, speech_lengths: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Apply frontend, specaug, normalize, and preencoder."""
        speech, speech_lengths = self._extract_feats(speech, speech_lengths)

        if self.specaug and self.training:
            speech, speech_lengths = self.specaug(speech, speech_lengths)

        if self.normalize:
            speech, speech_lengths = self.normalize(speech, speech_lengths)

        if self.preencoder:
            speech, speech_lengths = self.preencoder(speech, speech_lengths)

        return speech, speech_lengths

    def encode(
        self, speech: torch.Tensor, speech_lengths: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Encode speech to frame-level representations.

        When weighted_sum=True, returns a weighted sum of all encoder layers.
        Otherwise, calls the encoder without return_all_hs; if interctc_layer_idx
        is configured on the encoder, returns (final_out, [(layer_idx, tensor), ...]).
        """
        speech, speech_lengths = self._apply_preprocessing(speech, speech_lengths)
        pad_masks = make_pad_mask(speech_lengths).to(speech.device)
        if self.weighted_sum:
            encoder_out, encoder_out_lens, _ = self.encoder(
                speech, speech_lengths, masks=pad_masks, return_all_hs=True
            )
            hs_list = encoder_out[1]
            assert len(hs_list) == self.layer_weights.numel()
            w = torch.softmax(self.layer_weights, dim=0).to(
                hs_list[0].device, hs_list[0].dtype
            )
            hs = torch.stack(hs_list, dim=0)  # (L, B, T, D)
            return (w.view(-1, 1, 1, 1) * hs).sum(0), encoder_out_lens
        else:
            ctc_for_encoder = (
                self.ctc_aux
                if self.interctc_ctc_type == "ortho" and self.ctc_aux is not None
                else self.ctc
            )
            encoder_out, encoder_out_lens, _ = self.encoder(
                speech, speech_lengths, masks=pad_masks, ctc=ctc_for_encoder
            )
            return encoder_out, encoder_out_lens

    def ctc_collapse_batch(self, x: torch.Tensor, max_length: int, pad: int = -1):
        B, T = x.shape
        blank = self.blank_id
        x_prev = torch.cat(
            [torch.full((B, 1), blank, device=x.device, dtype=x.dtype), x[:, :-1]],
            dim=1,
        )
        keep = (x != blank) & ((x_prev == blank) | (x != x_prev))
        pos = keep.long().cumsum(1) - 1
        lengths = keep.sum(1)
        out = torch.full((B, T), pad, device=x.device, dtype=x.dtype)
        # Compute batch indices and output positions for kept elements
        batch_idx = (
            torch.arange(B, device=x.device, dtype=torch.long).unsqueeze(1).expand_as(x)
        )
        output_pos = pos.clone()
        # Only use positions where keep is True
        batch_idx_keep = batch_idx[keep]
        output_pos_keep = output_pos[keep]
        # Flatten the output and set values at correct positions
        flat_out = out.view(-1)
        flat_idx = batch_idx_keep * T + output_pos_keep
        flat_out[flat_idx] = x[keep]
        out = flat_out.view(B, T)
        ##### Trim to max_length from ground truth lengths
        out = out[:, :max_length]
        lengths = torch.clamp(lengths, max=max_length)
        return out, lengths

    def _calc_att_loss(self, encoder_out, encoder_out_lens, ys_pad, ys_pad_lens):
        ys_pad = torch.where(ys_pad == -1, self.ignore_id, ys_pad)
        ys = [y[y != self.ignore_id][:l] for y, l in zip(ys_pad, ys_pad_lens)]
        _sos = ys_pad.new([self.sos])
        _eos = ys_pad.new([self.eos])
        ys_in = [torch.cat([_sos, y]) for y in ys]
        ys_out = [torch.cat([y, _eos]) for y in ys]
        ys_in_pad = pad_list(ys_in, self.eos)
        ys_out_pad = pad_list(ys_out, self.ignore_id)
        ys_in_lens = torch.tensor([len(y) for y in ys_in], device=ys_pad.device)

        decoder_out, _ = self.decoder(
            encoder_out, encoder_out_lens, ys_in_pad, ys_in_lens
        )
        loss_att = self.criterion_att(decoder_out, ys_out_pad)
        acc_att = th_accuracy(
            decoder_out.view(-1, len(self.token_list)),
            ys_out_pad,
            ignore_label=self.ignore_id,
        )
        return loss_att, acc_att

    def _calc_ctc_loss(
        self, encoder_out, encoder_out_lens, ys_pad, ys_pad_lens, **kwargs
    ):
        ys_pad = torch.where(ys_pad == -1, self.ignore_id, ys_pad)
        ys_pad = ys_pad[:, : ys_pad_lens.max()]
        loss_ctc = self.ctc(
            encoder_out,
            encoder_out_lens,
            ys_pad,
            ys_pad_lens,
            lang_sym=kwargs.get("lang_sym"),
            accent_sym=kwargs.get("accent_sym"),
        )
        stats = {}
        assert self.error_calculator is not None, "ErrorCalculator not initialized"
        if not self.training:  # err calc, slow?
            with torch.no_grad():
                ys_hat = self.ctc.argmax(encoder_out).data  # greedy-top1
                # stats["cer_ctc"] = self.error_calculator(
                #     ys_hat.cpu(), ys_pad.cpu(), is_ctc=True
                # )
                metrics = self.error_calculator(
                    ys_hat.cpu(), ys_pad.cpu(), ys_pad_lens.cpu()
                )
                for k, v in metrics.items():
                    stats[k + "_ctc"] = v
        return loss_ctc, stats

    def ctc_logits(
        self, speech: torch.Tensor, speech_lengths: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Get CTC logits for inference."""
        encoder_out, encoder_out_lens = self.encode(speech, speech_lengths)
        if isinstance(encoder_out, tuple):
            encoder_out = encoder_out[0]
        return self.ctc.ctc_lo(encoder_out), encoder_out_lens

    def encoder_output_size(self) -> int:
        return self.encoder.output_size()

    def get_blank_id(self) -> int:
        return self.blank_id

    def get_frontend(self):
        return self.frontend

    def get_trainable_parameters(self):
        trainable_params = {"head": [], "encoder": []}
        for n, p in self.named_parameters():
            if (
                n.startswith("ctc")
                or n.startswith("decoder")
                or n.startswith("criterion_att")
            ):
                trainable_params["head"].append(p)
            elif n.startswith("encoder"):
                trainable_params["encoder"].append(p)
            elif n.startswith("frontend"):
                if self.freeze_frontend:
                    p.requires_grad = False
                else:
                    trainable_params["encoder"].append(p)
            else:
                # freeze other parts:
                p.requires_grad = False
        return trainable_params


if __name__ == "__main__":
    # python -m src.model.xeusphoneme.xeuspr_model
    import argparse
    from src.model.xeusphoneme.builders import build_xeus_pr_from_hf

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--work_dir",
        type=str,
        default="exp/cache/xeus",
        help="Working directory for model files",
    )
    parser.add_argument(
        "--config",
        type=str,
        required=False,
        default=None,
        help="Path to config file (overrides work_dir download)",
    )
    parser.add_argument(
        "--checkpoint",
        type=str,
        required=False,
        default=None,
        help="Path to checkpoint file (overrides work_dir download)",
    )

    args = parser.parse_args()

    model = build_xeus_pr_from_hf(
        work_dir=args.work_dir,
        hf_repo="espnet/xeus",
        force=False,
        config_file=args.config,
        checkpoint=args.checkpoint,
        weighted_sum=True,
    )
    print(model)
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Total parameters: {total_params}")

    dummy_input = torch.randn(2, 16000)
    dummy_lengths = torch.tensor([16000, 16000])
    encoder_out, encoder_out_lens = model.encode(dummy_input, dummy_lengths)
    print(f"Encoder output shape: {encoder_out.shape}")
