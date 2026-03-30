from pathlib import Path

from typing import Dict, List, Optional, Tuple, Union, Any
import torch
import torchaudio
from typeguard import typechecked

from espnet2.legacy.nets.e2e_asr_common import ErrorCalculator
from espnet2.legacy.nets.pytorch_backend.nets_utils import pad_list, th_accuracy
from espnet2.legacy.nets.pytorch_backend.transformer.label_smoothing_loss import (
    LabelSmoothingLoss,
)
from espnet2.asr.encoder.transformer_encoder import TransformerEncoder
from src.model.powsm.ctc import CTC
from src.model.powsm.utils import force_gatherable
from src.model.powsm.e_branchformer import EBranchformerEncoder
from src.model.powsm.transformer_decoder import TransformerDecoder
from src.model.powsm.builders_common import (
    load_token_list,
    load_config,
    build_frontend,
    build_specaug,
    build_normalize,
    build_ctc,
    resolve_model_paths,
    POWSM_REL_CONFIG,
    POWSM_REL_CKPT,
    POWSM_REL_STATS,
)
from src.utils import RankedLogger

log = RankedLogger(__name__, rank_zero_only=False)


class PowsmModel(torch.nn.Module):
    """CTC-attention hybrid Encoder-Decoder model"""

    @typechecked
    def __init__(
        self,
        vocab_size: int,
        token_list: Union[Tuple[str, ...], List[str]],
        frontend: Optional[torch.nn.Module],
        specaug: Optional[torch.nn.Module],
        normalize: Optional[torch.nn.Module],
        preencoder: Optional[torch.nn.Module],
        encoder: torch.nn.Module,
        postencoder: Optional[torch.nn.Module],
        decoder: Optional[torch.nn.Module],
        ctc: CTC,
        ctc_weight: float = 0.5,
        interctc_weight: float = 0.0,
        ignore_id: int = -1,
        lsm_weight: float = 0.0,
        length_normalized_loss: bool = False,
        report_cer: bool = True,
        report_wer: bool = True,
        sym_space: str = "<space>",
        sym_blank: str = "<blank>",
        sym_sos: str = "<sos>",
        sym_eos: str = "<eos>",
        sym_sop: str = "<sop>",
        sym_na: str = "<na>",
        extract_feats_in_collect_stats: bool = True,
    ):
        assert 0.0 <= ctc_weight <= 1.0, ctc_weight
        assert 0.0 <= interctc_weight < 1.0, interctc_weight

        super().__init__()

        self.blank_id = token_list.index(sym_blank)
        self.sos = token_list.index(sym_sos)
        self.eos = token_list.index(sym_eos)
        self.sop = token_list.index(sym_sop)
        self.na = token_list.index(sym_na)
        self.vocab_size = vocab_size
        self.ignore_id = ignore_id
        self.ctc_weight = ctc_weight
        self.interctc_weight = interctc_weight
        self.token_list = token_list.copy()

        self.frontend = frontend
        self.specaug = specaug
        self.normalize = normalize
        self.preencoder = preencoder  # Always None but kept for weight compatibility
        self.postencoder = postencoder  # Always None but kept for weight compatibility
        self.encoder = encoder

        if not hasattr(self.encoder, "interctc_use_conditioning"):
            self.encoder.interctc_use_conditioning = False
        if self.encoder.interctc_use_conditioning:
            self.encoder.conditioning_layer = torch.nn.Linear(
                vocab_size, self.encoder.output_size()
            )

        self.error_calculator = None

        if ctc_weight < 1.0:
            assert (
                decoder is not None
            ), "decoder should not be None when attention is used"
        else:
            decoder = None
            log.warning("Set decoder to none as ctc_weight==1.0")

        self.decoder = decoder

        self.criterion_att = LabelSmoothingLoss(
            size=vocab_size,
            padding_idx=ignore_id,
            smoothing=lsm_weight,
            normalize_length=length_normalized_loss,
        )

        if report_cer or report_wer:
            self.error_calculator = ErrorCalculator(
                token_list, sym_space, sym_blank, report_cer, report_wer
            )

        if ctc_weight == 0.0:
            self.ctc = None
        else:
            self.ctc = ctc

        self.extract_feats_in_collect_stats = extract_feats_in_collect_stats
        self.is_encoder_whisper = "Whisper" in type(self.encoder).__name__

        if self.is_encoder_whisper:
            assert (
                self.frontend is None
            ), "frontend should be None when using full Whisper model"

        # Fixed!
        self.sampling_rate = self.frontend.fs if self.frontend is not None else 16_000

    @torch.no_grad()
    def points_by_frames(self) -> int:
        """Get the ratio of input points to output frames."""
        return 640

    def forward(
        self,
        speech: torch.Tensor,
        speech_lengths: torch.Tensor,
        text: torch.Tensor,
        text_lengths: torch.Tensor,
        text_prev: torch.Tensor,
        text_prev_lengths: torch.Tensor,
        text_ctc: torch.Tensor,
        text_ctc_lengths: torch.Tensor,
        **kwargs,
    ) -> Dict[str, Any]:
        """Frontend + Encoder + Decoder + Calc loss"""
        assert text_lengths.dim() == 1, text_lengths.shape
        # Check that batch_size is unified
        assert (
            speech.shape[0]
            == speech_lengths.shape[0]
            == text.shape[0]
            == text_lengths.shape[0]
            == text_prev.shape[0]
            == text_prev_lengths.shape[0]
            == text_ctc.shape[0]
            == text_ctc_lengths.shape[0]
        ), (
            speech.shape,
            speech_lengths.shape,
            text.shape,
            text_lengths.shape,
            text_prev.shape,
            text_prev_lengths.shape,
            text_ctc.shape,
            text_ctc_lengths.shape,
        )
        batch_size = speech.shape[0]

        # -1 is used as padding index in collate fn
        text[text == -1] = self.ignore_id

        # for data-parallel
        text = text[:, : text_lengths.max()]

        # 1. Encoder
        encoder_out, encoder_out_lens = self.encode(speech, speech_lengths)
        intermediate_outs = None
        if isinstance(encoder_out, tuple):
            intermediate_outs = encoder_out[1]
            encoder_out = encoder_out[0]

        loss_att, acc_att, cer_att, wer_att = None, None, None, None
        loss_ctc, cer_ctc = None, None
        stats = dict()

        # 1. CTC branch
        if self.ctc_weight != 0.0:
            loss_ctc, cer_ctc = self._calc_ctc_loss(
                encoder_out, encoder_out_lens, text_ctc, text_ctc_lengths
            )

            # Collect CTC branch stats
            stats["loss_ctc"] = loss_ctc.detach() if loss_ctc is not None else None
            stats["cer_ctc"] = cer_ctc

        # Intermediate CTC (optional)
        loss_interctc = 0.0
        if self.interctc_weight != 0.0 and intermediate_outs is not None:
            for layer_idx, intermediate_out in intermediate_outs:
                # we assume intermediate_out has the same length & padding
                # as those of encoder_out
                loss_ic, cer_ic = self._calc_ctc_loss(
                    intermediate_out, encoder_out_lens, text_ctc, text_ctc_lengths
                )
                loss_interctc = loss_interctc + loss_ic

                # Collect Intermedaite CTC stats
                stats["loss_interctc_layer{}".format(layer_idx)] = (
                    loss_ic.detach() if loss_ic is not None else None
                )
                stats["cer_interctc_layer{}".format(layer_idx)] = cer_ic

            loss_interctc = loss_interctc / len(intermediate_outs)

            # calculate whole encoder loss
            loss_ctc = (
                1 - self.interctc_weight
            ) * loss_ctc + self.interctc_weight * loss_interctc

        # 2. Attention decoder branch
        if self.ctc_weight != 1.0:
            loss_att, acc_att, cer_att, wer_att = self._calc_att_loss(
                encoder_out,
                encoder_out_lens,
                text,
                text_lengths,
                text_prev,
                text_prev_lengths,
            )

        # 3. CTC-Att loss definition
        if self.ctc_weight == 0.0:
            loss = loss_att
        elif self.ctc_weight == 1.0:
            loss = loss_ctc
        else:
            loss = self.ctc_weight * loss_ctc + (1 - self.ctc_weight) * loss_att

        # Collect Attn branch stats
        stats["loss_att"] = loss_att.detach() if loss_att is not None else None
        stats["acc"] = acc_att
        stats["cer"] = cer_att
        stats["wer"] = wer_att

        # Collect total loss stats
        stats["loss"] = loss.detach()

        # force_gatherable: to-device and to-tensor if scalar for DataParallel
        loss, stats, weight = force_gatherable((loss, stats, batch_size), loss.device)
        model_output = {"loss": loss, "stats": stats, "weight": weight}
        return model_output

    def collect_feats(
        self,
        speech: torch.Tensor,
        speech_lengths: torch.Tensor,
        text: torch.Tensor,
        text_lengths: torch.Tensor,
        text_prev: torch.Tensor,
        text_prev_lengths: torch.Tensor,
        text_ctc: torch.Tensor,
        text_ctc_lengths: torch.Tensor,
        **kwargs,
    ) -> Dict[str, torch.Tensor]:
        feats, feats_lengths = self._extract_feats(speech, speech_lengths)
        return {"feats": feats, "feats_lengths": feats_lengths}

    def encode(
        self, speech: torch.Tensor, speech_lengths: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Frontend + Encoder. Note that this method is used by s2t_inference.py"""
        # with autocast(False):
        with torch.amp.autocast("cuda", enabled=False):
            # 1. Extract feats
            feats, feats_lengths = self._extract_feats(speech, speech_lengths)

            # 2. Data augmentation
            if self.specaug is not None and self.training:
                feats, feats_lengths = self.specaug(feats, feats_lengths)

            # 3. Normalization for feature: e.g. Global-CMVN, Utterance-CMVN
            if self.normalize is not None:
                feats, feats_lengths = self.normalize(feats, feats_lengths)

        # 4. Forward encoder
        # feats: (Batch, Length, Dim)
        # -> encoder_out: (Batch, Length2, Dim2)
        if self.encoder.interctc_use_conditioning:
            encoder_out, encoder_out_lens, _ = self.encoder(
                feats, feats_lengths, ctc=self.ctc
            )
        else:
            encoder_out, encoder_out_lens, _ = self.encoder(feats, feats_lengths)

        intermediate_outs = None
        if isinstance(encoder_out, tuple):
            intermediate_outs = encoder_out[1]
            encoder_out = encoder_out[0]

        assert encoder_out.size(0) == speech.size(0), (
            encoder_out.size(),
            speech.size(0),
        )
        if (
            getattr(self.encoder, "selfattention_layer_type", None) != "lf_selfattn"
            and not self.is_encoder_whisper
        ):
            assert encoder_out.size(-2) <= encoder_out_lens.max(), (
                encoder_out.size(),
                encoder_out_lens.max(),
            )

        if intermediate_outs is not None:
            return (encoder_out, intermediate_outs), encoder_out_lens

        return encoder_out, encoder_out_lens

    def _extract_feats(
        self, speech: torch.Tensor, speech_lengths: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        assert speech_lengths.dim() == 1, speech_lengths.shape

        # for data-parallel
        speech = speech[:, : speech_lengths.max()]

        if self.frontend is not None:
            # Frontend
            #  e.g. STFT and Feature extract
            #       data_loader may send time-domain signal in this case
            # speech (Batch, NSamples) -> feats: (Batch, NFrames, Dim)
            feats, feats_lengths = self.frontend(speech, speech_lengths)
        else:
            # No frontend and no feature extract
            feats, feats_lengths = speech, speech_lengths
        return feats, feats_lengths

    def _calc_att_loss(
        self,
        encoder_out: torch.Tensor,
        encoder_out_lens: torch.Tensor,
        ys_pad: torch.Tensor,
        ys_pad_lens: torch.Tensor,
        ys_prev_pad: torch.Tensor,
        ys_prev_lens: torch.Tensor,
    ):
        # 0. Prepare input and output with sos, eos, sop
        ys = [y[y != self.ignore_id] for y in ys_pad]
        ys_prev = [y[y != self.ignore_id] for y in ys_prev_pad]

        _sos = ys_pad.new([self.sos])
        _eos = ys_pad.new([self.eos])
        _sop = ys_pad.new([self.sop])
        ys_in = []
        ys_in_lens = []
        ys_out = []
        for y_prev, y in zip(ys_prev, ys):
            if self.na in y_prev:
                # Prev is not available in this case
                y_in = [_sos, y]
                y_in_len = len(y) + 1
                y_out = [y, _eos]
            else:
                y_in = [_sop, y_prev, _sos, y]
                y_in_len = len(y_prev) + len(y) + 2
                y_out = [self.ignore_id * ys_pad.new_ones(len(y_prev) + 1), y, _eos]

            ys_in.append(torch.cat(y_in))
            ys_in_lens.append(y_in_len)
            ys_out.append(torch.cat(y_out))

        ys_in_pad = pad_list(ys_in, self.eos)
        ys_in_lens = torch.tensor(ys_in_lens).to(ys_pad_lens)
        ys_out_pad = pad_list(ys_out, self.ignore_id)

        # 1. Forward decoder
        decoder_out, _ = self.decoder(
            encoder_out, encoder_out_lens, ys_in_pad, ys_in_lens
        )

        # 2. Compute attention loss
        loss_att = self.criterion_att(decoder_out, ys_out_pad)
        acc_att = th_accuracy(
            decoder_out.view(-1, self.vocab_size),
            ys_out_pad,
            ignore_label=self.ignore_id,
        )

        # Compute cer/wer using attention-decoder
        if self.training or self.error_calculator is None:
            cer_att, wer_att = None, None
        else:
            ys_hat = decoder_out.argmax(dim=-1)
            cer_att, wer_att = self.error_calculator(ys_hat.cpu(), ys_out_pad.cpu())

        return loss_att, acc_att, cer_att, wer_att

    def _calc_ctc_loss(
        self,
        encoder_out: torch.Tensor,
        encoder_out_lens: torch.Tensor,
        ys_pad: torch.Tensor,
        ys_pad_lens: torch.Tensor,
    ):
        # Filter out invalid samples where text is not available
        is_valid = [self.na not in y for y in ys_pad]
        if not any(is_valid):
            return torch.tensor(0.0), None

        encoder_out = encoder_out[is_valid]
        encoder_out_lens = encoder_out_lens[is_valid]
        ys_pad = ys_pad[is_valid]
        ys_pad_lens = ys_pad_lens[is_valid]

        # Calc CTC loss
        loss_ctc = self.ctc(encoder_out, encoder_out_lens, ys_pad, ys_pad_lens)

        # Calc CER using CTC
        cer_ctc = None
        if not self.training and self.error_calculator is not None:
            ys_hat = self.ctc.argmax(encoder_out).data
            cer_ctc = self.error_calculator(ys_hat.cpu(), ys_pad.cpu(), is_ctc=True)
        return loss_ctc, cer_ctc

    def encoder_output_size(self) -> int:
        """Get encoder output dimension"""
        return self.encoder.output_size()

    def ctc_logits(
        self, speech: torch.Tensor, speech_lengths: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Get CTC logits from encoder output

        Args:
            speech: (Batch, Length, ...)
            speech_lengths: (Batch,)
        Returns:
            Tuple[torch.Tensor, torch.Tensor]:
                - CTC logits: (Batch, Length, Vocab)
                - Encoder output lengths: (Batch,)
        """
        encoder_out, encoder_out_lens = self.encode(speech, speech_lengths)
        logits = self.ctc.ctc_lo(encoder_out)
        return logits, encoder_out_lens

    @torch.no_grad()
    def forced_align(self, speech, speech_lengths, text, text_lengths, utt_id=None):
        """Calculate frame-wise alignment from CTC probabilities.
        Only works with batch size 1.

        Args:
            speech: (Batch, Length, ...)
            speech_lengths: (Batch,)
            text: (Batch, Length)
            text_lengths: (Batch,)
            utt_id: Optional[str], utterance identifier for logging
        Returns:
            Tuple(tensor, tensor):
                - Label for each time step in the alignment path computed
                using forced alignment.
                - Log probability scores of the labels for each time
                step.
        """
        assert (
            self.ctc is not None
        ), "CTC is not used in this model. Cannot compute forced alignment."
        assert text_lengths.dim() == 1, text_lengths.shape
        # Check that batch_size is unified
        assert (
            speech.shape[0]
            == speech_lengths.shape[0]
            == text.shape[0]
            == text_lengths.shape[0]
        ), (
            speech.shape,
            speech_lengths.shape,
            text.shape,
            text_lengths.shape,
        )
        batch_size = speech.shape[0]
        assert batch_size == 1, "Forced alignment needs batch size 1."

        # -1 is used as padding index in collate fn
        text = torch.where(text == -1, self.ignore_id, text)
        text = text[:, : text_lengths.max()]  # for data-parallel
        encoder_out, encoder_out_lens = self.encode(speech, speech_lengths)
        log_probs = self.ctc.log_softmax(encoder_out)  # (B, Tmax, odim)
        assert log_probs.size(0) == 1, "Forced alignment needs batch size 1"
        assert not (text == self.blank_id).any(), "Target has blank tokens."
        if text_lengths.item() > encoder_out_lens.item():
            log.error(
                f"Target length {text_lengths.item()} is longer than "
                f"encoder output length {encoder_out_lens.item()}."
                f"Utterance id is :{utt_id}"
            )
        align_label, align_prob = torchaudio.functional.forced_align(
            log_probs, text, encoder_out_lens, text_lengths, blank=self.blank_id
        )
        return align_label, align_prob

    def get_blank_id(self) -> int:
        """Get blank id for CTC"""
        return self.blank_id


def build_powsm_from_files(
    config_file: str,
    model_file: str,
    stats_file: str,
) -> PowsmModel:
    """Build PowsmModel from config, model, and stats files.

    Args:
        config_file: Path to YAML configuration file.
        model_file: Path to model checkpoint file.
        stats_file: Path to feature statistics file.

    Returns:
        PowsmModel instance with loaded weights.
    """
    args = load_config(config_file)

    # Load token list
    token_list = load_token_list(args.token_list)
    args.token_list = token_list
    vocab_size = len(token_list)
    log.info(f"Vocabulary size: {vocab_size}")

    # 1. frontend
    frontend, input_size = build_frontend(args)

    # 2. Data augmentation for spectrogram
    specaug = build_specaug(args)

    # 3. Normalization layer
    normalize = build_normalize(args, stats_file)

    # 4. Encoder
    if args.encoder == "e_branchformer":  # , "Only Branchformer is supported!"
        encoder = EBranchformerEncoder(input_size=input_size, **args.encoder_conf)
    elif args.encoder == "transformer":
        encoder = TransformerEncoder(input_size=input_size, **args.encoder_conf)
    encoder_output_size = encoder.output_size()

    # 5. Decoder
    assert args.decoder == "transformer", "Only Transformer decoder is supported!"
    decoder = TransformerDecoder(
        vocab_size=vocab_size,
        encoder_output_size=encoder_output_size,
        **args.decoder_conf,
    )

    # 6. CTC
    ctc = build_ctc(vocab_size, encoder_output_size, args.ctc_conf)

    # 7. Build model
    model = PowsmModel(
        vocab_size=vocab_size,
        frontend=frontend,
        specaug=specaug,
        normalize=normalize,
        preencoder=None,
        encoder=encoder,
        postencoder=None,
        decoder=decoder,
        ctc=ctc,
        token_list=token_list,
        **args.model_conf,
    )

    # 8. Load weights
    state_dict = torch.load(model_file, map_location="cpu", weights_only=False)
    load_info = model.load_state_dict(state_dict, strict=True)
    log.info(f"Model loaded: {model_file} with info: {load_info}")
    model.training_args = args
    return model


def build_powsm(
    *,
    work_dir: str,
    hf_repo: Optional[str] = "espnet/powsm",
    force: bool = False,
    config_file: Optional[str] = None,
    model_file: Optional[str] = None,
    stats_file: Optional[str] = None,
):
    """Build Powsm model from local files or huggingface repo.
    Args:
        work_dir: Directory to store downloaded files from hf repo.
        hf_repo: Huggingface repo name. If None, load from local files.
        force: Whether to force re-download from hf repo.
        config_file: Path to config file. If None, use default path in hf repo.
          Takes precedence over hf_repo.
        model_file: Path to model file. If None, use default path in hf repo.
          Takes precedence over hf_repo.
        stats_file: Path to stats file. If None, use default path in hf repo.
          Takes precedence over hf_repo.

    Returns:
        PowsmModel instance.
    """
    cfg, mdl, stats = resolve_model_paths(
        work_dir=work_dir,
        hf_repo=hf_repo,
        force_download=force,
        config_file=config_file,
        model_file=model_file,
        stats_file=stats_file,
        rel_config=POWSM_REL_CONFIG,
        rel_ckpt=POWSM_REL_CKPT,
        rel_stats=POWSM_REL_STATS,
    )

    return build_powsm_from_files(cfg, mdl, stats)
