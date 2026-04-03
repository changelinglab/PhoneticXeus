"""Powsm inference class.
Usage:
    python -m src.model.powsm.powsm_inference
"""

from pathlib import Path
from typing import List, Optional, Tuple, Union, Dict, Any

import numpy as np
import torch
import torch.nn.functional as F
from typeguard import typechecked

from src.espnet_import.minimal_ctcattn_beamsearch import (
    BatchBeamSearch,
    CTCPrefixScorer,
    Hypothesis,
    LengthBonus,
)
from src.espnet_import.scorer_interface import BatchScorerInterface

from src.utils import RankedLogger
from src.model.sentencepieces_tokenizer import SentencepiecesTokenizer
from src.model.powsm.token_id_converter import TokenIDConverter
from src.model.powsm.utils import to_device
from src.model.powsm.powsm_model import build_powsm

log = RankedLogger(__name__, rank_zero_only=True)


class ScoreFilter(BatchScorerInterface, torch.nn.Module):
    """Filter scores based on pre-defined rules.

    See comments in the score method.

    """

    def __init__(
        self,
        notimestamps: int,
        first_time: int,
        last_time: int,
        sos: int,
        eos: int,
        vocab_size: int,
    ):
        super().__init__()

        self.notimestamps = notimestamps
        self.first_time = first_time
        self.last_time = last_time
        self.sos = sos
        self.eos = eos
        self.vocab_size = vocab_size

        # dummy param used to obtain the current dtype and device
        self.param = torch.nn.Parameter(torch.tensor(0.0, dtype=torch.float32))

    def score(
        self, y: torch.Tensor, state: Any, x: torch.Tensor
    ) -> Tuple[torch.Tensor, Any]:
        """Score new token (required).

        Args:
            y (torch.Tensor): 1D torch.int64 prefix tokens.
            state: Scorer state for prefix tokens
            x (torch.Tensor): The encoder feature that generates ys.

        Returns:
            tuple[torch.Tensor, Any]: Tuple of
                scores for next token that has a shape of `(n_vocab)`
                and next state for ys

        """

        score = torch.zeros(
            self.vocab_size, dtype=self.param.dtype, device=self.param.device
        )
        if self.notimestamps in y:
            # Suppress timestamp tokens if we don't predict time
            score[self.first_time : self.last_time + 1] = -np.inf
        elif y[-3] == self.sos:
            # The first token must be a timestamp if we predict time
            score[: self.first_time] = -np.inf
            score[self.last_time + 1 :] = -np.inf
        else:
            prev_times = y[torch.logical_and(y >= self.first_time, y <= self.last_time)]
            if len(prev_times) % 2 == 1:
                # there are an odd number of timestamps, so the sentence is incomplete
                score[self.eos] = -np.inf
                # timestamps are monotonic
                score[self.first_time : prev_times[-1] + 1] = -np.inf
            else:
                # there are an even number of timestamps (all are paired)
                if y[-1] >= self.first_time and y[-1] <= self.last_time:
                    # the next tokon should be a timestamp or eos
                    score[: y[-1]] = -np.inf
                    score[self.last_time + 1 :] = -np.inf
                    score[self.eos] = 0.0
                else:
                    # this is an illegal hyp
                    score[:] = -np.inf

        return score, None

    def batch_score(
        self, ys: torch.Tensor, states: List[Any], xs: torch.Tensor
    ) -> Tuple[torch.Tensor, List[Any]]:
        """Score new token batch (required).

        Args:
            ys (torch.Tensor): torch.int64 prefix tokens (n_batch, ylen).
            states (List[Any]): Scorer states for prefix tokens.
            xs (torch.Tensor):
                The encoder feature that generates ys (n_batch, xlen, n_feat).

        Returns:
            tuple[torch.Tensor, List[Any]]: Tuple of
                batchfied scores for next token with shape of `(n_batch, n_vocab)`
                and next state list for ys.

        """

        scores = list()
        outstates = list()
        for i, (y, state, x) in enumerate(zip(ys, states, xs)):
            score, outstate = self.score(y, state, x)
            outstates.append(outstate)
            scores.append(score)
        scores = torch.cat(scores, 0).view(ys.shape[0], -1)
        return scores, outstates


class PowsmInference:
    """Powsm inference class for speech-to-text conversion."""

    @typechecked
    def __init__(
        self,
        model: torch.nn.Module,
        tokenizer: SentencepiecesTokenizer,
        device: str = "cpu",
        maxlenratio: float = 0.0,
        minlenratio: float = 0.0,
        dtype: str = "float32",
        beam_size: int = 5,
        ctc_weight: float = 0.0,
        penalty: float = 0.0,
        nbest: int = 1,
        normalize_length: bool = False,
        # default values that can be overwritten in __call__
        lang_sym: str = "<unk>",
        task_sym: str = "<pr>",
    ):

        model.to(dtype=getattr(torch, dtype), device=device).eval()
        decoder = model.decoder
        ctc = CTCPrefixScorer(ctc=model.ctc, eos=model.eos)
        token_list = model.token_list
        scorers = dict(
            decoder=decoder,
            ctc=ctc,
            length_bonus=LengthBonus(len(token_list)),
            scorefilter=ScoreFilter(
                notimestamps=token_list.index(
                    model.training_args.preprocessor_conf["notime_symbol"]
                ),
                first_time=token_list.index(
                    model.training_args.preprocessor_conf["first_time_symbol"]
                ),
                last_time=token_list.index(
                    model.training_args.preprocessor_conf["last_time_symbol"]
                ),
                sos=model.sos,
                eos=model.eos,
                vocab_size=len(token_list),
            ),
        )

        weights = dict(
            decoder=1.0 - ctc_weight,
            ctc=ctc_weight,
            length_bonus=penalty,
            scorefilter=1.0,
        )
        beam_search = BatchBeamSearch(
            beam_size=beam_size,
            weights=weights,
            scorers=scorers,
            sos=model.sos,
            eos=model.eos,
            vocab_size=len(token_list),
            token_list=token_list,
            pre_beam_score_key=None if ctc_weight == 1.0 else "full",
            normalize_length=normalize_length,
        )
        beam_search.to(device=device, dtype=getattr(torch, dtype)).eval()
        converter = TokenIDConverter(token_list=token_list)

        log.info(f"Beam_search: {beam_search}")
        log.info(f"Decoding device={device}, dtype={dtype}")
        log.info(f"Text tokenizer: {tokenizer}")

        self.model = model
        self.training_args = model.training_args
        self.preprocessor_conf = model.training_args.preprocessor_conf
        self.converter = converter
        self.tokenizer = tokenizer
        self.beam_search = beam_search
        self.maxlenratio = maxlenratio
        self.minlenratio = minlenratio
        self.device = device
        self.dtype = dtype
        self.nbest = nbest
        self.lang_sym = lang_sym
        self.task_sym = task_sym

    @torch.no_grad()
    @typechecked
    def __call__(
        self,
        speech: Union[torch.Tensor, np.ndarray],
        *args,
        text_prev: Optional[Union[torch.Tensor, np.ndarray, str, List]] = None,
        lang_sym: Optional[str] = None,
        task_sym: Optional[str] = None,
        predict_time: Optional[bool] = False,
        **kwargs,
    ) -> Any:
        """Perform inference on a SINGLE input utterance.

        Args:
            speech: Input speech of shape (nsamples,)

        Returns:
            List of (text, tokens, token_ids, score) tuples for n-best hypotheses
        """
        lang_sym = lang_sym if lang_sym is not None else self.lang_sym
        task_sym = task_sym if task_sym is not None else self.task_sym
        predict_time = predict_time if predict_time is not None else self.predict_time

        lang_id = self.converter.token2id.get(lang_sym, self.converter.unk_id)
        task_id = self.converter.token2id[task_sym]
        notime_id = self.converter.token2id[self.preprocessor_conf["notime_symbol"]]

        # Prepare hyp_primer
        hyp_primer = [self.model.sos, lang_id, task_id]
        if not predict_time:
            hyp_primer.append(notime_id)
        if text_prev is not None:
            text_prev = self.converter.tokens2ids(self.tokenizer.text2tokens(text_prev))
            if self.model.na in text_prev:
                text_prev = None
        if text_prev is not None:
            hyp_primer = [self.model.sop] + text_prev + hyp_primer
        self.beam_search.set_hyp_primer(hyp_primer)

        # Preapre speech
        if isinstance(speech, np.ndarray):
            speech = torch.tensor(speech)

        if speech.dim() > 1:
            raise ValueError("Only single utterance decoding is supported.")

        model_speech_length = int(
            self.preprocessor_conf["fs"] * self.preprocessor_conf["speech_length"]
        )
        speech = speech.to(getattr(torch, self.dtype))
        # Pad or trim speech to the fixed length
        if speech.size(-1) >= model_speech_length:
            speech = speech[:model_speech_length]
        else:
            speech = F.pad(speech, (0, model_speech_length - speech.size(-1)))
        speech = speech.unsqueeze(0)  # (1, nsamples)
        speech_length = speech.new_full(
            [1], dtype=torch.long, fill_value=speech.size(1)
        )
        batch = {"speech": speech, "speech_lengths": speech_length}
        batch = to_device(batch, device=self.device)

        enc, enc_olens = self.model.encode(**batch)
        intermediate_outs = None
        if isinstance(enc, tuple):
            enc, intermediate_outs = enc
        enc[0] = enc[0][: enc_olens[0]]

        results = self._decode_single_sample(enc[0])

        if intermediate_outs is not None:
            encoder_interctc_res = self._decode_interctc(intermediate_outs)
            results = (results, encoder_interctc_res)

        return results

    @typechecked
    def _decode_interctc(
        self, intermediate_outs: List[Tuple[int, torch.Tensor]]
    ) -> Dict[int, List[List[str]]]:

        exclude_ids = [self.model.blank_id, self.model.sos, self.model.eos]
        res = {}
        token_list = self.beam_search.token_list

        for layer_idx, encoder_out in intermediate_outs:
            y = self.model.ctc.argmax(encoder_out)  # (B, Tmax)
            y = [predid.tolist() for predid in y]
            y = [[token_list[x] for x in pred] for pred in y]
            res[layer_idx] = y

        return res

    def _decode_single_sample(self, enc: torch.Tensor):
        nbest_hyps = self.beam_search(
            x=enc, maxlenratio=self.maxlenratio, minlenratio=self.minlenratio
        )
        nbest_hyps = nbest_hyps[: self.nbest]

        results = []
        for hyp in nbest_hyps:
            assert isinstance(hyp, Hypothesis), type(hyp)

            # remove sos/eos and get results
            last_pos = -1
            start_pos = 0
            if isinstance(hyp.yseq, list):
                token_int = hyp.yseq[start_pos:last_pos]
            else:
                token_int = hyp.yseq[start_pos:last_pos].tolist()

            token_int = token_int[token_int.index(self.model.sos) + 1 :]

            # remove blank symbol id
            token_int = list(filter(lambda x: x != self.model.blank_id, token_int))

            # Change integer-ids to tokens
            token = self.converter.ids2tokens(token_int)

            # remove special tokens (task, timestamp, etc.)
            token_nospecial = [x for x in token if not (x[0] == "<" and x[-1] == ">")]

            text, text_nospecial = None, None
            if self.tokenizer is not None:
                text = self.tokenizer.tokens2text(token)
                text_nospecial = self.tokenizer.tokens2text(token_nospecial)

            results.append(
                {
                    "processed_transcript": text_nospecial.split(">")[-1].replace(
                        "/", ""
                    ),
                    "predicted_transcript": text_nospecial,
                    # "text": text,
                    # "token": token,
                    # "token_int": token_int,
                    # "hyp": hyp, # full hypothesis object with scores
                }
            )

        return results


def build_powsm_inference(
    work_dir: str = "./powsm_model",
    hf_repo: Optional[str] = "espnet/powsm",
    force_download: bool = False,
    config_file: Optional[str] = None,
    model_file: Optional[str] = None,
    bpemodel: Optional[str] = None,
    stats_file: Optional[str] = None,
    device: str = "cpu",
    dtype: str = "float32",
    beam_size: int = 5,
    ctc_weight: float = 0.3,
    penalty: float = 0.0,
    nbest: int = 1,
    normalize_length: bool = False,
    maxlenratio: float = 0.0,
    minlenratio: float = 0.0,
) -> PowsmInference:
    """Build PowsmInference from Hugging Face repo or local files.

    Args:
        work_dir: Directory to store downloaded files
        hf_repo: Hugging Face repository ID (e.g., "espnet/powsm")
        force_download: Force re-download from HF repo
        config_file: Optional path to config file (overrides HF repo)
        model_file: Optional path to model checkpoint (overrides HF repo)
        bpemodel: Optional path to BPE model (overrides HF repo)
        stats_file: Optional path to stats file (overrides HF repo)
        device: Device to run on ("cpu" or "cuda")
        dtype: Data type ("float32", "float16", etc.)
        beam_size: Beam size for decoding
        ctc_weight: CTC weight (0.0 to 1.0)
        penalty: Length penalty
        nbest: Number of hypotheses to return
        normalize_length: Whether to normalize scores by length
        maxlenratio: Maximum length ratio
        minlenratio: Minimum length ratio

    Returns:
        PowsmInference object that wraps powsm model and can be called for decoding
    """

    model = build_powsm(
        work_dir=work_dir,
        hf_repo=hf_repo,
        force=force_download,
        config_file=config_file,
        model_file=model_file,
        stats_file=stats_file,
    )

    # Create tokenizer
    root = Path(work_dir)
    REL_BPE = "data/token_list/bpe_unigram40000/bpe.model"
    bpe = bpemodel or str(root / REL_BPE)
    tokenizer = SentencepiecesTokenizer(bpe)

    # Create inference object
    inference = PowsmInference(
        model=model,
        tokenizer=tokenizer,
        device=device,
        dtype=dtype,
        beam_size=beam_size,
        ctc_weight=ctc_weight,
        penalty=penalty,
        nbest=nbest,
        normalize_length=normalize_length,
        maxlenratio=maxlenratio,
        minlenratio=minlenratio,
    )

    return inference
