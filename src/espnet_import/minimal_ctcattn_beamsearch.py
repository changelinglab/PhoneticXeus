"""Merged beam search + CTC scorer module.

Sources:
  end_detect          <- espnet/nets/e2e_asr_common.py
  CTCPrefixScore      <- espnet/nets/ctc_prefix_score.py
  CTCPrefixScoreTH    <- espnet/nets/ctc_prefix_score.py
  Hypothesis          <- espnet/nets/beam_search.py
  BeamSearch          <- espnet/nets/beam_search.py
  BatchHypothesis     <- espnet/nets/batch_beam_search.py
  BatchBeamSearch     <- espnet/nets/batch_beam_search.py
  CTCPrefixScorer     <- espnet/nets/scorers/ctc.py
  LengthBonus         <- espnet/nets/scorers/length_bonus.py
"""

import logging
from itertools import chain
from typing import Any, Dict, List, NamedTuple, Tuple, Union

import numpy as np
import torch
from packaging.version import parse as V
from torch.nn.utils.rnn import pad_sequence

from src.espnet_import.scorer_interface import (
    BatchPartialScorerInterface,
    BatchScorerInterface,
    PartialScorerInterface,
    ScorerInterface,
)

logger = logging.getLogger(__name__)
is_torch_1_9_plus = V(torch.__version__) >= V("1.9.0")


# -- espnet/nets/e2e_asr_common.py --

def end_detect(ended_hyps, i, M=3, D_end=np.log(1 * np.exp(-10))):
    """End detection (Eq. 50, Watanabe et al. Hybrid CTC/Attention)."""
    if len(ended_hyps) == 0:
        return False
    count = 0
    best_hyp = sorted(ended_hyps, key=lambda x: x["score"], reverse=True)[0]
    for m in range(M):
        hyp_length = i - m
        hyps_same_length = [x for x in ended_hyps if len(x["yseq"]) == hyp_length]
        if len(hyps_same_length) > 0:
            best_hyp_same_length = sorted(
                hyps_same_length, key=lambda x: x["score"], reverse=True
            )[0]
            if best_hyp_same_length["score"] - best_hyp["score"] < D_end:
                count += 1
    return count == M


# -- espnet/nets/ctc_prefix_score.py --

class CTCPrefixScoreTH(object):
    """Batch CTC prefix scorer (PyTorch); Algorithm 2 in Watanabe et al."""

    def __init__(self, x, xlens, blank, eos, margin=0):
        self.logzero = -10000000000.0
        self.blank = blank
        self.eos = eos
        self.batch = x.size(0)
        self.input_length = x.size(1)
        self.odim = x.size(2)
        self.dtype = x.dtype
        if x.is_cuda:
            self.device = torch.device("cuda:%d" % x.get_device())
        elif x.device.type == "mps":
            self.device = x.device
        else:
            self.device = torch.device("cpu")
        for i, l in enumerate(xlens):
            if l < self.input_length:
                x[i, l:, :] = self.logzero
                x[i, l:, blank] = 0
        xn = x.transpose(0, 1)  # (B, T, O) -> (T, B, O)
        xb = xn[:, :, self.blank].unsqueeze(2).expand(-1, -1, self.odim)
        self.x = torch.stack([xn, xb])  # (2, T, B, O)
        self.end_frames = torch.as_tensor(xlens, device=self.device) - 1
        self.margin = margin
        if margin > 0:
            self.frame_ids = torch.arange(
                self.input_length, dtype=self.dtype, device=self.device
            )
        self.idx_bh = None
        self.idx_b = torch.arange(self.batch, device=self.device)
        self.idx_bo = (self.idx_b * self.odim).unsqueeze(1)

    def __call__(self, y, state, scoring_ids=None, att_w=None):
        """Compute CTC prefix scores for next labels."""
        output_length = len(y[0]) - 1
        last_ids = [yi[-1] for yi in y]
        n_bh = len(last_ids)
        n_hyps = n_bh // self.batch
        self.scoring_num = scoring_ids.size(-1) if scoring_ids is not None else 0
        if state is None:
            r_prev = torch.full(
                (self.input_length, 2, self.batch, n_hyps),
                self.logzero,
                dtype=self.dtype,
                device=self.device,
            )
            r_prev[:, 1] = torch.cumsum(
                self.x[0, :, :, self.blank], 0
            ).unsqueeze(2)
            r_prev = r_prev.view(-1, 2, n_bh)
            s_prev = 0.0
            f_min_prev = 0
            f_max_prev = 1
        else:
            r_prev, s_prev, f_min_prev, f_max_prev = state

        if self.scoring_num > 0:
            scoring_idmap = torch.full(
                (n_bh, self.odim), -1, dtype=torch.long, device=self.device
            )
            snum = self.scoring_num
            if self.idx_bh is None or n_bh > len(self.idx_bh):
                self.idx_bh = torch.arange(n_bh, device=self.device).view(-1, 1)
            scoring_idmap[self.idx_bh[:n_bh], scoring_ids] = torch.arange(
                snum, device=self.device
            )
            scoring_idx = (
                scoring_ids + self.idx_bo.repeat(1, n_hyps).view(-1, 1)
            ).view(-1)
            x_ = torch.index_select(
                self.x.view(2, -1, self.batch * self.odim), 2, scoring_idx
            ).view(2, -1, n_bh, snum)
        else:
            scoring_ids = None
            scoring_idmap = None
            snum = self.odim
            x_ = (
                self.x.unsqueeze(3).repeat(1, 1, 1, n_hyps, 1).view(2, -1, n_bh, snum)
            )

        r = torch.full(
            (self.input_length, 2, n_bh, snum),
            self.logzero,
            dtype=self.dtype,
            device=self.device,
        )
        if output_length == 0:
            r[0, 0] = x_[0, 0]

        r_sum = torch.logsumexp(r_prev, 1)
        log_phi = r_sum.unsqueeze(2).repeat(1, 1, snum)
        if scoring_ids is not None:
            for idx in range(n_bh):
                pos = scoring_idmap[idx, last_ids[idx]]
                if pos >= 0:
                    log_phi[:, idx, pos] = r_prev[:, 1, idx]
        else:
            for idx in range(n_bh):
                log_phi[:, idx, last_ids[idx]] = r_prev[:, 1, idx]

        if att_w is not None and self.margin > 0:
            f_arg = torch.matmul(att_w, self.frame_ids)
            f_min = max(int(f_arg.min().cpu()), f_min_prev)
            f_max = max(int(f_arg.max().cpu()), f_max_prev)
            start = min(
                f_max_prev, max(f_min - self.margin, output_length, 1)
            )
            end = min(f_max + self.margin, self.input_length)
        else:
            f_min = f_max = 0
            start = max(output_length, 1)
            end = self.input_length

        for t in range(start, end):
            rp = r[t - 1]
            rr = torch.stack([rp[0], log_phi[t - 1], rp[0], rp[1]]).view(
                2, 2, n_bh, snum
            )
            r[t] = torch.logsumexp(rr, 1) + x_[:, t]

        log_phi_x = (
            torch.cat((log_phi[0].unsqueeze(0), log_phi[:-1]), dim=0) + x_[0]
        )
        if scoring_ids is not None:
            log_psi = torch.full(
                (n_bh, self.odim), self.logzero, dtype=self.dtype, device=self.device
            )
            log_psi_ = torch.logsumexp(
                torch.cat(
                    (log_phi_x[start:end], r[start - 1, 0].unsqueeze(0)), dim=0
                ),
                dim=0,
            )
            for si in range(n_bh):
                log_psi[si, scoring_ids[si]] = log_psi_[si]
        else:
            log_psi = torch.logsumexp(
                torch.cat(
                    (log_phi_x[start:end], r[start - 1, 0].unsqueeze(0)), dim=0
                ),
                dim=0,
            )

        for si in range(n_bh):
            log_psi[si, self.eos] = r_sum[self.end_frames[si // n_hyps], si]

        if self.eos != self.blank:
            log_psi[:, self.blank] = self.logzero

        return (log_psi - s_prev), (r, log_psi, f_min, f_max, scoring_idmap)

    def index_select_state(self, state, best_ids):
        """Select CTC states according to best ids."""
        r, s, f_min, f_max, scoring_idmap = state
        n_bh = len(s)
        n_hyps = n_bh // self.batch
        vidx = (
            best_ids + (self.idx_b * (n_hyps * self.odim)).view(-1, 1)
        ).view(-1)
        s_new = torch.index_select(s.view(-1), 0, vidx)
        s_new = s_new.view(-1, 1).repeat(1, self.odim).view(n_bh, self.odim)
        if scoring_idmap is not None:
            snum = self.scoring_num
            hyp_idx = (
                best_ids // self.odim + (self.idx_b * n_hyps).view(-1, 1)
            ).view(-1)
            label_ids = torch.fmod(best_ids, self.odim).view(-1)
            score_idx = scoring_idmap[hyp_idx, label_ids]
            score_idx[score_idx == -1] = 0
            vidx = score_idx + hyp_idx * snum
        else:
            snum = self.odim
        r_new = torch.index_select(
            r.view(-1, 2, n_bh * snum), 2, vidx
        ).view(-1, 2, n_bh)
        return r_new, s_new, f_min, f_max

    def extend_prob(self, x):
        """Extend CTC prob for streaming decoding."""
        if self.x.shape[1] < x.shape[1]:
            xlens = [x.size(1)]
            for i, l in enumerate(xlens):
                if l < self.input_length:
                    x[i, l:, :] = self.logzero
                    x[i, l:, self.blank] = 0
            tmp_x = self.x
            xn = x.transpose(0, 1)
            xb = xn[:, :, self.blank].unsqueeze(2).expand(-1, -1, self.odim)
            self.x = torch.stack([xn, xb])
            self.x[:, : tmp_x.shape[1], :, :] = tmp_x
            self.input_length = x.size(1)
            self.end_frames = torch.as_tensor(xlens) - 1

    def extend_state(self, state):
        """Extend state for streaming decoding."""
        if state is None:
            return state
        r_prev, s_prev, f_min_prev, f_max_prev = state
        r_prev_new = torch.full(
            (self.input_length, 2),
            self.logzero,
            dtype=self.dtype,
            device=self.device,
        )
        start = max(r_prev.shape[0], 1)
        r_prev_new[0:start] = r_prev
        for t in range(start, self.input_length):
            r_prev_new[t, 1] = (
                r_prev_new[t - 1, 1] + self.x[0, t, :, self.blank]
            )
        return (r_prev_new, s_prev, f_min_prev, f_max_prev)


class CTCPrefixScore(object):
    """Numpy CTC prefix scorer; Algorithm 2 in Watanabe et al."""

    def __init__(self, x, blank, eos, xp):
        self.xp = xp
        self.logzero = -10000000000.0
        self.blank = blank
        self.eos = eos
        self.input_length = len(x)
        self.x = x

    def initial_state(self):
        """Obtain an initial CTC state."""
        r = self.xp.full((self.input_length, 2), self.logzero, dtype=np.float32)
        r[0, 1] = self.x[0, self.blank]
        for i in range(1, self.input_length):
            r[i, 1] = r[i - 1, 1] + self.x[i, self.blank]
        return r

    def __call__(self, y, cs, r_prev):
        """Compute CTC prefix scores for next labels."""
        output_length = len(y) - 1
        r = self.xp.ndarray((self.input_length, 2, len(cs)), dtype=np.float32)
        xs = self.x[:, cs]
        if output_length == 0:
            r[0, 0] = xs[0]
            r[0, 1] = self.logzero
        else:
            r[output_length - 1] = self.logzero

        r_sum = self.xp.logaddexp(r_prev[:, 0], r_prev[:, 1])
        last = y[-1]
        if output_length > 0 and last in cs:
            log_phi = self.xp.ndarray((self.input_length, len(cs)), dtype=np.float32)
            for i in range(len(cs)):
                log_phi[:, i] = r_sum if cs[i] != last else r_prev[:, 1]
        else:
            log_phi = r_sum

        start = max(output_length, 1)
        log_psi = r[start - 1, 0]
        for t in range(start, self.input_length):
            r[t, 0] = self.xp.logaddexp(r[t - 1, 0], log_phi[t - 1]) + xs[t]
            r[t, 1] = (
                self.xp.logaddexp(r[t - 1, 0], r[t - 1, 1]) + self.x[t, self.blank]
            )
            log_psi = self.xp.logaddexp(log_psi, log_phi[t - 1] + xs[t])

        eos_pos = self.xp.where(cs == self.eos)[0]
        if len(eos_pos) > 0:
            log_psi[eos_pos] = r_sum[-1]

        if self.eos != self.blank:
            blank_pos = self.xp.where(cs == self.blank)[0]
            if len(blank_pos) > 0:
                log_psi[blank_pos] = self.logzero

        return log_psi, self.xp.rollaxis(r, 2)


# -- espnet/nets/beam_search.py --

class Hypothesis(NamedTuple):
    """Hypothesis data type."""

    yseq: torch.Tensor
    score: Union[float, torch.Tensor] = 0
    scores: Dict[str, Union[float, torch.Tensor]] = dict()
    states: Dict[str, Any] = dict()
    hs: List[torch.Tensor] = []

    def asdict(self) -> dict:
        """Convert data to JSON-friendly dict."""
        return self._replace(
            yseq=self.yseq.tolist(),
            score=float(self.score),
            scores={k: float(v) for k, v in self.scores.items()},
        )._asdict()


class BeamSearch(torch.nn.Module):
    """Beam search implementation."""

    def __init__(
        self,
        scorers: Dict[str, ScorerInterface],
        weights: Dict[str, float],
        beam_size: int,
        vocab_size: int,
        sos: int,
        eos: int,
        token_list: List[str] = None,
        pre_beam_ratio: float = 1.5,
        pre_beam_score_key: str = None,
        return_hs: bool = False,
        hyp_primer: List[int] = None,
        normalize_length: bool = False,
    ):
        super().__init__()
        self.weights = weights
        self.scorers = dict()
        self.full_scorers = dict()
        self.part_scorers = dict()
        self.nn_dict = torch.nn.ModuleDict()
        for k, v in scorers.items():
            w = weights.get(k, 0)
            if w == 0 or v is None:
                continue
            assert isinstance(v, ScorerInterface), (
                f"{k} ({type(v)}) does not implement ScorerInterface"
            )
            self.scorers[k] = v
            if isinstance(v, PartialScorerInterface):
                self.part_scorers[k] = v
            else:
                self.full_scorers[k] = v
            if isinstance(v, torch.nn.Module):
                self.nn_dict[k] = v

        self.sos = sos
        self.eos = eos
        self.hyp_primer = hyp_primer
        self.token_list = token_list
        self.pre_beam_size = int(pre_beam_ratio * beam_size)
        self.beam_size = beam_size
        self.n_vocab = vocab_size
        if (
            pre_beam_score_key is not None
            and pre_beam_score_key != "full"
            and pre_beam_score_key not in self.full_scorers
        ):
            raise KeyError(f"{pre_beam_score_key} is not found in {self.full_scorers}")
        self.pre_beam_score_key = pre_beam_score_key
        self.do_pre_beam = (
            self.pre_beam_score_key is not None
            and self.pre_beam_size < self.n_vocab
            and len(self.part_scorers) > 0
        )
        self.return_hs = return_hs
        self.normalize_length = normalize_length

    def set_hyp_primer(self, hyp_primer: List[int] = None) -> None:
        """Set the primer sequence for decoding (OpenAI Whisper)."""
        self.hyp_primer = hyp_primer

    def init_hyp(self, x: torch.Tensor) -> List[Hypothesis]:
        """Get an initial hypothesis data."""
        init_states = dict()
        init_scores = dict()
        for k, d in self.scorers.items():
            init_states[k] = d.init_state(x)
            init_scores[k] = 0.0
        primer = [self.sos] if self.hyp_primer is None else self.hyp_primer
        return [
            Hypothesis(
                score=0.0,
                scores=init_scores,
                states=init_states,
                hs=[],
                yseq=torch.tensor(primer, device=x.device),
            )
        ]

    @staticmethod
    def append_token(xs: torch.Tensor, x: int) -> torch.Tensor:
        """Append new token to prefix tokens."""
        x = torch.tensor([x], dtype=xs.dtype, device=xs.device)
        return torch.cat((xs, x))

    def score_full(
        self,
        hyp: Hypothesis,
        x: torch.Tensor,
        pre_x: torch.Tensor = None,
    ) -> Tuple[Dict[str, torch.Tensor], Dict[str, Any]]:
        """Score new hypothesis by full scorers."""
        scores = dict()
        states = dict()
        for k, d in self.full_scorers.items():
            if "decoder" in k and self.return_hs:
                scores[k], hs, states[k] = d.score(
                    hyp.yseq, hyp.states[k], x, return_hs=self.return_hs
                )
            elif pre_x is not None:
                scores[k], states[k] = d.score(hyp.yseq, hyp.states[k], x, pre_x)
            else:
                scores[k], states[k] = d.score(hyp.yseq, hyp.states[k], x)
        if self.return_hs:
            return hs, scores, states
        return scores, states

    def score_partial(
        self,
        hyp: Hypothesis,
        ids: torch.Tensor,
        x: torch.Tensor,
    ) -> Tuple[Dict[str, torch.Tensor], Dict[str, Any]]:
        """Score new hypothesis by partial scorers."""
        scores = dict()
        states = dict()
        for k, d in self.part_scorers.items():
            scores[k], states[k] = d.score_partial(hyp.yseq, ids, hyp.states[k], x)
        return scores, states

    def beam(
        self, weighted_scores: torch.Tensor, ids: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Compute topk full and partial token ids."""
        if weighted_scores.size(0) == ids.size(0):
            top_ids = weighted_scores.topk(self.beam_size)[1]
            return top_ids, top_ids
        tmp = weighted_scores[ids]
        weighted_scores[:] = -float("inf")
        weighted_scores[ids] = tmp
        top_ids = weighted_scores.topk(self.beam_size)[1]
        local_ids = weighted_scores[ids].topk(self.beam_size)[1]
        return top_ids, local_ids

    @staticmethod
    def merge_scores(
        prev_scores: Dict[str, float],
        next_full_scores: Dict[str, torch.Tensor],
        full_idx: int,
        next_part_scores: Dict[str, torch.Tensor],
        part_idx: int,
    ) -> Dict[str, torch.Tensor]:
        """Merge scores for new hypothesis."""
        new_scores = dict()
        for k, v in next_full_scores.items():
            new_scores[k] = prev_scores[k] + v[full_idx]
        for k, v in next_part_scores.items():
            new_scores[k] = prev_scores[k] + v[part_idx]
        return new_scores

    def merge_states(self, states: Any, part_states: Any, part_idx: int) -> Any:
        """Merge states for new hypothesis."""
        new_states = dict()
        for k, v in states.items():
            new_states[k] = v
        for k, d in self.part_scorers.items():
            new_states[k] = d.select_state(part_states[k], part_idx)
        return new_states

    def search(
        self,
        running_hyps: List[Hypothesis],
        x: torch.Tensor,
        pre_x: torch.Tensor = None,
    ) -> List[Hypothesis]:
        """Search new tokens for running hypotheses."""
        best_hyps = []
        part_ids = torch.arange(self.n_vocab, device=x.device)
        for hyp in running_hyps:
            weighted_scores = torch.zeros(
                self.n_vocab, dtype=x.dtype, device=x.device
            )
            if self.return_hs:
                hs, scores, states = self.score_full(hyp, x, pre_x=pre_x)
            else:
                scores, states = self.score_full(hyp, x, pre_x=pre_x)
            for k in self.full_scorers:
                weighted_scores += self.weights[k] * scores[k]
            if self.do_pre_beam:
                pre_beam_scores = (
                    weighted_scores
                    if self.pre_beam_score_key == "full"
                    else scores[self.pre_beam_score_key]
                )
                part_ids = torch.topk(pre_beam_scores, self.pre_beam_size)[1]
            part_scores, part_states = self.score_partial(hyp, part_ids, x)
            for k in self.part_scorers:
                weighted_scores[part_ids] += self.weights[k] * part_scores[k]
            weighted_scores += hyp.score
            for j, part_j in zip(*self.beam(weighted_scores, part_ids)):
                new_hs = hyp.hs + [hs.squeeze(0)] if self.return_hs else []
                best_hyps.append(
                    Hypothesis(
                        score=weighted_scores[j],
                        yseq=self.append_token(hyp.yseq, j),
                        scores=self.merge_scores(
                            hyp.scores, scores, j, part_scores, part_j
                        ),
                        states=self.merge_states(states, part_states, part_j),
                        hs=new_hs,
                    )
                )
            best_hyps = sorted(best_hyps, key=lambda x: x.score, reverse=True)[
                : min(len(best_hyps), self.beam_size)
            ]
        return best_hyps

    def forward(
        self,
        x: torch.Tensor,
        maxlenratio: float = 0.0,
        minlenratio: float = 0.0,
        pre_x: torch.Tensor = None,
    ) -> List[Hypothesis]:
        """Perform beam search."""
        inp = pre_x if pre_x is not None else x
        if maxlenratio == 0:
            maxlen = inp.shape[0]
        elif maxlenratio < 0:
            maxlen = -1 * int(maxlenratio)
        else:
            maxlen = max(1, int(maxlenratio * inp.size(0)))
        minlen = (
            -1 * int(minlenratio) if minlenratio < 0 else int(minlenratio * inp.size(0))
        )
        logger.info("decoder input length: " + str(inp.shape[0]))
        logger.info("max output length: " + str(maxlen))
        logger.info("min output length: " + str(minlen))

        running_hyps = self.init_hyp(x if pre_x is None else pre_x)
        ended_hyps = []
        for i in range(maxlen):
            logger.debug("position " + str(i))
            best = self.search(running_hyps, x, pre_x=pre_x)
            running_hyps = self.post_process(
                i, maxlen, minlen, maxlenratio, best, ended_hyps
            )
            if maxlenratio == 0.0 and end_detect([h.asdict() for h in ended_hyps], i):
                logger.info(f"end detected at {i}")
                break
            if len(running_hyps) == 0:
                logger.info("no hypothesis. Finish decoding.")
                break
            else:
                logger.debug(f"remained hypotheses: {len(running_hyps)}")

        if self.normalize_length:
            nbest_hyps = sorted(
                ended_hyps,
                key=lambda x: x.score / (len(x.yseq) - 1),
                reverse=True,
            )
        else:
            nbest_hyps = sorted(ended_hyps, key=lambda x: x.score, reverse=True)

        if len(nbest_hyps) == 0:
            logger.warning(
                "there is no N-best results, perform recognition "
                "again with smaller minlenratio."
            )
            return (
                []
                if minlenratio < 0.1
                else self.forward(x, maxlenratio, max(0.0, minlenratio - 0.1))
            )

        best = nbest_hyps[0]
        for k, v in best.scores.items():
            logger.info(
                f"{v:6.2f} * {self.weights[k]:3} = "
                f"{v * self.weights[k]:6.2f} for {k}"
            )
        logger.info(f"total log probability: {best.score:.2f}")
        logger.info(
            f"normalized log probability: {best.score / len(best.yseq):.2f}"
        )
        logger.info(f"total number of ended hypotheses: {len(nbest_hyps)}")
        if self.token_list is not None:
            logger.info(
                "best hypo: "
                + "".join([self.token_list[x] for x in best.yseq[1:-1]])
                + "\n"
            )
        if best.yseq[1:-1].shape[0] == maxlen:
            logger.warning(
                "best hypo length: {} == max output length: {}".format(
                    best.yseq[1:-1].shape[0], maxlen
                )
            )
        return nbest_hyps

    def post_process(
        self,
        i: int,
        maxlen: int,
        minlen: int,
        maxlenratio: float,
        running_hyps: List[Hypothesis],
        ended_hyps: List[Hypothesis],
    ) -> List[Hypothesis]:
        """Perform post-processing of beam search iterations."""
        logger.debug(f"the number of running hypotheses: {len(running_hyps)}")
        if self.token_list is not None:
            logger.debug(
                "best hypo: "
                + "".join([self.token_list[x] for x in running_hyps[0].yseq[1:]])
            )
        if i == maxlen - 1:
            logger.info("adding <eos> in the last position in the loop")
            running_hyps = [
                h._replace(yseq=self.append_token(h.yseq, self.eos))
                for h in running_hyps
            ]
        remained_hyps = []
        for hyp in running_hyps:
            if hyp.yseq[-1] == self.eos:
                for k, d in chain(
                    self.full_scorers.items(), self.part_scorers.items()
                ):
                    s = d.final_score(hyp.states[k])
                    hyp.scores[k] += s
                    hyp = hyp._replace(score=hyp.score + self.weights[k] * s)
                if i >= minlen:
                    ended_hyps.append(hyp)
            else:
                remained_hyps.append(hyp)
        return remained_hyps


# -- espnet/nets/batch_beam_search.py --

class BatchHypothesis(NamedTuple):
    """Batchified hypothesis data type."""

    yseq: torch.Tensor = torch.tensor([])
    score: torch.Tensor = torch.tensor([])
    length: torch.Tensor = torch.tensor([])
    scores: Dict[str, torch.Tensor] = dict()
    states: Dict[str, Dict] = dict()
    hs: List[torch.Tensor] = []

    def __len__(self) -> int:
        return len(self.length)


class BatchBeamSearch(BeamSearch):
    """Batch beam search implementation."""

    def batchfy(self, hyps: List[Hypothesis]) -> BatchHypothesis:
        """Convert list to batch."""
        if len(hyps) == 0:
            return BatchHypothesis()
        hs = [h.hs for h in hyps] if self.return_hs else []
        return BatchHypothesis(
            yseq=pad_sequence(
                [h.yseq for h in hyps], batch_first=True, padding_value=self.eos
            ),
            length=torch.tensor([len(h.yseq) for h in hyps], dtype=torch.int64),
            score=torch.tensor([h.score for h in hyps]),
            scores={
                k: torch.tensor([h.scores[k] for h in hyps]) for k in self.scorers
            },
            states={k: [h.states[k] for h in hyps] for k in self.scorers},
            hs=hs,
        )

    def _batch_select(
        self, hyps: BatchHypothesis, ids: List[int]
    ) -> BatchHypothesis:
        hs = [hyps.hs[i] for i in ids] if self.return_hs else []
        return BatchHypothesis(
            yseq=hyps.yseq[ids],
            score=hyps.score[ids],
            length=hyps.length[ids],
            scores={k: v[ids] for k, v in hyps.scores.items()},
            states={
                k: [self.scorers[k].select_state(v, i) for i in ids]
                for k, v in hyps.states.items()
            },
            hs=hs,
        )

    def _select(self, hyps: BatchHypothesis, i: int) -> Hypothesis:
        return Hypothesis(
            yseq=hyps.yseq[i, : hyps.length[i]],
            score=hyps.score[i],
            scores={k: v[i] for k, v in hyps.scores.items()},
            states={
                k: self.scorers[k].select_state(v, i)
                for k, v in hyps.states.items()
            },
            hs=hyps.hs[i] if self.return_hs else [],
        )

    def unbatchfy(self, batch_hyps: BatchHypothesis) -> List[Hypothesis]:
        """Revert batch to list."""
        return [
            Hypothesis(
                yseq=batch_hyps.yseq[i][: batch_hyps.length[i]],
                score=batch_hyps.score[i],
                scores={k: batch_hyps.scores[k][i] for k in self.scorers},
                states={
                    k: v.select_state(batch_hyps.states[k], i)
                    for k, v in self.scorers.items()
                },
                hs=batch_hyps.hs[i] if self.return_hs else [],
            )
            for i in range(len(batch_hyps.length))
        ]

    def batch_beam(
        self, weighted_scores: torch.Tensor, ids: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Batch-compute topk full and partial token ids."""
        top_ids = weighted_scores.view(-1).topk(self.beam_size)[1]
        if is_torch_1_9_plus:
            prev_hyp_ids = torch.div(top_ids, self.n_vocab, rounding_mode="trunc")
        else:
            prev_hyp_ids = top_ids // self.n_vocab
        new_token_ids = top_ids % self.n_vocab
        return prev_hyp_ids, new_token_ids, prev_hyp_ids, new_token_ids

    def init_hyp(self, x: torch.Tensor) -> BatchHypothesis:
        """Get an initial hypothesis data."""
        init_states = dict()
        init_scores = dict()
        for k, d in self.scorers.items():
            init_states[k] = d.batch_init_state(x)
            init_scores[k] = 0.0
        primer = [self.sos] if self.hyp_primer is None else self.hyp_primer
        return self.batchfy(
            [
                Hypothesis(
                    score=0.0,
                    scores=init_scores,
                    states=init_states,
                    hs=[],
                    yseq=torch.tensor(primer, device=x.device),
                )
            ]
        )

    def score_full(
        self,
        hyp: BatchHypothesis,
        x: torch.Tensor,
        pre_x: torch.Tensor = None,
    ) -> Tuple[Dict[str, torch.Tensor], Dict[str, Any]]:
        """Score new hypothesis by full scorers (batch)."""
        scores = dict()
        states = dict()
        for k, d in self.full_scorers.items():
            if "decoder" in k and self.return_hs:
                (scores[k], hs), states[k] = d.batch_score(
                    hyp.yseq, hyp.states[k], x, return_hs=self.return_hs
                )
            elif "decoder" in k and pre_x is not None:
                scores[k], states[k] = d.batch_score(
                    hyp.yseq, hyp.states[k], x, pre_x
                )
            else:
                scores[k], states[k] = d.batch_score(hyp.yseq, hyp.states[k], x)
        if self.return_hs:
            return hs, scores, states
        return scores, states

    def score_partial(
        self,
        hyp: BatchHypothesis,
        ids: torch.Tensor,
        x: torch.Tensor,
        pre_x: torch.Tensor = None,
    ) -> Tuple[Dict[str, torch.Tensor], Dict[str, Any]]:
        """Score new hypothesis by partial scorers (batch)."""
        scores = dict()
        states = dict()
        for k, d in self.part_scorers.items():
            if "ctc" in k and pre_x is not None:
                scores[k], states[k] = d.batch_score_partial(
                    hyp.yseq, ids, hyp.states[k], pre_x
                )
            else:
                scores[k], states[k] = d.batch_score_partial(
                    hyp.yseq, ids, hyp.states[k], x
                )
        return scores, states

    def merge_states(self, states: Any, part_states: Any, part_idx: int) -> Any:
        """Merge states for new hypothesis (batch)."""
        new_states = dict()
        for k, v in states.items():
            new_states[k] = v
        for k, v in part_states.items():
            new_states[k] = v
        return new_states

    def search(
        self,
        running_hyps: BatchHypothesis,
        x: torch.Tensor,
        pre_x: torch.Tensor = None,
    ) -> BatchHypothesis:
        """Search new tokens for running hypotheses (batch)."""
        n_batch = len(running_hyps)
        part_ids = None
        weighted_scores = torch.zeros(
            n_batch, self.n_vocab, dtype=x.dtype, device=x.device
        )
        x_exp = x.expand(n_batch, *x.shape)
        pre_x_exp = pre_x.expand(n_batch, *pre_x.shape) if pre_x is not None else None
        if self.return_hs:
            hs, scores, states = self.score_full(running_hyps, x_exp, pre_x_exp)
        else:
            scores, states = self.score_full(running_hyps, x_exp, pre_x_exp)
        for k in self.full_scorers:
            weighted_scores += self.weights[k] * scores[k]
        if self.do_pre_beam:
            pre_beam_scores = (
                weighted_scores
                if self.pre_beam_score_key == "full"
                else scores[self.pre_beam_score_key]
            )
            part_ids = torch.topk(pre_beam_scores, self.pre_beam_size, dim=-1)[1]
        part_scores, part_states = self.score_partial(
            running_hyps, part_ids, x, pre_x
        )
        for k in self.part_scorers:
            weighted_scores += self.weights[k] * part_scores[k]
        weighted_scores += running_hyps.score.to(
            dtype=x.dtype, device=x.device
        ).unsqueeze(1)

        best_hyps = []
        prev_hyps = self.unbatchfy(running_hyps)
        for (
            full_prev_hyp_id,
            full_new_token_id,
            part_prev_hyp_id,
            part_new_token_id,
        ) in zip(*self.batch_beam(weighted_scores, part_ids)):
            prev_hyp = prev_hyps[full_prev_hyp_id]
            new_hs = (
                prev_hyp.hs + [hs[full_prev_hyp_id].squeeze(0)]
                if self.return_hs
                else []
            )
            best_hyps.append(
                Hypothesis(
                    score=weighted_scores[full_prev_hyp_id, full_new_token_id],
                    yseq=self.append_token(prev_hyp.yseq, full_new_token_id),
                    scores=self.merge_scores(
                        prev_hyp.scores,
                        {k: v[full_prev_hyp_id] for k, v in scores.items()},
                        full_new_token_id,
                        {k: v[part_prev_hyp_id] for k, v in part_scores.items()},
                        part_new_token_id,
                    ),
                    states=self.merge_states(
                        {
                            k: self.full_scorers[k].select_state(v, full_prev_hyp_id)
                            for k, v in states.items()
                        },
                        {
                            k: self.part_scorers[k].select_state(
                                v, part_prev_hyp_id, part_new_token_id
                            )
                            for k, v in part_states.items()
                        },
                        part_new_token_id,
                    ),
                    hs=new_hs,
                )
            )
        return self.batchfy(best_hyps)

    def post_process(
        self,
        i: int,
        maxlen: int,
        minlen: int,
        maxlenratio: float,
        running_hyps: BatchHypothesis,
        ended_hyps: List[Hypothesis],
    ) -> BatchHypothesis:
        """Perform post-processing of batch beam search iterations."""
        n_batch = running_hyps.yseq.shape[0]
        logger.debug(f"the number of running hypothes: {n_batch}")
        if self.token_list is not None:
            logger.debug(
                "best hypo: "
                + "".join(
                    [
                        self.token_list[x]
                        for x in running_hyps.yseq[0, 1 : running_hyps.length[0]]
                    ]
                )
            )
        if i == maxlen - 1:
            logger.info("adding <eos> in the last position in the loop")
            yseq_eos = torch.cat(
                (
                    running_hyps.yseq,
                    torch.full(
                        (n_batch, 1),
                        self.eos,
                        device=running_hyps.yseq.device,
                        dtype=torch.int64,
                    ),
                ),
                1,
            )
            running_hyps.yseq.resize_as_(yseq_eos)
            running_hyps.yseq[:] = yseq_eos
            running_hyps.length[:] = yseq_eos.shape[1]

        is_eos = (
            running_hyps.yseq[torch.arange(n_batch), running_hyps.length - 1]
            == self.eos
        )
        for b in torch.nonzero(is_eos, as_tuple=False).view(-1):
            hyp = self._select(running_hyps, b)
            if i >= minlen:
                ended_hyps.append(hyp)
        remained_ids = torch.nonzero(is_eos == 0, as_tuple=False).view(-1).cpu()
        return self._batch_select(running_hyps, remained_ids)


# -- espnet/nets/scorers/ctc.py --

class CTCPrefixScorer(BatchPartialScorerInterface):
    """Decoder interface wrapper for CTCPrefixScore."""

    def __init__(self, ctc: torch.nn.Module, eos: int):
        self.ctc = ctc
        self.eos = eos
        self.impl = None

    def init_state(self, x: torch.Tensor):
        """Get an initial state for decoding."""
        logp = self.ctc.log_softmax(x.unsqueeze(0)).detach().squeeze(0).cpu().numpy()
        self.impl = CTCPrefixScore(logp, 0, self.eos, np)
        return 0, self.impl.initial_state()

    def select_state(self, state, i, new_id=None):
        """Select state with relative ids in the main beam search."""
        if type(state) is tuple:
            if len(state) == 2:
                sc, st = state
                return sc[i], st[i]
            else:
                r, log_psi, f_min, f_max, scoring_idmap = state
                s = log_psi[i, new_id].expand(log_psi.size(1))
                if scoring_idmap is not None:
                    return r[:, :, i, scoring_idmap[i, new_id]], s, f_min, f_max
                else:
                    return r[:, :, i, new_id], s, f_min, f_max
        return None if state is None else state[i]

    def score_partial(self, y, ids, state, x):
        """Score new token."""
        prev_score, state = state
        presub_score, new_st = self.impl(y.cpu(), ids.cpu(), state)
        tscore = torch.as_tensor(
            presub_score - prev_score, device=x.device, dtype=x.dtype
        )
        return tscore, (presub_score, new_st)

    def batch_init_state(self, x: torch.Tensor):
        """Get an initial state for batch decoding."""
        logp = self.ctc.log_softmax(x.unsqueeze(0))
        xlen = torch.tensor([logp.size(1)])
        self.impl = CTCPrefixScoreTH(logp, xlen, 0, self.eos)
        return None

    def batch_score_partial(self, y, ids, state, x):
        """Score new token (batch)."""
        batch_state = (
            (
                torch.stack([s[0] for s in state], dim=2),
                torch.stack([s[1] for s in state]),
                state[0][2],
                state[0][3],
            )
            if state[0] is not None
            else None
        )
        return self.impl(y, batch_state, ids)

    def extend_prob(self, x: torch.Tensor):
        """Extend probs for streaming decoding."""
        logp = self.ctc.log_softmax(x.unsqueeze(0))
        self.impl.extend_prob(logp)

    def extend_state(self, state):
        """Extend state for streaming decoding."""
        return [self.impl.extend_state(s) for s in state]


# -- espnet/nets/scorers/length_bonus.py --

class LengthBonus(BatchScorerInterface):
    """Length bonus in beam search."""

    def __init__(self, n_vocab: int):
        self.n = n_vocab

    def score(self, y, state, x):
        """Score new token."""
        return (
            torch.tensor([1.0], device=x.device, dtype=x.dtype).expand(self.n),
            None,
        )

    def batch_score(
        self, ys: torch.Tensor, states: List[Any], xs: torch.Tensor
    ) -> Tuple[torch.Tensor, List[Any]]:
        """Score new token (batch)."""
        return (
            torch.tensor([1.0], device=xs.device, dtype=xs.dtype).expand(
                ys.shape[0], self.n
            ),
            None,
        )
