#!/usr/bin/env python3
"""
Minimal standalone S2T decoding — no espnet dependency.

Assumptions
-----------
- batch_size = 1; single utterance per call.
- CTC-only scoring (no attention decoder, no LM).
- Input speech: 1-D numpy array or 1-D torch.Tensor at the model's sample rate.
- model exposes:
    model.ctc.log_softmax(enc) -> (1, T, vocab)
    model.ctc.argmax(enc)      -> (1, T)
    model.encode(**batch)      -> enc (1, T, D), enc_olens
    model.sos, model.eos, model.blank_id, model.na, model.token_list
- tokenizer (or None) exposes .tokens2text(list[str]) -> str.
- For long-form greedy decoding, provide frames_per_sec + buffer_len_in_secs.

CTCPrefixScoreTH adapted from ESPnet (Apache 2.0),
Copyright 2018 Mitsubishi Electric Research Labs (Takaaki Hori).
"""

import argparse
import yaml

import numpy as np
import torch


# ── TokenIDConverter ──────────────────────────────────────────────────────────

class _TokenIDConverter:
    def __init__(self, token_list):
        self.token_list = list(token_list)
        self.token2id = {t: i for i, t in enumerate(self.token_list)}

    def ids2tokens(self, ids):
        return [self.token_list[i] for i in ids]

    def tokens2ids(self, tokens):
        return [self.token2id[t] for t in tokens if t in self.token2id]


# ── BPE tokenizer (sentencepiece) ─────────────────────────────────────────────

class _BpeTokenizer:
    """Thin sentencepiece wrapper matching ESPnet tokenizer API."""

    def __init__(self, model_path):
        import sentencepiece as spm
        self._sp = spm.SentencePieceProcessor()
        self._sp.load(model_path)

    def text2tokens(self, text):
        return self._sp.encode_as_pieces(text)

    def tokens2text(self, tokens):
        return self._sp.decode_pieces(tokens)


def _build_bpe_tokenizer(bpemodel):
    return _BpeTokenizer(bpemodel)


# ── Model builder ─────────────────────────────────────────────────────────────

def _build_model_from_file(config_file, model_file, device="cpu"):
    """Load config YAML, build model, load checkpoint; return (model, args)."""
    from pathlib import Path
    config_file = (
        Path(config_file) if config_file else Path(model_file).parent / "config.yaml"
    )
    with open(config_file, "r", encoding="utf-8") as f:
        args = argparse.Namespace(**yaml.safe_load(f))

    model = _build_s2t_ctc_model(args)
    model.to(device)

    if model_file is not None:
        state = torch.load(model_file, map_location=device, weights_only=False)
        if "module" in state:
            state = state["module"]
        model.load_state_dict(state, strict=False)

    return model, args


def _build_s2t_ctc_model(args):
    """Instantiate ESPnetS2TCTCModel from config args using local modules."""
    from model.powsm.builders_common import (
        load_token_list, build_frontend, build_specaug, build_normalize, build_ctc,
    )
    from model.powsm.e_branchformer_ctc import EBranchformerCTCEncoder
    from model.powsm.prompt_encoder import PromptEncoder
    from model.powsm.s2t_ctc_model import ESPnetS2TCTCModel

    token_list = load_token_list(args.token_list)
    vocab_size = len(token_list)

    frontend, input_size = build_frontend(args)
    specaug = build_specaug(args) if getattr(args, "specaug", None) else None
    normalize = build_normalize(args, args.normalize_conf["stats_file"])

    encoder = EBranchformerCTCEncoder(input_size=input_size, **args.encoder_conf)
    prompt_encoder = PromptEncoder(
        input_size=args.promptencoder_conf["output_size"],
        **args.promptencoder_conf,
    )
    ctc = build_ctc(vocab_size, encoder.output_size(), getattr(args, "ctc_conf", {}))

    return ESPnetS2TCTCModel(
        vocab_size=vocab_size,
        token_list=token_list,
        frontend=frontend,
        specaug=specaug,
        normalize=normalize,
        encoder=encoder,
        prompt_encoder=prompt_encoder,
        ctc=ctc,
        **getattr(args, "model_conf", {}),
    )


# ── CTCPrefixScoreTH ──────────────────────────────────────────────────────────

class CTCPrefixScoreTH:
    """Vectorised CTC prefix scoring for a beam of hypotheses."""

    def __init__(self, x, xlens, blank, eos):
        # x: (1, T, vocab) log-probs for one utterance
        self.logzero      = -10_000_000_000.0
        self.blank        = blank
        self.eos          = eos
        self.input_length = x.size(1)
        self.odim         = x.size(2)
        self.dtype        = x.dtype
        self.device       = x.device
        for i, l in enumerate(xlens):
            if l < self.input_length:
                x[i, l:, :] = self.logzero
                x[i, l:, blank] = 0
        xn       = x.transpose(0, 1)                             # (T, 1, vocab)
        xb       = xn[:, :, blank].unsqueeze(2).expand(-1, -1, self.odim)
        self.x   = torch.stack([xn, xb])                         # (2, T, 1, vocab)
        self.end_frame = int(xlens[0]) - 1
        self.idx_b     = torch.arange(1, device=self.device)     # batch=1
        self.idx_bo    = (self.idx_b * self.odim).unsqueeze(1)
        self.idx_bh    = None

    def __call__(self, y, state, scoring_ids=None):
        output_length    = len(y[0]) - 1
        last_ids         = [yi[-1] for yi in y]
        n_bh             = len(last_ids)
        self.scoring_num = scoring_ids.size(-1) if scoring_ids is not None else 0

        if state is None:
            r_prev = torch.full((self.input_length, 2, 1, n_bh),
                                self.logzero, dtype=self.dtype, device=self.device)
            r_prev[:, 1] = torch.cumsum(self.x[0, :, :, self.blank], 0).unsqueeze(2)
            r_prev    = r_prev.view(-1, 2, n_bh)
            s_prev    = 0.0
            f_min_prev, f_max_prev = 0, 1
        else:
            r_prev, s_prev, f_min_prev, f_max_prev = state

        if self.scoring_num > 0:
            scoring_idmap = torch.full((n_bh, self.odim), -1, dtype=torch.long, device=self.device)
            snum = self.scoring_num
            if self.idx_bh is None or n_bh > len(self.idx_bh):
                self.idx_bh = torch.arange(n_bh, device=self.device).view(-1, 1)
            scoring_idmap[self.idx_bh[:n_bh], scoring_ids] = torch.arange(snum, device=self.device)
            scoring_idx = (scoring_ids + self.idx_bo.repeat(1, n_bh).view(-1, 1)).view(-1)
            x_ = torch.index_select(
                self.x.view(2, -1, self.odim), 2, scoring_idx
            ).view(2, -1, n_bh, snum)
        else:
            scoring_ids = scoring_idmap = None
            snum = self.odim
            x_ = self.x.expand(-1, -1, n_bh, -1)   # (2, T, n_bh, vocab)

        r = torch.full((self.input_length, 2, n_bh, snum),
                       self.logzero, dtype=self.dtype, device=self.device)
        if output_length == 0:
            r[0, 0] = x_[0, 0]

        r_sum   = torch.logsumexp(r_prev, 1)
        log_phi = r_sum.unsqueeze(2).repeat(1, 1, snum)
        if scoring_ids is not None:
            for i in range(n_bh):
                pos = scoring_idmap[i, last_ids[i]]
                if pos >= 0:
                    log_phi[:, i, pos] = r_prev[:, 1, i]
        else:
            for i in range(n_bh):
                log_phi[:, i, last_ids[i]] = r_prev[:, 1, i]

        start = max(output_length, 1)
        for t in range(start, self.input_length):
            rp   = r[t - 1]
            rr   = torch.stack([rp[0], log_phi[t - 1], rp[0], rp[1]]).view(2, 2, n_bh, snum)
            r[t] = torch.logsumexp(rr, 1) + x_[:, t]

        log_phi_x = torch.cat((log_phi[0].unsqueeze(0), log_phi[:-1]), dim=0) + x_[0]
        if scoring_ids is not None:
            log_psi = torch.full((n_bh, self.odim), self.logzero, dtype=self.dtype, device=self.device)
            log_psi_ = torch.logsumexp(
                torch.cat((log_phi_x[start:], r[start - 1, 0].unsqueeze(0)), dim=0), dim=0)
            for i in range(n_bh):
                log_psi[i, scoring_ids[i]] = log_psi_[i]
        else:
            log_psi = torch.logsumexp(
                torch.cat((log_phi_x[start:], r[start - 1, 0].unsqueeze(0)), dim=0), dim=0)

        for i in range(n_bh):
            log_psi[i, self.eos] = r_sum[self.end_frame, i]
        log_psi[:, self.blank] = self.logzero

        return (log_psi - s_prev), (r, log_psi, 0, 0, scoring_idmap)

    def index_select_state(self, state, best_ids):
        """Pick per-hypothesis state for chosen token ids. best_ids: (1, k) tensor."""
        r, s, f_min, f_max, scoring_idmap = state
        n_bh = len(s)
        vidx = best_ids.view(-1)
        s_new = torch.index_select(s.view(-1), 0, vidx)
        s_new = s_new.view(-1, 1).repeat(1, self.odim).view(n_bh, self.odim)
        if scoring_idmap is not None:
            snum      = self.scoring_num
            label_ids = torch.fmod(best_ids, self.odim).view(-1)
            score_idx = scoring_idmap[torch.zeros_like(label_ids), label_ids]
            score_idx[score_idx == -1] = 0
            vidx      = score_idx
        else:
            snum = self.odim
        r_new = torch.index_select(r.view(-1, 2, n_bh * snum), 2, vidx).view(-1, 2, n_bh)
        return r_new, s_new, f_min, f_max


# ── Beam search ───────────────────────────────────────────────────────────────

def ctc_beam_search(log_probs, beam_size, sos, eos, blank_id, maxlenratio=0.0):
    """CTC prefix beam search. log_probs: (T, vocab). Returns list of token ids."""
    T      = log_probs.size(0)
    maxlen = T if maxlenratio == 0 else max(1, int(maxlenratio * T))

    impl = CTCPrefixScoreTH(
        log_probs.unsqueeze(0).clone(), torch.tensor([T]), blank_id, eos
    )

    # (cumulative_score, token_ids, ctc_state)
    beams = [(0.0, [sos], None)]
    ended = []

    for _ in range(maxlen):
        candidates = []
        for score, yseq, state in beams:
            tok_scores, state5 = impl([yseq], state)    # (1, vocab)
            tok_scores         = tok_scores[0]
            for tok in torch.topk(tok_scores, min(beam_size, tok_scores.size(0)))[1].tolist():
                new_state = impl.index_select_state(state5, torch.tensor([[tok]]))
                candidates.append((score + float(tok_scores[tok]), yseq + [tok], new_state))

        candidates.sort(key=lambda c: c[0], reverse=True)
        beams = []
        for c in candidates:
            (ended if c[1][-1] == eos else beams).append(c)
        beams = beams[:beam_size]
        if not beams:
            break

    best = max(ended or beams, key=lambda c: c[0])[1]
    best = best[1:]                                   # strip sos
    if best and best[-1] == eos:
        best = best[:-1]                              # strip eos
    return best


# ── Shared helpers ────────────────────────────────────────────────────────────

def _build_batch(speech, model, lang_id, task_id, dtype, device,
                 text_prev_ids=None):
    if isinstance(speech, np.ndarray):
        speech = torch.tensor(speech)
    speech = speech.unsqueeze(0).to(getattr(torch, dtype))
    B      = 1
    if text_prev_ids is None:
        text_prev_ids = [model.na]
    tp = torch.tensor([text_prev_ids], dtype=torch.long)
    batch  = dict(
        speech            = speech,
        speech_lengths    = torch.tensor([speech.size(1)], dtype=torch.long),
        text_prev         = tp,
        text_prev_lengths = torch.tensor([tp.size(1)], dtype=torch.long),
        prefix            = torch.tensor([[lang_id, task_id]], dtype=torch.long),
        prefix_lengths    = torch.tensor([2],                 dtype=torch.long),
    )
    return {k: v.to(device) for k, v in batch.items()}


def _ids_to_text(token_int, id2token, tokenizer):
    token     = [id2token[i] for i in token_int]
    nospecial = [t for t in token if not (t.startswith("<") and t.endswith(">"))]
    return tokenizer.tokens2text(nospecial) if tokenizer else " ".join(nospecial)


# ── Speech2Text (beam search) ─────────────────────────────────────────────────

class Speech2Text:
    """CTC beam-search decoding for a single utterance."""

    def __init__(self, model, tokenizer=None, device="cpu", dtype="float32",
                 beam_size=20, maxlenratio=0.0,
                 lang_sym="<nolang>", task_sym="<asr>", **_):
        self.model       = model.to(dtype=getattr(torch, dtype)).eval()
        self.tokenizer   = tokenizer
        self.device      = device
        self.dtype       = dtype
        self.beam_size   = beam_size
        self.maxlenratio = maxlenratio
        self.lang_id     = model.token_list.index(lang_sym)
        self.task_id     = model.token_list.index(task_sym)

    @torch.no_grad()
    def __call__(self, speech, lang_sym=None, task_sym=None):
        lang_id = self.model.token_list.index(lang_sym) if lang_sym else self.lang_id
        task_id = self.model.token_list.index(task_sym) if task_sym else self.task_id

        batch     = _build_batch(speech, self.model, lang_id, task_id, self.dtype, self.device)
        enc, _    = self.model.encode(**batch)
        enc       = enc[0] if not isinstance(enc, tuple) else enc[0][0]   # (T, D)
        log_probs = self.model.ctc.log_softmax(enc.unsqueeze(0))[0]       # (T, vocab)

        token_int = ctc_beam_search(
            log_probs, self.beam_size,
            self.model.sos, self.model.eos, self.model.blank_id, self.maxlenratio,
        )
        return _ids_to_text(token_int, self.model.token_list, self.tokenizer)


# ── Speech2TextGreedySearch ───────────────────────────────────────────────────

class Speech2TextGreedySearch:
    """CTC greedy (argmax) decoding; supports short- and long-form audio."""

    def __init__(
        self,
        s2t_train_config=None,
        s2t_model_file=None,
        token_type=None,
        bpemodel=None,
        device="cpu",
        dtype="float32",
        lang_sym="<nolang>",
        task_sym="<asr>",
        **_,
    ):
        s2t_model, s2t_train_args = _build_model_from_file(
            s2t_train_config, s2t_model_file, device
        )
        s2t_model.to(dtype=getattr(torch, dtype)).eval()

        if token_type is None:
            token_type = s2t_train_args.token_type
        if bpemodel is None:
            bpemodel = s2t_train_args.bpemodel
        tokenizer = _build_bpe_tokenizer(bpemodel) if bpemodel else None

        converter = _TokenIDConverter(token_list=s2t_model.token_list)

        self.model           = s2t_model   # used by _greedy / _decode_long
        self.s2t_model       = s2t_model
        self.s2t_train_args  = s2t_train_args
        self.preprocessor_conf = (
            getattr(s2t_train_args, "preprocessor_conf", None) or {}
        )
        self.tokenizer       = tokenizer
        self.converter       = converter
        self.device          = device
        self.dtype           = dtype
        self.lang_sym        = lang_sym
        self.task_sym        = task_sym
        self.lang_id         = converter.token2id.get(lang_sym, 0)
        self.task_id         = converter.token2id.get(task_sym, 0)

        # Derive sample_rate from frontend_conf (may be int or "16k"-style string)
        fs = s2t_train_args.frontend_conf.get("fs", 16000)
        if isinstance(fs, str):
            s = fs.strip().lower()
            fs = int(float(s[:-1]) * 1000) if s.endswith("k") else int(float(s))
        self.sample_rate = int(fs)

        # Frames-per-second after subsampling (used by long-form decoding)
        _subsample = {
            "conv2d1": 1, "conv2d2": 2, "conv2d": 4, "conv2d6": 6, "conv2d8": 8,
        }
        input_layer = s2t_train_args.encoder_conf.get("input_layer", "conv2d")
        hop_length  = s2t_train_args.frontend_conf.get("hop_length", 160)
        self.frames_per_sec = self.sample_rate / hop_length / _subsample.get(input_layer, 4)

        self.buffer_len_in_secs = float(
            self.preprocessor_conf.get("speech_length", 30.0)
        )

    @torch.no_grad()
    def __call__(self, speech, text_prev=None, lang_sym=None, task_sym=None):
        lang_id = self.converter.token2id.get(lang_sym, self.lang_id) if lang_sym \
            else self.lang_id
        task_id = self.converter.token2id.get(task_sym, self.task_id) if task_sym \
            else self.task_id
        text_prev_ids = self._resolve_text_prev(text_prev)
        batch = _build_batch(
            speech, self.model, lang_id, task_id, self.dtype, self.device,
            text_prev_ids=text_prev_ids,
        )
        enc, _ = self.model.encode(**batch)
        if isinstance(enc, tuple):
            enc = enc[0]
        return self._greedy_results(enc)

    def _resolve_text_prev(self, text_prev):
        """Convert text_prev to a list of token ids."""
        if text_prev is None or (isinstance(text_prev, str) and text_prev == "<na>"):
            return [self.model.na]
        if isinstance(text_prev, str):
            if self.tokenizer is not None:
                ids = self.converter.tokens2ids(self.tokenizer.text2tokens(text_prev))
            else:
                ids = []
            return ids if ids else [self.model.na]
        if isinstance(text_prev, torch.Tensor):
            return text_prev.view(-1).tolist()
        if isinstance(text_prev, np.ndarray):
            return text_prev.flatten().tolist()
        return [self.model.na]

    def _greedy(self, enc):
        # Returns plain string; used by batch_decode / _decode_long
        if enc.dim() == 2:
            enc = enc.unsqueeze(0)
        token_int = self.model.ctc.argmax(enc)[0]
        token_int = torch.unique_consecutive(token_int).tolist()
        token_int = [x for x in token_int if x != self.model.blank_id]
        return _ids_to_text(token_int, self.model.token_list, self.tokenizer)

    def _greedy_results(self, enc):
        # Returns ESPnet ListOfHypothesis: [(text, token, token_int, text_nospecial, hyp)]
        if enc.dim() == 2:
            enc = enc.unsqueeze(0)
        token_int = self.model.ctc.argmax(enc)[0]
        token_int = torch.unique_consecutive(token_int).tolist()
        token_int = [x for x in token_int if x != self.model.blank_id]
        token = [self.model.token_list[i] for i in token_int]
        nospecial = [t for t in token if not (t.startswith("<") and t.endswith(">"))]
        text_nospecial = (
            self.tokenizer.tokens2text(nospecial) if self.tokenizer else " ".join(nospecial)
        )
        return [(None, token, token_int, text_nospecial, None)]

    @torch.no_grad()
    def batch_decode(self, speech, batch_size=16, context_len_in_secs=4,
                     lang_sym=None, task_sym=None):
        """Decode one audio or a list of audios (short- or long-form)."""
        import librosa
        lang_id = self.model.token_list.index(lang_sym) if lang_sym else self.lang_id
        task_id = self.model.token_list.index(task_sym) if task_sym else self.task_id

        single = not isinstance(speech, list)
        if single:
            speech = [speech]

        results = []
        for sp in speech:
            if isinstance(sp, str):
                sp, _ = librosa.load(sp, sr=self.sample_rate)
            elif isinstance(sp, torch.Tensor):
                sp = sp.cpu().numpy()
            buf_samples = int(self.sample_rate * self.buffer_len_in_secs)
            if len(sp) <= buf_samples:
                batch  = _build_batch(sp, self.model, lang_id, task_id, self.dtype, self.device)
                enc, _ = self.model.encode(**batch)
                if isinstance(enc, tuple):
                    enc = enc[0]
                results.append(self._greedy(enc))
            else:
                results.append(self._decode_long(sp, batch_size, context_len_in_secs, lang_id, task_id))

        return results[0] if single else results

    def _decode_long(self, speech, batch_size, context_len_in_secs, lang_id, task_id):
        sr         = self.sample_rate
        fps        = self.frames_per_sec
        buf_secs   = self.buffer_len_in_secs
        chunk_secs = buf_secs - 2 * context_len_in_secs
        buf_len    = int(sr * buf_secs)
        chunk_len  = int(sr * chunk_secs)
        ctx_frames = int(fps * context_len_in_secs) if fps else None

        speech = np.pad(speech, (int(sr * context_len_in_secs),) * 2)
        buffers = []
        for i in range(0, len(speech), chunk_len):
            seg = speech[i: i + buf_len]
            buffers.append(np.pad(seg, (0, max(0, buf_len - len(seg)))))
            if len(seg) < buf_len:
                break

        speech_t = torch.tensor(np.array(buffers), dtype=getattr(torch, self.dtype))
        na       = self.model.na
        unmerged = []

        for i in range(0, speech_t.size(0), batch_size):
            cur = speech_t[i: i + batch_size].to(self.device)
            B   = cur.size(0)
            batch = dict(
                speech            = cur,
                speech_lengths    = cur.new_full([B], fill_value=cur.size(1)),
                text_prev         = torch.full([B, 1], na,            dtype=torch.long, device=self.device),
                text_prev_lengths = torch.ones([B],                    dtype=torch.long, device=self.device),
                prefix            = torch.tensor([lang_id, task_id],  dtype=torch.long, device=self.device).expand(B, -1),
                prefix_lengths    = torch.full([B], 2,                dtype=torch.long, device=self.device),
            )
            enc, _ = self.model.encode(**batch)
            if isinstance(enc, tuple):
                enc = enc[0]
            toks = self.model.ctc.argmax(enc)
            if ctx_frames:
                toks = toks[:, ctx_frames:-ctx_frames]
            unmerged.append(toks.reshape(-1))

        merged    = torch.unique_consecutive(torch.cat(unmerged)).tolist()
        token_int = [x for x in merged if x != self.model.blank_id]
        return _ids_to_text(token_int, self.model.token_list, self.tokenizer)