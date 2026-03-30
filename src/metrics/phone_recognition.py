"""Evaluate phone recognition output using panphon feature-based metrics.

Usage:
    python -m src.metrics.phone_recognition \
        --prediction_file exp/runs/inf_doreco_xeuspr/8job/transcription.json \
        --output_file exp/runs/inf_doreco_xeuspr/8job/results.csv \
        --gt_field target \
        --evaluation_name xeuspr \
        --key_field utt_id

    python -m src.metrics.phone_recognition --evaluation_name powsmctc \
        --prediction_file exp/runs/inf_doreco_powsm_ctc/8jobARR/transcription.json \
        --output_file exp/runs/inf_doreco_powsm_ctc/8jobARR/results.csv \
        --gt_field target \
        --key_field utt_id \
        --language_field lang_sym

    # Using a Kaldi-style ground truth file instead of gt_field in the JSON:
    python -m src.metrics.phone_recognition --evaluation_name xeuspr \
        --prediction_file exp/runs/inf_doreco_xeuspr/8job/transcription.json \
        --output_file exp/runs/inf_doreco_xeuspr/8job/results.csv \
        --gt_file data/doreco/text \
        --key_field utt_id
"""

import argparse
import csv
import io
import json
import os
import string
import unicodedata
from dataclasses import dataclass
from typing import Any, Dict, Tuple, Union

import panphon
import panphon.distance
from rich.console import Console
from rich.table import Table
from tqdm import tqdm

from src.utils import RankedLogger

log = RankedLogger(__name__, rank_zero_only=True)


def load_kaldi_text(path: str) -> Dict[str, str]:
    """Load a Kaldi-style text file (utt_id <space> transcription per line)."""
    utt2text = {}
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            parts = line.split(maxsplit=1)
            utt2text[parts[0]] = parts[1] if len(parts) > 1 else ""
    print(f"Loaded {len(utt2text)} ground truth entries from {path}")
    return utt2text


@dataclass
class PhoneRecognitionSummary:
    """Aggregate metrics for a phone recognition experiment."""

    PFER: float
    FED: float
    PER: float
    SUB: float
    INS: float
    DEL: float
    N: int
    phones: int


class PhoneRecognitionEvaluator:
    """Evaluates phone recognition output using panphon feature-based metrics.

    Metrics: PER, PFER, FED, and per-utterance breakdowns.
    Input format: {utt_id: {"prediction": str, "transcription": str}, ...}
    """

    def __init__(self, normalize_ipa: bool = True):
        self.normalize_ipa = normalize_ipa
        self.dst = panphon.distance.Distance()

    @staticmethod
    def clean_text(s: str) -> str:
        """Normalize IPA text: remove spaces/punct, NFC->NFD, fix 'g'→'ɡ'."""
        s = s.replace(" ", "").translate(str.maketrans("", "", string.punctuation))
        s = unicodedata.normalize("NFD", s)
        return s.replace("g", "ɡ").strip()

    def _prepare(self, text: str) -> str:
        return self.clean_text(text) if self.normalize_ipa else text

    def _compute_sid_metrics(self, hyp: str, ref: str) -> Tuple[int, int, int]:
        """Compute substitution, insertion, deletion counts via DP backtrack."""
        sub_errors = ins_errors = del_errors = 0
        Hlen, Rlen = len(hyp) + 1, len(ref) + 1
        D = [[0] * Rlen for _ in range(Hlen)]
        for hi in range(Hlen):
            D[hi][0] = hi
        for rj in range(Rlen):
            D[0][rj] = rj
        for hi in range(1, Hlen):
            for rj in range(1, Rlen):
                cost = 0 if hyp[hi - 1] == ref[rj - 1] else 1
                D[hi][rj] = min(
                    D[hi - 1][rj] + 1, D[hi][rj - 1] + 1, D[hi - 1][rj - 1] + cost
                )
        hi, rj = Hlen - 1, Rlen - 1
        while hi > 0 or rj > 0:
            if (
                hi > 0
                and rj > 0
                and D[hi][rj] == D[hi - 1][rj - 1]
                and hyp[hi - 1] == ref[rj - 1]
            ):
                hi -= 1
                rj -= 1
            elif hi > 0 and rj > 0 and D[hi][rj] == D[hi - 1][rj - 1] + 1:
                sub_errors += 1
                hi -= 1
                rj -= 1
            elif rj > 0 and D[hi][rj] == D[hi][rj - 1] + 1:
                del_errors += 1
                rj -= 1
            else:
                ins_errors += 1
                hi -= 1
        return sub_errors, ins_errors, del_errors

    def _compute_utterance_metrics(
        self, hyp: str, ref: str
    ) -> Dict[str, Union[int, float]]:
        """Compute all metrics for a single utterance."""
        hyp, ref = self._prepare(hyp), self._prepare(ref)
        fed = self.dst.feature_edit_distance(hyp, ref)
        hyp_segs = self.dst.fm.ipa_segs(hyp)
        ref_segs = self.dst.fm.ipa_segs(ref)
        n_phones = len(ref_segs)
        per_errors = self.dst.min_edit_distance(
            lambda v: 1,
            lambda v: 1,
            lambda x, y: 0 if x == y else 1,
            [[]],
            hyp_segs,
            ref_segs,
        )
        sub_errors, ins_errors, del_errors = self._compute_sid_metrics(
            hyp_segs, ref_segs
        )
        return {
            "metrics": {
                "pfer": float(fed / n_phones * 100) if n_phones > 0 else 0.0,
                "fed": float(fed),
                "per": float(per_errors / n_phones * 100) if n_phones > 0 else 0.0,
            },
            "fed": fed,
            "per_errors": per_errors,
            "sub_errors": sub_errors,
            "ins_errors": ins_errors,
            "del_errors": del_errors,
            "n_phones": n_phones,
        }

    def evaluate(
        self,
        test_data: Dict[str, Dict[str, Any]],
        tqdm_enabled: bool = True,
    ) -> Tuple[PhoneRecognitionSummary, Dict[str, Dict[str, float]]]:
        """Evaluate a full dataset.

        Returns:
            summary: PhoneRecognitionSummary
            instance_metrics: {utt_id: {"pfer", "fed", "per", "fer"}}
        """
        if not test_data:
            return (
                PhoneRecognitionSummary(
                    PFER=0.0,
                    FED=0.0,
                    PER=0.0,
                    SUB=0.0,
                    INS=0.0,
                    DEL=0.0,
                    N=0,
                    phones=0,
                ),
                {},
            )

        instance_metrics: Dict[str, Dict[str, float]] = {}
        fed_sum = per_err_sum = 0.0
        phones_sum = sub_err_sum = ins_err_sum = del_err_sum = n_utts = 0

        iterator = test_data.items()
        if tqdm_enabled:
            iterator = tqdm(
                iterator, total=len(test_data), desc="Evaluating", leave=False
            )

        for utt_id, sample in iterator:
            out = self._compute_utterance_metrics(
                sample.get("prediction", ""), sample.get("transcription", "")
            )
            instance_metrics[utt_id] = out["metrics"]
            fed_sum += out["fed"]
            per_err_sum += out["per_errors"]
            phones_sum += out["n_phones"]
            sub_err_sum += out["sub_errors"]
            ins_err_sum += out["ins_errors"]
            del_err_sum += out["del_errors"]
            n_utts += 1

        p = phones_sum
        return (
            PhoneRecognitionSummary(
                PFER=(fed_sum / p * 100) if p > 0 else 0.0,
                FED=fed_sum,
                PER=(per_err_sum / p * 100) if p > 0 else 0.0,
                SUB=(sub_err_sum / p * 100) if p > 0 else 0.0,
                INS=(ins_err_sum / p * 100) if p > 0 else 0.0,
                DEL=(del_err_sum / p * 100) if p > 0 else 0.0,
                N=n_utts,
                phones=phones_sum,
            ),
            instance_metrics,
        )

    @staticmethod
    def pretty_print(summary: PhoneRecognitionSummary, **_kwargs: Any) -> None:
        """Print a rich summary table then dump a CSV row to stdout."""
        console = Console()

        t = Table(title="Phone Recognition Results")
        t.add_column("Metric")
        t.add_column("Value", justify="right")
        t.add_row("Utterances (N)", str(summary.N))
        t.add_row("Total Phones", str(summary.phones))
        t.add_row("PFER (%)", f"{summary.PFER:.2f}")
        t.add_row("FED (total)", f"{summary.FED:.2f}")
        t.add_row("PER (%)", f"{summary.PER:.2f}")
        t.add_row("SUB (%)", f"{summary.SUB:.2f}")
        t.add_row("INS (%)", f"{summary.INS:.2f}")
        t.add_row("DEL (%)", f"{summary.DEL:.2f}")
        console.print(t)

        buf = io.StringIO()
        writer = csv.writer(buf)
        fields = ["N", "phones", "PFER", "FED", "PER", "SUB", "INS", "DEL"]
        writer.writerow(fields)
        writer.writerow(
            [
                summary.N,
                summary.phones,
                f"{summary.PFER:.2f}",
                f"{summary.FED:.2f}",
                f"{summary.PER:.2f}",
                f"{summary.SUB:.2f}",
                f"{summary.INS:.2f}",
                f"{summary.DEL:.2f}",
            ]
        )
        console.print(buf.getvalue())

    def write_to_csv(
        self,
        summary: PhoneRecognitionSummary,
        evalname: str,
        output_file: str,
        language: str,
    ) -> None:
        """Append summary metrics to a CSV file."""
        os.makedirs(os.path.dirname(output_file), exist_ok=True)
        write_header = (
            not os.path.exists(output_file) or os.path.getsize(output_file) == 0
        )
        with open(output_file, mode="a", newline="") as csvfile:
            writer = csv.writer(csvfile)
            if write_header:
                writer.writerow(
                    [
                        "eval_name",
                        "language",
                        "N",
                        "phones",
                        "PFER (%)",
                        "FED",
                        "PER (%)",
                        "SUB (%)",
                        "INS (%)",
                        "DEL (%)",
                    ]
                )
            writer.writerow(
                [
                    evalname,
                    language,
                    summary.N,
                    summary.phones,
                    f"{summary.PFER:.2f}",
                    f"{summary.FED:.2f}",
                    f"{summary.PER:.2f}",
                    f"{summary.SUB:.2f}",
                    f"{summary.INS:.2f}",
                    f"{summary.DEL:.2f}",
                ]
            )


def _load_raw_predictions(pred_file: str) -> Dict[str, Any]:
    """Load predictions from a single JSON or JSONL file.

    Supports:
      - JSON: single dict loaded with json.load
      - JSONL: one JSON object per line (distributed inference output)
    """
    with open(pred_file, "r") as f:
        first_char = f.read(1)
    if not first_char:
        return {}
    # JSONL: each line is {idx: {pred, passthrough}}
    if first_char == "{":
        with open(pred_file, "r") as f:
            try:
                return json.load(f)
            except json.JSONDecodeError:
                pass
        # Fall through to JSONL
    data = {}
    with open(pred_file, "r") as f:
        for line in f:
            line = line.strip()
            if line:
                data.update(json.loads(line))
    return data


def _load_predictions(
    pred_file: str,
    language_field: str = None,
    gt_file: str = None,
    key_field: str = "utt_id",
    gt_field: str = "masked_phones",
    pred_field: str = "processed_transcript",
    noisy_pr: bool = False,
) -> Dict[str, Dict[str, Dict[str, str]]]:
    """Load predictions into {language: {utt_id: {prediction, transcription}}}.

    pred_file can be:
      - A single JSON or JSONL file
      - A glob pattern matching multiple shards (e.g., "exp/preds.*.jsonl")
    """
    import glob as glob_mod

    matched = sorted(glob_mod.glob(pred_file))
    if not matched:
        raise FileNotFoundError(f"No files matched: {pred_file}")
    data = {}
    for path in matched:
        data.update(_load_raw_predictions(path))
    original_len = len(data)
    data = {k: v for k, v in data.items() if k != "__error__"}
    print(
        f"Loaded {len(data)} entries from {len(matched)} file(s) "
        f"(removed {original_len - len(data)} errors)"
    )

    gt_lookup = load_kaldi_text(gt_file) if gt_file is not None else None

    if language_field is not None:
        assert (
            language_field in next(iter(data.values()))["passthrough"]
        ), f"Language field '{language_field}' not found in prediction file."
        all_languages = sorted(
            {item["passthrough"][language_field] for item in data.values()}
        )
    else:
        all_languages = ["combined"]

    print(f"Found {len(all_languages)} languages: {all_languages}")
    return_data = {}
    for lang in tqdm(all_languages, desc="Loading predictions"):
        D = {}
        for item in data.values():
            if item["passthrough"].get(language_field, "combined") != lang:
                continue
            utt_id = item["passthrough"][key_field]
            prediction = item["pred"][0][pred_field]
            if gt_lookup is not None and utt_id in gt_lookup:
                transcription = gt_lookup[utt_id]
            elif not noisy_pr:
                transcription = item["passthrough"][gt_field]
            else:
                transcription = "".join(
                    n for n in item["passthrough"]["masked_phones"] if n != "[NOISE]"
                )
            D[utt_id] = {"prediction": prediction, "transcription": transcription}
        return_data[lang] = D

    if gt_lookup is not None:
        matched = sum(
            1 for ld in return_data.values() for uid in ld if uid in gt_lookup
        )
        total = sum(len(ld) for ld in return_data.values())
        print(f"Ground truth file matched {matched}/{total} utterances")

    return return_data


def add_args(parser: argparse.ArgumentParser) -> None:
    """Add phone recognition evaluation arguments to an argparse parser."""
    parser.add_argument(
        "--prediction_file", required=True, help="Path to prediction JSON file"
    )
    parser.add_argument(
        "--gt_field",
        type=str,
        default="masked_phones",
        help="Ground truth field name in the prediction file",
    )
    parser.add_argument(
        "--gt_file",
        type=str,
        default=None,
        help="Kaldi-style text file to use as ground truth (overrides --gt_field)",
    )
    parser.add_argument(
        "--pred_field",
        type=str,
        default="processed_transcript",
        help="Predicted transcription field name",
    )
    parser.add_argument(
        "--key_field", type=str, default="utt_id", help="Utterance ID field name"
    )
    parser.add_argument(
        "--noisy_pr", action="store_true", help="Evaluate noisy phone recognition"
    )
    parser.add_argument(
        "--output_file", type=str, default=None, help="CSV file to append results to"
    )
    parser.add_argument(
        "--language_field",
        type=str,
        default=None,
        help="Field for per-language evaluation",
    )
    parser.add_argument(
        "--evaluation_name", type=str, help="Name for this evaluation run"
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    add_args(parser)
    args = parser.parse_args()
    if args.output_file:
        assert args.evaluation_name is not None, "Please provide --evaluation_name"

    loaded_predictions = _load_predictions(
        args.prediction_file,
        args.language_field,
        gt_file=args.gt_file,
        key_field=args.key_field,
        gt_field=args.gt_field,
        pred_field=args.pred_field,
        noisy_pr=args.noisy_pr,
    )
    print(
        f"Loaded predictions for {len(loaded_predictions)} languages, "
        f"{sum(len(v) for v in loaded_predictions.values())} utterances."
    )
    for lang, preds in tqdm(loaded_predictions.items(), desc="Evaluating languages"):
        evaluator = PhoneRecognitionEvaluator(normalize_ipa=True)
        summary, _ = evaluator.evaluate(preds)
        PhoneRecognitionEvaluator.pretty_print(summary)
        if args.output_file:
            evaluator.write_to_csv(
                summary, args.evaluation_name, args.output_file, lang
            )
            print(f"Appended results to {args.output_file}")
