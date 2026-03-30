# ~2-3x Faster error calculator!
import torch
import numpy as np
from rapidfuzz.distance import Levenshtein
from typing import List
from src.metrics.phone_recognition import PhoneRecognitionEvaluator


class ErrorCalculator:
    def __init__(
        self,
        token_list: List[str],
        blank_id: int,
        sym_space: str = "<space>",
        ignore_id: int = -1,
        log_phone_metrics=True,
    ):
        self.token_list = np.array(token_list)
        self.blank_id = blank_id
        self.ignore_id = ignore_id
        self.idx_space = (
            token_list.index(sym_space) if sym_space in token_list else None
        )
        self.ignore_set = {ignore_id, blank_id, self.idx_space}
        if log_phone_metrics:
            self.evaluator = PhoneRecognitionEvaluator(normalize_ipa=True)

    def _ctc_collapse_batch(self, x: torch.Tensor):
        if x.numel() == 0:
            return x, torch.zeros(0, device=x.device, dtype=torch.long)

        mask = torch.ones_like(x, dtype=torch.bool)
        mask[:, 1:] = x[:, 1:] != x[:, :-1]

        lengths = mask.sum(1)
        max_len = int(lengths.max().item())
        out = torch.full(
            (x.size(0), max_len), self.ignore_id, device=x.device, dtype=x.dtype
        )

        pos = mask.long().cumsum(1) - 1
        batch_offsets = torch.arange(x.size(0), device=x.device).unsqueeze(1) * max_len
        flat_dest_indices = (pos + batch_offsets)[mask]
        out.view(-1)[flat_dest_indices] = x[mask]

        return out, lengths

    def _ids_to_strs(self, ids_cpu: np.ndarray, lens_cpu: np.ndarray) -> List[str]:
        """Lookup-Table optimized string builder."""
        results = []
        # Optimization: Use a local variable for the array
        t_list = self.token_list
        i_id, b_id, s_id = self.ignore_id, self.blank_id, self.idx_space

        for i in range(len(ids_cpu)):
            row = ids_cpu[i, : lens_cpu[i]]
            # Vectorized filtering via Numpy is much faster than Python 'if' for large rows
            filtered = row[(row != i_id) & (row != b_id) & (row != s_id)]
            results.append("".join(t_list[filtered]))
        return results

    def __call__(
        self,
        ys_hat: torch.Tensor,
        ys_pad: torch.Tensor,
        ys_pad_lens: torch.Tensor,
    ):

        # 1. GPU Collapse
        collapsed_hat, hat_lens = self._ctc_collapse_batch(ys_hat)
        if ys_hat.is_cuda:
            torch.cuda.synchronize()

        # 2. Transfer
        hat_np = collapsed_hat.cpu().numpy()
        hat_lens_np = hat_lens.cpu().numpy()
        ref_np = ys_pad.cpu().numpy()
        ref_lens_np = ys_pad_lens.cpu().numpy()

        # 3. Stringify
        hyps = self._ids_to_strs(hat_np, hat_lens_np)
        refs = self._ids_to_strs(ref_np, ref_lens_np)

        # 4. Edit Distance
        # Filter pairs where ref is not empty
        pairs = [(h, r) for h, r in zip(hyps, refs) if len(r) > 0]

        metrics = {"cer": 0.0, "pfer": 0.0, "per": 0.0}
        if not pairs:
            return metrics

        v_hyp, v_ref = zip(*pairs)
        distances = [Levenshtein.distance(h, r) for h, r in zip(v_hyp, v_ref)]

        total_dist = sum(distances)
        total_ref_len = sum(len(r) for r in v_ref)
        cer = total_dist / total_ref_len
        metrics["cer"] = cer * 100

        if self.evaluator is not None:

            test_data = {
                i: {"prediction": h, "transcription": r}
                for i, (h, r) in enumerate(pairs)
            }
            summary, _ = self.evaluator.evaluate(
                test_data, tqdm_enabled=False,
            )
            metrics["per"] = summary.FER
            metrics["pfer"] = float(summary.PER)
            metrics["ins"] = summary.INS
            metrics["del"] = summary.DEL
            metrics["sub"] = summary.SUB

        return metrics
