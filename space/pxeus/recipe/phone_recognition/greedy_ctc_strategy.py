import torch
from typing import List, Dict, Any, Union


def ctc_collapse_vectorized(
    ids: torch.Tensor, blank_id: int, ignore_id: int = -1
) -> List[List[int]]:
    """Optimized CTC collapse for batch tensors."""
    mask = torch.ones_like(ids, dtype=torch.bool)
    mask[:, 1:] = ids[:, 1:] != ids[:, :-1]
    mask &= ids != blank_id
    if ignore_id != -1:
        mask &= ids != ignore_id

    return [ids[i][mask[i]].tolist() for i in range(ids.size(0))]


class GreedyCTCInference:
    """A scalable inference engine for any CTC-based phone recognizer."""

    def __init__(self, token_list: List[str], blank_id: int):
        self.token_list = token_list
        self.blank_id = blank_id

    @torch.no_grad()
    def __call__(
        self,
        model: torch.nn.Module,
        speech: torch.Tensor,
        speech_lengths: torch.Tensor,
        **kwargs
    ) -> List[Dict[str, Any]]:
        # 1. Standardized Forward pass
        # Works as long as model has .encode() and .ctc
        encoder_out, _ = model.encode(speech, speech_lengths)
        if isinstance(encoder_out, tuple):
            encoder_out = encoder_out[0]
        logits = model.ctc.ctc_lo(encoder_out)

        # 2. Greedy search
        y_hat = torch.argmax(logits, dim=-1)

        # 3. Collapse
        collapsed_ids = ctc_collapse_vectorized(y_hat, self.blank_id)

        # 4. Map to text
        results = []
        for ids in collapsed_ids:
            tokens = [self.token_list[i] for i in ids]
            raw_text = "/".join(tokens)
            # Filter special tokens
            clean_tokens = [
                t for t in tokens if not (t.startswith("<") and t.endswith(">"))
            ]
            processed = "".join(clean_tokens).strip()  # replace(self.sym_space, " ")

            results.append(
                {
                    "processed_transcript": processed,
                    "predicted_transcript": raw_text,
                }
            )
        return results
