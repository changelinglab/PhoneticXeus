import torch
import numpy as np
import pytest
from typing import List

# REFERENCE
from src.espnet_import.error_calculator import ErrorCalculator as ESPnetErrorCalculator
from src.recipe.phone_recognition.error_calculator import (
    ErrorCalculator as CustomErrorCalculator,
)


def setup_calculators(token_list: List[str], blank_id: int):
    """Initializes both calculators with the same parameters."""
    blank_sym = token_list[blank_id]
    space_sym = "<space>" if "<space>" in token_list else " "

    # Initialize ESPnet Reference
    # Note: report_cer=True is required for it to calculate during __call__
    ref_calc = ESPnetErrorCalculator(
        char_list=token_list, sym_space=space_sym, sym_blank=blank_sym, report_cer=True
    )

    # Initialize Optimized Calculator
    device = "cuda" if torch.cuda.is_available() else "cpu"
    opt_calc = CustomErrorCalculator(
        token_list=token_list, blank_id=blank_id, sym_space=space_sym, ignore_id=-1
    )

    return ref_calc, opt_calc, device


@pytest.mark.parametrize("iteration", range(1000))
def test_randomized_fuzzing(iteration):
    """Runs 1,000 randomized tests with varying batch sizes and lengths."""
    # Setup vocab
    token_list = ["<blank>", "<space>", "a", "b", "c", "sh", "th", "p", "t", "k"]
    blank_id = 0
    ref_calc, opt_calc, device = setup_calculators(token_list, blank_id)

    # Random batch parameters
    batch_size = np.random.randint(1, 16)
    max_len_hat = np.random.randint(20, 100)
    max_len_pad = np.random.randint(10, 50)

    # Generate random predictions and targets
    # Indices range from -1 (ignore) to len(token_list)-1
    ys_hat = torch.randint(-1, len(token_list), (batch_size, max_len_hat)).to(device)
    ys_pad = torch.randint(-1, len(token_list), (batch_size, max_len_pad)).to(device)

    # Randomize reference lengths and apply padding mask
    ys_pad_lens = torch.zeros(batch_size, dtype=torch.long).to(device)
    for i in range(batch_size):
        length = np.random.randint(0, max_len_pad + 1)
        ys_pad_lens[i] = length
        ys_pad[i, length:] = -1  # Standard padding index

    # Run Reference (Requires numpy/CPU)
    # is_ctc=True triggers calculate_cer_ctc
    with torch.no_grad():
        ref_cer = ref_calc(ys_hat.cpu().numpy(), ys_pad.cpu().numpy(), is_ctc=True)

        # Run Optimized (GPU/Native)
        opt_cer = opt_calc(ys_hat, ys_pad, ys_pad_lens)
        if isinstance(opt_cer, torch.Tensor):
            opt_cer = opt_cer.item()

    # Assert equality with tolerance for float precision
    assert opt_cer == pytest.approx(ref_cer if ref_cer is not None else 0.0, abs=1e-6)


def test_hard_edge_cases():
    """Specific cases that often break CTC logic."""
    token_list = ["<blank>", "<space>", "a", "b", "c"]
    blank_id = 0
    ref_calc, opt_calc, device = setup_calculators(token_list, blank_id)

    cases = [
        {
            "name": "Repeats separated by blanks (Should NOT collapse)",
            "hat": [[2, 0, 2]],  # a, blank, a -> "aa"
            "pad": [[2, 2]],  # "aa"
            "lens": [2],
            "expected": 0.0,
        },
        {
            "name": "Consecutive repeats (Should collapse)",
            "hat": [[2, 2, 2, 0, 3]],  # a, a, a, blank, b -> "ab"
            "pad": [[2, 3]],  # "ab"
            "lens": [2],
            "expected": 0.0,
        },
        {
            "name": "Empty reference (Should return 0.0)",
            "hat": [[2, 3, 4]],
            "pad": [[-1, -1, -1]],
            "lens": [0],
            "expected": 0.0,
        },
        {
            "name": "Mixed padding and blanks",
            "hat": [[0, 0, -1, 2, 2]],  # blank, blank, ignore, a, a -> "a"
            "pad": [[2]],  # "a"
            "lens": [1],
            "expected": 0.0,
        },
    ]

    for case in cases:
        ys_hat = torch.tensor(case["hat"]).to(device)
        ys_pad = torch.tensor(case["pad"]).to(device)
        ys_pad_lens = torch.tensor(case["lens"]).to(device)

        ref_cer = ref_calc(ys_hat.cpu().numpy(), ys_pad.cpu().numpy(), is_ctc=True)
        opt_cer = opt_calc(ys_hat, ys_pad, ys_pad_lens)
        if isinstance(opt_cer, torch.Tensor):
            opt_cer = opt_cer.item()

        assert opt_cer == pytest.approx(
            ref_cer if ref_cer is not None else 0.0, abs=1e-6
        ), f"Failed case: {case['name']}"


if __name__ == "__main__":
    # If running directly without pytest
    print("Running hard edge cases...")
    test_hard_edge_cases()
    print("Running 1000 randomized tests...")
    for i in range(1000):
        test_randomized_fuzzing(i)
        if i % 100 == 0:
            print(f"Progress: {i}/10")
    print("All tests passed!")
