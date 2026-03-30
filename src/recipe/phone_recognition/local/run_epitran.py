"""Read a Kaldi-style text file and output IPA phoneme sequences using epitran.

Usage:
    python -m src.recipe.phone_recognition.local.run_epitran \
          path/to/text.good --output exp/data/epitran_outputs/gmuaccent.epitran

given two files, each in kaldi style and containing phone sequences for same uttterances (reference and epitran) use the above code (from src.metrics.phone_recognition) to align them based on edit distance such that cost of insertion and deletion is more than the cost of substitution of related phones (you can check if panphon does this in some way, you can reuse the function). After this create a mapping of phones from epitran to their most substituted counterpart phone in the reference.
"""

import argparse
import sys
import epitran
from tqdm import tqdm


def load_kaldi_text(filepath: str) -> list[tuple[str, str]]:
    """Load a Kaldi-style text file.

    Returns a list of (utterance_id, transcript) tuples.
    """
    utterances = []
    with open(filepath, "r", encoding="utf-8") as f:
        for lineno, line in enumerate(f, 1):
            line = line.strip()
            parts = line.split(maxsplit=1)
            utt_id = parts[0]
            transcript = parts[1] if len(parts) > 1 else ""
            # transcript = "please call stella ask her to bring these things with her from the store six spoons of fresh snow peas five thick slabs of blue cheese and maybe a snack for her brother bob we also need a small plastic snake and a big toy frog for the kids she can scoop these things into three red bags and we will go meet her wednesday at the train station"
            utterances.append((utt_id, transcript.strip()))
    return utterances


def text_to_phonemes(epi: epitran.Epitran, text: str, delimiter: str = "") -> str:
    """Convert text to a space-delimited IPA phoneme sequence."""
    words = text.strip().split()
    word_phonemes = []
    for word in words:
        ipa = epi.transliterate(word)
        word_phonemes.append(ipa)
    return delimiter.join(word_phonemes)


def main():
    parser = argparse.ArgumentParser(
        description="Convert Kaldi text file to IPA phoneme sequences via epitran."
    )
    parser.add_argument("input", help="Path to Kaldi-style text file")
    parser.add_argument(
        "--output", nargs="?", default=None, help="Output file (default: stdout)"
    )
    parser.add_argument(
        "--lang", default="eng-Latn", help="Epitran language code (default: eng-Latn)"
    )
    parser.add_argument("--delimiter", default="", help="Delimiter between words")
    args = parser.parse_args()

    print(f"Loading epitran for language: {args.lang}", file=sys.stderr)
    epi = epitran.Epitran(args.lang)

    utterances = load_kaldi_text(args.input)
    print(utterances[:5])  # Debug: print first 5 utterances
    print(f"Loaded {len(utterances)} utterances from {args.input}", file=sys.stderr)

    out = open(args.output, "w", encoding="utf-8") if args.output else sys.stdout

    for utt_id, transcript in tqdm(utterances):
        phoneme_seq = text_to_phonemes(epi, transcript, delimiter=args.delimiter)
        out.write(f"{utt_id} {phoneme_seq}\n")

    if args.output:
        out.close()
        print(f"Written to {args.output}", file=sys.stderr)


if __name__ == "__main__":
    main()
