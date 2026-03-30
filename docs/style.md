# Python Style Reference

Follows the [Google Python Style Guide](https://google.github.io/styleguide/pyguide.html).
Line length: **80** chars (enforced by Black).

---

## Imports

```python
# Order: __future__ → stdlib → third-party → local.
# Blank line between each group.
from __future__ import annotations

import os
import sys

import numpy as np
import torch

from src.metrics import phone_recognition
```

- Import **modules**, not names: prefer `from pkg import module` where the
  imported name is itself a module.
- `from x import y as z` only for name conflicts or very long module paths.
- Standard abbreviations are fine: `import numpy as np`.
- **No relative imports.** Use full package paths:
  `from src.recipe.phone_recognition import model_module`, not `from . import model_module`.

---

## Naming

| Entity | Style | Example |
|---|---|---|
| Package / Module | `lower_with_under` | `model_module.py` |
| Class / Exception | `CapWords` | `PhoneRecognitionModel`, `ValueError` |
| Function / Method | `lower_with_under()` | `build_inference()` |
| Constant | `CAPS_WITH_UNDER` | `DEFAULT_BEAM_SIZE` |
| Instance var / param | `lower_with_under` | `ctc_weight` |
| Internal (private) | leading `_` | `_parse_gt()`, `_cache` |

- No abbreviations unless domain-standard (`ctc`, `ipa`, `utt` are fine here).
- No type in the name: `phones` not `phone_list`.
- Single-char names only for: loop counters (`i`, `j`), exception (`e`),
  file handle (`f`).

---

## Docstrings

Google format. Mandatory for all public functions and classes.

```python
def align_phones(
    ref_segs: list[str], hyp_segs: list[str]
) -> list[tuple[str, str, str]]:
    """Align hypothesis phones to reference using edit distance.

    Args:
        ref_segs: Reference phone sequence.
        hyp_segs: Hypothesis phone sequence.

    Returns:
        List of (op, ref, hyp) tuples; op is one of C/S/D/I.

    Raises:
        ValueError: If either sequence is empty.
    """
```

- `"""` not `'''`.
- Summary line ends with `.`, `?`, or `!`; fits on one line.
- Generators use `Yields:` not `Returns:`.
- Classes: summary + `Attributes:` section listing public attributes.
- `@override` methods may omit docstring unless behavior materially differs.

---

## Type Annotations

Required for all public functions and complex/error-prone code.

```python
from collections.abc import Mapping, Sequence
from typing import Any, TypeAlias

# Use abstract types for parameters
def evaluate(
    utt_data: Mapping[str, dict[str, str]],
) -> PhoneRecognitionSummary: ...

# Type aliases for complex types
_UttData: TypeAlias = dict[str, dict[str, str]]

# Explicit Optional — never implicit None default without annotation
def foo(x: int | None = None) -> str | None: ...
```

- Don't annotate `self`/`cls`; don't annotate `__init__` return.
- Prefer `collections.abc.Sequence`/`Mapping` over `list`/`dict` for params.
- Use `X | None` (Python 3.10+) instead of `Optional[X]`.

---

## Functions

```python
# No mutable defaults — use None sentinel
def build(items: Sequence[str] | None = None) -> list[str]:
    if items is None:
        items = []

# Not this (list shared across all calls):
def build(items=[]):  # wrong
```

- Prefer functions ≤ 40 lines; extract named helpers for longer logic.
- No `staticmethod` — write a module-level function instead.
- `classmethod` only for named constructors or class-level routines.
- Lambda only for simple one-liners (≤ 60–80 chars); use `def` otherwise.

---

## Exceptions

```python
# Use specific built-ins for programming errors
raise ValueError(f"ctc_weight must be in [0, 1], got {ctc_weight}")
raise TypeError(f"Expected str, got {type(x).__name__}")

# Catch specific types; always re-raise or handle meaningfully
try:
    result = risky_call()
except SpecificError as e:
    logger.warning("Failed: %s", e)
    raise
```

- Never bare `except:` or `except Exception:` unless re-raising or at an
  explicit isolation boundary.
- Minimize code inside `try` blocks.
- Custom exceptions inherit from a built-in type; names end in `Error`.
- `assert` only in tests (may be compiled away).

---

## Strings

```python
# f-strings for interpolation
msg = f"Processing {utt_id} with {len(phones)} phones"

# Logging: %-format (lazy evaluation, no f-string)
logger.info("Processing %s with %d phones", utt_id, len(phones))

# Building strings in loops: join, not +=
result = " ".join(str(p) for p in phones)
```

---

## Boolean and None Checks

```python
if items:           # not: if len(items) > 0
if not items:       # not: if len(items) == 0
if x is None:       # not: if x == None
if x is not None:   # not: if x != None
if flag:            # not: if flag == True
if not flag:        # not: if flag == False
```

- Integers: compare to `0` explicitly (`if count == 0:`), not `if not count:`.

---

## Classes and Properties

```python
class PhoneRecognitionModel:
    """CTC-Attention hybrid phone recognition model.

    Attributes:
        ctc_weight: Interpolation weight for CTC vs attention loss.
    """

    @property
    def device(self) -> torch.device:
        """Current inference device."""
        return self._device
```

- `@property` for attribute access that involves computation.
- Don't hide side effects inside properties.
- Don't add getters/setters for plain pass-through — make the attribute public.
- Avoid mutable global state; module-level constants use `CAPS_WITH_UNDER`.

---

## Files and Resources

```python
# Always use context manager
with open(filepath) as fh:
    for line in fh:
        process(line)
```

---

## Comments

```python
# Explain *why*, not *what* — the code shows what.

# Weights decay after warmup to avoid oscillation near optimum.
scheduler.step()

# TODO: github.com/org/repo/issues/42 - Switch to torchaudio forced_align.
```

- Block comments: full sentences, capital first letter, period at end.
- Inline: two spaces before `#`, one space after.
- `# TODO: <link> - <explanation>` format with a bug/issue reference.

---

## Layout

- Two blank lines between top-level definitions.
- One blank line between methods; between class docstring and first method.
- No blank line immediately after a `def` line.
- One statement per line; no semicolons.
- No backslash continuation — use implicit joining inside `()`:

```python
result = (
    some_long_function_call(arg1, arg2)
    + another_call(arg3)
)
```

- Trailing comma when the closing bracket is on its own line.

---

## Main Block

```python
def main() -> None:
    """Entry point."""
    ...


if __name__ == "__main__":
    main()
```
