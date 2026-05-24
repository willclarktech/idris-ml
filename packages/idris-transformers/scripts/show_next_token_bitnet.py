"""Decode the next predicted token from `Example/HfBitNetInference --dump-logits`.

Reads the Idris example's stdout (one float per line for the 128256
last-position logits, interleaved with `[stage] ...` diagnostic lines),
argmax's, and decodes the resulting token id via the local HF tokenizer
for `microsoft/bitnet-b1.58-2B-4T`. Echoes the stage lines live to
stdout so the user still sees progress while the example runs; the
flood of 128256 floats is suppressed.

Usage (typically piped from the Idris example):

    ./build/.../hf-bitnet-inference --dump-logits \\
      | uv run python scripts/show_next_token_bitnet.py

Or with a captured file:

    uv run python scripts/show_next_token_bitnet.py <captured-stdout>

Output (final two lines):

    Argmax token id: <int>
    Next token:      <decoded text>
"""

from __future__ import annotations

import sys
from pathlib import Path

# Avoid heavy ML imports until we need to decode — tokenizer load takes
# ~100ms and we want stage-echo to feel instant.
SCRIPT_DIR  = Path(__file__).resolve().parent
REPO_ROOT   = SCRIPT_DIR.parent.parent.parent
MODEL_DIR   = REPO_ROOT / "models" / "microsoft" / "bitnet-b1.58-2B-4T"
EXPECTED_VOCAB = 128256


def parse_stream(stream):
    """Read line-by-line; echo [stage] lines live, accumulate floats.
    Returns (argmax_token_id, count_of_floats)."""
    best_v = float("-inf")
    best_i = -1
    n = 0
    for line in stream:
        s = line.rstrip("\n")
        if not s.strip():
            continue
        if s.lstrip().startswith("[") or s.startswith(" "):
            # stage line, ERR line, anything bracket-prefixed —
            # just pass through to the user's terminal
            print(s, flush=True)
            continue
        # Try to parse as float; if it doesn't parse, pass through.
        try:
            v = float(s)
        except ValueError:
            print(s, flush=True)
            continue
        if v > best_v:
            best_v = v
            best_i = n
        n += 1
    return best_i, n


def main() -> int:
    if len(sys.argv) >= 2 and sys.argv[1] not in ("-", "/dev/stdin"):
        path = Path(sys.argv[1])
        if not path.is_file():
            print(f"ERR: {path} not found", file=sys.stderr)
            return 1
        with path.open("r") as fh:
            argmax_id, n_floats = parse_stream(fh)
    else:
        argmax_id, n_floats = parse_stream(sys.stdin)

    if argmax_id < 0:
        print("ERR: no float lines found in input", file=sys.stderr)
        return 1
    if n_floats != EXPECTED_VOCAB:
        print(f"WARN: expected {EXPECTED_VOCAB} logits, got {n_floats}",
              file=sys.stderr)

    # Tokenizer load deferred until after we've consumed the stream so
    # the [stage] echoes feel responsive.
    if not MODEL_DIR.is_dir():
        print(f"ERR: {MODEL_DIR} not found — fetch via the hf-download script first",
              file=sys.stderr)
        return 1
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(str(MODEL_DIR))
    text = tok.decode([argmax_id], skip_special_tokens=False,
                       clean_up_tokenization_spaces=False)
    print()
    print(f"Argmax token id: {argmax_id}")
    print(f"Next token:      {text!r}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
