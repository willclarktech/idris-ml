"""Compare Idris `Example/HfBertInference` stdout against the
`save_oracle.py`-produced safetensors fixture.

Usage:
    python compare_inference.py <idris_stdout_file> <oracle.safetensors> [tol] [--argmax-match]

Optional `--argmax-match` adds a stricter check: argmax(idris) must
equal argmax(oracle). Useful for LM-style gates where the absolute
tolerance is loose (BF16 + many-layer accumulation noise) but the
top-1 prediction is what semantically matters.

Exit codes:
    0  max-abs-diff < tol (and argmax matches if --argmax-match) — passing
    1  shape, value, or argmax mismatch — failing
"""

from __future__ import annotations

import sys
from pathlib import Path

from safetensors.torch import load_file


def main() -> None:
    args = sys.argv[1:]
    check_argmax = "--argmax-match" in args
    args = [a for a in args if a != "--argmax-match"]
    if len(args) < 2:
        print(__doc__, file=sys.stderr)
        sys.exit(2)
    stdout_path = Path(args[0])
    oracle_path = Path(args[1])
    tol = float(args[2]) if len(args) > 2 else 1e-2

    # Idris dumped one float per line to stdout. Filter out `[stage] ...`
    # diagnostic lines (added by stageStamp in the HF inference examples
    # for perf-log JSONL parsing) — they are non-numeric.
    with stdout_path.open() as f:
        idris_vals = [
            float(line.strip())
            for line in f
            if line.strip() and not line.lstrip().startswith("[stage]")
        ]

    oracle_tensor = load_file(str(oracle_path))["output"]
    oracle_vals = oracle_tensor.tolist()

    n_idris = len(idris_vals)
    n_oracle = len(oracle_vals)
    if n_idris != n_oracle:
        print(
            f"FAIL: length mismatch (idris={n_idris}, oracle={n_oracle})",
            file=sys.stderr,
        )
        sys.exit(1)

    diffs = [abs(a - b) for a, b in zip(idris_vals, oracle_vals)]
    max_diff = max(diffs)
    max_idx = diffs.index(max_diff)

    if max_diff > tol:
        print(
            f"FAIL: max-abs-diff {max_diff:.6e} > tol {tol:.6e}",
            file=sys.stderr,
        )
        print(f"  worst at index {max_idx}:", file=sys.stderr)
        print(f"    idris  = {idris_vals[max_idx]:+.10f}", file=sys.stderr)
        print(f"    oracle = {oracle_vals[max_idx]:+.10f}", file=sys.stderr)
        print(f"  first 5 idris:  {idris_vals[:5]}", file=sys.stderr)
        print(f"  first 5 oracle: {oracle_vals[:5]}", file=sys.stderr)
        sys.exit(1)

    if check_argmax:
        idris_argmax = max(range(n_idris), key=lambda i: idris_vals[i])
        oracle_argmax = max(range(n_oracle), key=lambda i: oracle_vals[i])
        if idris_argmax != oracle_argmax:
            print(
                f"FAIL: argmax mismatch (idris={idris_argmax}, oracle={oracle_argmax})",
                file=sys.stderr,
            )
            print(f"  idris[{idris_argmax}]  = {idris_vals[idris_argmax]:+.6f}",
                  file=sys.stderr)
            print(f"  oracle[{oracle_argmax}] = {oracle_vals[oracle_argmax]:+.6f}",
                  file=sys.stderr)
            sys.exit(1)
        print(f"PASS: max-abs-diff {max_diff:.6e} < tol {tol:.6e}  "
              f"argmax matches ({idris_argmax})")
    else:
        print(f"PASS: max-abs-diff {max_diff:.6e} < tol {tol:.6e}")


if __name__ == "__main__":
    main()
