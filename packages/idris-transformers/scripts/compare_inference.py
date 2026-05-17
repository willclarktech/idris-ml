"""Compare Idris `Example/HfBertInference` stdout against the
`save_oracle.py`-produced safetensors fixture.

Usage:
    python compare_inference.py <idris_stdout_file> <oracle.safetensors> [tol]

Exit codes:
    0  max-abs-diff < tol — passing
    1  shape or value mismatch — failing
"""

from __future__ import annotations

import sys
from pathlib import Path

from safetensors.torch import load_file


def main() -> None:
    if len(sys.argv) < 3:
        print(__doc__, file=sys.stderr)
        sys.exit(2)
    stdout_path = Path(sys.argv[1])
    oracle_path = Path(sys.argv[2])
    tol = float(sys.argv[3]) if len(sys.argv) > 3 else 1e-2

    # Idris dumped one float per line to stdout.
    with stdout_path.open() as f:
        idris_vals = [float(line.strip()) for line in f if line.strip()]

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

    print(f"PASS: max-abs-diff {max_diff:.6e} < tol {tol:.6e}")


if __name__ == "__main__":
    main()
