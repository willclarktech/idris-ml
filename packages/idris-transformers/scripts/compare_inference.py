"""Compare Idris `Example/HfBertInference` stdout against the
`save_oracle.py`-produced safetensors fixture.

Usage:
    python compare_inference.py <idris_stdout_file> <oracle.safetensors> [tol] [--argmax-match]
    python compare_inference.py <idris_stdout_file> <oracle.safetensors> --token-sequence

Float modes (default):
- Idris dumps one float per line. Compare element-wise against the
  oracle's `output` key, asserting max-abs-diff < tol.
- Optional `--argmax-match` adds a stricter check: argmax(idris) must
  equal argmax(oracle). Useful for LM-style gates where the absolute
  tolerance is loose (BF16 + many-layer accumulation noise) but the
  top-1 prediction is what semantically matters.

Sequence mode (`--token-sequence`):
- Idris dumps one *integer token id* per line. Compare element-wise
  against the oracle's `token_ids` key (int64 tensor), asserting
  exact equality. No tolerance — tokens are discrete.
- Used by `test-hf-llama-generate-roundtrip` (and any future
  multi-token generation gate) to catch drift accumulating across
  greedy decode steps that single-forward gates can't see.

Exit codes:
    0  comparison passes — float-mode max-abs-diff < tol (and argmax
       matches if --argmax-match), or sequence-mode element-wise
       integer equality
    1  shape, value, argmax, or sequence-element mismatch
"""

from __future__ import annotations

import sys
from pathlib import Path

# safetensors' stub types load_file's path parameter as PathLike[Unknown];
# the return (Dict[str, Tensor]) is fully typed, so call sites are safe.
from safetensors.torch import load_file  # pyright: ignore[reportUnknownVariableType]


def _read_lines_filtered(path: Path) -> list[str]:
    """Idris stdout is one value per line plus stage/diagnostic lines
    that must be filtered out: `[stage] ...`, `[perf] ...` (added by
    perf-run.sh op-count probes), and the human-facing banner lines
    emitted by the example wrappers."""
    out: list[str] = []
    with path.open() as f:
        for raw in f:
            line = raw.strip()
            if not line:
                continue
            stripped = line.lstrip()
            if stripped.startswith("[stage]"):
                continue
            if stripped.startswith("[perf]"):
                continue
            out.append(line)
    return out


def _run_token_sequence(stdout_path: Path, oracle_path: Path) -> None:
    """Sequence-mode comparator. Reads Nat per line from Idris stdout;
    asserts element-wise equality against the oracle's `token_ids`
    int64 tensor."""
    raw_lines = _read_lines_filtered(stdout_path)
    # Idris dumps token IDs as Nats. Anything that isn't a parseable
    # int is a wrapping-text line the filter didn't catch — fail loudly
    # rather than silently dropping it (mid-run drift gives garbage on
    # one line and the comparator would otherwise hide the failure).
    idris_ids: list[int] = []
    for line in raw_lines:
        try:
            idris_ids.append(int(line))
        except ValueError:
            print(
                f"FAIL: unparseable line in {stdout_path}: {line!r}\n"
                f"  Expected one integer token id per line. "
                f"Diagnostic lines should be prefixed `[stage]` / `[perf]`.",
                file=sys.stderr,
            )
            sys.exit(1)

    tensors = load_file(str(oracle_path))
    if "token_ids" not in tensors:
        print(
            f"FAIL: oracle {oracle_path} missing 'token_ids' key; keys: {list(tensors)}",
            file=sys.stderr,
        )
        sys.exit(1)
    # Tensor.tolist() is typed list[Unknown] in torch's stubs; these are int64 token ids.
    oracle_ids: list[int] = tensors["token_ids"].tolist()  # pyright: ignore[reportUnknownVariableType, reportUnknownMemberType]

    n_idris = len(idris_ids)
    n_oracle = len(oracle_ids)
    if n_idris != n_oracle:
        print(
            f"FAIL: length mismatch (idris={n_idris}, oracle={n_oracle})\n"
            f"  idris  ids: {idris_ids}\n"
            f"  oracle ids: {oracle_ids}",
            file=sys.stderr,
        )
        sys.exit(1)

    for i, (a, b) in enumerate(zip(idris_ids, oracle_ids, strict=True)):
        if a != b:
            print(
                f"FAIL: token sequence mismatch at position {i}: idris={a} oracle={b}",
                file=sys.stderr,
            )
            print(f"  idris  ids: {idris_ids}", file=sys.stderr)
            print(f"  oracle ids: {oracle_ids}", file=sys.stderr)
            sys.exit(1)

    print(f"PASS: token sequence matches ({n_idris} tokens)")
    print(f"  ids: {idris_ids}")


def main() -> None:
    args = sys.argv[1:]
    token_sequence = "--token-sequence" in args
    args = [a for a in args if a != "--token-sequence"]
    check_argmax = "--argmax-match" in args
    args = [a for a in args if a != "--argmax-match"]
    if len(args) < 2:
        print(__doc__, file=sys.stderr)
        sys.exit(2)
    stdout_path = Path(args[0])
    oracle_path = Path(args[1])

    if token_sequence:
        if check_argmax:
            print(
                "FAIL: --token-sequence and --argmax-match are mutually exclusive",
                file=sys.stderr,
            )
            sys.exit(2)
        _run_token_sequence(stdout_path, oracle_path)
        return

    tol = float(args[2]) if len(args) > 2 else 1e-2

    raw_lines = _read_lines_filtered(stdout_path)
    idris_vals = [float(line) for line in raw_lines]

    oracle_tensor = load_file(str(oracle_path))["output"]
    # Tensor.tolist() is typed list[Unknown] in torch's stubs; the oracle output is float.
    oracle_vals: list[float] = oracle_tensor.tolist()  # pyright: ignore[reportUnknownVariableType, reportUnknownMemberType]

    n_idris = len(idris_vals)
    n_oracle = len(oracle_vals)
    if n_idris != n_oracle:
        print(
            f"FAIL: length mismatch (idris={n_idris}, oracle={n_oracle})",
            file=sys.stderr,
        )
        sys.exit(1)

    diffs = [abs(a - b) for a, b in zip(idris_vals, oracle_vals, strict=True)]
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
            print(f"  idris[{idris_argmax}]  = {idris_vals[idris_argmax]:+.6f}", file=sys.stderr)
            print(f"  oracle[{oracle_argmax}] = {oracle_vals[oracle_argmax]:+.6f}", file=sys.stderr)
            sys.exit(1)
        print(f"PASS: max-abs-diff {max_diff:.6e} < tol {tol:.6e}  argmax matches ({idris_argmax})")
    else:
        print(f"PASS: max-abs-diff {max_diff:.6e} < tol {tol:.6e}")


if __name__ == "__main__":
    main()
