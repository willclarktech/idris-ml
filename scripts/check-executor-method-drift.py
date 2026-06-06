#!/usr/bin/env python3
"""check-executor-method-drift — guard against method-set drift between
the three built-in Executor backends (`Tape` / `Torch` / `Mlx`).

Parses `Executor/{Tape,Torch,Mlx}.idr` and extracts the set of method
names implemented in each `UserExecutor*` instance block. Reports any
method present in one backend but not all three.

The line-level DRY-up of these files is cosmetic; this drift gate is the
real protection against the bug class where someone wires a new method
into one backend's instance block and silently forgets the others.

Exit codes:
  0 — no drift detected
  1 — drift found (one or more methods present in <3 backends)
  2 — parse error

CI wiring: invoked by `make check-executor-method-drift`, run in the
existing `make check-*` group alongside `check-rename-headers` and
`check-ffi-wrap-template`.
"""

from __future__ import annotations

import os
import re
import sys
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
EXECUTOR_FILES = {
    "tape": ROOT / "packages/idris-ml/src/Executor/Tape.idr",
    "torch": ROOT / "packages/idris-ml/src/Executor/Torch.idr",
    "mlx": ROOT / "packages/idris-ml/src/Executor/Mlx.idr",
}

# Methods that are legitimately allowed to differ. Empty for now; populate
# only when a real divergence is intentional (e.g. one backend has a
# kernel the other two can't support).
ALLOWLIST: set[tuple[str, str]] = set()  # (slice, method) pairs

# Slices where backends opt in independently — drift across backends is
# expected and intentional.
#
# Two kinds of slices live here:
#   * **Opt-in interfaces** (`Streamed`, `MemoryHygiene`, `Diagnostics`,
#     ...) — backends with the relevant hardware concept implement them;
#     others don't even declare an instance. Useful only to backends with
#     that concept.
#   * **Default-impl interfaces** (`Optimizations`) — every backend gets
#     working semantics via interface-level defaults; per-backend impls
#     are partial, overriding only the methods they natively accelerate.
#     Drift across overrides is the whole point.
#
# Methods in these slices are not checked for cross-backend presence.
# The drift gate remains intact for mandatory slices (`Core`, `Linear`,
# `NN`, `Conv`, `Autograd`, `ParamRegistry`, `Optimizer`, `Serialize`,
# `Profiling`, `TensorCreate`). Other gates (the per-slice numeric
# verification test for `Optimizations`; the existing example smoke
# gates) cover correctness of the opt-in surfaces.
OPT_IN_SLICES: set[str] = {
    "Optimizations",
    "Streamed",
    "MemoryHygiene",
    "Diagnostics",
}


INTERFACE_HEAD_RE = re.compile(
    r"^(?:\{[^}]*\}\s*->\s*)?"  # optional implicit binder
    r"UserExecutor(?P<slice>\w+)\s+"  # UserExecutorCore / Linear / etc.
    r"\([^)]*\)",  # (TapeExecutor) / (TorchExecutor d) / ...
    re.MULTILINE,
)

INSTANCE_HEAD_RE = re.compile(
    r"^(?:\{[^}]*\}\s*->\s*)?"
    r"UserExecutor(?P<slice>\w+)\s+"
    r"(?P<head>(?:\(\s*\w+(?:\s+\w+|\s+\([^)]*\))?\s*\)|\w+))\s+"
    r"where\s*$",
    re.MULTILINE,
)

METHOD_RE = re.compile(
    r"^\s+"  # indented (inside instance block)
    r"(?P<method>prim\w+|backendTag|hardwareClass|deviceName|deviceStreamTag)"
    r"\b"  # word boundary (not part of larger ident)
    r"[\s\w]*"  # optional args (space-separated names)
    r"=",  # method definition (= for impl, : for sig)
    re.MULTILINE,
)


def parse_methods(text: str) -> dict[str, set[str]]:
    """Return {slice_name → set[method_names]} for the given Executor file."""
    by_slice: dict[str, set[str]] = defaultdict(set)
    pos = 0
    for m in INSTANCE_HEAD_RE.finditer(text):
        slice_name = m.group("slice")
        # Collect methods until the next top-level definition (blank line
        # followed by an unindented line) — a coarse end-of-block heuristic.
        start = m.end()
        end_match = re.search(r"\n(?=\S)", text[start:])
        end = start + end_match.start() if end_match else len(text)
        block = text[start:end]
        for mm in METHOD_RE.finditer(block):
            by_slice[slice_name].add(mm.group("method"))
        pos = end
    return dict(by_slice)


def main() -> int:
    parsed: dict[str, dict[str, set[str]]] = {}
    for backend, path in EXECUTOR_FILES.items():
        if not path.exists():
            print(f"error: missing {path}", file=sys.stderr)
            return 2
        text = path.read_text()
        parsed[backend] = parse_methods(text)

    # Union of all (slice, method) pairs across the three backends.
    all_keys: set[tuple[str, str]] = set()
    for backend_methods in parsed.values():
        for slice_name, methods in backend_methods.items():
            for method in methods:
                all_keys.add((slice_name, method))

    drift: list[tuple[str, str, list[str]]] = []
    for slice_name, method in sorted(all_keys):
        if (slice_name, method) in ALLOWLIST:
            continue
        if slice_name in OPT_IN_SLICES:
            continue
        present = [b for b in EXECUTOR_FILES if method in parsed.get(b, {}).get(slice_name, set())]
        if len(present) < len(EXECUTOR_FILES):
            missing = sorted(set(EXECUTOR_FILES) - set(present))
            drift.append((slice_name, method, missing))

    if drift:
        print(
            "Drift detected: method present in some Executor backends but not all.\n",
            file=sys.stderr,
        )
        for slice_name, method, missing in drift:
            print(
                f"  UserExecutor{slice_name}.{method} missing from: {', '.join(missing)}",
                file=sys.stderr,
            )
        print(f"\nTotal: {len(drift)} drift(s).", file=sys.stderr)
        return 1

    print(f"OK — no method drift across {len(EXECUTOR_FILES)} Executor backends.", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
