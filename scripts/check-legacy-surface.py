#!/usr/bin/env python3
"""Ratchet gate: the legacy `Layer/`/`Network`/`epoch*`/`DataPoint`/`native*`-optimizer surface must stay deleted.

The v1 API rework replaced the legacy training surface with the `Nn`
models-as-records surface (`Module`/`Params`/`Seq`), the `fit` driver,
and `Dataset`/`DataStream`. The transitional scaffolding's concrete
collapse (per `feedback_no_backcompat`) deleted the old modules. This
gate keeps them gone: zero code references to the legacy identifiers,
zero imports of the deleted modules, across every package's `src`.

Comments are stripped before matching, so historical prose references
in kept modules (`Fit.idr`, `Train/Engine.idr`, `Nn/Module.idr` all
explain what they replaced) don't trip the gate — only live code does.

Run: python3 scripts/check-legacy-surface.py
"""

import glob
import re
import sys

SRC_GLOBS = [
    "packages/idris-ml/src/**/*.idr",
    "packages/idris-ml-examples/src/**/*.idr",
    "packages/idris-ml-notebook/src/**/*.idr",
    "packages/idris-transformers/src/**/*.idr",
]

# Legacy code identifiers (the deleted surface). LayerAny family has no
# leading word boundary because the prefix varies (linearLayerAny, …).
IDENT = re.compile(
    r"\w*LayerAny"
    r"|\bOutputLayer\b"
    r"|\bAnyLayer\b"
    r"|\bLayerLike\b"
    r"|\bepochVar\w*"
    r"|\bepochRecurrentVar\b"
    r"|\bepochTwoPhaseVar\b"
    r"|\brunTraining\w*"
    r"|\bforwardVar\w*"
    r"|\bforwardTwoPhase\b"
    r"|\bRecurrentDataPoint\b"
    r"|\bTwoPhaseDataPoint\b"
    r"|\bTensorDataPoint\b"
    r"|\bDataPoint\b"
    r"|\bDataLoader\b"
    # Superseded `native*` optimizer constructors — replaced by the typed
    # `sgd`/`rmsprop`/`adam`/`adamW` (Optimizer.idr) + `Train.Freeze` scoping.
    # `nativeAdam\w*` covers GlobalClip/Group/W; `nativeTrainStepScaled` (the
    # mixed-precision step prim wrapper, kept) is deliberately not matched.
    r"|\bnativeSgd\b"
    r"|\bnativeRmsprop\b"
    r"|\bnativeAdam\w*"
)

# Imports of the deleted modules (Layer barrel, Layer.*, Backprop, …).
IMPORT = re.compile(
    r"^\s*import\s+(?:public\s+)?"
    r"(?:Layer(?:\.\w+)*|Backprop|Curriculum|DataPoint|DataLoader)\b"
)

# `--` line/inline comment (preceded by start or whitespace) and `|||` docs.
LINE_COMMENT = re.compile(r"(^|\s)--.*$")
DOC_COMMENT = re.compile(r"^\s*\|\|\|")


def strip_comments(line: str) -> str:
    if DOC_COMMENT.match(line):
        return ""
    return LINE_COMMENT.sub(r"\1", line)


def main() -> int:
    failures: list[str] = []
    for pattern in SRC_GLOBS:
        for path in sorted(glob.glob(pattern, recursive=True)):
            with open(path) as f:
                in_block = False
                for n, raw in enumerate(f, 1):
                    # crude {- … -} block-comment skip (whole-line spans only)
                    if in_block:
                        if "-}" in raw:
                            in_block = False
                        continue
                    if raw.lstrip().startswith("{-") and "-}" not in raw:
                        in_block = True
                        continue
                    line = strip_comments(raw)
                    m = IMPORT.search(line) or IDENT.search(line)
                    if m:
                        ref = m.group(0).strip()
                        failures.append(f"{path}:{n}: legacy surface reference `{ref}`")
    if failures:
        for msg in failures:
            print(f"FAIL: {msg}", file=sys.stderr)
        print(
            f"\n{len(failures)} legacy-surface reference(s) — the Layer/epoch*/DataPoint/native* "
            "surface is deleted; use Nn/fit/Dataset + sgd/adam/adamW instead.",
            file=sys.stderr,
        )
        return 1
    print("legacy-surface ratchet OK (no Layer/epoch*/DataPoint/native* references)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
