#!/usr/bin/env python3
"""Targeted `d` → `ex` rename for the Executor type-parameter binder.

Goal: every place `d` was bound as `Executor` becomes `ex`. We use `ex`
rather than `e` to free `d` AND `e` for use as positional binders
elsewhere in the codebase (the original code overloaded `e` for 4th-dim
Nat in tparam4dNormal etc., which clashed with a `d → e` rename).

We can't safely do a global `\\bd\\b` substitution — `d` is used as a
value binding in many local contexts (`let d = abs (a - b)`, lambda
parameters, pattern matches). So instead we target a curated set of
high-confidence patterns:

  1. Binder shapes: `(0 d : Executor)`, `{d : Executor}`, etc.
  2. Typeclass constraint shapes: `Linked d`, `Compatible d dt`,
     `UserExecutor* d`, `HardwareClassed d`.
  3. Named applications: `{d}` and `{d=...}` for Executor-pinned slots.
  4. Tensor / Network / record-application shapes: `Tensor [..] d dt g`
     and `Tensor [..] d dt g`-style applications.

After running, build the project and fix any remaining unbound-d errors
manually — those are the rare local-binding cases the script
deliberately skipped.

Run from repo root: `python3 scripts/rename_d_to_ex.py`.
"""
from __future__ import annotations

import os
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
os.chdir(ROOT)


PATTERNS: list[tuple[str, str]] = [
    # ---- Binder shapes ----
    (r"\(\s*0\s+d\s*:\s*Executor\s*\)",        "(0 ex : Executor)"),
    (r"\{\s*0\s+d\s*:\s*Executor\s*\}",        "{0 ex : Executor}"),
    (r"\(\s*d\s*:\s*Executor\s*\)",            "(ex : Executor)"),
    (r"\{\s*d\s*:\s*Executor\s*\}",            "{ex : Executor}"),
    (r"\(\s*auto\s+0\s+d\s*:\s*Executor\s*\)", "(auto 0 ex : Executor)"),
    (r"\{\s*auto\s+0\s+d\s*:\s*Executor\s*\}", "{auto 0 ex : Executor}"),

    # ---- Typeclass constraints (binder usage in signatures). ----
    (r"\bLinked\s+d\b",                        "Linked ex"),
    (r"\bCompatible\s+d\s+(?=\w)",             "Compatible ex "),
    (r"\bUserExecutorCore\s+d\b",              "UserExecutorCore ex"),
    (r"\bUserExecutorLinear\s+d\b",            "UserExecutorLinear ex"),
    (r"\bUserExecutorNN\s+d\b",                "UserExecutorNN ex"),
    (r"\bUserExecutorConv\s+d\b",              "UserExecutorConv ex"),
    (r"\bUserExecutorTraining\s+d\b",          "UserExecutorTraining ex"),
    (r"\bUserExecutorTransfer\s+d\b",          "UserExecutorTransfer ex"),
    (r"\bUserExecutorQuant\s+d\b",             "UserExecutorQuant ex"),
    (r"\bHardwareClassed\s+d\b",               "HardwareClassed ex"),

    # ---- Named application syntax {d=…} pinning Executor slots. ----
    (r"\{d\s*=\s*TapeExecutor(\s*\})",                                 r"{ex=TapeExecutor\1"),
    (r"\{d\s*=\s*TorchExecutor\s+(TCpu|TMps|\(TCuda\s+\w+\))(\s*\})",  r"{ex=TorchExecutor \1\2"),
    (r"\{d\s*=\s*MlxExecutor\s+(MCpu|MGpu)(\s*\})",                    r"{ex=MlxExecutor \1\2"),
    (r"\{d\s*=\s*TestExecutor(\s*\})",                                 r"{ex=TestExecutor\1"),
    (r"\{d\s*=\s*ExampleExecutor(\s*\})",                              r"{ex=ExampleExecutor\1"),
    (r"\{d\s*=\s*MlxCpu(\s*\})",                                       r"{ex=MlxCpu\1"),
    (r"\{d\s*=\s*MlxGpu(\s*\})",                                       r"{ex=MlxGpu\1"),
    # Generic {d = <ident>} catch-all for local variables.
    (r"\{d\s*=\s*(\w+)\s*\}",                  r"{ex=\1}"),

    # ---- Plain {d} named application. ----
    (r"\{d\}",                                 "{ex}"),

    # ---- Tensor / Network / record-state shape applications. ----
    (r"\bTensor(\s+\[[^\]]*\])\s+d\s+(\w+)\s+(\w+)\b", r"Tensor\1 ex \2 \3"),
    (r"\bTensor\s+d\s+(\w+)\s+(\w+)\b",                r"Tensor ex \1 \2"),
    (r"\bNetwork\s+(\w+)\s+(\w+)\s+(\w+)\s+d\s+(\w+)\b",
     r"Network \1 \2 \3 ex \4"),
    (r"\b([A-Z]\w*State)(\s+\w[\w\s]*)\s+d\s+(\w+)\s+(\w+)\b",
     r"\1\2 ex \3 \4"),
]


TEXT_GLOBS = [
    "packages/**/*.idr",
    "packages/**/*.ipkg",
    "packages/**/*.in",
]


EXCLUDE_PATHS = {
    "scripts/rename_d_to_ex.py",
    "scripts/rename_device_to_executor.py",
}


def discover_files() -> list[Path]:
    files: set[Path] = set()
    for pattern in TEXT_GLOBS:
        for p in ROOT.glob(pattern):
            if p.is_file() and str(p.relative_to(ROOT)) not in EXCLUDE_PATHS:
                files.add(p)
    return sorted(files)


def rewrite_file(path: Path) -> tuple[int, bool]:
    original = path.read_text()
    new = original
    total = 0
    for pat, repl in PATTERNS:
        new, n = re.subn(pat, repl, new)
        total += n
    if new != original:
        path.write_text(new)
        return total, True
    return 0, False


def main() -> int:
    files = discover_files()
    print(f"Scanning {len(files)} files…", file=sys.stderr)

    touched: list[Path] = []
    total = 0
    for f in files:
        n, did = rewrite_file(f)
        total += n
        if did:
            touched.append(f)

    print(f"Substitutions: {total}; files touched: {len(touched)}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
