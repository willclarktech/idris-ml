#!/usr/bin/env python3
"""Targeted `d` → `ex` rename for the Executor type-parameter binder.

Goal: every place `d` was bound as `Executor` becomes `ex`. We use `ex`
rather than `e` to free `d` AND `e` for use as positional binders
elsewhere in the codebase (the original code overloaded `e` for 4th-dim
Nat in tparam4dNormal etc., which clashed with a `d → e` rename).

We can't safely do a global `\\bd\\b` substitution — `d` is used as a
value binding in many local contexts (`let d = abs (a - b)`, lambda
parameters, monadic `d <- dataSrc`). So instead we target a curated set of
high-confidence patterns that only ever appear in type-expression context.

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


# Each pattern targets a specific Executor-binder usage shape. Patterns
# are applied in order, top to bottom, in a single sweep per file.
PATTERNS: list[tuple[str, str]] = [
    # ------------------------------------------------------------------
    # Binder shapes
    # ------------------------------------------------------------------
    (r"\(\s*0\s+d\s*:\s*Executor\s*\)",        "(0 ex : Executor)"),
    (r"\{\s*0\s+d\s*:\s*Executor\s*\}",        "{0 ex : Executor}"),
    (r"\(\s*d\s*:\s*Executor\s*\)",            "(ex : Executor)"),
    (r"\{\s*d\s*:\s*Executor\s*\}",            "{ex : Executor}"),
    (r"\(\s*auto\s+0\s+d\s*:\s*Executor\s*\)", "(auto 0 ex : Executor)"),
    (r"\{\s*auto\s+0\s+d\s*:\s*Executor\s*\}", "{auto 0 ex : Executor}"),

    # ------------------------------------------------------------------
    # Typeclass constraints (binder usage in signatures)
    # ------------------------------------------------------------------
    (r"\bLinked\s+d\b",                        "Linked ex"),
    (r"\bCompatible\s+d\b",                    "Compatible ex"),
    (r"\bUserExecutorCore\s+d\b",              "UserExecutorCore ex"),
    (r"\bUserExecutorLinear\s+d\b",            "UserExecutorLinear ex"),
    (r"\bUserExecutorNN\s+d\b",                "UserExecutorNN ex"),
    (r"\bUserExecutorConv\s+d\b",              "UserExecutorConv ex"),
    (r"\bUserExecutorTraining\s+d\b",          "UserExecutorTraining ex"),
    (r"\bUserExecutorTransfer\s+d\b",          "UserExecutorTransfer ex"),
    (r"\bUserExecutorQuant\s+d\b",             "UserExecutorQuant ex"),
    (r"\bUserExecutorAutograd\s+d\b",          "UserExecutorAutograd ex"),
    (r"\bUserExecutorParamRegistry\s+d\b",     "UserExecutorParamRegistry ex"),
    (r"\bUserExecutorOptimizer\s+d\b",         "UserExecutorOptimizer ex"),
    (r"\bUserExecutorSerialize\s+d\b",         "UserExecutorSerialize ex"),
    (r"\bUserExecutorProfiling\s+d\b",         "UserExecutorProfiling ex"),
    (r"\bUserExecutorTensorCreate\s+d\b",      "UserExecutorTensorCreate ex"),
    (r"\bUserExecutorInference\s+d\b",         "UserExecutorInference ex"),
    (r"\bHardwareClassed\s+d\b",               "HardwareClassed ex"),
    (r"\bRunsOn\s+d\b",                        "RunsOn ex"),
    (r"\bRunsVia\s+d\b",                       "RunsVia ex"),

    # ------------------------------------------------------------------
    # Type-level `Tensor [..] d dt g` / `Tensor d dt g`
    # ------------------------------------------------------------------
    (r"\bTensor(\s+\[[^\]]*\])\s+d\b",         r"Tensor\1 ex"),
    (r"\bTensor\s+d\s+(\w+)\s+(\w+)\b",        r"Tensor ex \1 \2"),

    # ------------------------------------------------------------------
    # Type aliases TVec / TMat
    # ------------------------------------------------------------------
    (r"\bTVec\s+(\S+)\s+d\b",                  r"TVec \1 ex"),
    (r"\bTMat\s+(\S+)\s+(\S+)\s+d\b",          r"TMat \1 \2 ex"),

    # ------------------------------------------------------------------
    # Network / record-state shape applications
    # ------------------------------------------------------------------
    (r"\bNetwork\s+(\w+)\s+(\w+)\s+(\w+)\s+d\b", r"Network \1 \2 \3 ex"),
    (r"\bAnyLayer\s+d\b",                      "AnyLayer ex"),
    (r"\bGradScaler\s+d\b",                    "GradScaler ex"),
    (r"\bNativeOptimizer\s+d\b",               "NativeOptimizer ex"),
    (r"\bStage\s+d\b",                         "Stage ex"),

    # Capitalized identifier `Foo <one-or-more-word-args> d <dt> <g>` —
    # catches state records like `LinearState i o d dt g`,
    # `BertEmbedding vocab dim d dt g`, `LstmState i h d dt g`, etc.
    (r"\b([A-Z]\w+)((?:\s+\w+)+)\s+d\s+(\w+)\s+(\w+)\b",
     r"\1\2 ex \3 \4"),
    # …and `Foo <args> d <dt>` (two slots only)
    (r"\b([A-Z]\w+)((?:\s+\w+)+)\s+d\s+(\w+)\b",
     r"\1\2 ex \3"),

    # ------------------------------------------------------------------
    # Bare ` d dt g` / ` d <dt> NoGrad|WithGrad` suffix — catches type
    # expressions where the `d` slot is preceded by shape args that
    # contain non-\w characters (e.g. brackets, parens, ::) and the prior
    # capitalized-name pattern can't see past. Highly anchored to the
    # downstream dtype/grad-mode slot so it doesn't fire on value-context
    # bindings.
    # ------------------------------------------------------------------
    (r"\bd\s+dt\s+g\b",                        "ex dt g"),
    (r"\bd\s+dt\s+NoGrad\b",                   "ex dt NoGrad"),
    (r"\bd\s+dt\s+WithGrad\b",                 "ex dt WithGrad"),
    (r"\bd\s+ty\s+g\b",                        "ex ty g"),
    (r"\bd\s+ty\s+NoGrad\b",                   "ex ty NoGrad"),
    (r"\bd\s+ty\s+WithGrad\b",                 "ex ty WithGrad"),
    (r"\bd\s+(F16|F32|F64|BF16|I32|I64|U32|U64|Bool|Ternary)\s+(NoGrad|WithGrad|g)\b",
     r"ex \1 \2"),
    # `d <dtype>` end-of-expression (no trailing grad slot)
    (r"\bd\s+(F16|F32|F64|BF16|I32|I64|U32|U64|Bool|Ternary)\b",
     r"ex \1"),
    # Trailing `d` followed by `NoGrad`/`WithGrad` directly (Tensor [..] d NoGrad style)
    (r"\bd\s+(NoGrad|WithGrad)\b",             r"ex \1"),
    # `d g` at the very tail (Tensor [..] d g, no dtype slot)
    (r"\b(Tensor\s+\[[^\]]*\])\s+d\s+g\b",     r"\1 ex g"),

    # Mixed-precision slot suffix `d pDt cDt g` / `d pDt cDt NoGrad` etc.
    (r"\bd\s+pDt\s+cDt\s+g\b",                 "ex pDt cDt g"),
    (r"\bd\s+pDt\s+cDt\s+NoGrad\b",            "ex pDt cDt NoGrad"),
    (r"\bd\s+pDt\s+cDt\s+WithGrad\b",          "ex pDt cDt WithGrad"),

    # ------------------------------------------------------------------
    # Named application syntax {d=…} pinning Executor slots
    # ------------------------------------------------------------------
    (r"\{d\s*=\s*TapeExecutor(\s*\})",                                 r"{ex=TapeExecutor\1"),
    (r"\{d\s*=\s*TorchExecutor\s+(TCpu|TMps|\(TCuda\s+\w+\))(\s*\})",  r"{ex=TorchExecutor \1\2"),
    (r"\{d\s*=\s*MlxExecutor\s+(MCpu|MGpu)(\s*\})",                    r"{ex=MlxExecutor \1\2"),
    (r"\{d\s*=\s*TestExecutor(\s*\})",                                 r"{ex=TestExecutor\1"),
    (r"\{d\s*=\s*ExampleExecutor(\s*\})",                              r"{ex=ExampleExecutor\1"),
    (r"\{d\s*=\s*MlxCpu(\s*\})",                                       r"{ex=MlxCpu\1"),
    (r"\{d\s*=\s*MlxGpu(\s*\})",                                       r"{ex=MlxGpu\1"),
    # Generic {d = <ident>} catch-all for local variables
    (r"\{d\s*=\s*(\w+)\s*\}",                  r"{ex=\1}"),

    # Plain {d} named application
    (r"\{d\}",                                 "{ex}"),
]


TEXT_GLOBS = [
    "packages/**/*.idr",
    "packages/**/*.ipkg",
    "packages/**/*.in",
]


# Doc files: apply ONLY binder + named-app patterns (not the State/Network
# pattern, which could match general English in prose). The binder shapes
# are anchored enough to be safe in prose.
DOC_GLOBS = [
    "docs/**/*.md",
    "packages/**/*.md",
    "CLAUDE.md",
    "README.md",
]


DOC_PATTERN_PREFIXES = (
    "(0 ex :", "{0 ex :", "(ex :", "{ex :",
    "(auto 0 ex", "{auto 0 ex",
    "Linked ex", "Compatible ex",
    "UserExecutorCore ex", "UserExecutorLinear ex", "UserExecutorNN ex",
    "UserExecutorConv ex", "UserExecutorTraining ex", "UserExecutorTransfer ex",
    "UserExecutorQuant ex", "UserExecutorAutograd ex", "UserExecutorParamRegistry ex",
    "UserExecutorOptimizer ex", "UserExecutorSerialize ex", "UserExecutorProfiling ex",
    "UserExecutorTensorCreate ex", "UserExecutorInference ex",
    "HardwareClassed ex", "RunsOn ex", "RunsVia ex",
    "Tensor", "TVec", "TMat",
    "{ex=", "{ex}",
)


EXCLUDE_PATHS = {
    "scripts/rename_d_to_ex.py",
    "scripts/rename_device_to_executor.py",
    # Plan + memory files (not codebase artifacts):
}


def discover_idris_files() -> list[Path]:
    files: set[Path] = set()
    for pattern in TEXT_GLOBS:
        for p in ROOT.glob(pattern):
            if p.is_file() and str(p.relative_to(ROOT)) not in EXCLUDE_PATHS:
                files.add(p)
    return sorted(files)


def discover_doc_files() -> list[Path]:
    files: set[Path] = set()
    for pattern in DOC_GLOBS:
        for p in ROOT.glob(pattern):
            if p.is_file() and str(p.relative_to(ROOT)) not in EXCLUDE_PATHS:
                files.add(p)
    return sorted(files)


def rewrite_file(path: Path, patterns: list[tuple[str, str]]) -> tuple[int, bool]:
    """Apply patterns line-by-line.

    Earlier global-scope application discovered `\\s+` in patterns like
    `\\b([A-Z]\\w+)((?:\\s+\\w+)+)\\s+d\\s+(\\w+)\\b` traverses across
    newline boundaries, accidentally matching a doc-comment identifier
    (e.g. `UserExecutorCore`) to a value-position `d` 3 lines below. Per
    line application keeps matches bounded to a single source line.

    Multi-line continuation type signatures don't need cross-line matching
    because each line individually has its own `Tensor [..] d dt g` or
    `UserExecutor* d` shape.
    """
    original = path.read_text()
    lines = original.split("\n")
    new_lines: list[str] = []
    total = 0
    for line in lines:
        new = line
        for pat, repl in patterns:
            new, n = re.subn(pat, repl, new)
            total += n
        new_lines.append(new)
    new_text = "\n".join(new_lines)
    if new_text != original:
        path.write_text(new_text)
        return total, True
    return 0, False


# Doc-safe patterns: only the binder shapes + named applications. The
# Network/State/Tensor patterns could spuriously match prose.
DOC_PATTERNS = [p for p in PATTERNS if not p[1].startswith((
    r"\1\2", r"\1", "Network ", "AnyLayer ex", "GradScaler ex",
    "NativeOptimizer ex", "Stage ex",
)) and not p[0].startswith(r"\bTensor")]


def main() -> int:
    idris_files = discover_idris_files()
    print(f"Scanning {len(idris_files)} Idris files…", file=sys.stderr)

    idris_touched: list[Path] = []
    idris_total = 0
    for f in idris_files:
        n, did = rewrite_file(f, PATTERNS)
        idris_total += n
        if did:
            idris_touched.append(f)

    print(
        f"Idris substitutions: {idris_total}; files touched: {len(idris_touched)}",
        file=sys.stderr,
    )

    doc_files = discover_doc_files()
    print(f"Scanning {len(doc_files)} doc files…", file=sys.stderr)

    doc_touched: list[Path] = []
    doc_total = 0
    for f in doc_files:
        n, did = rewrite_file(f, DOC_PATTERNS)
        doc_total += n
        if did:
            doc_touched.append(f)

    print(
        f"Doc substitutions:   {doc_total}; files touched: {len(doc_touched)}",
        file=sys.stderr,
    )

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
