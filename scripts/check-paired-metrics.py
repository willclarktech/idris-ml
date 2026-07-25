#!/usr/bin/env python3
"""Gate: paired examples report the same metric keys on their RESULT line.

Why this exists: `Example.SeqClassify` reported `loss` while its reference
reported `accuracy`, and neither reported the other. The two sides had run
side by side for months with no shared number, so no threshold could compare
them and nobody noticed the Idris example evaluated nothing at all. A metric
the reference reports and the example does not is an alignment hole that
convergence runs cannot see, because each side passes its own gate.

Static, like `check-paired-defaults.py`: the Idris side is a regex over
`formatResult [("k", …), …]`, the Python side an AST walk for
`format_result([...])`. Keys chosen at runtime (gpt picks `bpc` or `val_bpc`
by corpus) come back as `<dynamic>` and are ignored — declare the real key
in `metrics_only_*` if a genuine asymmetry needs recording.

Usage: scripts/check-paired-metrics.py [--json]
Exit 0 = keys agree, 1 = divergence, 2 = parse failure.
"""

from __future__ import annotations

import argparse
import ast
import json
import re
import sys
from dataclasses import asdict, dataclass
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from paired_examples import EXAMPLES, REPO_ROOT  # noqa: E402

# Keys that identify a run rather than measure it. Present on both sides by
# convention and uninteresting to compare.
BOOKKEEPING = frozenset({"seed", "epochs", "steps"})


@dataclass
class Report:
    """One pair's metric-key comparison."""

    name: str
    shared: list[str]
    only_idris: list[str]
    only_python: list[str]


DYNAMIC = "<dynamic>"


def parse_idris(path: Path) -> set[str]:
    """Keys from every `formatResult [("k", …), …]` literal in the file."""
    text = path.read_text()
    keys: set[str] = set()
    for m in re.finditer(r"formatResult\s*\[", text):
        # Walk to the matching close bracket so multi-line literals are whole.
        depth, i = 0, m.end() - 1
        while i < len(text):
            if text[i] == "[":
                depth += 1
            elif text[i] == "]":
                depth -= 1
                if depth == 0:
                    break
            i += 1
        body = text[m.end() : i]
        keys.update(re.findall(r'\(\s*"([a-zA-Z_][a-zA-Z0-9_]*)"\s*,', body))
    return keys


def parse_python(path: Path) -> set[str]:
    """Keys from every `format_result([...])` call, via the AST.

    A tuple whose first element is not a string constant contributes
    `<dynamic>` rather than being dropped silently.
    """
    tree = ast.parse(path.read_text())
    keys: set[str] = set()
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        fn = node.func
        name = fn.attr if isinstance(fn, ast.Attribute) else getattr(fn, "id", None)
        if name != "format_result" or not node.args:
            continue
        arg = node.args[0]
        if not isinstance(arg, (ast.List, ast.Tuple)):
            continue
        for elt in arg.elts:
            if isinstance(elt, (ast.Tuple, ast.List)) and elt.elts:
                first = elt.elts[0]
                if isinstance(first, ast.Constant) and isinstance(first.value, str):
                    keys.add(first.value)
                else:
                    keys.add(DYNAMIC)
    return keys


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--json", action="store_true", help="machine-readable report")
    args = parser.parse_args()

    reports: list[Report] = []
    failed = False
    for spec in EXAMPLES:
        try:
            idris = parse_idris(REPO_ROOT / spec["idris"])
            python = parse_python(REPO_ROOT / spec["python"])
        except (OSError, SyntaxError, ValueError) as exc:
            print(f"PARSE FAIL {spec['name']}: {exc}", file=sys.stderr)
            return 2

        idris -= BOOKKEEPING
        python -= BOOKKEEPING | {DYNAMIC}
        allowed_idris = set(spec.get("metrics_only_idris", []))
        allowed_python = set(spec.get("metrics_only_python", []))

        only_idris = sorted(idris - python - allowed_idris)
        only_python = sorted(python - idris - allowed_python)
        if only_idris or only_python:
            failed = True
        reports.append(
            Report(
                name=spec["name"],
                shared=sorted(idris & python),
                only_idris=only_idris,
                only_python=only_python,
            )
        )

    if args.json:
        print(json.dumps([asdict(r) for r in reports], indent=2))
        return 1 if failed else 0

    for report in reports:
        flags: list[str] = []
        if report.only_idris:
            flags.append(f"{len(report.only_idris)} idris-only")
        if report.only_python:
            flags.append(f"{len(report.only_python)} python-only")
        status = f"[{' · '.join(flags)}]" if flags else "[OK]"
        print(f"{report.name:<20} {status}")
        for key in report.only_idris:
            print(f"    idris-only  {key}")
        for key in report.only_python:
            print(f"    python-only {key}")

    print("")
    if failed:
        print(
            "Metric keys diverge. A key one side reports and the other does not means",
            file=sys.stderr,
        )
        print(
            "no shared threshold can compare them — add the metric to the side that",
            file=sys.stderr,
        )
        print(
            "lacks it (the better practice on both, per the alignment policy), or",
            file=sys.stderr,
        )
        print(
            "record the asymmetry in metrics_only_idris / metrics_only_python with a",
            file=sys.stderr,
        )
        print("reason.", file=sys.stderr)
        return 1

    print(f"All {len(EXAMPLES)} paired examples report the same metric keys.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
