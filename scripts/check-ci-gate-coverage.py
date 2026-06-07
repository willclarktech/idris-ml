#!/usr/bin/env python3
"""Cross-check CI workflows against the Makefile test taxonomy.

Two invariants (see CLAUDE.md "Test gates must run in CI" and the CI
design principle that jobs mirror the make taxonomy):

1. Resolvability: every `make <target>` invoked by any workflow or
   composite action must exist in the Makefile. Catches the bug class
   where a target is renamed/removed but the CI spec keeps invoking it
   (the `test-unit-safetensors` landmine: target deleted 2026-06-05,
   spec still invoked it twice).

2. Aggregator coverage: every leaf of the canonical aggregators
   (`test-unit`, `test-integration`, `test-e2e`, `test-coverage`) must
   be invoked by some workflow, listed in the spec's
   `ci_coverage_exceptions` (permanent, with a reason), or listed in
   `ci_coverage_known_holes` (temporary; warns but passes — each entry
   is a tracked TODO, emptied as gates get wired in). Catches the bug
   class where a gate exists locally but never runs in CI
   (`test-unit-gym`, `test-e2e-pytorch-ref`).

Make invocations are extracted from every `run:` block in
`.github/workflows/*.yml` and `.github/actions/**/*.yml`. Heuristic:
once a run block `cd`s away from the repo root, subsequent `make`
calls in that block are foreign (e.g. the Idris2 / pack bootstrap
clones) and are skipped.

Usage: scripts/check-ci-gate-coverage.py   # exit 1 on violation
Wrapped by `make test-integration-lint-ci-coverage`.
"""

from __future__ import annotations

import json
import re
import subprocess
import sys
from pathlib import Path
from typing import cast

ROOT = Path(__file__).resolve().parent.parent
SPEC = ROOT / ".github" / "workflows" / "test.yml.spec.json"
AGGREGATORS = ["test-unit", "test-integration", "test-e2e", "test-coverage"]

# Lines like `        run: <cmd>` or `        run: |` (block scalar).
RUN_RE = re.compile(r"^(\s*)(?:-\s+)?run:\s*(.*)$")
# A make invocation segment: everything after `make` up to a shell
# connective. Targets are lowercase-kebab tokens; VAR=VAL and -flags
# are filtered out below.
MAKE_RE = re.compile(r"(?:^|[\s;(])make\s+([^&|;]*)")
CD_RE = re.compile(r"(?:^|[\s;&(])cd\s+\S")


def workflow_files() -> list[Path]:
    files = sorted((ROOT / ".github" / "workflows").glob("*.yml"))
    files += sorted((ROOT / ".github" / "actions").glob("**/*.yml"))
    return files


def extract_run_blocks(text: str) -> list[str]:
    """Return each `run:` block's shell text (single- or multi-line)."""
    blocks: list[str] = []
    lines = text.splitlines()
    i = 0
    while i < len(lines):
        m = RUN_RE.match(lines[i])
        if not m:
            i += 1
            continue
        indent, rest = m.group(1), m.group(2)
        if rest and not rest.startswith(("|", ">")):
            blocks.append(rest)
            i += 1
            continue
        # Block scalar: collect lines indented deeper than the `run:` key.
        body: list[str] = []
        i += 1
        while i < len(lines):
            line = lines[i]
            if line.strip() and len(line) - len(line.lstrip()) <= len(indent):
                break
            body.append(line)
            i += 1
        blocks.append("\n".join(body))
    return blocks


def extract_make_targets(block: str) -> set[str]:
    """Make targets invoked at repo root within one run block."""
    targets: set[str] = set()
    cd_seen = False
    for line in block.splitlines():
        if line.lstrip().startswith("#"):  # shell comment
            continue
        if CD_RE.search(line):
            cd_seen = True
        if cd_seen:
            continue
        # GH expression interpolations contain spaces (`${{ matrix.x }}`);
        # collapse them so they don't tokenize as targets. The collapsed
        # form keeps its VAR= prefix and is filtered below.
        line = re.sub(r"\$\{\{[^}]*\}\}", "${GH_EXPR}", line)
        for seg in MAKE_RE.findall(line):
            for tok in seg.split():
                if tok.startswith("-") or "=" in tok or tok.startswith("$"):
                    continue
                targets.add(tok)
    return targets


def make_database() -> tuple[dict[str, list[str]], set[str]]:
    """(target -> prerequisites, targets with a recipe), from `make -qp`.

    The recipe distinction matters for coverage recursion: a pure
    aggregator (prereqs, no recipe) is covered when all its prereqs
    are; a recipe-bearing target only counts when invoked itself —
    its prereqs (source files, build stamps) are inputs, not subtests.
    """
    proc = subprocess.run(
        ["make", "-qp"],
        cwd=ROOT,
        capture_output=True,
        text=True,
        check=False,  # question mode exits 1 when targets are out of date
    )
    db: dict[str, list[str]] = {}
    has_recipe: set[str] = set()
    prev = ""
    current = ""
    for line in proc.stdout.splitlines():
        if line.startswith("\t") and current:
            has_recipe.add(current)
        elif (
            line
            and not line.startswith(("#", ".", " "))
            and ":" in line
            and "=" not in line.split(":", 1)[0]
            and prev != "# Not a target:"
        ):
            name, _, deps = line.partition(":")
            if not deps.startswith(":"):  # skip double-colon edge cases
                current = name.strip()
                db[current] = deps.split()
        elif not line.strip():
            current = ""
        prev = line
    return db, has_recipe


def missing_leaves(
    target: str,
    invoked: set[str],
    excused: set[str],
    db: dict[str, list[str]],
    has_recipe: set[str],
    seen: set[str],
) -> list[str]:
    """Deepest uncovered targets under `target` (empty = covered). A
    target is covered if invoked/excused directly, or if it is a pure
    aggregator (no recipe) all of whose prerequisites are covered."""
    if target in invoked or target in excused:
        return []
    if target in seen:
        return [target]
    seen.add(target)
    prereqs = db.get(target, [])
    if not prereqs or target in has_recipe:
        return [target]
    out: list[str] = []
    for p in prereqs:
        out.extend(missing_leaves(p, invoked, excused, db, has_recipe, seen))
    return out


def main() -> None:
    spec: dict[str, object] = json.loads(SPEC.read_text())
    exceptions_raw = spec.get("ci_coverage_exceptions", {})
    exceptions: dict[str, str] = (
        {str(k): str(v) for k, v in cast("dict[object, object]", exceptions_raw).items()}
        if isinstance(exceptions_raw, dict)
        else {}
    )
    holes_raw = spec.get("ci_coverage_known_holes", [])
    holes: list[str] = (
        [str(h) for h in cast("list[object]", holes_raw)] if isinstance(holes_raw, list) else []
    )

    invoked: set[str] = set()
    for f in workflow_files():
        for block in extract_run_blocks(f.read_text()):
            invoked |= extract_make_targets(block)

    db, has_recipe = make_database()
    failures: list[str] = []

    for t in sorted(invoked):
        if t not in db:
            failures.append(f"unresolvable: CI invokes `make {t}` but no such target exists")

    excused = set(exceptions) | set(holes)
    for agg in AGGREGATORS:
        reported: set[str] = set()
        for prereq in db.get(agg, []):
            for leaf in missing_leaves(prereq, invoked, excused, db, has_recipe, set()):
                if leaf not in reported:
                    reported.add(leaf)
                    failures.append(
                        f"uncovered: `{agg}` leaf `{leaf}` runs in no workflow "
                        f"(add to a workflow, or to ci_coverage_exceptions with a reason)"
                    )

    for h in holes:
        print(f"WARN: known hole `{h}` not yet wired into CI", file=sys.stderr)
    stale = [h for h in holes if not missing_leaves(h, invoked, set(), db, has_recipe, set())]
    for h in stale:
        failures.append(f"stale known hole: `{h}` is now covered — remove it from the list")

    if failures:
        for msg in failures:
            print(f"FAIL: {msg}", file=sys.stderr)
        sys.exit(1)
    n_exc = len(exceptions)
    print(
        f"OK: {len(invoked)} CI make targets resolve; "
        f"aggregator leaves covered ({n_exc} named exceptions, {len(holes)} known holes)."
    )


if __name__ == "__main__":
    main()
