#!/usr/bin/env python3
"""Generate the make-invocation blocks of .github/workflows/test.yml from
.github/workflows/test.yml.spec.json.

The spec's `jobs` map holds one invocation list per workflow job; each
job in the workflow carries a marker-bounded region

  # >>> GENERATED (<job>) FROM test.yml.spec.json >>>
  ...
  # <<< END GENERATED (<job>) <<<

that this script rewrites. Job scaffolding (checkout, composite setup,
cache saves, artifact upload) stays hand-written.

Every generated step gets `!cancelled()` merged into its condition:
this CI is a detector, not a gate (results are read after the fact on
a ~weekly publication cadence), so one red gate must never skip its
siblings — a masked failure costs a week.

Adding a new gate: append to the right job in the spec, run this
script, commit both.

Usage:
  scripts/gen-ci-workflow.py            # rewrites the workflow in place
  scripts/gen-ci-workflow.py --check    # exits 1 if workflow would change

The `make test-integration-lint-ci-workflow` target wraps --check; CI fails
if a hand-edit of test.yml diverges from the spec.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parent.parent
SPEC = ROOT / ".github" / "workflows" / "test.yml.spec.json"
WORKFLOW = ROOT / ".github" / "workflows" / "test.yml"

STEP_INDENT = "      "  # 6 spaces — matches the job step lists

OS_TO_IF = {
    "ubuntu": "matrix.os == 'ubuntu-latest'",
    "macos": "matrix.os == 'macos-latest'",
}


def begin_marker(job: str) -> str:
    return f"# >>> GENERATED ({job}) FROM test.yml.spec.json >>>"


def end_marker(job: str) -> str:
    return f"# <<< END GENERATED ({job}) <<<"


def render_invocation(inv: dict[str, Any]) -> list[str]:
    """Render one invocation as a list of YAML lines (no trailing newlines)."""
    lines = [f"{STEP_INDENT}- name: {inv['name']}"]

    # Block comment (# lines) preceding the step body.
    comments: list[str] = inv.get("comment") or []
    for c in comments:
        lines.append(f"{STEP_INDENT}  # {c}" if c else f"{STEP_INDENT}  #")

    # `if:` combining the `os` filter, an explicit `if`, and the
    # unconditional `!cancelled()` (run even after a sibling failed).
    cond_parts: list[str] = []
    if "os" in inv:
        os_val = inv["os"]
        if os_val not in OS_TO_IF:
            raise SystemExit(f"unknown os filter: {os_val!r} (use 'ubuntu' or 'macos')")
        cond_parts.append(OS_TO_IF[os_val])
    if inv.get("if"):
        cond_parts.append(inv["if"])
    cond_parts.append("!cancelled()")
    # ${{ }} wrapper: a bare leading `!` would parse as a YAML tag.
    lines.append(f"{STEP_INDENT}  if: ${{{{ {' && '.join(cond_parts)} }}}}")

    # env block.
    env: dict[str, Any] | None = inv.get("env")
    if env:
        lines.append(f"{STEP_INDENT}  env:")
        for k, v in env.items():
            lines.append(f"{STEP_INDENT}    {k}: {v}")

    # continue-on-error (advisory step that shouldn't block the matrix).
    if inv.get("continue-on-error"):
        lines.append(f"{STEP_INDENT}  continue-on-error: true")

    # run line.
    lines.append(f"{STEP_INDENT}  run: {inv['run']}")
    return lines


def render_block(invocations: list[dict[str, Any]]) -> str:
    """Render one job's generated block (between markers), ending in '\n'."""
    out_lines: list[str] = []
    for i, inv in enumerate(invocations):
        if i > 0:
            out_lines.append("")  # blank line between steps
        out_lines.extend(render_invocation(inv))
    return "\n".join(out_lines) + "\n"


def splice_workflow(workflow_text: str, job: str, generated_block: str) -> str:
    """Replace the job's BEGIN..END marker region with generated_block."""
    begin = begin_marker(job)
    end = end_marker(job)
    begin_idx = workflow_text.find(begin)
    end_idx = workflow_text.find(end)
    if begin_idx == -1 or end_idx == -1:
        raise SystemExit(
            f"missing markers for job {job!r} in {WORKFLOW.relative_to(ROOT)}: "
            f"expected {begin!r} and {end!r}"
        )
    if begin_idx > end_idx:
        raise SystemExit(f"BEGIN marker appears after END marker for job {job!r}")
    # Find the start of the BEGIN line (rewind to preceding newline) and the
    # end of the END line (advance to following newline). Replace everything
    # between (exclusive of the marker lines themselves) with the block.
    line_end_of_begin = workflow_text.find("\n", begin_idx) + 1
    line_start_of_end = workflow_text.rfind("\n", 0, end_idx) + 1
    return workflow_text[:line_end_of_begin] + generated_block + workflow_text[line_start_of_end:]


def main() -> None:
    check = "--check" in sys.argv[1:]
    if not SPEC.exists():
        raise SystemExit(f"spec not found: {SPEC}")
    if not WORKFLOW.exists():
        raise SystemExit(f"workflow not found: {WORKFLOW}")
    spec: dict[str, Any] = json.loads(SPEC.read_text())
    jobs: dict[str, list[dict[str, Any]]] = spec.get("jobs", {})
    if not jobs:
        raise SystemExit("spec has no `jobs` map")
    workflow_text = WORKFLOW.read_text()
    new_text = workflow_text
    total = 0
    for job, invocations in jobs.items():
        new_text = splice_workflow(new_text, job, render_block(invocations))
        total += len(invocations)
    if check:
        if new_text != workflow_text:
            print(
                f"FAIL: {WORKFLOW.relative_to(ROOT)} disagrees with "
                f"{SPEC.relative_to(ROOT)}. Run scripts/gen-ci-workflow.py to regenerate.",
                file=sys.stderr,
            )
            sys.exit(1)
        print(f"OK: {WORKFLOW.relative_to(ROOT)} is in sync with spec.")
        return
    if new_text != workflow_text:
        WORKFLOW.write_text(new_text)
        print(
            f"Regenerated {WORKFLOW.relative_to(ROOT)}: "
            f"{total} spec entries across {len(jobs)} jobs."
        )
    else:
        print(f"No changes; {WORKFLOW.relative_to(ROOT)} already in sync.")


if __name__ == "__main__":
    main()
