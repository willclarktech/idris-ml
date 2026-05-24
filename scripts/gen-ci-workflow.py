#!/usr/bin/env python3
"""Generate the test-invocation block of .github/workflows/test.yml from
.github/workflows/test.yml.spec.json.

The workflow's setup boilerplate (Chez install, idris2 cache+build, MLX/torch
install, artifact upload) stays hand-written. Only the steps between the
"# >>> GENERATED FROM test.yml.spec.json >>>" and "# <<< END GENERATED <<<"
marker comments are emitted from the spec.

Adding a new gate: append to the spec, run this script, commit both.

Usage:
  scripts/gen-ci-workflow.py            # rewrites the workflow in place
  scripts/gen-ci-workflow.py --check    # exits 1 if workflow would change

The `make test-integration-lint-ci-workflow` target wraps --check; CI fails
if a hand-edit of test.yml diverges from the spec.
"""

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
SPEC = ROOT / ".github" / "workflows" / "test.yml.spec.json"
WORKFLOW = ROOT / ".github" / "workflows" / "test.yml"

BEGIN_MARKER = "# >>> GENERATED FROM test.yml.spec.json >>>"
END_MARKER = "# <<< END GENERATED <<<"
STEP_INDENT = "      "  # 6 spaces — matches the existing test-examples job

OS_TO_IF = {
    "ubuntu": "matrix.os == 'ubuntu-latest'",
    "macos": "matrix.os == 'macos-latest'",
}


def render_invocation(inv: dict) -> list[str]:
    """Render one invocation as a list of YAML lines (no trailing newlines)."""
    lines = [f"{STEP_INDENT}- name: {inv['name']}"]

    # Block comment (# lines) preceding the step body.
    for c in inv.get("comment") or []:
        lines.append(f"{STEP_INDENT}  # {c}" if c else f"{STEP_INDENT}  #")

    # `if:` combining `os` filter and explicit `if`.
    cond_parts = []
    if "os" in inv:
        os_val = inv["os"]
        if os_val not in OS_TO_IF:
            raise SystemExit(f"unknown os filter: {os_val!r} (use 'ubuntu' or 'macos')")
        cond_parts.append(OS_TO_IF[os_val])
    if inv.get("if"):
        cond_parts.append(inv["if"])
    if cond_parts:
        lines.append(f"{STEP_INDENT}  if: {' && '.join(cond_parts)}")

    # env block.
    env = inv.get("env")
    if env:
        lines.append(f"{STEP_INDENT}  env:")
        for k, v in env.items():
            lines.append(f"{STEP_INDENT}    {k}: {v}")

    # run line.
    lines.append(f"{STEP_INDENT}  run: {inv['run']}")
    return lines


def render_block(spec: dict) -> str:
    """Render the full generated block (between markers) as a string ending in '\n'."""
    out_lines = []
    invocations = spec.get("invocations", [])
    for i, inv in enumerate(invocations):
        if i > 0:
            out_lines.append("")  # blank line between steps
        out_lines.extend(render_invocation(inv))
    return "\n".join(out_lines) + "\n"


def splice_workflow(workflow_text: str, generated_block: str) -> str:
    """Return workflow_text with the BEGIN..END marker region replaced by generated_block."""
    begin_idx = workflow_text.find(BEGIN_MARKER)
    end_idx = workflow_text.find(END_MARKER)
    if begin_idx == -1 or end_idx == -1:
        raise SystemExit(
            f"missing markers in {WORKFLOW.relative_to(ROOT)}: "
            f"expected {BEGIN_MARKER!r} and {END_MARKER!r}"
        )
    if begin_idx > end_idx:
        raise SystemExit("BEGIN marker appears after END marker")
    # Find the start of the BEGIN line (rewind to preceding newline) and the
    # end of the END line (advance to following newline). Replace everything
    # between (exclusive of the marker lines themselves) with the block.
    line_start_of_begin = workflow_text.rfind("\n", 0, begin_idx) + 1
    line_end_of_begin = workflow_text.find("\n", begin_idx) + 1
    line_start_of_end = workflow_text.rfind("\n", 0, end_idx) + 1
    return (
        workflow_text[:line_end_of_begin]
        + generated_block
        + workflow_text[line_start_of_end:]
    )


def main():
    check = "--check" in sys.argv[1:]
    if not SPEC.exists():
        raise SystemExit(f"spec not found: {SPEC}")
    if not WORKFLOW.exists():
        raise SystemExit(f"workflow not found: {WORKFLOW}")
    spec = json.loads(SPEC.read_text())
    workflow_text = WORKFLOW.read_text()
    generated = render_block(spec)
    new_text = splice_workflow(workflow_text, generated)
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
        n = len(spec.get("invocations", []))
        print(f"Regenerated {WORKFLOW.relative_to(ROOT)} from {n} spec entries.")
    else:
        print(f"No changes; {WORKFLOW.relative_to(ROOT)} already in sync.")


if __name__ == "__main__":
    main()
