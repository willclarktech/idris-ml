#!/usr/bin/env python3
"""Gate: every convergence-campaign example has a PyTorch reference peer.

Why this exists: the paired table in `paired_examples.py` is hand-maintained,
and the two gates that read it (`check-paired-defaults.py`,
`check-paired-metrics.py`) only check the pairs they are *given*. An example
missing from the table is silently exempt from both. `example-double-dqn` and
`example-sac` were in exactly that state from the day each landed until
2026-07-31 — their defaults had never been compared with the reference's.

So this asks make for the campaign list rather than re-deriving it, resolves
each target's Idris source through `make -n` (the target name and the source
basename do not always agree — `example-dnc-recall` builds
`DncAssociativeRecall.idr`), and fails if any of them is absent from the table
or names a file that does not exist.

Usage: scripts/check-example-pairing.py
Exit 0 = every campaign example is paired, 1 = gaps.
"""

from __future__ import annotations

import re
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from paired_examples import EXAMPLES, REPO_ROOT  # noqa: E402

SOURCE_RE = re.compile(r"Example/([A-Za-z0-9]+)\.idr")


def make_var(name: str) -> list[str]:
    """Read a make variable. `print-%` is prerequisite-free (mk/config.mk)."""
    out = subprocess.run(
        ["make", "-s", "--no-print-directory", f"print-{name}"],
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
        check=True,
    )
    return out.stdout.split()


def idris_source_for(target: str) -> str | None:
    """The Example/*.idr a target builds, per make's own recipe.

    `-n` prints without executing, so this neither builds nor races a running
    campaign. Returns the LAST match: prerequisite recipes (install) are
    printed first and may mention other sources.
    """
    out = subprocess.run(
        ["make", "-n", target],
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
        check=False,
    )
    matches = SOURCE_RE.findall(out.stdout)
    return matches[-1] if matches else None


def main() -> int:
    campaign = make_var("CONVERGENCE_CAMPAIGN_EXAMPLES")
    if not campaign:
        print(
            "FAIL: CONVERGENCE_CAMPAIGN_EXAMPLES is empty — did the variable move?",
            file=sys.stderr,
        )
        return 1

    by_source = {Path(spec["idris"]).stem: spec for spec in EXAMPLES}

    unpaired: list[tuple[str, str]] = []
    unresolved: list[str] = []
    for target in campaign:
        source = idris_source_for(target)
        if source is None:
            unresolved.append(target)
            continue
        if source not in by_source:
            unpaired.append((target, source))

    missing_files: list[str] = []
    for spec in EXAMPLES:
        for side in ("idris", "python"):
            path = REPO_ROOT / spec[side]
            if not path.exists():
                missing_files.append(f"{spec['name']}: {spec[side]}")

    if unresolved:
        print("FAIL: could not resolve an Idris source for:", file=sys.stderr)
        for target in unresolved:
            print(f"  - {target} (no Example/*.idr in `make -n {target}`)", file=sys.stderr)

    if unpaired:
        print("FAIL: campaign examples with no entry in paired_examples.EXAMPLES:", file=sys.stderr)
        for target, source in unpaired:
            print(f"  - {target} (builds {source}.idr)", file=sys.stderr)
        print("", file=sys.stderr)
        print(
            "Without an entry, check-paired-defaults.py and check-paired-metrics.py",
            file=sys.stderr,
        )
        print(
            "skip the example entirely — its hyperparameters and reported metrics",
            file=sys.stderr,
        )
        print("are never compared against the reference. Add a row, or drop the", file=sys.stderr)
        print("example from CONVERGENCE_CAMPAIGN_EXAMPLES in mk/e2e.mk.", file=sys.stderr)

    if missing_files:
        print("FAIL: paired-table entries naming files that do not exist:", file=sys.stderr)
        for entry in missing_files:
            print(f"  - {entry}", file=sys.stderr)

    if unresolved or unpaired or missing_files:
        return 1

    print(f"check-example-pairing: OK ({len(campaign)} campaign examples, all paired)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
