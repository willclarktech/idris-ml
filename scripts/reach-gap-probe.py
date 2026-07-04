#!/usr/bin/env python3
"""Idris reachability gap-finder (advisory).

Reports top-level definitions in `packages/idris-ml/src` that are NEVER
reachable from any test or example entry point — i.e. code no part of the
suite exercises. This is a GAP FINDER, not a coverage percentage:
"reachable" means a static call path exists from a compiled `main`, not
that a test ran the code or asserted on it, so it is an upper bound on
real coverage. The actionable artifact is the unreachable LIST.

Usage:
    python3 scripts/reach-gap-probe.py [BUILDDIR]
    make test-coverage-reach-gap        # runs reach-dump first, then this

Inputs:
  - `<BUILDDIR>/reach/*.cases` — `idris2 --dumpcases` dumps of the test
    main + each example (produced by the `reach-dump` Make target). Their
    union is the reachable set.
  - `packages/idris-ml/src/**/*.idr` (excluding Test/) — the universe.
  - `scripts/reach-exclusions.txt` — FQNs that legitimately never appear
    in a dump even when used (inlined / erased / type-level / %foreign).

Output:
  - `<BUILDDIR>/reach-gap.csv` (fqn, module, source_file).
  - Summary + sample gap rows to stdout.

Exit code: always 0 in v1 (advisory). The ratchet gate (fail on NEW
unreachable defs vs a committed baseline) is a documented follow-up; see
docs/develop/reachability-policy.md.
"""

from __future__ import annotations

import io
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "scripts"))

from mltools.idris_parser import (  # noqa: E402
    is_excluded,
    parse_exclusions,
    parse_reachable,
    scan_universe,
)

SRC_ROOT = ROOT / "packages" / "idris-ml" / "src"
EXCLUSIONS_FILE = ROOT / "scripts" / "reach-exclusions.txt"


def collect_reachable(reach_dir: Path) -> tuple[set[str], int]:
    """Union of reachable FQNs across every `*.cases` dump; also the dump
    count (so the caller can warn if reach-dump produced nothing)."""
    reachable: set[str] = set()
    dumps = sorted(reach_dir.glob("*.cases"))
    for dump in dumps:
        try:
            reachable |= parse_reachable(dump.read_text())
        except (OSError, UnicodeDecodeError):
            continue
    return reachable, len(dumps)


def build_gap_rows(gap: list[str], universe: dict[str, str]) -> list[dict[str, str]]:
    rows: list[dict[str, str]] = []
    for fqn in gap:
        module = fqn.rsplit(".", 1)[0]
        rows.append({"fqn": fqn, "module": module, "source_file": universe[fqn]})
    return rows


def write_csv(path: Path, rows: list[dict[str, str]], fieldnames: list[str]) -> None:
    import csv

    buf = io.StringIO()
    w = csv.DictWriter(buf, fieldnames=fieldnames, lineterminator="\n")
    w.writeheader()
    for row in rows:
        w.writerow(row)
    path.write_text(buf.getvalue())


def print_summary(
    universe: dict[str, str],
    reachable: set[str],
    excluded_hits: set[str],
    gap_rows: list[dict[str, str]],
    n_dumps: int,
    csv_path: Path,
) -> None:
    print()
    print("=== Idris reachability gap-finder (advisory) ===")
    print(f"dumps unioned (reach/*.cases): {n_dumps}")
    print(f"universe (idris-ml/src, non-Test top-level defs): {len(universe)}")
    pct = 100 * (len(universe) - len(gap_rows)) // max(1, len(universe))
    print(f"reachable from suite: {len(universe) - len(gap_rows)} ({pct}%)")
    print(f"excluded (reach-exclusions.txt applied): {len(excluded_hits)}")
    print(f"GAP (unreachable, non-excluded): {len(gap_rows)}")
    print()
    print(f"Report: {csv_path}")

    if n_dumps == 0:
        print()
        print("WARNING: no *.cases dumps found — run `make reach-dump` first.")

    # Per-module gap tally (most-unreachable modules first) — the useful
    # at-a-glance "where are the holes" view.
    by_mod: dict[str, int] = {}
    for r in gap_rows:
        by_mod[r["module"]] = by_mod.get(r["module"], 0) + 1
    if by_mod:
        print()
        print("Top modules by gap count:")
        for mod, n in sorted(by_mod.items(), key=lambda kv: (-kv[1], kv[0]))[:15]:
            print(f"  {n:4d}  {mod}")

    if gap_rows:
        print()
        print("Sample unreachable defs:")
        for r in gap_rows[:20]:
            print(f"  {r['fqn']}   [{r['source_file']}]")
        if len(gap_rows) > 20:
            print(f"  ... and {len(gap_rows) - 20} more — see CSV")


def main(argv: list[str]) -> int:
    builddir = Path(argv[1]) if len(argv) > 1 else ROOT / "build"
    builddir.mkdir(parents=True, exist_ok=True)
    reach_dir = builddir / "reach"
    csv_path = builddir / "reach-gap.csv"

    reachable, n_dumps = collect_reachable(reach_dir)
    universe = scan_universe(SRC_ROOT)
    empty: tuple[frozenset[str], tuple[str, ...]] = (frozenset(), ())
    exact, prefixes = (
        parse_exclusions(EXCLUSIONS_FILE.read_text())
        if EXCLUSIONS_FILE.exists()
        else empty
    )

    excluded_hits = {u for u in universe if is_excluded(u, exact, prefixes)}
    gap = sorted(u for u in universe if u not in reachable and u not in excluded_hits)
    gap_rows = build_gap_rows(gap, universe)

    write_csv(csv_path, gap_rows, ["fqn", "module", "source_file"])
    print_summary(universe, reachable, excluded_hits, gap_rows, n_dumps, csv_path)

    # Advisory: never fails CI in v1. Flip to a ratchet (NEW vs baseline)
    # once the gap list is pruned and trusted — see reachability-policy.md.
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
