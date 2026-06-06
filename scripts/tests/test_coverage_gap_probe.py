#!/usr/bin/env python3
"""Regression test for the coverage-gap-probe TEST_ROOTS fix.

Before the fix, every OP_* was reported MISSING because the probe
scanned `packages/backends/test/` (a directory that no longer exists).
After the fix, tape OPs with colocated tests under
`packages/backends/backend_tape/<area>/test_*.c` resolve correctly.

Run via:
    python3 scripts/tests/test_coverage_gap_probe.py
"""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "scripts"))

from mltools.header_parser import grep_word_in_dirs  # noqa: E402


def test_tape_add_has_test_coverage() -> None:
    """`tensor_add` is the canonical tape OP_ADD entry point and has
    a dedicated test under `backend_tape/core/elementwise/`. If the
    probe can't find it, TEST_ROOTS is misconfigured.
    """
    backends = ROOT / "packages" / "backends"
    roots = [
        backends / "backend_tape",
        ROOT / "packages" / "idris-test-c" / "src",
    ]
    hits = grep_word_in_dirs("tensor_add", roots)
    assert hits, (
        "tensor_add has no test hits under tape roots — "
        "TEST_ROOTS is pointing at the wrong directory"
    )


def test_tape_backward_has_test_coverage() -> None:
    """`tensor_backward` is exercised by virtually every gradient
    assertion. If this is zero, the search is broken.
    """
    backends = ROOT / "packages" / "backends"
    roots = [
        backends / "backend_tape",
        ROOT / "packages" / "idris-test-c" / "src",
    ]
    hits = grep_word_in_dirs("tensor_backward", roots)
    assert len(hits) >= 5, (
        f"tensor_backward only has {len(hits)} test hits; "
        "expected ≥5 across the colocated tape suite"
    )


def test_missing_root_is_silently_skipped() -> None:
    """`grep_word_in_dirs` must tolerate roots that don't exist —
    e.g. backend_torch has no colocated tests today, so its root
    should contribute zero hits without raising.
    """
    bogus = [ROOT / "this_path_does_not_exist"]
    assert grep_word_in_dirs("tensor_add", bogus) == []


if __name__ == "__main__":
    test_tape_add_has_test_coverage()
    test_tape_backward_has_test_coverage()
    test_missing_root_is_silently_skipped()
    print("OK")
