#!/usr/bin/env python3
"""Regression tests for the coverage-gap-probe search.

Two historical bugs are guarded here:

1. TEST_ROOTS once pointed at `packages/backends/test/` (a directory that
   no longer exists), so every OP_* was reported MISSING. Tape OPs with
   colocated tests under `backend_tape/<area>/test_*.c` must resolve.

2. The probe searched the DEFAULT `*.c` glob across the backend trees,
   which hold BOTH implementation and test sources — so every symbol
   self-matched its own definition and was counted "covered" whether or
   not a test exercised it. The probe now passes `suffixes=("test_*.c",)`;
   these tests mirror that and assert the impl self-match is excluded.

Run via:
    python3 scripts/tests/test_coverage_gap_probe.py
"""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "scripts"))

from mltools.header_parser import grep_word_in_dirs  # noqa: E402

# Mirror the probe's test-only restriction (coverage-gap-probe.py
# TEST_FILE_GLOB). Searches that count coverage MUST use this, not the
# default `*.c`, or impl files self-match.
TEST_GLOB = ("test_*.c",)


def test_tape_add_has_test_coverage() -> None:
    """`tensor_add` is the canonical tape OP_ADD entry point and has
    a dedicated test under `backend_tape/core/elementwise/`. If the
    probe can't find it (test-only glob), TEST_ROOTS is misconfigured.
    """
    backends = ROOT / "packages" / "backends"
    roots = [
        backends / "backend_tape",
        ROOT / "packages" / "idris-test-c" / "src",
    ]
    hits = grep_word_in_dirs("tensor_add", roots, suffixes=TEST_GLOB)
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
    hits = grep_word_in_dirs("tensor_backward", roots, suffixes=TEST_GLOB)
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
    assert grep_word_in_dirs("tensor_add", bogus, suffixes=TEST_GLOB) == []


def test_test_only_glob_excludes_impl_self_match() -> None:
    """The core of bug #2: with the default `*.c` glob, `tensor_add`
    matches its own implementation file (`core/elementwise/add.c`); with
    the test-only glob every hit must be a `test_*.c` file. This is what
    stops an implemented-but-untested symbol from being scored "covered".
    """
    backends = ROOT / "packages" / "backends"
    roots = [backends / "backend_tape"]

    default_hits = grep_word_in_dirs("tensor_add", roots)  # all *.c
    assert any(not p.name.startswith("test_") for p in default_hits), (
        "expected the default glob to self-match tensor_add's impl file — "
        "if this fails the test's premise no longer holds"
    )

    test_only_hits = grep_word_in_dirs("tensor_add", roots, suffixes=TEST_GLOB)
    assert test_only_hits, "tensor_add should still have a real test_*.c hit"
    assert all(p.name.startswith("test_") for p in test_only_hits), (
        "test-only glob leaked a non-test file — self-match not excluded"
    )


def test_mlx_ops_covered_by_cross_backend_tests() -> None:
    """The tape-colocated `test_avg_pool2d.c` (and similar) exercises
    mlx's `tensor_avg_pool2d` when the binary is built with
    `-DBACKEND_MLX` because the Makefile globs `test_*.c` across all
    three backend trees. The probe MUST scan all trees for OP coverage
    (not narrow to the matching backend's tree) or it will falsely flag
    every cross-backend test as MISSING — the regression behind the
    `16f99d94` follow-up.
    """
    backends = ROOT / "packages" / "backends"
    all_trees = [backends / d for d in ("backend_tape", "backend_torch", "backend_mlx")]
    all_trees.append(ROOT / "packages" / "idris-test-c" / "src")
    for sym in (
        "tensor_avg_pool2d",
        "tensor_gelu",
        "tensor_masked_fill",
        "tensor_rms_norm_2d",
        "tensor_swiglu_2d",
        "tensor_tile_2d",
        "tensor_max_pool2d_batched",
    ):
        hits = grep_word_in_dirs(sym, all_trees, suffixes=TEST_GLOB)
        assert hits, f"{sym} has no test hits — cross-backend tree search is broken"


if __name__ == "__main__":
    test_tape_add_has_test_coverage()
    test_tape_backward_has_test_coverage()
    test_missing_root_is_silently_skipped()
    test_test_only_glob_excludes_impl_self_match()
    test_mlx_ops_covered_by_cross_backend_tests()
    print("OK")
