#!/usr/bin/env python3
"""Tests for the perf-log `kind="compile"` record (compilation-time capture).

The compile timer (`scripts/perf-compile.sh`) measures how long `idris2`
spends on a full cold build of a unit (the idris-ml library via
`--build`, or a single example module via `-o` — elaborate + Chez codegen
+ executable), and logs one `kind="compile"` JSONL record per measurement.
Compilation is the developer-felt cost the linear-types migration can move
(runtime is linearity-invariant) — see linear-types-and-effects.md.

Run via:
    python3 scripts/tests/test_perf_log_compile.py
"""

from __future__ import annotations

import json
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "scripts"))

from mltools.perf_log import append_compile, iter_entries  # noqa: E402


def test_append_compile_writes_one_record() -> None:
    with tempfile.TemporaryDirectory() as d:
        log = Path(d) / "perf-log.jsonl"
        entry = append_compile(
            unit="idris-ml",
            backend="tape",
            device="cpu",
            commit="abc1234+dirty",
            compile_ms=123456,
            compile_human="2m 3s",
            exit_code=0,
            log_path=log,
        )
        # Returned dict has the canonical shape.
        assert entry["kind"] == "compile"
        assert entry["unit"] == "idris-ml"
        assert entry["backend"] == "tape"
        assert entry["device"] == "cpu"
        assert entry["commit"] == "abc1234+dirty"
        assert entry["compile_ms"] == 123456
        assert entry["compile_human"] == "2m 3s"
        assert entry["exit"] == 0
        assert entry["cold"] is True  # default: timer always cold-builds
        assert "ts" in entry and "date" in entry

        # Exactly one line was appended, and it round-trips as JSON.
        rows = list(iter_entries(log))
        assert len(rows) == 1
        assert rows[0] == entry
        lines = log.read_text().splitlines()
        assert len(lines) == 1
        json.loads(lines[0])


def test_append_compile_records_failure_exit() -> None:
    with tempfile.TemporaryDirectory() as d:
        log = Path(d) / "perf-log.jsonl"
        entry = append_compile(
            unit="example-mnist",
            backend="tape",
            device="cpu",
            commit="deadbeef",
            compile_ms=9000,
            compile_human="9.000s",
            exit_code=1,
            log_path=log,
        )
        assert entry["unit"] == "example-mnist"
        assert entry["exit"] == 1


def test_append_compile_is_append_only() -> None:
    with tempfile.TemporaryDirectory() as d:
        log = Path(d) / "perf-log.jsonl"
        for i in range(3):
            append_compile(
                unit=f"example-{i}",
                backend="tape",
                device="cpu",
                commit="c",
                compile_ms=i,
                compile_human=f"{i}ms",
                exit_code=0,
                log_path=log,
            )
        assert len(list(iter_entries(log))) == 3


if __name__ == "__main__":
    fns = [v for k, v in sorted(globals().items()) if k.startswith("test_") and callable(v)]
    for fn in fns:
        fn()
        print(f"ok: {fn.__name__}")
    print(f"all {len(fns)} tests passed")
