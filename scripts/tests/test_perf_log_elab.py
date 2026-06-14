#!/usr/bin/env python3
"""Tests for the perf-log `kind="elab"` record (elaboration-time capture).

The elaboration timer (`scripts/perf-elab.sh`) measures how long `idris2`
spends typechecking a unit (the idris-ml library, or a single example
module) cold, and logs one `kind="elab"` JSONL record per measurement.
This is the one perf axis the linear-types migration can move (linearity
is compile-time only, so it can't touch runtime) — see
docs/develop/linear-types-and-effects.md.

Run via:
    python3 scripts/tests/test_perf_log_elab.py
"""

from __future__ import annotations

import json
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "scripts"))

from mltools.perf_log import append_elab, iter_entries  # noqa: E402


def test_append_elab_writes_one_record() -> None:
    with tempfile.TemporaryDirectory() as d:
        log = Path(d) / "perf-log.jsonl"
        entry = append_elab(
            unit="idris-ml",
            backend="tape",
            device="cpu",
            commit="abc1234+dirty",
            elab_ms=12345,
            elab_human="12.345s",
            exit_code=0,
            log_path=log,
        )
        # Returned dict has the canonical shape.
        assert entry["kind"] == "elab"
        assert entry["unit"] == "idris-ml"
        assert entry["backend"] == "tape"
        assert entry["device"] == "cpu"
        assert entry["commit"] == "abc1234+dirty"
        assert entry["elab_ms"] == 12345
        assert entry["elab_human"] == "12.345s"
        assert entry["exit"] == 0
        assert entry["cold"] is True  # default: timer always cold-elaborates
        assert "ts" in entry and "date" in entry

        # Exactly one line was appended, and it round-trips as JSON.
        rows = list(iter_entries(log))
        assert len(rows) == 1
        assert rows[0] == entry
        # And it's valid JSONL on disk (one object per line).
        lines = log.read_text().splitlines()
        assert len(lines) == 1
        json.loads(lines[0])


def test_append_elab_records_failure_exit() -> None:
    with tempfile.TemporaryDirectory() as d:
        log = Path(d) / "perf-log.jsonl"
        entry = append_elab(
            unit="example-mnist",
            backend="tape",
            device="cpu",
            commit="deadbeef",
            elab_ms=999,
            elab_human="0.999s",
            exit_code=1,
            log_path=log,
        )
        assert entry["unit"] == "example-mnist"
        assert entry["exit"] == 1


def test_append_elab_is_append_only() -> None:
    with tempfile.TemporaryDirectory() as d:
        log = Path(d) / "perf-log.jsonl"
        for i in range(3):
            append_elab(
                unit=f"example-{i}",
                backend="tape",
                device="cpu",
                commit="c",
                elab_ms=i,
                elab_human=f"{i}ms",
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
