"""Kernel-layer do_execute tests — the retry seam for the Usage collision.

`Control.Linear.LIO` (re-exported by Notebook.Prelude) ships
`fromInteger : (x : Integer) -> Either (x = 0) (x = 1) => Usage`, so a
bare `0`/`1` literal in an open-typed expression commits elaboration to
`Usage` and `1 + 1` fails with "Can't find an implementation for Num
Usage". A module-level `%hide` doesn't reach the interactive context
(the REPL resets a loaded module's hide list), so the kernel retries
the specific collision with Idris's `with`-disambiguation.

Run via:
    make test-e2e-jupyter
"""

import asyncio
import os
from collections.abc import Iterator
from pathlib import Path
from typing import Any

import pytest
from idris_ml_kernel.kernel import Idris2Kernel

PROJECT_ROOT = Path(__file__).resolve().parents[3]


@pytest.fixture(scope="module")
def kernel() -> Iterator[Idris2Kernel]:
    os.environ.setdefault("IDRIS_ML_ROOT", str(PROJECT_ROOT))
    k = Idris2Kernel()
    yield k
    k.repl.close()


def run(k: Idris2Kernel, code: str) -> dict[str, Any]:
    return asyncio.run(k.do_execute(code, silent=True))


def test_unit_literal_arithmetic_is_ok(kernel: Idris2Kernel) -> None:
    """`1 + 1` is the first thing a new user types; it must not error."""
    reply = run(kernel, "1 + 1")
    assert reply["status"] == "ok", reply.get("evalue", "")


def test_zero_literal_arithmetic_is_ok(kernel: Idris2Kernel) -> None:
    reply = run(kernel, "0 + 2")
    assert reply["status"] == "ok", reply.get("evalue", "")


def test_unit_literals_inside_exec_are_ok(kernel: Idris2Kernel) -> None:
    reply = run(kernel, ":exec printLn (1 + 1)")
    assert reply["status"] == "ok", reply.get("evalue", "")


def test_ordinary_literals_still_ok(kernel: Idris2Kernel) -> None:
    reply = run(kernel, "2 + 3")
    assert reply["status"] == "ok", reply.get("evalue", "")


def test_genuine_errors_still_reported(kernel: Idris2Kernel) -> None:
    """The retry must not swallow unrelated failures."""
    reply = run(kernel, "definitelyNotDefined + 1")
    assert reply["status"] == "error"
