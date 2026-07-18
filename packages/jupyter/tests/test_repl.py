"""Integration tests for Idris2REPL — requires built backend + idris2."""

from collections.abc import Iterator
from pathlib import Path

import pytest
from idris_ml_kernel.repl import Idris2REPL

PROJECT_ROOT = Path(__file__).resolve().parents[3]


@pytest.fixture(scope="module")
def repl() -> Iterator[Idris2REPL]:
    """Start a REPL once for the whole test module."""
    r = Idris2REPL(project_root=PROJECT_ROOT)
    yield r
    r.close()


def test_repl_starts(repl: Idris2REPL) -> None:
    """REPL spawns and reaches a prompt."""
    assert repl.is_alive()


def test_type_query(repl: Idris2REPL) -> None:
    result = repl.send(":t MkTensor")
    assert "AnyPtr" in result
    assert "Tensor" in result


def test_doc_query(repl: Idris2REPL) -> None:
    result = repl.send(":doc MkTensor")
    assert "Tensor" in result


def test_module_import(repl: Idris2REPL) -> None:
    result = repl.send(":module Ml.Nn.Linear")
    assert "Imported" in result


def test_let_definition(repl: Idris2REPL) -> None:
    repl.send(":let myTestFn : Int -> Int")
    repl.send(":let myTestFn x = x + 42")
    result = repl.send(":t myTestFn")
    assert "Int -> Int" in result


def test_expression_eval(repl: Idris2REPL) -> None:
    result = repl.send("2 + 3")
    assert "5" in result


# The :exec expressions below anchor the dylib load with a
# `primIO (primParamCount ...)` call: idris2's :exec only emits
# `load-shared-object "libidrisml.dylib"` for `%foreign "C:...,libidrisml"`
# declarations that survive DCE, and the Tensor-touching prims are all
# scheme-wrapped (the C symbol is a string inside Scheme, invisible to
# the loader). primParamCount is a plain C decl, so threading one call
# through the IO chain forces the load.
_LIB_ANCHOR = "_ <- primIO (primParamCount {ex=TapeExecutor}); "


def test_ffi_tensor_create(repl: Idris2REPL) -> None:
    """Core FFI test: create a scalar tensor and read its value."""
    result = repl.send(
        ":exec do { " + _LIB_ANCHOR + "putStrLn (show (primItem {ex=TapeExecutor} "
        "(primCreateScalar {ex=TapeExecutor} 3.14 0))) }"
    )
    # The error path quotes the source line (which contains 3.14), so a
    # bare substring check can false-pass — also require a clean result.
    assert "3.14" in result
    assert "Error" not in result and "Exception" not in result


def test_ffi_tensor_arithmetic(repl: Idris2REPL) -> None:
    """FFI test: tensor multiply (6.0 appears nowhere in the source)."""
    result = repl.send(
        ":exec do { " + _LIB_ANCHOR + "putStrLn (show (primItem {ex=TapeExecutor} "
        "(primMul {ex=TapeExecutor} "
        "(primCreateScalar {ex=TapeExecutor} 2.0 1) "
        "(primCreateScalar {ex=TapeExecutor} 3.0 0)))) }"
    )
    assert "6.0" in result


def test_ffi_exec_after_let(repl: Idris2REPL) -> None:
    """Verify :let definitions are usable in :exec."""
    repl.send(":let double : Double -> Double")
    repl.send(":let double x = x * 2.0")
    result = repl.send(":exec putStrLn (show (double 21.0))")
    assert "42.0" in result


def test_error_handling(repl: Idris2REPL) -> None:
    """Type error should return error text, not crash."""
    result = repl.send(":t nonexistentName")
    assert "Error" in result or "Undefined" in result
    # REPL should still be alive. (Not `1 + 1`: at the raw-REPL layer the
    # literal 1 elaborates to the linear Usage type, which has no Num. The
    # kernel layer retries that collision — see test_kernel_exec.py — but
    # this suite talks to the REPL directly, below the retry.)
    result2 = repl.send("3 + 4")
    assert "7" in result2


def test_browse(repl: Idris2REPL) -> None:
    result = repl.send(":browse Ml.Tensor")
    assert "MkTensor" in result
    assert "NativeOptimizer" in result
