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
    result = repl.send(":t Var")
    assert "AnyPtr" in result
    assert "Variable" in result


def test_doc_query(repl: Idris2REPL) -> None:
    result = repl.send(":doc Var")
    assert "Variable" in result


def test_module_import(repl: Idris2REPL) -> None:
    result = repl.send(":module Layer.Core")
    assert "Imported" in result


def test_let_definition(repl: Idris2REPL) -> None:
    repl.send(":let myTestFn : Int -> Int")
    repl.send(":let myTestFn x = x + 42")
    result = repl.send(":t myTestFn")
    assert "Int -> Int" in result


def test_expression_eval(repl: Idris2REPL) -> None:
    result = repl.send("2 + 3")
    assert "5" in result


def test_ffi_tensor_create(repl: Idris2REPL) -> None:
    """Core FFI test: create a scalar tensor and read its value."""
    result = repl.send(
        ":exec (let t = prim__createScalar 3.14 0 in putStrLn (show (prim__item t)))"
    )
    assert "3.14" in result


def test_ffi_tensor_arithmetic(repl: Idris2REPL) -> None:
    """FFI test: tensor multiply."""
    result = repl.send(
        ":exec (let a = prim__createScalar 2.0 1 in "
        "let b = prim__createScalar 3.0 0 in "
        "let c = prim__mul a b in "
        "putStrLn (show (prim__item c)))"
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
    # REPL should still be alive
    result2 = repl.send("1 + 1")
    assert "2" in result2


def test_browse(repl: Idris2REPL) -> None:
    result = repl.send(":browse Variable")
    assert "Var" in result
    assert "NativeOptimizer" in result
