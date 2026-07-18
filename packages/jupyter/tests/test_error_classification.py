"""Tests for the kernel's cell-failure classification.

The unit fixtures are real REPL outputs captured while authoring
models/bert.ipynb (2026-07-26), when a notebook whose cells were all
parse errors passed `make test-e2e-notebooks`: the kernel only flagged
outputs containing "Error:" or "Exception:", and neither parse failures
nor module-load failures carry those markers.

Run via:
    make test-e2e-jupyter
"""

from idris_ml_kernel.kernel import is_error_output

# --- real captured failure outputs (must classify as errors) ----------

PARSE_ERROR = (
    "Couldn't parse any alternatives:\n"
    "1: Expected 'case', 'if', 'do', application or operator expression.\n"
    "\n"
    "(Interactive):1:1--1:7\n"
    " 1 | import Transformers.Bert\n"
    "     ^^^^^^\n"
    "... (47 others)"
)

MODULE_LOAD_ERROR = (
    "Error loading module Transformers.Bert: Module Language.Reflection.Util not found"
)

UNCAUGHT_ERROR = "Uncaught error: Can't find package idris-ml (any)"

TYPED_ERROR = (
    "Error: While processing right hand side of bad. When unifying:\n"
    "    Tensor [4] TapeExecutor F64 NoGrad\n"
    "and:\n"
    "    Tensor [3] TapeExecutor F64 NoGrad"
)

RUNTIME_EXCEPTION = "Exception: invalid memory reference.  Some debugging context lost"


def test_parse_failure_is_error() -> None:
    assert is_error_output(PARSE_ERROR)


def test_module_load_failure_is_error() -> None:
    assert is_error_output(MODULE_LOAD_ERROR)


def test_uncaught_error_is_error() -> None:
    assert is_error_output(UNCAUGHT_ERROR)


def test_typed_error_is_error() -> None:
    assert is_error_output(TYPED_ERROR)


def test_runtime_exception_is_error() -> None:
    assert is_error_output(RUNTIME_EXCEPTION)


# --- healthy outputs (must NOT classify as errors) ---------------------


def test_clean_outputs_are_ok() -> None:
    for ok in (
        "Imported module Transformers.Bert",
        "pooled[0..2] = -0.9999995783617099, 0.11944057093063778",
        "one SGD step done; loss = 1.0463",
        "Tensor.MkTensor : AnyPtr -> Maybe String -> Tensor dims ex dt g",
        "5",
        "",
    ):
        assert not is_error_output(ok), ok
