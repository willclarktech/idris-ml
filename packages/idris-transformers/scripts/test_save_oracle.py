"""Pytest paired with save_oracle.py — runs the generator end-to-end
and asserts the resulting oracle is well-formed.

Invoke via the Makefile:
    make test-transformers-oracle

Or manually:
    cd packages/pytorch && uv run pytest \\
        ../idris-transformers/scripts/test_save_oracle.py -v
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import pytest
import torch
from safetensors.torch import load_file

SCRIPT = Path(__file__).resolve().parent / "save_oracle.py"
ORACLE = Path(__file__).resolve().parent.parent / "models" / "bert-tiny-oracle.safetensors"
HIDDEN = 128


@pytest.fixture(scope="module")
def oracle_path() -> Path:
    """Run save_oracle.py once for the module and return the resulting path."""
    # Always regenerate so a stale fixture from a previous run can't
    # mask a regression in the generator itself.
    if ORACLE.exists():
        ORACLE.unlink()
    result = subprocess.run(
        [sys.executable, str(SCRIPT)],
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, (
        f"save_oracle.py failed (exit {result.returncode}):\n"
        f"stdout:\n{result.stdout}\n\nstderr:\n{result.stderr}"
    )
    assert ORACLE.exists(), f"script claimed success but {ORACLE} missing"
    return ORACLE


def test_oracle_shape(oracle_path: Path) -> None:
    """The pooled output is the [CLS] token's projection, shape [hidden=128]."""
    tensors = load_file(str(oracle_path))
    assert "output" in tensors, f"oracle missing 'output' key; keys: {list(tensors)}"
    out = tensors["output"]
    assert out.shape == (HIDDEN,), f"shape {out.shape} != [{HIDDEN}]"


def test_oracle_finite(oracle_path: Path) -> None:
    """No NaN / Inf in the pooled output."""
    out = load_file(str(oracle_path))["output"]
    assert torch.isfinite(out).all(), "oracle has non-finite values"


def test_oracle_dtype(oracle_path: Path) -> None:
    """Oracle is F32 — matches the model's on-disk dtype, no surprise widening."""
    out = load_file(str(oracle_path))["output"]
    assert out.dtype == torch.float32, f"oracle dtype {out.dtype} != float32"


def test_oracle_nontrivial(oracle_path: Path) -> None:
    """Pooled output isn't all-zeros (would mean we ran a zero-init untrained
    model by accident, or the tokenizer fed [PAD] through)."""
    out = load_file(str(oracle_path))["output"]
    assert out.abs().max().item() > 1e-3, (
        f"oracle is suspiciously close to zero (max abs = {out.abs().max().item()})"
    )
