"""Pytest paired with save_oracle_gpt2.py — runs the generator end-to-end
and asserts the resulting oracle is well-formed.

Mirrors `test_save_oracle.py` (the BERT counterpart). The same four
properties matter here:
  - the safetensors file exists and contains the expected key
  - the shape matches the GPT-2 model's hidden dim
  - values are finite (no NaN / Inf from a broken forward)
  - values are non-trivial (catches an accidentally zero-init / pad-fed run)

Invoke via the Makefile:
    make test-transformers-oracle-gpt2

Or manually:
    cd packages/pytorch && uv run pytest \\
        ../idris-transformers/scripts/test_save_oracle_gpt2.py -v
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import pytest
import torch
from safetensors.torch import load_file

SCRIPT = Path(__file__).resolve().parent / "save_oracle_gpt2.py"
ORACLE = Path(__file__).resolve().parent.parent / "models" / "tiny-gpt2-oracle.safetensors"
# `hf-internal-testing/tiny-random-gpt2` — random-init GPT-2 fixture
# that ships safetensors. hidden=32.
HIDDEN = 32


@pytest.fixture(scope="module")
def oracle_path() -> Path:
    """Run save_oracle_gpt2.py once for the module and return the resulting path."""
    if ORACLE.exists():
        ORACLE.unlink()
    result = subprocess.run(
        [sys.executable, str(SCRIPT)],
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, (
        f"save_oracle_gpt2.py failed (exit {result.returncode}):\n"
        f"stdout:\n{result.stdout}\n\nstderr:\n{result.stderr}"
    )
    assert ORACLE.exists(), f"script claimed success but {ORACLE} missing"
    return ORACLE


def test_oracle_shape(oracle_path: Path) -> None:
    """Final hidden state of last position — [hidden=32]."""
    tensors = load_file(str(oracle_path))
    assert "output" in tensors, f"oracle missing 'output' key; keys: {list(tensors)}"
    out = tensors["output"]
    assert out.shape == (HIDDEN,), f"shape {out.shape} != [{HIDDEN}]"


def test_oracle_finite(oracle_path: Path) -> None:
    """No NaN / Inf in the hidden state."""
    out = load_file(str(oracle_path))["output"]
    assert torch.isfinite(out).all(), "oracle has non-finite values"


def test_oracle_dtype(oracle_path: Path) -> None:
    """Oracle is F32 — matches the model's on-disk dtype."""
    out = load_file(str(oracle_path))["output"]
    assert out.dtype == torch.float32, f"oracle dtype {out.dtype} != float32"


def test_oracle_nontrivial(oracle_path: Path) -> None:
    """Hidden state isn't all-zeros (would mean we ran an untrained
    zero-init model or fed [PAD] tokens through)."""
    out = load_file(str(oracle_path))["output"]
    assert out.abs().max().item() > 1e-3, (
        f"oracle is suspiciously close to zero (max abs = {out.abs().max().item()})"
    )
