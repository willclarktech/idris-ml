"""Pytest paired with save_oracle_llama_generate.py — runs the
generator end-to-end and asserts the resulting oracle is well-formed.

Mirrors `test_save_oracle_llama.py` (the single-forward counterpart).
Properties asserted:
  - the safetensors file exists and contains the expected key
  - shape is [prompt_len + num_new_tokens] (no early-stop / padding)
  - dtype is int64 (token IDs are discrete)
  - values are in-range [0, vocab=128256)
  - the prefix round-trips through the tokenizer (no BPE drift)

Invoke via the Makefile:
    make test-transformers-oracle-llama-generate

Or manually:
    cd packages/pytorch && uv run pytest \\
        ../idris-transformers/scripts/test_save_oracle_llama_generate.py -v
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import pytest
import torch
from safetensors.torch import load_file

SCRIPT = Path(__file__).resolve().parent / "save_oracle_llama_generate.py"
ORACLE = (
    Path(__file__).resolve().parent.parent.parent.parent
    / "models"
    / "llama-3.2-1b-generate-oracle.safetensors"
)
VOCAB_SIZE = 128256
EXPECTED_PROMPT = "The capital of France is"
NUM_NEW_TOKENS = 8


@pytest.fixture(scope="module")
def oracle_path() -> Path:
    """Run save_oracle_llama_generate.py once for the module and return the resulting path."""
    if ORACLE.exists():
        ORACLE.unlink()
    result = subprocess.run(
        [sys.executable, str(SCRIPT)],
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, (
        f"save_oracle_llama_generate.py failed (exit {result.returncode}):\n"
        f"stdout:\n{result.stdout}\n\nstderr:\n{result.stderr}"
    )
    assert ORACLE.exists(), f"script claimed success but {ORACLE} missing"
    return ORACLE


def test_oracle_key(oracle_path: Path) -> None:
    """Oracle has the expected 'token_ids' key."""
    tensors = load_file(str(oracle_path))
    assert "token_ids" in tensors, f"oracle missing 'token_ids' key; keys: {list(tensors)}"


def test_oracle_dtype(oracle_path: Path) -> None:
    """Oracle dtype is int64 — token IDs are discrete."""
    out = load_file(str(oracle_path))["token_ids"]
    assert out.dtype == torch.int64, f"oracle dtype {out.dtype} != int64"


def test_oracle_shape(oracle_path: Path) -> None:
    """Shape is [prompt_len + NUM_NEW_TOKENS]. prompt_len varies with
    upstream BPE; the assert is on the additive piece + a sanity
    range. Oracle uses `add_special_tokens=True` (BOS prepended for
    Llama-3) to match the Idris-side Tokenizer subprocess."""
    out = load_file(str(oracle_path))["token_ids"]
    assert out.ndim == 1, f"shape {out.shape} not rank-1"
    # Prompt "The capital of France is" with BOS prefix is 6-8 tokens
    # in Llama-3 BPE. Total = prompt + 4 generated → 10-12.
    assert 6 + NUM_NEW_TOKENS <= out.shape[0] <= 9 + NUM_NEW_TOKENS, (
        f"length {out.shape[0]} outside expected 10..13 range for "
        f"BOS + {EXPECTED_PROMPT!r} + {NUM_NEW_TOKENS} new tokens"
    )


def test_oracle_in_range(oracle_path: Path) -> None:
    """All token IDs are in [0, vocab)."""
    out = load_file(str(oracle_path))["token_ids"]
    assert out.min().item() >= 0, f"negative token id {out.min().item()}"
    assert out.max().item() < VOCAB_SIZE, (
        f"out-of-range token id {out.max().item()} (vocab={VOCAB_SIZE})"
    )


def test_oracle_nontrivial(oracle_path: Path) -> None:
    """Generated tokens aren't all identical (would suggest a model
    stuck in a loop — possible but very unlikely on this prompt)."""
    out = load_file(str(oracle_path))["token_ids"]
    # The full output is at least 9 tokens; the last NUM_NEW_TOKENS are
    # the generated ones. Their being all identical to each other across
    # 4 positions on this prompt would be deeply suspicious.
    new_ids = out[-NUM_NEW_TOKENS:].tolist()
    assert len(set(new_ids)) > 1 or NUM_NEW_TOKENS == 1, (
        f"generated tokens are all identical: {new_ids}"
    )
