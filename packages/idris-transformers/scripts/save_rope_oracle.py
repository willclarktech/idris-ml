"""Produce a RoPE table oracle for `Test.RoPE`.

Computes Llama 3's NTK-aware RoPE inv_freq + cos/sin tables for the
`Llama-3.2-1B` config (head_dim=64, rope_theta=500000, scaling factor
32 with low=1.0 high=4.0 original_max_position=8192). Saves a tiny
slice that the Idris-side unit tests pin against:

  - inv_freq[0..31]                    — full Llama-3-scaled inv_freq
  - cos_table[0..3, 0..31]             — first 4 positions × 32 dims
  - sin_table[0..3, 0..31]

The Idris side's `Layer/RoPE.idr` materialises the same numbers via
pure Idris Double math (`baseInvFreq` + `applyLlamaFreqScaling` +
`tabulate`). The oracle gate catches formula transcription errors at
the *inv_freq* step — before any tensor / FFI machinery enters.

Usage:
    cd packages/pytorch && uv run python \\
        ../idris-transformers/scripts/save_rope_oracle.py
"""

from __future__ import annotations

import math
from pathlib import Path

import torch
from safetensors.torch import save_file

SCRIPT_DIR = Path(__file__).resolve().parent
PKG_DIR    = SCRIPT_DIR.parent
MODELS_DIR = PKG_DIR / "models"
ORACLE_PATH = MODELS_DIR / "llama3-rope-oracle.safetensors"

# Llama 3.2 1B's RoPE config.
HEAD_DIM    = 64
ROPE_BASE   = 500000.0
FACTOR      = 32.0
LOW_FREQ    = 1.0
HIGH_FREQ   = 4.0
ORIG_MAXPOS = 8192


def _compute_llama3_parameters(inv_freq: torch.Tensor) -> torch.Tensor:
    """Apply Llama 3 NTK-aware scaling to the base inv_freq tensor.

    Direct port of HF transformers' `_compute_llama3_parameters` from
    `models/llama/modeling_rope_utils.py`. Operates element-wise on
    the 1-D `inv_freq` tensor.
    """
    wavelen = 2 * math.pi / inv_freq
    low_band  = ORIG_MAXPOS / HIGH_FREQ
    high_band = ORIG_MAXPOS / LOW_FREQ
    scaled    = inv_freq / FACTOR
    smooth    = (ORIG_MAXPOS / wavelen - LOW_FREQ) / (HIGH_FREQ - LOW_FREQ)
    interp    = (1 - smooth) * scaled + smooth * inv_freq

    out = inv_freq.clone()
    high_freq_mask = wavelen < low_band
    low_freq_mask  = wavelen > high_band
    mid_freq_mask  = ~high_freq_mask & ~low_freq_mask
    out = torch.where(high_freq_mask, inv_freq, out)
    out = torch.where(low_freq_mask,  scaled,   out)
    out = torch.where(mid_freq_mask,  interp,   out)
    return out


def main() -> None:
    MODELS_DIR.mkdir(parents=True, exist_ok=True)

    half_dim = HEAD_DIM // 2

    # Base inv_freq: 1 / base^(2i/d) for i in 0..half_dim
    i_range  = torch.arange(half_dim, dtype=torch.float64)
    base_inv = 1.0 / (ROPE_BASE ** (2.0 * i_range / HEAD_DIM))

    # Llama 3 NTK-aware scaling.
    inv_freq = _compute_llama3_parameters(base_inv)

    # cos / sin tables for positions 0..3.
    n_positions = 4
    positions   = torch.arange(n_positions, dtype=torch.float64)
    angles      = positions.unsqueeze(1) * inv_freq.unsqueeze(0)  # [n_pos, half_dim]
    cos_tab     = torch.cos(angles)
    sin_tab     = torch.sin(angles)

    assert inv_freq.shape == (half_dim,)
    assert cos_tab.shape  == (n_positions, half_dim)
    assert sin_tab.shape  == (n_positions, half_dim)
    assert torch.isfinite(inv_freq).all()
    assert torch.isfinite(cos_tab).all()
    assert torch.isfinite(sin_tab).all()

    save_file(
        {
            "inv_freq": inv_freq.contiguous(),
            "cos_table": cos_tab.contiguous(),
            "sin_table": sin_tab.contiguous(),
        },
        str(ORACLE_PATH),
    )
    print(f"wrote {ORACLE_PATH}")
    print(f"  inv_freq[0]   = {inv_freq[0].item():.17e}")
    print(f"  inv_freq[16]  = {inv_freq[16].item():.17e}")
    print(f"  inv_freq[31]  = {inv_freq[31].item():.17e}")
    print(f"  cos_tab[1, 0] = {cos_tab[1, 0].item():.17e}")
    print(f"  sin_tab[1, 0] = {sin_tab[1, 0].item():.17e}")


if __name__ == "__main__":
    main()
