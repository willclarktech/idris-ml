"""BitNet b1.58 building-blocks — small reference oracle for the
cross-language gate.

This module composes the existing BitLinear forward (bitlinear.py)
with RMSNorm and SiLU into the BitNet MLP block. Per HF's
`BitNetMLP(GemmaMLP)` (transformers.models.bitnet.modular_bitnet):

    down_proj(ffn_sub_norm(silu(gate_proj(x)) * up_proj(x)))

The `ffn_sub_norm` (RMSNorm over the intermediate dim) is BitNet's
addition over plain Llama / Gemma — it's there so the BitLinear in
`down_proj` sees normalised activations.

Semantic choice for this oracle: our per-row absmean BitLinear
(see `bitlinear.py` and `tensor_bitlinear_fwd` in
`packages/backends/.../bitlinear.c`), NOT HF's scalar-weight_scale
+ activation-quantization variant. The scalar/act-quant variant is
what HF transformers actually computes for `microsoft/bitnet-b1.58-2B-4T`
and lands in B4.3 as a separate primitive. The per-row variant is
the one the existing `tensor_bitlinear_fwd` kernel implements, and
this oracle gates that kernel composed into a multi-layer block.

The fixture is tiny (hidden=4, intermediate=6) so the cross-
language test runs in <1s on every backend. Expected outputs are
deterministically computable from the fixture; the C-side test
embeds them as constants.
"""

from __future__ import annotations

import torch
from torch import Tensor

from .bitlinear import absmean_ternary_quant, bitlinear_forward

# --------------------------------------------------------------------
# Primitives
# --------------------------------------------------------------------


def rms_norm(x: Tensor, weight: Tensor, eps: float = 1e-5) -> Tensor:
    """Llama-style RMSNorm.

    y = x * rsqrt(mean(x^2) + eps) * weight

    Args:
        x:      [..., n] in compute dtype.
        weight: [n]     in compute dtype.
        eps:    numerical stabiliser, default 1e-5 (Llama 3 default).

    Returns:
        y: same shape as x.
    """
    if weight.shape != x.shape[-1:]:
        raise ValueError(f"rms_norm weight must be [{x.shape[-1]}], got {tuple(weight.shape)}")
    var = x.pow(2).mean(dim=-1, keepdim=True)
    x_normed = x * torch.rsqrt(var + eps)
    return x_normed * weight


# --------------------------------------------------------------------
# BitNet MLP block
# --------------------------------------------------------------------


def bitnet_mlp_forward(
    x: Tensor,
    gate_weight: tuple[Tensor, Tensor],
    up_weight: tuple[Tensor, Tensor],
    down_weight: tuple[Tensor, Tensor],
    ffn_sub_norm_weight: Tensor,
    gate_bias: Tensor | None = None,
    up_bias: Tensor | None = None,
    down_bias: Tensor | None = None,
    eps: float = 1e-5,
) -> Tensor:
    """BitNet MLP block — three BitLinears + SiLU gate + sub-norm.

    Implements `BitNetMLP.forward` from HF's modular_bitnet.py with
    BitLinear semantics swapped from HF's variant to ours (per-row
    absmean scale, no activation quant). Composes:

        gate     = bitlinear(x, gate_weight)        -- [hidden] -> [intermediate]
        up       = bitlinear(x, up_weight)          -- [hidden] -> [intermediate]
        gated    = silu(gate) * up                  -- [intermediate]
        normed   = rms_norm(gated, ffn_sub_norm_weight)
        out      = bitlinear(normed, down_weight)   -- [intermediate] -> [hidden]

    Args:
        x: [hidden] in compute dtype.
        gate_weight, up_weight, down_weight: each is a
          (ternary, scale) pair as produced by `absmean_ternary_quant`.
          - gate / up shapes: ternary [intermediate, hidden] int8,
                              scale [intermediate] compute-dtype
          - down shapes:      ternary [hidden, intermediate] int8,
                              scale [hidden] compute-dtype
        ffn_sub_norm_weight: [intermediate] compute-dtype.
        gate_bias, up_bias, down_bias: optional [intermediate] / [intermediate]
          / [hidden] bias vectors.
        eps: RMSNorm epsilon.

    Returns:
        y: [hidden] in compute dtype.
    """
    gate_t, gate_s = gate_weight
    up_t, up_s = up_weight
    down_t, down_s = down_weight

    gate = bitlinear_forward(gate_t, gate_s, x, gate_bias)  # [intermediate]
    up = bitlinear_forward(up_t, up_s, x, up_bias)  # [intermediate]
    gated = torch.nn.functional.silu(gate) * up  # [intermediate]
    normed = rms_norm(gated, ffn_sub_norm_weight, eps=eps)  # [intermediate]
    return bitlinear_forward(down_t, down_s, normed, down_bias)


# --------------------------------------------------------------------
# Small deterministic fixture
# --------------------------------------------------------------------
#
# Sized so the cross-language oracle runs in well under a second:
#   hidden       = 4
#   intermediate = 6
#
# Weights are seeded from a fixed PRNG (torch.manual_seed) then run
# through absmean_ternary_quant to get the (ternary, scale) pairs.
# The fixture seed + sizes are stable across torch versions because
# we use torch.Generator-based random not the global one.


FIXTURE_HIDDEN: int = 4
FIXTURE_INTERMEDIATE: int = 6


def fixture_inputs(dtype: torch.dtype = torch.float64) -> dict[str, Tensor]:
    """Generate the deterministic fixture for the BitNet MLP block.

    Returns a dict with keys: x, gate_t, gate_s, up_t, up_s, down_t,
    down_s, ffn_sub_norm_weight, gate_bias, up_bias, down_bias.
    """
    g = torch.Generator()
    g.manual_seed(411)  # the GitHub issue number — make it grep-able

    h, m = FIXTURE_HIDDEN, FIXTURE_INTERMEDIATE

    # Input vector (small magnitudes — keep the BitLinear product tame
    # so we don't get fp comparisons dominated by overflow rounding).
    x = torch.randn(h, generator=g, dtype=dtype) * 0.5

    # Three weight matrices, each ternarised from a random F-dtype init.
    gate_w_raw = torch.randn(m, h, generator=g, dtype=dtype)
    up_w_raw = torch.randn(m, h, generator=g, dtype=dtype)
    down_w_raw = torch.randn(h, m, generator=g, dtype=dtype)
    gate_t, gate_s = absmean_ternary_quant(gate_w_raw)
    up_t, up_s = absmean_ternary_quant(up_w_raw)
    down_t, down_s = absmean_ternary_quant(down_w_raw)

    # FFN sub-norm scale (close to 1.0, like a learned RmsNorm).
    ffn_sub_norm_weight = torch.ones(m, dtype=dtype) + 0.1 * torch.randn(
        m, generator=g, dtype=dtype
    )

    # Biases — small, nonzero so they exercise the bias-add path.
    gate_bias = 0.05 * torch.randn(m, generator=g, dtype=dtype)
    up_bias = 0.05 * torch.randn(m, generator=g, dtype=dtype)
    down_bias = 0.05 * torch.randn(h, generator=g, dtype=dtype)

    return {
        "x": x,
        "gate_t": gate_t,
        "gate_s": gate_s,
        "up_t": up_t,
        "up_s": up_s,
        "down_t": down_t,
        "down_s": down_s,
        "ffn_sub_norm_weight": ffn_sub_norm_weight,
        "gate_bias": gate_bias,
        "up_bias": up_bias,
        "down_bias": down_bias,
    }


def fixture_expected_y(dtype: torch.dtype = torch.float64) -> list[float]:
    """Compute the expected MLP-block output for the embedded fixture."""
    fx = fixture_inputs(dtype=dtype)
    y = bitnet_mlp_forward(
        fx["x"],
        gate_weight=(fx["gate_t"], fx["gate_s"]),
        up_weight=(fx["up_t"], fx["up_s"]),
        down_weight=(fx["down_t"], fx["down_s"]),
        ffn_sub_norm_weight=fx["ffn_sub_norm_weight"],
        gate_bias=fx["gate_bias"],
        up_bias=fx["up_bias"],
        down_bias=fx["down_bias"],
    )
    return y.tolist()


if __name__ == "__main__":
    # Hand-runnable: dump the F64 expected output so the cross-language
    # gate can regenerate constants if the fixture seed ever changes.
    y = fixture_expected_y()
    print("FIXTURE_EXPECTED_Y_F64 =", y)
