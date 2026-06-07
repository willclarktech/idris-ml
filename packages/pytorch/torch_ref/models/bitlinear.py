"""BitNet b1.58 BitLinear forward — reference oracle.

This module implements the inference-time forward of a BitLinear
layer: y = (W_ternary * w_scale.unsqueeze(1)) @ x + bias. Per the
2024 BitNet b1.58 paper (https://arxiv.org/abs/2402.17764) weights
are restricted to {-1, 0, +1} and dequantised to compute dtype by
multiplying with a per-row absmean scale.

`bitlinear_forward` takes the already-quantised pieces (W_ternary,
w_scale) so the oracle is deterministic — no implicit absmean
recompute — which is what the cross-language gate needs. The
`absmean_ternary_quant` helper turns a real-valued weight matrix
into the (W_ternary, w_scale) pair using the paper's recipe.

The reference is intentionally written in pure PyTorch without
nn.Module wrapping — it's a function, not a layer. The Idris-side
oracle test passes byte-identical W_ternary / w_scale / x / bias
through `tensor_bitlinear_fwd` on each backend and asserts the
result matches within 1e-4.
"""

from __future__ import annotations

from typing import cast

import torch
from torch import Tensor


def absmean_ternary_quant(weight: Tensor) -> tuple[Tensor, Tensor]:
    """Per-row absmean ternary quantization.

    For each row j of `weight`, computes scale[j] = mean(|weight[j]|)
    then ternary[j] = round(weight[j] / scale[j]).clamp(-1, 1).

    Rows whose absmean is 0 (all-zero rows) get ternary == 0 + scale == 0
    rather than NaN-from-division; this matches the calloc-zero default
    of the underlying ternary buffer.

    Args:
        weight: [o, i] in any float dtype.

    Returns:
        (ternary, scale) where
          ternary: [o, i] int8 with values in {-1, 0, +1}
          scale:   [o]    same dtype as `weight`
    """
    if weight.ndim != 2:
        raise ValueError(f"weight must be 2D, got shape {tuple(weight.shape)}")
    scale = weight.abs().mean(dim=1)  # [o]
    safe_scale = scale.clamp(min=1e-12).unsqueeze(1)  # [o, 1], no /0
    ternary_float = torch.round(weight / safe_scale).clamp(-1, 1)
    ternary_float = torch.where(
        scale.unsqueeze(1) > 0, ternary_float, torch.zeros_like(ternary_float)
    )
    return ternary_float.to(torch.int8), scale


def bitlinear_forward(
    w_ternary: Tensor,
    w_scale: Tensor,
    x: Tensor,
    bias: Tensor | None = None,
) -> Tensor:
    """BitLinear inference forward.

    y = (W_ternary.to(compute_dtype) * w_scale.unsqueeze(1)) @ x + bias

    Args:
        w_ternary: [o, i] int8 with values in {-1, 0, +1}.
        w_scale:   [o] float (compute dtype).
        x:         [i] or [b, i] float (compute dtype).
        bias:      [o] float or None.

    Returns:
        y: [o] or [b, o] in compute dtype (= w_scale.dtype).
    """
    if w_ternary.ndim != 2:
        raise ValueError(f"w_ternary must be 2D, got {tuple(w_ternary.shape)}")
    o, i = w_ternary.shape
    if w_scale.shape != (o,):
        raise ValueError(f"w_scale must be [{o}], got {tuple(w_scale.shape)}")
    if bias is not None and bias.shape != (o,):
        raise ValueError(f"bias must be [{o}], got {tuple(bias.shape)}")

    compute_dtype = w_scale.dtype
    w_dequant = w_ternary.to(compute_dtype) * w_scale.unsqueeze(1)  # [o, i]
    if x.ndim == 1:
        if x.shape != (i,):
            raise ValueError(f"x must be [{i}], got {tuple(x.shape)}")
        y = w_dequant @ x  # [o]
    elif x.ndim == 2:
        _b, i2 = x.shape
        if i2 != i:
            raise ValueError(f"x must be [b, {i}], got {tuple(x.shape)}")
        y = x @ w_dequant.t()  # [b, o]
    else:
        raise ValueError(f"x must be 1D or 2D, got ndim={x.ndim}")
    if bias is not None:
        y = y + bias
    return y


# Fixed test vector — also embedded in
# `packages/backends/test/common/core/kernels/test_bitlinear_fwd.c`
# verbatim so the C unit test and this oracle agree on bytes.
#
# Shapes: o=3, i=4.
# Pattern: nontrivial ternary mix (zeros + ±1), nontrivial scale,
# nonzero input + bias. Computed values below in F64 for cross-
# backend determinism.
FIXTURE_W_TERNARY: list[list[int]] = [
    [1, 0, -1, 1],  # row 0
    [-1, 1, 1, 0],  # row 1
    [0, -1, 0, 1],  # row 2
]
FIXTURE_W_SCALE: list[float] = [0.5, 0.25, 0.75]
FIXTURE_X: list[float] = [1.0, 2.0, -0.5, 0.25]
FIXTURE_BIAS: list[float] = [0.1, -0.2, 0.3]


def fixture_expected_y() -> list[float]:
    """Compute the expected forward output for the embedded fixture."""
    w_t = torch.tensor(FIXTURE_W_TERNARY, dtype=torch.int8)
    s = torch.tensor(FIXTURE_W_SCALE, dtype=torch.float64)
    x = torch.tensor(FIXTURE_X, dtype=torch.float64)
    b = torch.tensor(FIXTURE_BIAS, dtype=torch.float64)
    y = bitlinear_forward(w_t, s, x, b)
    # Tensor.tolist() is list[Unknown] in the torch stubs; this is a 1-D float tensor.
    return cast("list[float]", y.tolist())  # pyright: ignore[reportUnknownMemberType]


if __name__ == "__main__":
    # Hand-verifiable: print the fixture's expected output so the C
    # test's embedded constants can be regenerated if the fixture
    # ever changes.
    y = fixture_expected_y()
    print("FIXTURE_EXPECTED_Y =", y)
