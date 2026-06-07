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

from typing import cast

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


def _as_floats(t: Tensor) -> list[float]:
    """Typed view of Tensor.tolist() for 1-D float tensors (stub returns list[Unknown])."""
    return cast("list[float]", t.tolist())  # pyright: ignore[reportUnknownMemberType]


def _as_int_rows(t: Tensor) -> list[list[int]]:
    """Typed view of Tensor.tolist() for 2-D integer tensors (stub returns list[Unknown])."""
    return cast("list[list[int]]", t.tolist())  # pyright: ignore[reportUnknownMemberType]


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
    return _as_floats(y)


def _pack_row_2bit(row: list[int]) -> list[int]:
    """Pack one row of ternary values into our 2-bit-per-slot byte layout.

    Encoding: {0 -> 00, 1 -> 01, -1 -> 11} (two's-complement codes),
    slot 0 in low two bits. Matches `decode_slot` in
    packages/backends/.../bitlinear.c. Rows are padded with zeros
    (code 00) to fill the last byte; the byte count per row is
    (len(row) + 3) // 4.
    """
    n = len(row)
    nbytes = (n + 3) // 4
    out = [0] * nbytes
    for k, v in enumerate(row):
        code = {0: 0, 1: 1, -1: 3}[int(v)]
        byte_idx = k >> 2
        slot = k & 0x3
        out[byte_idx] |= code << (slot * 2)
    return out


def _fmt_doubles(xs: list[float]) -> str:
    """Idris Vect-of-Double literal body: `[d0, d1, ...]` with no wrapper."""
    return "[" + ", ".join(f"{float(v):.17g}" for v in xs) + "]"


def _fmt_bytes(bs: list[int]) -> str:
    """Idris Vect-of-Int literal body."""
    return "[" + ", ".join(str(b) for b in bs) + "]"


def dump_idris_fixture() -> None:
    """Emit the fixture as Idris constants for the cross-language test.

    Prints a block of `let ... = the (Vect _ ...) [...]` declarations
    that the Idris BitNet block oracle test pastes in verbatim. Run
    after the fixture seed / size changes; the Idris side is then
    re-built against the new constants.
    """
    fx = fixture_inputs(dtype=torch.float64)
    h, m = FIXTURE_HIDDEN, FIXTURE_INTERMEDIATE

    # Per-row packed bytes for each ternary weight matrix.
    gate_rows = _as_int_rows(fx["gate_t"])  # [m, h]
    up_rows = _as_int_rows(fx["up_t"])
    down_rows = _as_int_rows(fx["down_t"])  # [h, m]
    gate_packed: list[list[int]] = [_pack_row_2bit(r) for r in gate_rows]
    up_packed: list[list[int]] = [_pack_row_2bit(r) for r in up_rows]
    down_packed: list[list[int]] = [_pack_row_2bit(r) for r in down_rows]

    print("-- generated by `python -m torch_ref.models.bitnet --dump-idris`")
    print(f"-- hidden={h}, intermediate={m}, seed=411")
    print("-- DO NOT hand-edit: re-run the dumper if the fixture changes")
    print()
    print("FIXTURE_HIDDEN : Nat")
    print(f"FIXTURE_HIDDEN = {h}")
    print()
    print("FIXTURE_INTERMEDIATE : Nat")
    print(f"FIXTURE_INTERMEDIATE = {m}")
    print()
    print(f"FIXTURE_X : Vect {h} Double")
    print(f"FIXTURE_X = {_fmt_doubles(_as_floats(fx['x']))}")
    print()

    # Each ternary matrix is dumped row-by-row as a list-of-bytes; the
    # Idris side concatenates and shoves through prim__allocBytes /
    # prim__setByte to form the packed buffer.
    bytes_per_h = (h + 3) // 4
    bytes_per_m = (m + 3) // 4

    print(f"-- gate_proj weight: shape [{m}, {h}], packed bytes per row = {bytes_per_h}")
    flat_gate = [b for row in gate_packed for b in row]
    print(f"FIXTURE_GATE_W_BYTES : Vect {m * bytes_per_h} Int")
    print(f"FIXTURE_GATE_W_BYTES = {_fmt_bytes(flat_gate)}")
    print(f"FIXTURE_GATE_S : Vect {m} Double")
    print(f"FIXTURE_GATE_S = {_fmt_doubles(_as_floats(fx['gate_s']))}")
    print(f"FIXTURE_GATE_B : Vect {m} Double")
    print(f"FIXTURE_GATE_B = {_fmt_doubles(_as_floats(fx['gate_bias']))}")
    print()

    print(f"-- up_proj weight: shape [{m}, {h}], packed bytes per row = {bytes_per_h}")
    flat_up = [b for row in up_packed for b in row]
    print(f"FIXTURE_UP_W_BYTES : Vect {m * bytes_per_h} Int")
    print(f"FIXTURE_UP_W_BYTES = {_fmt_bytes(flat_up)}")
    print(f"FIXTURE_UP_S : Vect {m} Double")
    print(f"FIXTURE_UP_S = {_fmt_doubles(_as_floats(fx['up_s']))}")
    print(f"FIXTURE_UP_B : Vect {m} Double")
    print(f"FIXTURE_UP_B = {_fmt_doubles(_as_floats(fx['up_bias']))}")
    print()

    print(f"-- down_proj weight: shape [{h}, {m}], packed bytes per row = {bytes_per_m}")
    flat_down = [b for row in down_packed for b in row]
    print(f"FIXTURE_DOWN_W_BYTES : Vect {h * bytes_per_m} Int")
    print(f"FIXTURE_DOWN_W_BYTES = {_fmt_bytes(flat_down)}")
    print(f"FIXTURE_DOWN_S : Vect {h} Double")
    print(f"FIXTURE_DOWN_S = {_fmt_doubles(_as_floats(fx['down_s']))}")
    print(f"FIXTURE_DOWN_B : Vect {h} Double")
    print(f"FIXTURE_DOWN_B = {_fmt_doubles(_as_floats(fx['down_bias']))}")
    print()

    print(f"FIXTURE_FFN_SUB_NORM : Vect {m} Double")
    print(f"FIXTURE_FFN_SUB_NORM = {_fmt_doubles(_as_floats(fx['ffn_sub_norm_weight']))}")
    print()
    print(f"FIXTURE_EXPECTED_Y : Vect {h} Double")
    print(f"FIXTURE_EXPECTED_Y = {_fmt_doubles(fixture_expected_y())}")


if __name__ == "__main__":
    import sys

    if "--dump-idris" in sys.argv:
        dump_idris_fixture()
    else:
        # Hand-runnable: dump the F64 expected output so the cross-
        # language gate can regenerate constants if the fixture seed
        # ever changes.
        y = fixture_expected_y()
        print("FIXTURE_EXPECTED_Y_F64 =", y)
