"""Quantization primitives — BitLinear, ternary pack/unpack, per-row mean."""

from .._entry import Entry


ENTRIES = {
    "tensor_absmean_per_row_2d": Entry(
        args=("T",),
        ret="T",
        slice="UserExecutorQuant",
        idris_method="primAbsmeanPerRow2d",
        mlx="bespoke",
    ),
    "tensor_bitlinear_fwd_hf_quant": Entry(
        args=("T", "d", "T", "T", "i", "T", "d"),
        ret="T",
        slice="UserExecutorQuant",
        idris_method="primBitlinearFwdHfQuant",
        mlx="bespoke",
    ),
    "tensor_bitlinear_fwd": Entry(
        args=("T", "T", "T", "T"),
        ret="T",
        slice="UserExecutorQuant",
        idris_method="primBitlinearFwd",
        mlx="bespoke",
    ),
    "tensor_create_ternary_from_hf_packed_2d": Entry(
        args=("R", "i", "i"),
        ret="T",
        slice="UserExecutorQuant",
        idris_method="primCreateTernaryFromHfPacked2d",
        mlx="bespoke",
    ),
    "tensor_create_ternary_packed_2d": Entry(
        args=("R", "i", "i", "i", "i"),
        ret="T",
        slice="UserExecutorQuant",
        idris_method="primCreateTernaryPacked2d",
        mlx="bespoke",
    ),
    "tensor_ternary_quant_with_scale_2d": Entry(
        args=("T", "T"),
        ret="T",
        slice="UserExecutorQuant",
        idris_method="primTernaryQuantWithScale2d",
        mlx="bespoke",
    ),
}
