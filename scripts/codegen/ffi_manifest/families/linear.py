"""Linear-algebra primitives — matmul, reshape, sort, reductions."""

from .._entry import Entry

ENTRIES = {
    "tensor_argsort": Entry(
        args=("T", "i", "i"), ret="T", slice="UserExecutorLinear", idris_method="primArgsort"
    ),
    "tensor_bmm": Entry(
        args=("T", "T"), ret="T", slice="UserExecutorLinear", idris_method="primBmm"
    ),
    "tensor_cat": Entry(
        args=("R", "i", "i"), ret="T", slice="UserExecutorLinear", idris_method="primCat"
    ),
    "tensor_cat2": Entry(
        args=("T", "T"), ret="T", slice="UserExecutorLinear", idris_method="primCat2"
    ),
    "tensor_concat_2d_axis1": Entry(
        args=("T", "T"), ret="T", slice="UserExecutorLinear", idris_method="primConcat2dAxis1"
    ),
    "tensor_cumprod": Entry(
        args=("T", "i"), ret="T", slice="UserExecutorLinear", idris_method="primCumprod"
    ),
    "tensor_dot": Entry(
        args=("T", "T"), ret="T", slice="UserExecutorLinear", idris_method="primDot"
    ),
    "tensor_gather": Entry(
        args=("T", "T", "i"), ret="T", slice="UserExecutorLinear", idris_method="primGather"
    ),
    "tensor_linear_2d": Entry(
        args=("T", "T", "T"), ret="T", slice="UserExecutorLinear", idris_method="primLinear2d"
    ),
    "tensor_linear": Entry(
        args=("T", "T", "T"), ret="T", slice="UserExecutorLinear", idris_method="primLinear"
    ),
    "tensor_matmul": Entry(
        args=("T", "T"), ret="T", slice="UserExecutorLinear", idris_method="primMatmul"
    ),
    "tensor_mean": Entry(args=("T",), ret="T", slice="UserExecutorLinear", idris_method="primMean"),
    "tensor_mm": Entry(args=("T", "T"), ret="T", slice="UserExecutorLinear", idris_method="primMm"),
    "tensor_mv": Entry(args=("T", "T"), ret="T", slice="UserExecutorLinear", idris_method="primMv"),
    "tensor_narrow": Entry(
        args=("T", "i", "i", "i"), ret="T", slice="UserExecutorLinear", idris_method="primNarrow"
    ),
    "tensor_outer": Entry(
        args=("T", "T"), ret="T", slice="UserExecutorLinear", idris_method="primOuter"
    ),
    "tensor_reshape_1d": Entry(
        args=("T", "i"), ret="T", slice="UserExecutorLinear", idris_method="primReshape1d"
    ),
    "tensor_reshape_2d": Entry(
        args=("T", "i", "i"), ret="T", slice="UserExecutorLinear", idris_method="primReshape2d"
    ),
    "tensor_reshape_3d": Entry(
        args=("T", "i", "i", "i"), ret="T", slice="UserExecutorLinear", idris_method="primReshape3d"
    ),
    "tensor_reshape_4d": Entry(
        args=("T", "i", "i", "i", "i"),
        ret="T",
        slice="UserExecutorLinear",
        idris_method="primReshape4d",
    ),
    "tensor_scatter_add": Entry(
        args=("T", "T", "i"), ret="T", slice="UserExecutorLinear", idris_method="primScatterAdd"
    ),
    "tensor_select": Entry(
        args=("T", "i", "i"), ret="T", slice="UserExecutorLinear", idris_method="primSelect"
    ),
    "tensor_squeeze": Entry(
        args=("T", "i"), ret="T", slice="UserExecutorLinear", idris_method="primSqueeze"
    ),
    "tensor_stack": Entry(
        args=("R", "i", "i"), ret="T", slice="UserExecutorLinear", idris_method="primStack"
    ),
    "tensor_sum_dim": Entry(
        args=("T", "i", "i"), ret="T", slice="UserExecutorLinear", idris_method="primSumDim"
    ),
    "tensor_sum": Entry(args=("T",), ret="T", slice="UserExecutorLinear", idris_method="primSum"),
    "tensor_tensor_max": Entry(
        args=("T",),
        ret="T",
        slice="UserExecutorLinear",
        idris_method="primTensorMax",
        c_symbol="tensor_max",
    ),
    "tensor_tensor_min": Entry(
        args=("T",),
        ret="T",
        slice="UserExecutorLinear",
        idris_method="primTensorMin",
        c_symbol="tensor_min",
    ),
    "tensor_transpose_2d": Entry(
        args=("T",), ret="T", slice="UserExecutorLinear", idris_method="primTranspose2d"
    ),
    "tensor_transpose_last2": Entry(
        args=("T",), ret="T", slice="UserExecutorLinear", idris_method="primTransposeLast2"
    ),
    "tensor_unsqueeze": Entry(
        args=("T", "i"), ret="T", slice="UserExecutorLinear", idris_method="primUnsqueeze"
    ),
    "tensor_view_1d": Entry(
        args=("T", "i"), ret="T", slice="UserExecutorLinear", idris_method="primView1d"
    ),
    "tensor_view_2d": Entry(
        args=("T", "i", "i"), ret="T", slice="UserExecutorLinear", idris_method="primView2d"
    ),
}
