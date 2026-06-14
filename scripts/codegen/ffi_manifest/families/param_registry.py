"""Param metadata — count, name, grad access, register/erase."""

from .._entry import Entry

ENTRIES = {
    "param_count": Entry(
        args=(),
        ret="i",
        slice="UserExecutorParamRegistry",
        idris_method="primParamCount",
        mlx="direct",
    ),
    "param_grad_item_at": Entry(
        args=("i", "i"),
        ret="d",
        slice="UserExecutorParamRegistry",
        idris_method="primParamGradItemAt",
        mlx="direct",
    ),
    "param_name": Entry(
        args=("i",),
        ret="s",
        slice="UserExecutorParamRegistry",
        idris_method="primParamName",
        mlx="direct",
    ),
    "param_register": Entry(
        args=("s", "T"),
        ret="T",
        slice="UserExecutorParamRegistry",
        idris_method="primParamRegister",
        c_symbol="param_register_return",
        mlx="direct",
    ),
    "param_register_buffer": Entry(
        args=("s", "T"),
        ret="T",
        slice="UserExecutorParamRegistry",
        idris_method="primParamRegisterBuffer",
        c_symbol="param_register_buffer_return",
        mlx="direct",
    ),
    "param_is_buffer": Entry(
        args=("i",),
        ret="i",
        slice="UserExecutorParamRegistry",
        idris_method="primParamIsBuffer",
        mlx="direct",
    ),
    "param_zero_all": Entry(
        args=(),
        ret="v",
        slice="UserExecutorParamRegistry",
        idris_method="primParamZeroAll",
        c_symbol="param_zero_all_grads",
        mlx="direct",
    ),
    "param_erase_by_prefix": Entry(
        args=("s",),
        ret="v",
        slice="UserExecutorParamRegistry",
        idris_method="primParamEraseByPrefix",
        mlx="direct",
    ),
}
