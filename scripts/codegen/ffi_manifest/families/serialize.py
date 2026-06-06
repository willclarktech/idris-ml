"""Param / optimizer load + save primitives (safetensors)."""

from .._entry import Entry

ENTRIES = {
    "optimizer_load": Entry(
        args=("R", "s"),
        ret="i",
        slice="UserExecutorSerialize",
        idris_method="primOptimizerLoad",
        mlx="direct",
    ),
    "optimizer_save": Entry(
        args=("R", "s"),
        ret="i",
        slice="UserExecutorSerialize",
        idris_method="primOptimizerSave",
        mlx="direct",
    ),
    "param_load_with_policy": Entry(
        args=("s", "i"),
        ret="i",
        slice="UserExecutorSerialize",
        idris_method="primParamLoadWithPolicy",
        mlx="direct",
    ),
    "param_load_with_prefix": Entry(
        args=("s", "i", "s"),
        ret="i",
        slice="UserExecutorSerialize",
        idris_method="primParamLoadWithPrefix",
        mlx="direct",
    ),
    "param_load": Entry(
        args=("s",),
        ret="i",
        slice="UserExecutorSerialize",
        idris_method="primParamLoad",
        mlx="direct",
    ),
    "param_save": Entry(
        args=("s",),
        ret="i",
        slice="UserExecutorSerialize",
        idris_method="primParamSave",
        mlx="direct",
    ),
    "param_save_by_name": Entry(
        args=("s", "s", "i"),
        ret="i",
        slice="UserExecutorSerialize",
        idris_method="primParamSaveByName",
        mlx="direct",
    ),
    "param_save_by_name_renamed": Entry(
        args=("s", "s", "s", "i"),
        ret="i",
        slice="UserExecutorSerialize",
        idris_method="primParamSaveByNameRenamed",
        mlx="direct",
    ),
}
