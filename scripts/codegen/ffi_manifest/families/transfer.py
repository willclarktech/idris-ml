"""Host ↔ device transfer + intra-backend migration primitives."""

from .._entry import Entry

ENTRIES = {
    "tensor_alloc_host": Entry(
        args=("i",),
        ret="T",
        slice="UserExecutorTransfer",
        idris_method="primAllocHost",
        c_symbol="tensor_alloc_doubles",
        mlx="direct",
    ),
    "tensor_alloc_int_host": Entry(
        args=("i",),
        ret="T",
        slice="UserExecutorTransfer",
        idris_method="primAllocIntHost",
        c_symbol="tensor_alloc_ints",
        mlx="direct",
    ),
    "tensor_create_from_host": Entry(
        args=("R", "R", "i", "i"),
        ret="T",
        slice="UserExecutorTransfer",
        idris_method="primCreateFromHost",
        c_symbol="tensor_create",
        torch="bespoke",
        mlx="direct",
    ),
    "tensor_free_host": Entry(
        args=("T",),
        ret="v",
        slice="UserExecutorTransfer",
        idris_method="primFreeHost",
        c_symbol="tensor_free_doubles",
        mlx="direct",
    ),
    "tensor_free_int_host": Entry(
        args=("T",),
        ret="v",
        slice="UserExecutorTransfer",
        idris_method="primFreeIntHost",
        c_symbol="tensor_free_ints",
        mlx="direct",
    ),
    "tensor_intra_migrate": Entry(
        args=("T", "s"),
        ret="T",
        slice="UserExecutorTransfer",
        idris_method="primIntraMigrate",
        c_symbol="tensor_to_device",
        torch="bespoke",
        mlx="direct",
    ),
    "tensor_set_int_host": Entry(
        args=("T", "i", "i"),
        ret="T",
        slice="UserExecutorTransfer",
        idris_method="primSetIntHost",
        c_symbol="tensor_write_int_return",
        mlx="direct",
    ),
    "tensor_to_host": Entry(
        args=("T", "R"),
        ret="R",
        slice="UserExecutorTransfer",
        idris_method="primToHost",
        c_symbol="tensor_to_doubles_return",
        mlx="direct",
    ),
}
