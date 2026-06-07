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
    # All-bespoke: each backend hand-writes a thin wrapper over its
    # existing `prim__createStreamed<B>` binding (the dtag-dispatch
    # create), fixing the stream/hw routing per backend — tape pins
    # stream 0, mlx threads `streamTag s`, torch creates on CPU then
    # migrates to the hw variant. The trailing arg is the RuntimeDType
    # dtag, so cross-backend `toExecutor` hops construct destination
    # storage matching the type-level `dt`.
    "tensor_create_from_host": Entry(
        args=("R", "R", "i", "i", "i"),
        ret="T",
        slice="UserExecutorTransfer",
        idris_method="primCreateFromHost",
        c_symbol="tensor_create_streamed",
        tape="bespoke",
        torch="bespoke",
        mlx="bespoke",
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
