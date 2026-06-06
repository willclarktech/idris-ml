"""Arena reset / epoch begin-end / persistent buffer release."""

from .._entry import Entry

ENTRIES = {
    "tensor_epoch_begin": Entry(
        args=(),
        ret="v",
        slice="UserExecutorMemoryHygiene",
        idris_method="primEpochBegin",
        mlx="direct",
    ),
    "tensor_epoch_end": Entry(
        args=(),
        ret="v",
        slice="UserExecutorMemoryHygiene",
        idris_method="primEpochEnd",
        mlx="direct",
    ),
    "tensor_release_all_persistent": Entry(
        args=(),
        ret="v",
        slice="UserExecutorMemoryHygiene",
        idris_method="primReleaseAllPersistent",
        c_symbol="backend_release_all_persistent",
        mlx="direct",
    ),
    "tensor_reset_for_eval": Entry(
        args=(),
        ret="v",
        slice="UserExecutorMemoryHygiene",
        idris_method="primResetForEval",
        c_symbol="backend_reset_for_eval",
        mlx="direct",
    ),
}
