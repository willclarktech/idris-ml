"""Backward + grad-mode + requires-grad primitives."""

from .._entry import Entry


ENTRIES = {
    "tensor_backward": Entry(
        args=("T",),
        ret="v",
        slice="UserExecutorAutograd",
        idris_method="primBackward",
        mlx="direct",
    ),
    "tensor_detach": Entry(
        args=("T",), ret="T", slice="UserExecutorAutograd", idris_method="primDetach"
    ),
    "tensor_no_grad_begin": Entry(
        args=(), ret="v", slice="UserExecutorAutograd", idris_method="primNoGradBegin", mlx="direct"
    ),
    "tensor_no_grad_end": Entry(
        args=(), ret="v", slice="UserExecutorAutograd", idris_method="primNoGradEnd", mlx="direct"
    ),
    "tensor_requires_grad": Entry(
        args=("T",),
        ret="i",
        slice="UserExecutorAutograd",
        idris_method="primRequiresGrad",
        mlx="direct",
    ),
    "tensor_set_requires_grad": Entry(
        args=("T", "i"),
        ret="v",
        slice="UserExecutorAutograd",
        idris_method="primSetRequiresGrad",
        mlx="direct",
    ),
    "tensor_with_grad": Entry(
        args=("T",), ret="T", slice="UserExecutorAutograd", idris_method="primWithGrad"
    ),
}
