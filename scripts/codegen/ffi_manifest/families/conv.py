"""Convolution + pooling primitives."""

from .._entry import Entry

ENTRIES = {
    "tensor_avg_pool1d": Entry(
        args=("T", "i", "i"), ret="T", slice="UserExecutorConv", idris_method="primAvgPool1d"
    ),
    "tensor_avg_pool2d": Entry(
        args=("T", "i", "i", "i", "i"),
        ret="T",
        slice="UserExecutorConv",
        idris_method="primAvgPool2d",
    ),
    "tensor_conv1d_circular": Entry(
        args=("T", "T"), ret="T", slice="UserExecutorConv", idris_method="primConv1dCircular"
    ),
    "tensor_conv1d": Entry(
        args=("T", "T", "T", "i", "i"), ret="T", slice="UserExecutorConv", idris_method="primConv1d"
    ),
    "tensor_conv2d_batched": Entry(
        args=("T", "T", "T", "i", "i", "i", "i"),
        ret="T",
        slice="UserExecutorConv",
        idris_method="primConv2dBatched",
    ),
    "tensor_conv2d": Entry(
        args=("T", "T", "T", "i", "i", "i", "i"),
        ret="T",
        slice="UserExecutorConv",
        idris_method="primConv2d",
    ),
    "tensor_max_pool1d": Entry(
        args=("T", "i", "i"), ret="T", slice="UserExecutorConv", idris_method="primMaxPool1d"
    ),
    "tensor_max_pool2d_batched": Entry(
        args=("T", "i", "i", "i", "i"),
        ret="T",
        slice="UserExecutorConv",
        idris_method="primMaxPool2dBatched",
    ),
    "tensor_max_pool2d": Entry(
        args=("T", "i", "i", "i", "i"),
        ret="T",
        slice="UserExecutorConv",
        idris_method="primMaxPool2d",
    ),
}
