/* tensor_transpose_2d / tensor_transpose_last2 for the torch backend.
 *
 * Both return a stride-metadata view — `t()` / `transpose()` are O(1)
 * in libtorch and don't allocate. The earlier version of this file
 * called `.contiguous()` on the result, forcing a full element copy;
 * at Llama-3.2-1B scale on MPS that materialised many GB of redundant
 * tensors and hit MPS's ~18 GiB allocation cap inside the forward
 * pass (`empty_like → empty_mps → MPSAllocator::allocate` OOM at
 * `tensor_transpose_2d_torch` frame, observed 2026-05-27).
 *
 * libtorch ops on the returned view (matmul, add, softmax, etc.)
 * already materialise their own contiguous copy when the kernel
 * needs it, so most callers never pay the copy. The host-side
 * accessors (`item1d` / `item2d` / `tensor_to_host_*`) likewise call
 * `.contiguous()` themselves before reading raw bytes. The only
 * caller that *would* need contiguous storage on this path would be
 * one that bypasses libtorch and reads the data pointer directly —
 * none exist in this backend today. */
#include "../../tensor.h"

extern "C" TensorHandle tensor_transpose_2d(TensorHandle h) {
    return from_tensor(to_tensor(h)->t());
}

extern "C" TensorHandle tensor_transpose_last2(TensorHandle h) {
    return from_tensor(to_tensor(h)->transpose(-2, -1));
}
