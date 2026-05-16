/* tensor_expand_mask for the torch backend. Expands a [m, n] mask to
 * [B, m, n] for broadcast against a batched attention matrix.
 * contiguous() makes the result safe to feed into FFI consumers that
 * expect a row-major buffer (Tensor's `view`/`expand` are non-owning). */
#include "../../tensor.h"

extern "C" TensorHandle tensor_expand_mask(TensorHandle hmask, int B) {
    return from_tensor(to_tensor(hmask)->unsqueeze(0).expand({(int64_t)B, -1, -1}).contiguous());
}
