/* tensor_narrow for the torch backend.
 *
 * Forwards `dim` straight through to libtorch's at::Tensor::narrow; the
 * historical "flatten then narrow axis 0" behaviour was a silent shape
 * lie when called with dim > 0 (used by HfBert's per-head Q/K/V split).
 * Pinned by `linear_shape_narrow::axis1_correctness_rank2` in the
 * common-backend test suite. */
#include "../../tensor.h"

extern "C" TensorHandle tensor_narrow(TensorHandle h, int dim, int start, int len) {
    auto t = to_tensor(h)->narrow(dim, start, len).contiguous();
    return from_tensor(std::move(t));
}
