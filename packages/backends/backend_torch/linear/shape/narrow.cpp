/* tensor_narrow for the torch backend.
 *
 * Forwards `dim` straight through to libtorch's at::Tensor::narrow; the
 * historical "flatten then narrow axis 0" behaviour was a silent shape
 * lie when called with dim > 0 (used by HfBert's per-head Q/K/V split).
 * Pinned by `linear_shape_narrow::axis1_correctness_rank2` in the
 * common-backend test suite.
 *
 * No `.contiguous()` — the strided view from narrow() composes
 * correctly with every downstream op (matmul, add, softmax, etc.).
 * The previous `.contiguous()` materialized ~16K view-only slices per
 * Llama 8-token forward, each triggering an MPS allocator + Metal
 * command buffer submission; mirrors the `ea90238` fix in
 * transpose.cpp. Host-side reads that need a row-major buffer go
 * through accessors.cpp's `cpu().to(...).contiguous()` path. */
#include "../../tensor.h"

extern "C" TensorHandle tensor_narrow(TensorHandle h, int dim, int start, int len) {
	auto t = to_tensor(h)->narrow(dim, start, len);
	return from_tensor(std::move(t));
}
