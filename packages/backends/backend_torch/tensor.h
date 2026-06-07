/* Tensor-handle conversion helpers for the torch backend's modular tree.
 *
 * Per-op .cpp files under backend_torch/ include this header to bridge
 * the public C `TensorHandle` ABI to libtorch's `at::Tensor`. The bridge
 * is split:
 *
 *   - to_tensor: defined inline here. State-free static_cast — kept inline
 *     to avoid a function call on every op (the libtorch op itself is
 *     orders of magnitude more expensive, but inline is free).
 *
 *   - from_tensor / from_tensor_persistent: declared here, defined in
 *     backend_torch.cpp. They touch the monolith-private intermediates
 *     vector and peak-live counter; keeping the state encapsulated means
 *     per-op .cpp files don't see those globals.
 *
 * `from_tensor` participates in the intermediate-tensor tracking that
 * `optimizer_step` sweeps at end-of-iteration. `from_tensor_persistent`
 * is the param-creator variant — never tracked, survives the sweep.
 */
#ifndef IDRISML_BACKEND_TORCH_TENSOR_H
#define IDRISML_BACKEND_TORCH_TENSOR_H

#include <torch/torch.h>
#include "../backend.h"

static inline at::Tensor* to_tensor(TensorHandle h) {
	return static_cast<at::Tensor*>(h);
}

TensorHandle from_tensor(at::Tensor t);
TensorHandle from_tensor_persistent(at::Tensor t);

#endif /* IDRISML_BACKEND_TORCH_TENSOR_H */
