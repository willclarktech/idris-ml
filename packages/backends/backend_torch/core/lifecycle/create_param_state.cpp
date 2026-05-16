/* Persistent-leaf creators — torch (F64 base).
 *
 * Houses the F64 base param + state creators (1d/2d/3d/4d for param,
 * 1d/2d for state) plus the small view selectors. F32 equivalents live
 * in training/dtype_dispatch.cpp; the F64 paths here are what the unified
 * dispatchers fall through to.
 *
 *   - tensor_create_{1d,2d}      raw 1d/2d creators (F64 from a host
 *                                buffer; optional requires_grad).
 *   - tensor_create_param_{1,2,3,4}d
 *                                grad-eligible parameter leaves; uses
 *                                make_param_leaf for the cast-before-grad
 *                                discipline (cast-after-grad yields a
 *                                non-leaf with no .grad).
 *   - tensor_create_state_{1,2}d persistent leaf, no grad.
 *   - tensor_view_{1d,2d}        0-dim view sharing storage with the
 *                                parent (persistent — survives
 *                                free_intermediates).
 */
#include "../../tensor.h"
#include "../../training/dtype_dispatch.h"
#include <torch/torch.h>
#include <cstdlib>

extern "C" TensorHandle tensor_create_1d(int n, double* data, int requires_grad) {
    auto t = torch::from_blob(data, {(int64_t)n}, torch::kFloat64).clone();
    free(data);
    if (requires_grad) t.requires_grad_(true);
    return from_tensor(std::move(t));
}

extern "C" TensorHandle tensor_create_2d(int rows, int cols, double* data, int requires_grad) {
    auto t = torch::from_blob(data, {(int64_t)rows, (int64_t)cols}, torch::kFloat64).clone();
    free(data);
    if (requires_grad) t.requires_grad_(true);
    return from_tensor(std::move(t));
}

extern "C" TensorHandle tensor_create_param_1d(int n, double* data) {
    return make_param_leaf(data, {(int64_t)n}, torch::kFloat64);
}

extern "C" TensorHandle tensor_create_param_2d(int rows, int cols, double* data) {
    return make_param_leaf(data, {(int64_t)rows, (int64_t)cols}, torch::kFloat64);
}

extern "C" TensorHandle tensor_create_param_3d(int d0, int d1, int d2, double* data) {
    return make_param_leaf(data, {(int64_t)d0, (int64_t)d1, (int64_t)d2}, torch::kFloat64);
}

extern "C" TensorHandle tensor_create_param_4d(int d0, int d1, int d2, int d3, double* data) {
    return make_param_leaf(data, {(int64_t)d0, (int64_t)d1, (int64_t)d2, (int64_t)d3}, torch::kFloat64);
}

extern "C" TensorHandle tensor_create_state_1d(int n, double* data) {
    auto t = torch::from_blob(data, {(int64_t)n}, torch::kFloat64).clone();
    return from_tensor_persistent(std::move(t));
}

extern "C" TensorHandle tensor_create_state_2d(int rows, int cols, double* data) {
    auto t = torch::from_blob(data, {(int64_t)rows, (int64_t)cols}, torch::kFloat64).clone();
    return from_tensor_persistent(std::move(t));
}

extern "C" TensorHandle tensor_view_2d(TensorHandle h, int row, int col) {
    /* Returns a 0-dim view that shares storage with the parent tensor.
       Must be persistent — views into param tensors survive free_intermediates. */
    return from_tensor_persistent(to_tensor(h)->select(0, row).select(0, col));
}

extern "C" TensorHandle tensor_view_1d(TensorHandle h, int idx) {
    return from_tensor_persistent(to_tensor(h)->select(0, idx));
}
