/* Autograd surface — torch.
 *
 * libtorch owns backward propagation, so this TU is small: a thin
 * wrapper around `at::Tensor::backward()` plus accessors and no-grad
 * scope management.
 *
 *   - tensor_backward         (calls libtorch's backward + diagnostic dump)
 *   - tensor_grad / tensor_zero_grad / tensor_requires_grad
 *   - tensor_detach           (libtorch's .detach())
 *   - tensor_with_grad        (.detach().clone() + requires_grad_(true))
 *   - tensor_set_requires_grad (gated by idrisml_is_floating_st; torch
 *     throws on int/bool autograd)
 *   - tensor_no_grad_begin / tensor_no_grad_end (nesting counter +
 *     thread_local NoGradGuard)
 *   - tensor_epoch_begin / tensor_epoch_end (no-ops; torch has no
 *     paravirt-Metal buffer ceiling so no per-epoch generation sweep)
 */
#include "../tensor.h"
#include "dtype_dispatch.h"
#include "profiling.h"
#include <torch/torch.h>
#include <memory>

extern "C" void _dbg_dump_param_grads_if_enabled_torch(void);

void tensor_backward(TensorHandle h) {
    double t0 = _wall_ms_torch();
    to_tensor(h)->backward();
    prof_backward_ms_torch += _wall_ms_torch() - t0;
    /* Per-param gradient L2-norm dump (DEBUG_PARAM_GRADS=1).
       Implementation lives in backend_torch/training/diagnostics.cpp. */
    _dbg_dump_param_grads_if_enabled_torch();
}

TensorHandle tensor_grad(TensorHandle h) {
    auto& g = to_tensor(h)->grad();
    if (!g.defined()) return nullptr;
    return from_tensor(g);
}

void tensor_zero_grad(TensorHandle h) {
    auto& t = *to_tensor(h);
    if (t.grad().defined()) {
        t.grad().zero_();
    }
}

int tensor_requires_grad(TensorHandle h) {
    return to_tensor(h)->requires_grad() ? 1 : 0;
}

TensorHandle tensor_detach(TensorHandle h) {
    return from_tensor(to_tensor(h)->detach());
}

TensorHandle tensor_with_grad(TensorHandle h) {
    auto t = to_tensor(h)->detach().clone();
    t.requires_grad_(true);
    return from_tensor(std::move(t));
}

void tensor_set_requires_grad(TensorHandle h, int requires_grad) {
    auto t = to_tensor(h);
    /* torch throws if you request grad on a non-floating tensor. Inference
       dtypes (int/bool) can be registered (e.g. via registerParam, for
       serialization) but can't carry gradients — silently leave grad off
       rather than abort. */
    if (requires_grad && !idrisml_is_floating_st(t->scalar_type())) return;
    t->requires_grad_(requires_grad != 0);
}

/* No-grad scope. Counter (not bool) so nested withNoGrad scopes
   nest correctly — only the outermost begin creates the guard,
   only the outermost end releases it. */
static thread_local int no_grad_depth = 0;
static thread_local std::unique_ptr<torch::NoGradGuard> no_grad_guard;

void tensor_no_grad_begin(void) {
    if (no_grad_depth == 0) {
        no_grad_guard = std::make_unique<torch::NoGradGuard>();
    }
    no_grad_depth++;
}

void tensor_no_grad_end(void) {
    if (no_grad_depth > 0) {
        no_grad_depth--;
        if (no_grad_depth == 0) {
            no_grad_guard.reset();
        }
    }
}

/* No buffer ceiling on torch; per-epoch generation free is a no-op. */
void tensor_epoch_begin(void) {}
void tensor_epoch_end(void) {}
