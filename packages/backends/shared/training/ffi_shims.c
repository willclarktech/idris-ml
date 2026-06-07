/* shared/training/ffi_shims.c — *_return FFI helpers.
 *
 * Idris-side bindings route side-effect-only operations through these
 * `*_return` shims so they return a value the typed FFI can consume —
 * the elaborator drops `let _ = x` for unused side-effect calls. Each
 * shim does the work and returns either the input tensor (for opaque
 * IO chains), a freshly-computed integer/double, or a passthrough.
 *
 * Backend-agnostic: every called function (tensor_backward,
 * param_register, tensor_set_requires_grad, tensor_to_doubles,
 * tensor_requires_grad, backend_reset_for_eval, backend_profile_*) is
 * declared in backend.h and exported by every backend. The shared TU
 * compiles once per backend in TRAINING_ADAPTER_BACKENDS with that
 * backend's rename header, so the unsuffixed name resolves to that
 * backend's implementation at link time.
 */

#include <stddef.h>
#include "../../backend.h"

TensorHandle tensor_backward_return(TensorHandle t) {
	tensor_backward(t);
	return t;
}

TensorHandle param_register_return(const char* name, TensorHandle t) {
	tensor_set_requires_grad(t, 1);
	param_register(name, t);
	return t;
}

int param_zero_all_grads_return(int dummy) {
	(void)dummy;
	param_zero_all_grads();
	return 0;
}

double* tensor_to_doubles_return(TensorHandle h, double* buf) {
	tensor_to_doubles(h, buf);
	return buf;
}

int tensor_backward_conditional(TensorHandle t) {
	if (tensor_requires_grad(t)) tensor_backward(t);
	return param_count();
}

double tensor_backward_return_loss(TensorHandle loss_ptr, double loss_val) {
	if (tensor_requires_grad(loss_ptr)) tensor_backward(loss_ptr);
	return loss_val;
}

void* idrisml_seq(void* a, void* b) {
	(void)a;
	return b;
}

int backend_reset_for_eval_return(int dummy) {
	(void)dummy;
	backend_reset_for_eval();
	return dummy;
}

int backend_profile_reset_return(int dummy) {
	(void)dummy;
	backend_profile_reset();
	return dummy;
}

int backend_profile_report_return(int dummy) {
	(void)dummy;
	backend_profile_report();
	return dummy;
}
