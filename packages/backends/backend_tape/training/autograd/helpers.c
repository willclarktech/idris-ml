/* training/autograd/helpers.c — thin autograd-surface helpers.
 *
 * Bundles the small, self-contained autograd helpers that
 * the `UserExecutorTraining` interface exposes:
 *   - tensor_grad, tensor_zero_grad
 *   - tensor_requires_grad, tensor_set_requires_grad
 *   - tensor_detach (= clone), tensor_with_grad
 *   - tensor_no_grad_begin / _end (depth-tracking around graph append)
 *   - tensor_epoch_begin / _end (no-op on tape — arena resets per-tape)
 *   - tensor_to_device / tensor_device (cpu-only on tape)
 *
 * tensor_backward (the big switch-driver) is the only autograd surface
 * that lives elsewhere — see backward.c.
 *
 * no_grad_depth is owned by tape.c (extern declared in tape.h).
 */

#include <string.h>
#include "../../tape.h"
#include "../../arena.h"
#include "../../tensor.h"
#include "../../../backend.h"

TensorHandle tensor_grad(TensorHandle h) {
    Tensor* t = (Tensor*)h;
    if (!t->grad) return NULL;
    return make_scalar(tape_grad_load_d(t, 0), 0);
}

void tensor_zero_grad(TensorHandle h) {
    Tensor* t = (Tensor*)h;
    if (t->grad) memset(t->grad, 0, t->numel * sizeof(double));
}

int tensor_requires_grad(TensorHandle h) {
    return ((Tensor*)h)->requires_grad;
}

TensorHandle tensor_detach(TensorHandle h) {
    return tensor_clone(h);
}

TensorHandle tensor_with_grad(TensorHandle h) {
    Tensor* t = (Tensor*)h;
    Tensor* r = make_scalar(((double*)t->data)[0], 1);
    tape_append(OP_CONST, r, NULL, NULL, 0);
    return r;
}

void tensor_set_requires_grad(TensorHandle h, int rg) {
    Tensor* t = (Tensor*)h;
    t->requires_grad = rg;
    if (rg && t->tape_idx < 0) {
        tape_append(OP_CONST, t, NULL, NULL, 0);
    }
}

void tensor_no_grad_begin(void) { no_grad_depth++; }
void tensor_no_grad_end(void)   { if (no_grad_depth > 0) no_grad_depth--; }

/* No buffer ceiling on tape; per-epoch generation free is a no-op. */
void tensor_epoch_begin(void) {}
void tensor_epoch_end(void)   {}

/* Device (CPU-only on tape) */
TensorHandle tensor_to_device(TensorHandle t, const char* device) { (void)device; return t; }
const char*  tensor_device(TensorHandle t)                        { (void)t; return "cpu"; }
