/* training/shims.c — small system/debug/FFI shims.
 *
 * Phase 1e.11 closeout. Collects everything that wasn't worth its own
 * file: the per-eval reset, live-count probes, mlx_compile no-op
 * accessors (tape backend never compiles), tensor_print debug
 * helper, the *_return RefC-compat wrappers, and the
 * idrisml_seq passthrough.
 */

#include <stdio.h>
#include <string.h>
#include "../tape.h"
#include "../tensor.h"
#include "../../backend.h"

extern long g_tape_peak;

/* System */
void backend_reset_for_eval(void) {
    tape_reset();
    /* Re-register params so they have valid tape indices */
    for (int j = 0; j < param_count(); j++) {
        Tensor* t = (Tensor*)param_tensor(j);
        t->tape_idx = -1;
        if (t->grad) memset(t->grad, 0, t->numel * sizeof(double));
        tape_append(OP_CONST, t, NULL, NULL, 0);
    }
}

int tensor_live_count(int dummy)      { (void)dummy; return (int)tape_size; }
int tensor_peak_live_count(int dummy) { (void)dummy; return (int)g_tape_peak; }

/* Debug */
const char* backend_name(void) { return "tape"; }

/* mx::compile is mlx-only; tape backend always reports disabled
 * regardless of MLX_COMPILE env var. */
int  tensor_mlx_compile_enabled(void)         { return 0; }
int  tensor_mlx_compile_invocations(void)     { return 0; }
void tensor_mlx_compile_reset_stats(void)     { }

void tensor_print(TensorHandle h) {
    Tensor* t = (Tensor*)h;
    if (t->rank == 0) {
        printf("%.6f\n", ((double*)t->data)[0]);
    } else {
        printf("[");
        for (int i = 0; i < t->numel; i++) {
            if (i > 0) printf(", ");
            printf("%.6f", ((double*)t->data)[i]);
        }
        printf("]\n");
    }
}

/* Portable FFI helpers (for RefC compatibility). Idris-side bindings
 * route side-effect-only operations through *_return shims so they
 * return a value the typed FFI can consume — `let _ = x` is dropped by
 * the compiler. */

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
    if (tensor_requires_grad(t))
        tensor_backward(t);
    return param_count();
}

double tensor_backward_return_loss(TensorHandle loss_ptr, double loss_val) {
    if (tensor_requires_grad(loss_ptr))
        tensor_backward(loss_ptr);
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

/* dropout_random_seed lives in shared_utils.c. */
