/* backend_tape/training/shims.c — tape-specific system/debug shims.
 *
 * What stays here: per-eval reset (tape_reset + param re-registration
 * on the fresh tape), live-count probes that read tape_size / g_tape_peak
 * directly, the constant `backend_name()` answer, the mlx_compile no-op
 * accessors (mx::compile is mlx-only — every other backend reports
 * disabled), and tensor_print's tape-specific F64-direct dump.
 *
 * The backend-agnostic `*_return` FFI wrappers + idrisml_seq live in
 * shared/training/ffi_shims.c.
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

/* dropout_random_seed lives in shared_utils.c. */
