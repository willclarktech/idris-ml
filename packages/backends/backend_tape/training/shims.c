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
#include <stdlib.h>
#include <string.h>
#include "../arena.h"
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

/* See backend.h: explicit pre-exit cleanup. Three-phase free pass:
 *
 *   (1) tape_reset() — drains the autograd tape, freeing per-entry
 *       malloc'd op_meta (LayerNormMeta x_hat/rstd, DropoutMeta mask,
 *       EmbeddingMeta indices, OP_STACK inputs arrays, etc.) and
 *       non-persistent grad arrays. Must run first because tape entries
 *       hold `e->result` pointers into the bump arena — dereferencing
 *       them after arena_free_all would touch freed memory.
 *   (2) walk param_registry_arr and free every persistent param tensor's
 *       malloc'd struct + data + shape + grad. `tensor_release_handle`
 *       is a no-op on tape (params are non-arena heap allocations from
 *       param_create.c; tape's handle ABI doesn't refcount), so the
 *       buffers leak until process exit unless we free them here.
 *       `param_clear` after the loop zeroes the registry count.
 *   (3) arena_free_all() — walks the bump arena chunk linked list,
 *       free(c->data) + free(c) per chunk. arena_reset only zeroes
 *       `used` counters; the chunks themselves only got freed at
 *       process exit by libc previously.
 *
 * On HfLlama-1B F32 the leaked param + arena heap is multi-GB; macOS
 * libmalloc is slow on teardowns at that scale, which caused a ~17 min
 * post-main tail observed in commit `e924b5e`. Running the work inside
 * main() makes it bounded and timeable via the existing stage stamps.
 *
 * Safety: only call from end-of-main once Idris-land is done touching
 * tensor handles. Every previously-returned arena pointer becomes
 * dangling after step (3). */
void backend_release_all_persistent(void) {
    tape_reset();
    int n = param_count();
    for (int i = 0; i < n; i++) {
        Tensor* t = (Tensor*)param_tensor(i);
        if (!t) continue;
        free(t->shape);
        free(t->data);
        free(t->grad);
        free(t);
    }
    param_clear();
    arena_free_all();
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
