/* core/lifecycle/free.c — Tensor lifecycle teardown.
 *
 * Phase 1a.1. tensor_free is a no-op on tape: the tape holds non-owning
 * pointers into arena memory, so freeing per-tensor while the tape is
 * still live would invalidate them. Real teardown happens in bulk at
 * tape_reset / arena_reset (per epoch).
 *
 * tensor_retain_handle / tensor_release_handle are wrap-side refcount
 * shims required for ABI parity with mlx (which DOES count); tape's
 * arena lifecycle subsumes them, so these are no-ops too.
 */

#include "../../../backend.h"

void tensor_free(TensorHandle h) {
    /* Tape holds non-owning pointers; freeing here would dangle them.
     * Arena reset (in tape_reset) tears down the whole arena at once. */
    (void)h;
}

void tensor_retain_handle(TensorHandle h) { (void)h; }
void tensor_release_handle(TensorHandle h) { (void)h; }
