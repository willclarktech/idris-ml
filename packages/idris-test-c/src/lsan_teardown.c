/* lsan_teardown.c — drain the tape backend's per-process transient
 * state before LeakSanitizer's exit-time check, so the C unit-test
 * ASan lane reports only genuine leaks.
 *
 * Why this exists: on the tape backend `tensor_free` is a deliberate
 * no-op — the tape holds non-owning pointers into arena memory, so
 * per-tensor frees would dangle. Real teardown is bulk, per-epoch, via
 * tape_reset() + arena_free_all(). The bare Criterion suites run no
 * training loop, so nothing resets the tape and every arena chunk /
 * op_meta / grad buffer lives until process exit, where LSan flags it.
 *
 * tape_reset() frees the per-op heap (LayerNorm/Dropout/Embedding/Stack
 * op_meta, OP_STACK input arrays) and every non-persistent grad buffer;
 * arena_free_all() frees the bump-arena chunks. Both touch only memory
 * the tape/arena themselves malloc'd, so neither can bad-free on a
 * test-registered handle (unlike backend_release_all_persistent, whose
 * param-registry walk assumes every registered param's data is a
 * standalone heap allocation — false for tensors a test registers from
 * arena buffers). What remains after this are the genuinely
 * unfreeable-on-tape persistent allocations (unregistered scalars,
 * params, NTM state, f32 persistents); those stay covered by the narrow
 * suppression file packages/idris-test-c/lsan.supp.
 *
 * Mechanism: Criterion forks a worker per Test(). The worker inherits
 * the atexit handler registered by the constructor below (it runs once,
 * pre-fork). LeakSanitizer registers its own leak check during
 * sanitizer init (before main), and atexit is LIFO, so our
 * later-registered handler runs first — the documented way to free
 * intentional allocations ahead of the check.
 *
 * Compiled only under ASan on the tape backend (a no-op TU otherwise),
 * so normal test runs and the mlx/torch binaries are unaffected.
 */

#if (defined(__SANITIZE_ADDRESS__) ||                                                              \
     (defined(__has_feature) && __has_feature(address_sanitizer))) &&                              \
    defined(BACKEND_TAPE)

#include <stdlib.h>

/* Internal tape symbols (exported, unrenamed — not part of backend.h's
   public ABI). Declared here rather than pulling the private tape
   headers into the test tree. */
extern void tape_reset(void);
extern void arena_free_all(void);

static void idrisml_tape_teardown(void) {
	tape_reset();     /* op_meta + non-persistent grads + arena_reset */
	arena_free_all(); /* free the bump-arena chunks */
}

__attribute__((constructor)) static void idrisml_register_tape_teardown(void) {
	atexit(idrisml_tape_teardown);
}

#endif
