/* backend_tape/training/adapter.c — tape's shared-port implementation.
 *
 * Defines the `g_active_port` instance the shared training-side TUs
 * under shared/training/ dereference. Methods that have a tape-side
 * implementation downcast `void*` to `Tensor*` from backend_tape/tensor.h
 * and call into the arena's dtype-aware element accessors. Slots whose
 * shared consumer has not yet been lifted from backend_tape/training/
 * stay as abort stubs so any premature wiring fails loudly.
 *
 * The grad buffer is always F64 (see backend_tape/arena.c `ensure_grad`),
 * regardless of the param's storage dtype — so grad_read/grad_write hit
 * `((double*)t->grad)[i]` directly. data_read/data_write route through
 * tape_load_d / tape_store_d so F32 storage narrows/widens correctly.
 */

#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <stddef.h>
#include <string.h>
#include "../tape.h"
#include "../tensor.h"
#include "../arena.h"
#include "../../shared/training/port.h"
#include "../../backend.h"

/* From training/profiling.c — tape-internal accumulators that the
   adapter's epoch_boundary updates. */
extern double _wall_ms(void);
extern double prof_optimizer_ms;
extern int    prof_forward_ops;
extern int    prof_epochs;
extern double prof_epoch_start;
extern double prof_op_t_prev;

/* From training/diagnostics.c — DEBUG_LSTM_TRAJ pre-reset dump. */
extern void _dbg_dump_lstm_traj_if_enabled(void);

/* ----------------------------------------------------------------------
   Tensor introspection.
   ---------------------------------------------------------------------- */
static int tape_tensor_numel(void* h)         { return ((Tensor*)h)->numel; }
static int tape_tensor_requires_grad(void* h) { return ((Tensor*)h)->requires_grad; }
static int tape_tensor_has_grad(void* h)      { return ((Tensor*)h)->grad != NULL; }

/* ----------------------------------------------------------------------
   Per-element data access — dtype-aware via tape_load_d / tape_store_d.
   Grad is always F64.
   ---------------------------------------------------------------------- */
static double tape_data_read(void* h, int i)            { return tape_load_d((Tensor*)h, i); }
static void   tape_data_write(void* h, int i, double v) { tape_store_d((Tensor*)h, i, v); }
static double tape_grad_read(void* h, int i)            { return ((double*)((Tensor*)h)->grad)[i]; }
static void   tape_grad_write(void* h, int i, double v) { ((double*)((Tensor*)h)->grad)[i] = v; }

/* ----------------------------------------------------------------------
   Bulk grad zero. memset over the F64 grad buffer.
   ---------------------------------------------------------------------- */
static void tape_zero_grad(void* h) {
    Tensor* t = (Tensor*)h;
    if (t->grad) memset(t->grad, 0, t->numel * sizeof(double));
}

/* ----------------------------------------------------------------------
   Bulk data load — element-wise through the dtype-aware store so F32
   storage gets the narrowing. For F64 the loop is equivalent to memcpy
   (same result, byte-identical).
   ---------------------------------------------------------------------- */
static void tape_load_doubles(void* h, const double* src, int n) {
    Tensor* t = (Tensor*)h;
    for (int i = 0; i < n; i++) tape_store_d(t, i, src[i]);
}

static void tape_load_int64(void* h, const int64_t* src, int n) {
    Tensor* t = (Tensor*)h;
    for (int i = 0; i < n; i++) tape_store_d(t, i, (double)src[i]);
}

/* ----------------------------------------------------------------------
   Backward driver. Delegates to tape's tensor_backward, which walks the
   Wengert list in reverse via the op-dispatch table.
   ---------------------------------------------------------------------- */
static void tape_adapter_backward(void* loss) { tensor_backward((TensorHandle)loss); }

/* ----------------------------------------------------------------------
   Wall-clock provider for the shared profiler / optimizer. _wall_ms
   is gettimeofday-based and shared by every prof_* accumulator.
   ---------------------------------------------------------------------- */
static double tape_adapter_wall_ms(void) { return _wall_ms(); }

/* ----------------------------------------------------------------------
   Epoch boundary. Called by the shared optimizer at the end of step().
   Sequence intentionally matches the monolithic optimizer_step tail
   verbatim — F64 byte-identical depends on:
     1. DEBUG_LSTM_TRAJ dump BEFORE tape_reset (it reads param data
        values, which survive reset, but dumping pre-reset matches
        the legacy step order so the dump's epoch index is unchanged).
     2. prof_forward_ops snapshotted from tape_size BEFORE reset.
     3. tape_reset then re-register each param via OP_CONST so its
        tape_idx is valid for the next forward pass.
     4. prof_optimizer_ms += elapsed (covers per-param loop + this
        hygiene; matches legacy).
     5. prof_epochs++ and restart epoch + per-op timers.
   ---------------------------------------------------------------------- */
static void tape_adapter_epoch_boundary(double t0_opt_start) {
    _dbg_dump_lstm_traj_if_enabled();
    prof_forward_ops = tape_size;
    tape_reset();
    for (int j = 0; j < param_count(); j++) {
        Tensor* t = (Tensor*)param_tensor(j);
        t->tape_idx = -1;
        if (t->grad) memset(t->grad, 0, t->numel * sizeof(double));
        tape_append(OP_CONST, t, NULL, NULL, 0);
    }
    prof_optimizer_ms += _wall_ms() - t0_opt_start;
    prof_epochs++;
    double t_next = _wall_ms();
    prof_epoch_start = t_next;
    prof_op_t_prev = t_next;
}

const BackendPort g_active_port = {
  .tensor_numel         = tape_tensor_numel,
  .tensor_requires_grad = tape_tensor_requires_grad,
  .tensor_has_grad      = tape_tensor_has_grad,
  .data_read            = tape_data_read,
  .data_write           = tape_data_write,
  .grad_read            = tape_grad_read,
  .grad_write           = tape_grad_write,
  .zero_grad            = tape_zero_grad,
  .load_doubles         = tape_load_doubles,
  .load_int64           = tape_load_int64,
  .backward             = tape_adapter_backward,
  .epoch_boundary       = tape_adapter_epoch_boundary,
  .wall_ms              = tape_adapter_wall_ms,
};
