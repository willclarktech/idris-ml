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
#include "../tensor.h"
#include "../arena.h"
#include "../../shared/training/port.h"

#define TAPE_ADAPTER_STUB(name) \
  do { \
    fprintf(stderr, \
      "tape adapter stub: " #name \
      " invoked before its shared-port implementation landed — aborting.\n"); \
    abort(); \
  } while (0)

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
   Slots whose shared consumer has not been lifted yet stay as abort
   stubs so any premature wiring fails loudly.
   ---------------------------------------------------------------------- */
static void   stub_backward(void* loss)        { (void)loss; TAPE_ADAPTER_STUB(backward); }
static void   stub_epoch_boundary(void)        { TAPE_ADAPTER_STUB(epoch_boundary); }
static double stub_wall_ms(void)               { TAPE_ADAPTER_STUB(wall_ms); }

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
  .backward             = stub_backward,
  .epoch_boundary       = stub_epoch_boundary,
  .wall_ms              = stub_wall_ms,
};
