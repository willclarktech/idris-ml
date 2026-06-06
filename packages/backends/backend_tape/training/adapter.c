/* backend_tape/training/adapter.c — tape's shared-port implementation.
 *
 * Defines the `g_active_port` instance the shared training-side TUs
 * under shared/training/ dereference. Methods that have a tape-side
 * implementation downcast `void*` to `Tensor*` from backend_tape/tensor.h
 * and call into the arena's dtype-aware element accessors, or
 * delegate to tape's specialized TUs (training/dtype_dispatch.c for
 * the dtag-streamed creators, training/optimizer.c for the per-element
 * optimizer math + tape epoch hygiene).
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
#include "../../backend.h"

/* ----------------------------------------------------------------------
   Tensor introspection.
   ---------------------------------------------------------------------- */
static int tape_tensor_numel(void* h)         { return ((Tensor*)h)->numel; }
static int tape_tensor_requires_grad(void* h) { return ((Tensor*)h)->requires_grad; }
static int tape_tensor_has_grad(void* h)      { return ((Tensor*)h)->grad != NULL; }

/* ----------------------------------------------------------------------
   Per-element data + grad access — dtype-aware via tape_load_d /
   tape_store_d (data) and tape_grad_load_d / tape_grad_store_d (grad).
   The grad-side dispatch is part of the Row 38 symmetric-F32-grads
   migration: F32 tensors get F32 grad buffers (4 bytes/elem) and
   tape_grad_*_d narrows on store / widens on load.
   ---------------------------------------------------------------------- */
static double tape_data_read(void* h, int i)            { return tape_load_d((Tensor*)h, i); }
static void   tape_data_write(void* h, int i, double v) { tape_store_d((Tensor*)h, i, v); }
static double tape_grad_read(void* h, int i)            { return tape_grad_load_d((Tensor*)h, i); }
static void   tape_grad_write(void* h, int i, double v) { tape_grad_store_d((Tensor*)h, i, v); };

/* ----------------------------------------------------------------------
   Bulk grad zero. memset over the typed grad buffer (size matches
   tape_grad_elem_size(t->dtype_tag)).
   ---------------------------------------------------------------------- */
static void tape_zero_grad(void* h) {
    Tensor* t = (Tensor*)h;
    if (t->grad) memset(t->grad, 0, (size_t)t->numel * tape_grad_elem_size(t->dtype_tag));
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
   Wall-clock provider — gettimeofday-based monotonic-ish millisecond
   reading. Delegates to the unified `_wall_ms` in shared_utils.c.
   ---------------------------------------------------------------------- */
extern double _wall_ms(void);
static double tape_adapter_wall_ms(void) { return _wall_ms(); }

/* ----------------------------------------------------------------------
   Dtag-streamed creators — bound from training/dtype_dispatch.c.
   ---------------------------------------------------------------------- */
extern TensorHandle tape_create_scalar_dtag(double v, int rg, int dtag);
extern TensorHandle tape_create_dtag(double* data, int* shape, int rank, int rg, int dtag);
extern TensorHandle tape_create_1d_dtag(int n, double* data, int rg, int dtag);
extern TensorHandle tape_create_2d_dtag(int rows, int cols, double* data, int rg, int dtag);
extern TensorHandle tape_create_param_1d_dtag(int n, double* data, int dtag);
extern TensorHandle tape_create_param_2d_dtag(int rows, int cols, double* data, int dtag);
extern TensorHandle tape_create_param_3d_dtag(int d0, int d1, int d2, double* data, int dtag);
extern TensorHandle tape_create_param_4d_dtag(int d0, int d1, int d2, int d3, double* data, int dtag);
extern TensorHandle tape_create_state_1d_dtag(int n, double* data, int dtag);
extern TensorHandle tape_create_state_2d_dtag(int rows, int cols, double* data, int dtag);
extern TensorHandle tape_cast_dtype_dtag(TensorHandle src, int dtag);

/* ----------------------------------------------------------------------
   Fused param create + init — bound from training/dtype_init.c.
   These provide the in-place init kernel surface the shared
   dtype_streamed.c trampolines require; mirrors torch's
   dtype_init.cpp surface.
   ---------------------------------------------------------------------- */
extern TensorHandle tape_create_param_1d_normal_dtag(int n,                                  double mean, double std, int dtag);
extern TensorHandle tape_create_param_2d_normal_dtag(int rows, int cols,                     double mean, double std, int dtag);
extern TensorHandle tape_create_param_3d_normal_dtag(int d0, int d1, int d2,                 double mean, double std, int dtag);
extern TensorHandle tape_create_param_4d_normal_dtag(int d0, int d1, int d2, int d3,         double mean, double std, int dtag);
extern TensorHandle tape_create_param_1d_const_dtag (int n,                                  double value,            int dtag);
extern TensorHandle tape_create_param_2d_const_dtag (int rows, int cols,                     double value,            int dtag);
extern TensorHandle tape_create_param_3d_const_dtag (int d0, int d1, int d2,                 double value,            int dtag);
extern TensorHandle tape_create_param_4d_const_dtag (int d0, int d1, int d2, int d3,         double value,            int dtag);
extern void         tape_set_init_seed(uint64_t seed);

static void* tape_port_create_scalar(double v, int rg, int dtag)                                 { return tape_create_scalar_dtag(v, rg, dtag); }
static void* tape_port_create(double* d, int* s, int r, int rg, int dtag)                        { return tape_create_dtag(d, s, r, rg, dtag); }
static void* tape_port_create_1d(int n, double* d, int rg, int dtag)                             { return tape_create_1d_dtag(n, d, rg, dtag); }
static void* tape_port_create_2d(int rows, int cols, double* d, int rg, int dtag)                { return tape_create_2d_dtag(rows, cols, d, rg, dtag); }
static void* tape_port_create_param_1d(int n, double* d, int dtag)                               { return tape_create_param_1d_dtag(n, d, dtag); }
static void* tape_port_create_param_2d(int rows, int cols, double* d, int dtag)                  { return tape_create_param_2d_dtag(rows, cols, d, dtag); }
static void* tape_port_create_param_3d(int d0, int d1, int d2, double* d, int dtag)              { return tape_create_param_3d_dtag(d0, d1, d2, d, dtag); }
static void* tape_port_create_param_4d(int d0, int d1, int d2, int d3, double* d, int dtag)      { return tape_create_param_4d_dtag(d0, d1, d2, d3, d, dtag); }
static void* tape_port_create_state_1d(int n, double* d, int dtag)                               { return tape_create_state_1d_dtag(n, d, dtag); }
static void* tape_port_create_state_2d(int rows, int cols, double* d, int dtag)                  { return tape_create_state_2d_dtag(rows, cols, d, dtag); }
static void* tape_port_cast_dtype(void* src, int dtag)                                            { return tape_cast_dtype_dtag((TensorHandle)src, dtag); }

/* Fused param create + init trampolines — forward to dtype_init.c. */
static void* tape_port_create_param_1d_normal(int n, double mean, double std, int dtag)                                { return tape_create_param_1d_normal_dtag(n, mean, std, dtag); }
static void* tape_port_create_param_2d_normal(int rows, int cols, double mean, double std, int dtag)                   { return tape_create_param_2d_normal_dtag(rows, cols, mean, std, dtag); }
static void* tape_port_create_param_3d_normal(int d0, int d1, int d2, double mean, double std, int dtag)               { return tape_create_param_3d_normal_dtag(d0, d1, d2, mean, std, dtag); }
static void* tape_port_create_param_4d_normal(int d0, int d1, int d2, int d3, double mean, double std, int dtag)       { return tape_create_param_4d_normal_dtag(d0, d1, d2, d3, mean, std, dtag); }
static void* tape_port_create_param_1d_const (int n, double value, int dtag)                                           { return tape_create_param_1d_const_dtag(n, value, dtag); }
static void* tape_port_create_param_2d_const (int rows, int cols, double value, int dtag)                              { return tape_create_param_2d_const_dtag(rows, cols, value, dtag); }
static void* tape_port_create_param_3d_const (int d0, int d1, int d2, double value, int dtag)                          { return tape_create_param_3d_const_dtag(d0, d1, d2, value, dtag); }
static void* tape_port_create_param_4d_const (int d0, int d1, int d2, int d3, double value, int dtag)                  { return tape_create_param_4d_const_dtag(d0, d1, d2, d3, value, dtag); }
static void  tape_port_set_init_seed(uint64_t seed)                                                                    { tape_set_init_seed(seed); }

/* ----------------------------------------------------------------------
   Optimizer surface — bound from training/optimizer.c.
   ---------------------------------------------------------------------- */
extern void* tape_optimizer_create_sgd(double lr);
extern void* tape_optimizer_create_rmsprop(double lr, double alpha, double eps,
                                            double weight_decay, double momentum);
extern void* tape_optimizer_create_adam(double lr, double beta1, double beta2, double eps);
extern void* tape_optimizer_create_adam_group(double lr, double beta1, double beta2,
                                               double eps, const char* prefix);
extern void* tape_optimizer_create_adamw(double lr, double beta1, double beta2, double eps,
                                          double weight_decay);
extern void  tape_optimizer_free(void* opt);
extern void  tape_optimizer_set_lr(void* opt, double lr);
extern void  tape_optimizer_set_param_lr(void* opt, const char* name, double lr);
extern void  tape_optimizer_step(void* opt);
extern void   tape_optimizer_clip_grad_value_filtered(void* opt, double max_val);
extern double tape_optimizer_clip_grad_norm_filtered(void* opt, double max_norm);
extern int   tape_optimizer_buf_count(void* opt);
extern void  tape_optimizer_get_m(void* opt, int idx, double* out);
extern void  tape_optimizer_get_v(void* opt, int idx, double* out);
extern void  tape_optimizer_set_m(void* opt, int idx, const double* in);
extern void  tape_optimizer_set_v(void* opt, int idx, const double* in);
extern void  tape_optimizer_get_meta(void* opt, double* out9);
extern void  tape_optimizer_set_meta(void* opt, const double* in9);

const BackendPort g_active_port = {
  .tensor_numel              = tape_tensor_numel,
  .tensor_requires_grad      = tape_tensor_requires_grad,
  .tensor_has_grad           = tape_tensor_has_grad,
  .data_read                 = tape_data_read,
  .data_write                = tape_data_write,
  .grad_read                 = tape_grad_read,
  .grad_write                = tape_grad_write,
  .zero_grad                 = tape_zero_grad,
  .load_doubles              = tape_load_doubles,
  .load_int64                = tape_load_int64,
  .backward                  = tape_adapter_backward,
  .wall_ms                   = tape_adapter_wall_ms,
  .create_scalar             = tape_port_create_scalar,
  .create                    = tape_port_create,
  .create_1d                 = tape_port_create_1d,
  .create_2d                 = tape_port_create_2d,
  .create_param_1d           = tape_port_create_param_1d,
  .create_param_2d           = tape_port_create_param_2d,
  .create_param_3d           = tape_port_create_param_3d,
  .create_param_4d           = tape_port_create_param_4d,
  .create_state_1d           = tape_port_create_state_1d,
  .create_state_2d           = tape_port_create_state_2d,
  .cast_dtype                = tape_port_cast_dtype,
  .create_param_1d_normal    = tape_port_create_param_1d_normal,
  .create_param_2d_normal    = tape_port_create_param_2d_normal,
  .create_param_3d_normal    = tape_port_create_param_3d_normal,
  .create_param_4d_normal    = tape_port_create_param_4d_normal,
  .create_param_1d_const     = tape_port_create_param_1d_const,
  .create_param_2d_const     = tape_port_create_param_2d_const,
  .create_param_3d_const     = tape_port_create_param_3d_const,
  .create_param_4d_const     = tape_port_create_param_4d_const,
  .set_init_seed             = tape_port_set_init_seed,
  .optimizer_create_sgd      = tape_optimizer_create_sgd,
  .optimizer_create_rmsprop  = tape_optimizer_create_rmsprop,
  .optimizer_create_adam     = tape_optimizer_create_adam,
  .optimizer_create_adam_group = tape_optimizer_create_adam_group,
  .optimizer_create_adamw    = tape_optimizer_create_adamw,
  .optimizer_free            = tape_optimizer_free,
  .optimizer_set_lr          = tape_optimizer_set_lr,
  .optimizer_set_param_lr    = tape_optimizer_set_param_lr,
  .optimizer_step            = tape_optimizer_step,
  .optimizer_clip_grad_value_filtered = tape_optimizer_clip_grad_value_filtered,
  .optimizer_clip_grad_norm_filtered  = tape_optimizer_clip_grad_norm_filtered,
  .optimizer_buf_count       = tape_optimizer_buf_count,
  .optimizer_get_m           = tape_optimizer_get_m,
  .optimizer_get_v           = tape_optimizer_get_v,
  .optimizer_set_m           = tape_optimizer_set_m,
  .optimizer_set_v           = tape_optimizer_set_v,
  .optimizer_get_meta        = tape_optimizer_get_meta,
  .optimizer_set_meta        = tape_optimizer_set_meta,
};
