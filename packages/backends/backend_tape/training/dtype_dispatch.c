/* backend_tape/training/dtype_dispatch.c — tape's dtag-streamed creators.
 *
 * The dtag-dispatched create / cast primitives that the tape adapter
 * binds into the shared port (port.create_scalar / create / create_1d
 * / create_2d / create_param_*d / create_state_*d / cast_dtype). The
 * shared wrapper file (shared/training/dtype_streamed.c) is a thin
 * stream-tag-strip-then-dispatch shell; this file owns the actual
 * per-storage-variant logic plus the helpers that round / re-tag /
 * widen between dtypes.
 *
 * Under the kind-major dtag layout (closed 2026-05-23): dtag 15 = F64,
 * dtag 14 = F32. All other valid dtags (Bool=1, U8=4, I8/I16/I32/I64,
 * F16=13, BF16=17) route through tape_retag_round's lingua-franca
 * path (double storage, dtype_tag set, values rounded into the
 * target dtype's representable precision). Invalid dtags abort via
 * tape_tag_from_dtag's default arm.
 *
 * Lint exemption: these wrappers are hand-maintained (not manifest-
 * generated) — see reference_dtype_ffi_wrappers in user memory.
 */

#include <stdlib.h>
#include <string.h>
#include "../tape.h"
#include "../arena.h"
#include "../tensor.h"
#include "../../backend.h"
#include "autograd/op_dispatch.h"

static TensorHandle tape_retag_round(TensorHandle h, int dtag) {
    Tensor* t = (Tensor*)h;
    int tag = tape_tag_from_dtag(dtag);
    for (int i = 0; i < t->numel; i++)
        ((double*)t->data)[i] = tape_round_to_dtype(((double*)t->data)[i], tag);
    t->dtype_tag = tag;
    return h;
}

/* Build a real-F32-storage arena tensor from F64 source data; the source
   buffer is freed (matches the F64 streamed-create convention where the
   underlying *_f64 creator owns + frees its `data` argument). */
static TensorHandle tape_arena_f32_from_doubles(int* shape, int rank,
                                                double* data, int rg) {
    int numel = 1;
    for (int i = 0; i < rank; i++) numel *= shape[i];
    float* arena_d = arena_alloc(numel * sizeof(float));
    for (int i = 0; i < numel; i++) arena_d[i] = (float)data[i];
    free(data);
    return make_tensor_arena_f32(arena_d, numel, shape, rank, rg);
}

/* Same, but persistent (malloc'd) — for params / state. tape_append is
   called only when requires_grad; mirrors tensor_create_param_*_f64. */
static TensorHandle tape_persistent_f32_from_doubles(int* shape, int rank,
                                                     double* data, int rg) {
    int numel = 1;
    for (int i = 0; i < rank; i++) numel *= shape[i];
    Tensor* t = calloc(1, sizeof(Tensor));
    t->data = malloc(numel * sizeof(float));
    for (int i = 0; i < numel; i++) ((float*)t->data)[i] = (float)data[i];
    free(data);
    t->shape = malloc(rank * sizeof(int));
    memcpy(t->shape, shape, rank * sizeof(int));
    t->rank = rank;
    t->numel = numel;
    t->requires_grad = rg;
    t->tape_idx = -1;
    t->persistent = 1;
    t->dtype_tag = DT_F32;
    if (rg) tape_append(OP_CONST, t, NULL, NULL, 0);
    return t;
}

/* ----------------------------------------------------------------------
   Port-bound creators. Exported (non-static) so the adapter can take
   their addresses; the rename header maps each name per backend.
   ---------------------------------------------------------------------- */

TensorHandle tape_create_scalar_dtag(double v, int rg, int dtag) {
    if (dtag == 15) return tensor_create_scalar_f64(v, rg);
    if (dtag == 14) return make_scalar_f32(v, rg);
    return tape_retag_round(tensor_create_scalar_f64(v, rg), dtag);
}

TensorHandle tape_create_dtag(double* data, int* shape, int rank, int rg, int dtag) {
    if (dtag == 15) return tensor_create_f64(data, shape, rank, rg);
    if (dtag == 14) {
        /* tensor_create_f64 copies + frees the input; mirror that contract.
           We need our own free since we bypass the f64 creator. */
        int numel = 1;
        for (int i = 0; i < rank; i++) numel *= shape[i];
        float* arena_d = arena_alloc(numel * sizeof(float));
        for (int i = 0; i < numel; i++) arena_d[i] = (float)data[i];
        return make_tensor_arena_f32(arena_d, numel, shape, rank, rg);
    }
    return tape_retag_round(tensor_create_f64(data, shape, rank, rg), dtag);
}

TensorHandle tape_create_1d_dtag(int n, double* data, int rg, int dtag) {
    if (dtag == 15) return tensor_create_1d_f64(n, data, rg);
    if (dtag == 14) { int s[] = {n}; return tape_arena_f32_from_doubles(s, 1, data, rg); }
    return tape_retag_round(tensor_create_1d_f64(n, data, rg), dtag);
}

TensorHandle tape_create_2d_dtag(int rows, int cols, double* data, int rg, int dtag) {
    if (dtag == 15) return tensor_create_2d_f64(rows, cols, data, rg);
    if (dtag == 14) { int s[] = {rows, cols}; return tape_arena_f32_from_doubles(s, 2, data, rg); }
    return tape_retag_round(tensor_create_2d_f64(rows, cols, data, rg), dtag);
}

TensorHandle tape_create_param_1d_dtag(int n, double* data, int dtag) {
    if (dtag == 15) return tensor_create_param_1d_f64(n, data);
    if (dtag == 14) { int s[] = {n}; return tape_persistent_f32_from_doubles(s, 1, data, /*rg=*/1); }
    return tape_retag_round(tensor_create_param_1d_f64(n, data), dtag);
}

TensorHandle tape_create_param_2d_dtag(int rows, int cols, double* data, int dtag) {
    if (dtag == 15) return tensor_create_param_2d_f64(rows, cols, data);
    if (dtag == 14) { int s[] = {rows, cols}; return tape_persistent_f32_from_doubles(s, 2, data, /*rg=*/1); }
    return tape_retag_round(tensor_create_param_2d_f64(rows, cols, data), dtag);
}

TensorHandle tape_create_param_3d_dtag(int d0, int d1, int d2, double* data, int dtag) {
    if (dtag == 15) return tensor_create_param_3d_f64(d0, d1, d2, data);
    if (dtag == 14) { int s[] = {d0, d1, d2}; return tape_persistent_f32_from_doubles(s, 3, data, /*rg=*/1); }
    return tape_retag_round(tensor_create_param_3d_f64(d0, d1, d2, data), dtag);
}

TensorHandle tape_create_param_4d_dtag(int d0, int d1, int d2, int d3, double* data, int dtag) {
    if (dtag == 15) return tensor_create_param_4d_f64(d0, d1, d2, d3, data);
    if (dtag == 14) { int s[] = {d0, d1, d2, d3}; return tape_persistent_f32_from_doubles(s, 4, data, /*rg=*/1); }
    return tape_retag_round(tensor_create_param_4d_f64(d0, d1, d2, d3, data), dtag);
}

TensorHandle tape_create_state_1d_dtag(int n, double* data, int dtag) {
    if (dtag == 15) return tensor_create_state_1d_f64(n, data);
    if (dtag == 14) { int s[] = {n}; return tape_persistent_f32_from_doubles(s, 1, data, /*rg=*/0); }
    return tape_retag_round(tensor_create_state_1d_f64(n, data), dtag);
}

TensorHandle tape_create_state_2d_dtag(int rows, int cols, double* data, int dtag) {
    if (dtag == 15) return tensor_create_state_2d_f64(rows, cols, data);
    if (dtag == 14) { int s[] = {rows, cols}; return tape_persistent_f32_from_doubles(s, 2, data, /*rg=*/0); }
    return tape_retag_round(tensor_create_state_2d_f64(rows, cols, data), dtag);
}

/* Cast is locally linear (the rounding is value-level, not gradient-level),
   so backward passes the upstream gradient through unchanged. Both r->grad
   and a->grad are F64 buffers via ensure_grad regardless of value storage,
   so the loop is dtype-agnostic. */
static void tape_backward_cast_dtype(TapeEntry* e) {
    Tensor* r = e->result;
    Tensor* a = e->arg1;
    if (!a) return;
    ensure_grad(a);
    ensure_grad(r);
    for (int i = 0; i < r->numel; i++) {
        ((double*)a->grad)[i] += ((double*)r->grad)[i];
    }
}
TAPE_REGISTER_OP(OP_CAST_DTYPE, tape_backward_cast_dtype)

TensorHandle tape_cast_dtype_dtag(TensorHandle src, int dtag) {
    Tensor* s = (Tensor*)src;
    int tag = tape_tag_from_dtag(dtag);
    int rg = s->requires_grad;
    /* F64 → F64 stays observational identity (preserves autograd-through-cast).
       The shortcut applies only when the *source* is already F64 — casting a
       non-F64 source up to F64 must still produce a fresh F64-tagged tensor. */
    if (tag == DT_F64 && s->dtype_tag == DT_F64) return tensor_cast_dtype_f64(src);
    /* For all other directions, propagate `requires_grad` from the source and
       record an OP_CAST_DTYPE entry so backward flows through. Pre-A1 this
       hardcoded `rg=0` on every non-F64-identity path, which silently dropped
       autograd lineage at every dtype boundary — broke mixed-precision
       training in the typed-layer path. */
    Tensor* result = NULL;
    /* F32 target: real F32 storage (4 bytes/elem), matching the streamed-create
       path. Skipping this and falling through to tape_retag_round would produce
       a lingua-franca F32 (double storage, DT_F32 tag) — internally consistent
       for tensor_item_1d (reads as double*) but garbage for tensor_to_doubles
       / tape_load_d / the F32 kernels (all assume 4-byte-per-elem float
       storage). tape_load_d on the read side normalizes real-F32 vs lingua-
       franca sources. */
    if (tag == DT_F32) {
        if (s->rank == 0) {
            double v = tape_load_d(s, 0);
            result = make_scalar_f32(v, rg);
        } else {
            float* arena_d = arena_alloc(s->numel * sizeof(float));
            for (int i = 0; i < s->numel; i++) arena_d[i] = (float)tape_load_d(s, i);
            result = make_tensor_arena_f32(arena_d, s->numel, s->shape, s->rank, rg);
        }
    } else {
        /* Lingua-franca path: fresh F64-storage tensor with the values rounded
           into the target dtype's representable precision, then retagged. */
        if (s->rank == 0) {
            double v = tape_load_d(s, 0);
            result = (Tensor*)tape_retag_round(make_scalar(v, rg), dtag);
        } else {
            double* arena_d = arena_alloc(s->numel * sizeof(double));
            for (int i = 0; i < s->numel; i++) arena_d[i] = tape_load_d(s, i);
            Tensor* t = make_tensor_arena(arena_d, s->numel, s->shape, s->rank, rg);
            result = (Tensor*)tape_retag_round((TensorHandle)t, dtag);
        }
    }
    if (rg) tape_append(OP_CAST_DTYPE, result, s, NULL, 0.0);
    return (TensorHandle)result;
}
