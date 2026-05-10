/* training/dtype_streamed.c — dtag-streamed creation + cast wrappers.
 *
 * The unified FFI dispatch surface that landed on
 * 2026-05-22: `tensor_create_<shape>_streamed(..., int dtag)` takes
 * a runtime dtype tag and routes to the appropriate concrete storage
 * (F64 lingua franca, real F32 arena/persistent, or one of the
 * inference-only dtypes via tape_retag_round).
 *
 * Static helpers (tape_retag_round, tape_arena_f32_from_doubles,
 * tape_persistent_f32_from_doubles) are TU-private to this file —
 * only the streamed wrappers consume them.
 *
 * Under the kind-major dtag layout (closed 2026-05-23): dtag 15 = F64,
 * dtag 14 = F32. All other valid dtags (Bool=1, U8=4, I8/I16/I32/I64,
 * F16=13, BF16=17) route through tape_retag_round's lingua-franca
 * path. Invalid dtags abort via tape_tag_from_dtag's default arm.
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

TensorHandle tensor_create_scalar_streamed(double value, int requires_grad, int stream_tag, int dtag) {
    (void)stream_tag;
    if (dtag == 15) return tensor_create_scalar_f64(value, requires_grad);
    if (dtag == 14) return make_scalar_f32(value, requires_grad);
    return tape_retag_round(tensor_create_scalar_f64(value, requires_grad), dtag);
}
TensorHandle tensor_create_streamed(double* data, int* shape, int rank, int requires_grad, int stream_tag, int dtag) {
    (void)stream_tag;
    if (dtag == 15) return tensor_create_f64(data, shape, rank, requires_grad);
    if (dtag == 14) {
        /* tensor_create_f64 copies + then frees the input; mirror that contract.
           We need our own free since we bypass the f64 creator. */
        int numel = 1;
        for (int i = 0; i < rank; i++) numel *= shape[i];
        float* arena_d = arena_alloc(numel * sizeof(float));
        for (int i = 0; i < numel; i++) arena_d[i] = (float)data[i];
        return make_tensor_arena_f32(arena_d, numel, shape, rank, requires_grad);
    }
    return tape_retag_round(tensor_create_f64(data, shape, rank, requires_grad), dtag);
}
TensorHandle tensor_create_1d_streamed(int n, double* data, int requires_grad, int stream_tag, int dtag) {
    (void)stream_tag;
    if (dtag == 15) return tensor_create_1d_f64(n, data, requires_grad);
    if (dtag == 14) { int s[] = {n}; return tape_arena_f32_from_doubles(s, 1, data, requires_grad); }
    return tape_retag_round(tensor_create_1d_f64(n, data, requires_grad), dtag);
}
TensorHandle tensor_create_2d_streamed(int rows, int cols, double* data, int requires_grad, int stream_tag, int dtag) {
    (void)stream_tag;
    if (dtag == 15) return tensor_create_2d_f64(rows, cols, data, requires_grad);
    if (dtag == 14) { int s[] = {rows, cols}; return tape_arena_f32_from_doubles(s, 2, data, requires_grad); }
    return tape_retag_round(tensor_create_2d_f64(rows, cols, data, requires_grad), dtag);
}
TensorHandle tensor_create_param_1d_streamed(int n, double* data, int stream_tag, int dtag) {
    (void)stream_tag;
    if (dtag == 15) return tensor_create_param_1d_f64(n, data);
    if (dtag == 14) { int s[] = {n}; return tape_persistent_f32_from_doubles(s, 1, data, /*rg=*/1); }
    return tape_retag_round(tensor_create_param_1d_f64(n, data), dtag);
}
TensorHandle tensor_create_param_2d_streamed(int rows, int cols, double* data, int stream_tag, int dtag) {
    (void)stream_tag;
    if (dtag == 15) return tensor_create_param_2d_f64(rows, cols, data);
    if (dtag == 14) { int s[] = {rows, cols}; return tape_persistent_f32_from_doubles(s, 2, data, /*rg=*/1); }
    return tape_retag_round(tensor_create_param_2d_f64(rows, cols, data), dtag);
}
TensorHandle tensor_create_param_3d_streamed(int d0, int d1, int d2, double* data, int stream_tag, int dtag) {
    (void)stream_tag;
    if (dtag == 15) return tensor_create_param_3d_f64(d0, d1, d2, data);
    if (dtag == 14) { int s[] = {d0, d1, d2}; return tape_persistent_f32_from_doubles(s, 3, data, /*rg=*/1); }
    return tape_retag_round(tensor_create_param_3d_f64(d0, d1, d2, data), dtag);
}
TensorHandle tensor_create_param_4d_streamed(int d0, int d1, int d2, int d3, double* data, int stream_tag, int dtag) {
    (void)stream_tag;
    if (dtag == 15) return tensor_create_param_4d_f64(d0, d1, d2, d3, data);
    if (dtag == 14) { int s[] = {d0, d1, d2, d3}; return tape_persistent_f32_from_doubles(s, 4, data, /*rg=*/1); }
    return tape_retag_round(tensor_create_param_4d_f64(d0, d1, d2, d3, data), dtag);
}
TensorHandle tensor_create_state_1d_streamed(int n, double* data, int stream_tag, int dtag) {
    (void)stream_tag;
    if (dtag == 15) return tensor_create_state_1d_f64(n, data);
    if (dtag == 14) { int s[] = {n}; return tape_persistent_f32_from_doubles(s, 1, data, /*rg=*/0); }
    return tape_retag_round(tensor_create_state_1d_f64(n, data), dtag);
}
TensorHandle tensor_create_state_2d_streamed(int rows, int cols, double* data, int stream_tag, int dtag) {
    (void)stream_tag;
    if (dtag == 15) return tensor_create_state_2d_f64(rows, cols, data);
    if (dtag == 14) { int s[] = {rows, cols}; return tape_persistent_f32_from_doubles(s, 2, data, /*rg=*/0); }
    return tape_retag_round(tensor_create_state_2d_f64(rows, cols, data), dtag);
}
TensorHandle tensor_cast_dtype_streamed(TensorHandle src, int stream_tag, int dtag) {
    (void)stream_tag;
    Tensor* s = (Tensor*)src;
    int tag = tape_tag_from_dtag(dtag);
    /* F64 → F64 stays observational identity (preserves autograd-through-cast).
       The shortcut applies only when the *source* is already F64 — casting a
       non-F64 source up to F64 must still produce a fresh F64-tagged tensor. */
    if (tag == DT_F64 && s->dtype_tag == DT_F64) return tensor_cast_dtype_f64(src);
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
            return make_scalar_f32(v, 0);
        }
        float* arena_d = arena_alloc(s->numel * sizeof(float));
        for (int i = 0; i < s->numel; i++) arena_d[i] = (float)tape_load_d(s, i);
        return make_tensor_arena_f32(arena_d, s->numel, s->shape, s->rank, 0);
    }
    /* Otherwise: a fresh non-grad tensor cloning src's values, rounded into the
       target dtype. (Inference / precision-demo casts are NoGrad.) Source is
       read via tape_load_d so a real-F32 source casts correctly into the
       lingua-franca target. */
    if (s->rank == 0) {
        double v = tape_load_d(s, 0);
        return tape_retag_round(make_scalar(v, 0), dtag);
    }
    double* arena_d = arena_alloc(s->numel * sizeof(double));
    for (int i = 0; i < s->numel; i++) arena_d[i] = tape_load_d(s, i);
    Tensor* t = make_tensor_arena(arena_d, s->numel, s->shape, s->rank, 0);
    return tape_retag_round((TensorHandle)t, dtag);
}
