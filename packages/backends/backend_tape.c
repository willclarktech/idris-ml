/* backend_tape.c — Tape-based autograd backend implementing backend.h.
 *
 * Each tensor is a scalar double with an index into a global tape.
 * Multi-dimensional tensors are flat double arrays with shape metadata.
 * Forward ops append to the tape; backward walks it in reverse.
 *
 * Design: arena-allocated tape + Accelerate BLAS for linalg.
 */

#include "backend.h"
#include "shared_utils.h"  /* bf16/f16 bit-conv helpers (Phase 4 round-trip) */
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <stdio.h>
#include <stdint.h>

#ifdef __APPLE__
#include <Accelerate/Accelerate.h>
#include <sys/resource.h>
#include <mach/mach.h>
#endif



/* ================================================================
   Tape backend modular includes (Phase 1.0.4: per-TU compile).
   Implementations live in backend_tape subdirs — compiled as separate
   TUs and linked into libidrisml. Headers below give backend_tape.c
   the type defs + extern decls it needs to reference them.
   ================================================================ */
#include "backend_tape/tensor.h"
#include "backend_tape/arena.h"
#include "backend_tape/tape.h"
#include "backend_tape/broadcast.h"
#include "backend_tape/training/autograd/op_dispatch.h"
#include "backend_tape/core/elementwise/_helpers.h"

/* Forward decl: _wall_ms lives later in the profiling section but is
 * called from the elementwise kernel includes (line 311 area). Defined
 * non-static so tape.c (and any future profiling-using TU) can extern
 * it via tape.h. */
double _wall_ms(void);


/* ================================================================
   Lifecycle — Phase 1a.1: extracted to backend_tape/core/lifecycle/
   {create_scalar,create,clone,free,item}.c
   ================================================================ */

/* ================================================================
   Accessors
   ================================================================ */


int tensor_numel(TensorHandle h) {
    return ((Tensor*)h)->numel;
}

int tensor_dim(TensorHandle h) {
    return ((Tensor*)h)->rank;
}

int tensor_size(TensorHandle h, int dim) {
    Tensor* t = (Tensor*)h;
    if (dim < t->rank) return t->shape[dim];
    return 0;
}

void tensor_to_doubles(TensorHandle h, double* out) {
    Tensor* t = (Tensor*)h;
    if (t->dtype_tag == DT_F32) {
        for (int i = 0; i < t->numel; i++) out[i] = (double)((float*)t->data)[i];
    } else {
        memcpy(out, t->data, t->numel * sizeof(double));
    }
}

/* Byte-level I64 readout — declared in backend.h with the byte-exact
   contract honoured only on backends with native int64 storage. Tape
   has no native int storage (integer dtypes ride in `double*` via the
   lingua-franca path, rounded on store), so this is a per-element cast
   from the dtype-uniform double view. Matches the value the safetensors
   double path was producing pre-row-20; no regression. */
void tensor_to_int64(TensorHandle h, int64_t* out) {
    Tensor* t = (Tensor*)h;
    for (int i = 0; i < t->numel; i++) {
        out[i] = (int64_t)tape_load_d(t, i);
    }
}

void tensor_to_floats(TensorHandle h, float* out) {
    Tensor* t = (Tensor*)h;
    if (t->dtype_tag == DT_F32) {
        memcpy(out, t->data, t->numel * sizeof(float));
    } else {
        for (int i = 0; i < t->numel; i++) out[i] = (float)((double*)t->data)[i];
    }
}

const char* tensor_dtype_name(TensorHandle h) {
    switch (((Tensor*)h)->dtype_tag) {
        case DT_F32:  return "F32";
        case DT_BF16: return "BF16";
        case DT_F16:  return "F16";
        case DT_I8:   return "I8";
        case DT_I16:  return "I16";
        case DT_I32:  return "I32";
        case DT_I64:  return "I64";
        case DT_U8:   return "U8";
        case DT_BOOL: return "BOOL";
        default:      return "F64";  /* DT_F64 */
    }
}

/* ================================================================
   Scalar arithmetic (forward only — backward in tape walk)
   ================================================================ */

#define SCALAR_BINOP(name, op_tag, expr) \
TensorHandle name(TensorHandle ha, TensorHandle hb) { \
    Tensor* a = (Tensor*)ha; Tensor* b = (Tensor*)hb; \
    double val = (expr); \
    Tensor* r = make_scalar(val, a->requires_grad || b->requires_grad); \
    if (r->requires_grad) tape_append(op_tag, r, a, b, 0); \
    return r; \
}

#define SCALAR_UNOP(name, op_tag, expr) \
TensorHandle name(TensorHandle ha) { \
    Tensor* a = (Tensor*)ha; \
    double val = (expr); \
    Tensor* r = make_scalar(val, a->requires_grad); \
    if (r->requires_grad) tape_append(op_tag, r, a, NULL, 0); \
    return r; \
}

/* ----------------------------------------------------------------
   Numpy-style broadcast helpers for elementwise binary ops.
   Used by binop_elementwise forward and OP_ADD/SUB/MUL/DIV/POW
   backward grad-reduction.
   ---------------------------------------------------------------- */

/* Broadcast helpers — declared in backend_tape/broadcast.h; defined
 * here (still in the monolith) but non-static so per-op files can
 * call them. MAX_BCAST_RANK comes via the header. */

/* True if `a`'s shape exactly matches `r`'s shape (no broadcast). */
int shapes_equal(Tensor* a, Tensor* r) {
    if (a->numel != r->numel || a->rank != r->rank) return 0;
    for (int k = 0; k < a->rank; k++) {
        if (a->shape[k] != r->shape[k]) return 0;
    }
    return 1;
}

/* Compute broadcast output shape from a and b (right-aligned, numpy rules).
   Returns 1 on success, 0 on incompatible shapes. */
int compute_bcast_shape(Tensor* a, Tensor* b,
                        int* r_shape, int* r_rank, int* r_numel) {
    int rank = a->rank > b->rank ? a->rank : b->rank;
    if (rank > MAX_BCAST_RANK) return 0;
    int numel = 1;
    for (int k = rank - 1; k >= 0; k--) {
        int ai = k - (rank - a->rank);
        int bi = k - (rank - b->rank);
        int sa = (ai >= 0) ? a->shape[ai] : 1;
        int sb = (bi >= 0) ? b->shape[bi] : 1;
        if (sa != sb && sa != 1 && sb != 1) return 0;
        int so = sa > sb ? sa : sb;
        r_shape[k] = so;
        numel *= so;
    }
    *r_rank = rank;
    *r_numel = numel;
    return 1;
}

/* Compute right-aligned broadcast strides for `a` w.r.t. output shape r_shape.
   out_strides[k] is the increment to a's flat index when output dim k advances;
   0 means broadcast on that dim. r_rank may exceed a->rank (rank-padding). */
void compute_bcast_strides(Tensor* a, int r_rank, int* r_shape,
                           int* out_strides) {
    int a_rank = a->rank;
    int natural[MAX_BCAST_RANK];
    int s = 1;
    for (int k = a_rank - 1; k >= 0; k--) { natural[k] = s; s *= a->shape[k]; }
    int offset = r_rank - a_rank;
    for (int j = 0; j < r_rank; j++) {
        int ai = j - offset;
        if (ai < 0) {
            out_strides[j] = 0;  /* phantom dim from rank-padding */
        } else {
            int sa = a->shape[ai];
            int sr = r_shape[j];
            out_strides[j] = (sa == 1 && sr > 1) ? 0 : natural[ai];
        }
    }
}

/* Element-wise binary + unary kernel bodies live in backend_tape_kernels.inc
   so the same source compiles for F64 and (later, step 6) F32 storage.
   Today: single include with SCALAR=double / SFX(name)=name##_f64.
   The kernels (binop_elementwise_inner_f64 / unop_elementwise_f64) are
   the implementation; the wrappers below are the public entry points
   step 7 will tag-dispatch from. */
/* F64 stamping. */
#define SCALAR    double
#define SFX(name) name##_f64
#define VDSP_VADD vDSP_vaddD
#define VDSP_VSUB vDSP_vsubD
#define VDSP_VMUL vDSP_vmulD
#define VDSP_VDIV vDSP_vdivD
#define VDSP_VNEG vDSP_vnegD
#define VV_EXP    vvexp
#define VV_LOG    vvlog
#define VV_SQRT   vvsqrt
#define VV_TANH   vvtanh
#define VV_FABS   vvfabs
#include "backend_tape_kernels.inc"
#undef SCALAR
#undef SFX
#undef VDSP_VADD
#undef VDSP_VSUB
#undef VDSP_VMUL
#undef VDSP_VDIV
#undef VDSP_VNEG
#undef VV_EXP
#undef VV_LOG
#undef VV_SQRT
#undef VV_TANH
#undef VV_FABS

/* F32 stamping — uses cblas_s* / vDSP_* (no `D` suffix) / vv*f forms. */
#define SCALAR    float
#define SFX(name) name##_f32
#define VDSP_VADD vDSP_vadd
#define VDSP_VSUB vDSP_vsub
#define VDSP_VMUL vDSP_vmul
#define VDSP_VDIV vDSP_vdiv
#define VDSP_VNEG vDSP_vneg
#define VV_EXP    vvexpf
#define VV_LOG    vvlogf
#define VV_SQRT   vvsqrtf
#define VV_TANH   vvtanhf
#define VV_FABS   vvfabsf
#include "backend_tape_kernels.inc"
#undef SCALAR
#undef SFX
#undef VDSP_VADD
#undef VDSP_VSUB
#undef VDSP_VMUL
#undef VDSP_VDIV
#undef VDSP_VNEG
#undef VV_EXP
#undef VV_LOG
#undef VV_SQRT
#undef VV_TANH
#undef VV_FABS

TensorHandle binop_elementwise(TensorHandle ha, TensorHandle hb, int op_tag,
                               double (*scalar_fn)(double, double)) {
    extern double prof_binop_inside_ms[];
    extern int prof_binop_inside_count[];
    double _b0 = _wall_ms();
    TensorHandle r = binop_elementwise_inner_f64(ha, hb, op_tag, scalar_fn);
    if (op_tag >= 0 && op_tag < OP_COUNT) {
        prof_binop_inside_ms[op_tag] += _wall_ms() - _b0;
        prof_binop_inside_count[op_tag]++;
    }
    return r;
}

TensorHandle binop_elementwise_f32_disp(TensorHandle ha, TensorHandle hb, int op_tag,
                                        float (*scalar_fn)(float, float)) {
    extern double prof_binop_inside_ms[];
    extern int prof_binop_inside_count[];
    double _b0 = _wall_ms();
    TensorHandle r = binop_elementwise_inner_f32(ha, hb, op_tag, scalar_fn);
    if (op_tag >= 0 && op_tag < OP_COUNT) {
        prof_binop_inside_ms[op_tag] += _wall_ms() - _b0;
        prof_binop_inside_count[op_tag]++;
    }
    return r;
}

void tape_abort_mixed_dtype(const char* op) {
    fprintf(stderr,
        "[tape backend] %s: mixed-dtype inputs forbidden — both operands must "
        "share a dtype_tag (cast first via tcast / tensor_cast_dtype_streamed).\n", op);
    abort();
}

static double fn_add(double a, double b) { return a + b; }
static double fn_sub(double a, double b) { return a - b; }
static double fn_mul(double a, double b) { return a * b; }
static double fn_div(double a, double b) { return a / b; }
static double fn_pow(double a, double b) { return pow(a, b); }

/* F32 scalar function counterparts — the F32 stamping of binop_elementwise_inner
   expects float (*)(float, float). */
static float fn_add_f32(float a, float b) { return a + b; }
static float fn_sub_f32(float a, float b) { return a - b; }
static float fn_mul_f32(float a, float b) { return a * b; }
static float fn_div_f32(float a, float b) { return a / b; }
static float fn_pow_f32(float a, float b) { return powf(a, b); }

/* Forward dispatch: both-F32 → F32 stamping; both-F64 → F64 stamping;
   mixed → abort. The F64 path is the default fallthrough so any inference
   dtype that ever leaks (it shouldn't, given the Idris-side Compatible
   gate) doesn't silently take an unintended branch. */
#define TAPE_BINOP_DISPATCH(name, op_tag, fn64, fn32) \
TensorHandle name(TensorHandle a, TensorHandle b) { \
    Tensor* ta = (Tensor*)a; Tensor* tb = (Tensor*)b; \
    if (ta->dtype_tag == DT_F32 || tb->dtype_tag == DT_F32) { \
        if (ta->dtype_tag != tb->dtype_tag) tape_abort_mixed_dtype(#name); \
        return binop_elementwise_f32_disp(a, b, op_tag, fn32); \
    } \
    return binop_elementwise(a, b, op_tag, fn64); \
}
/* tensor_add, tensor_sub, tensor_mul, tensor_div, tensor_pow: moved to backend_tape/core/elementwise/ (Phase 1a.2-7). */
#undef TAPE_BINOP_DISPATCH

/* Unary ops: support both scalar (rank 0) and multi-element tensors */
static double fn_neg(double x) { return -x; }
static double fn_abs(double x) { return fabs(x); }
static double fn_exp_d(double x) { return exp(x); }
static double fn_log_d(double x) { return log(x); }
static double fn_sqrt_d(double x) { return sqrt(x); }
static double fn_sigmoid(double x) { return 1.0 / (1.0 + exp(-x)); }
static double fn_tanh_d(double x) { return tanh(x); }
/* GELU(x) = x * 0.5 * (1 + tanh(sqrt(2/pi) * (x + 0.044715 * x^3))) */
static double fn_gelu_d(double x) {
    double c = 0.7978845608028654;  /* sqrt(2/pi) */
    double inner = c * (x + 0.044715 * x * x * x);
    return 0.5 * x * (1.0 + tanh(inner));
}
/* F32 unary scalar functions — paired with unop_elementwise_f32. */
static float fn_neg_f32(float x) { return -x; }
static float fn_abs_f32(float x) { return fabsf(x); }
static float fn_exp_f32(float x) { return expf(x); }
static float fn_log_f32(float x) { return logf(x); }
static float fn_sqrt_f32(float x) { return sqrtf(x); }
static float fn_sigmoid_f32(float x) { return 1.0f / (1.0f + expf(-x)); }
static float fn_tanh_f32(float x) { return tanhf(x); }
static float fn_gelu_f32(float x) {
    float c = 0.7978845608028654f;
    float inner = c * (x + 0.044715f * x * x * x);
    return 0.5f * x * (1.0f + tanhf(inner));
}

TensorHandle unop_elementwise(TensorHandle ha, int op, double (*fn)(double)) {
    /* Body lives in backend_tape_kernels.inc as unop_elementwise_f64. */
    return unop_elementwise_f64(ha, op, fn);
}

/* Symmetric F32 dispatch wrapper (mirror of binop_elementwise_f32_disp).
 * unop_elementwise_f32 itself is static (in the .inc stamping); this
 * non-static wrapper exposes it to per-op files in core/elementwise/. */
TensorHandle unop_elementwise_f32_disp(TensorHandle ha, int op_tag, float (*fn)(float)) {
    return unop_elementwise_f32(ha, op_tag, fn);
}

/* Forward dispatch: tag-aware unop wrappers — F32 input routes to the F32
   stamping, else falls through to F64. The F32 stamping picks up the
   matching fn_*_f32 helper. */
#define TAPE_UNOP_DISPATCH(name, op_tag, fn64, fn32) \
TensorHandle name(TensorHandle ha) { \
    Tensor* a = (Tensor*)ha; \
    if (a->dtype_tag == DT_F32) return unop_elementwise_f32(ha, op_tag, fn32); \
    return unop_elementwise(ha, op_tag, fn64); \
}
/* tensor_neg/abs/exp/log/sqrt: moved to backend_tape/core/elementwise/ (Phase 1a.6). */
/* tensor_sigmoid, tensor_tanh: moved to backend_tape/core/elementwise/ (Phase 1a.8). */
#undef TAPE_UNOP_DISPATCH
/* tensor_gelu: moved to backend_tape/nn/activation/gelu.c (Phase 1c.2). */

/* LeakyReLU: max(alpha*x, x). Uses scalar_arg to store alpha. F32 forward
   uses real F32 arena storage; backward (OP_LEAKY_RELU) reads a->data via
   tape_load_d so both dtypes share the same case body. */
/* tensor_leaky_relu_f32: moved to backend_tape/nn/activation/leaky_relu.c (Phase 1c.2). */

/* tensor_leaky_relu, tensor_silu: moved to backend_tape/nn/activation/ (Phase 1c.2). */

/* tensor_softplus: moved to backend_tape/core/elementwise/softplus.c (Phase 1a.8). */

/* tensor_add_scalar / tensor_mul_scalar / tensor_clamp_min:
 * moved to backend_tape/core/scalar/ (Phase 1a.9). */

/* ================================================================
   Reduction
   ================================================================ */

/* Reductions — dtype-aware scalar output. tape_load_d covers the data read;
   the dtype-matched make_scalar / make_scalar_f32 sets the result tag.
   Backward (OP_SUM, OP_MEAN) only writes input grads (always F64) and
   doesn't read input data, so the existing cases work for both dtypes. */
/* tensor_sum, tensor_sum_dim, tensor_mean, tensor_min, tensor_max:
 * moved to backend_tape/linear/reduction/ (Phase 1b.3). */

/* ================================================================
   Linear algebra
   ================================================================ */

/* F32 stamping of tensor_mv. F64 grads are kept (asymmetric), so the
   backward case still reads a double* x_vals cache — we convert from
   the F32 vec on store. Output uses make_tensor_arena_f32 + tags F32. */
/* tensor_mv: moved to backend_tape/linear/linalg/mv.c (Phase 1b.4.b). */

/* Fused batched linear: Y[B,o] = X[B,i] @ W[o,i]^T + bias[o].
   Single allocation, single tape entry. W: [o, i], X: [B, i], bias: [o] (or NULL). */
/* tensor_linear_2d: moved to backend_tape/linear/linalg/linear_2d.c (Phase 1b.5). */

/* Concatenate along axis 1: A[m,n] ++ B[m,k] -> [m, n+k].
   Single tape entry; backward scatters dY back to dA / dB by column split. */
/* tensor_concat_2d_axis1: moved to backend_tape/linear/concat/concat_2d_axis1.c (Phase 1b.2.c). */

/* F32 stamping of tensor_linear. F32 mat/vec/bias → F32 output via
   cblas_sgemv + vDSP_vadd. Meta caches x_vals as double* (converted on
   store) so the existing backward case can read it uniformly. */
/* tensor_linear: moved to backend_tape/linear/linalg/linear.c (Phase 1b.5). */

/* tensor_dot: moved to backend_tape/linear/linalg/dot.c (Phase 1b.4). */

/* tensor_matmul: moved to backend_tape/linear/linalg/matmul.c (Phase 1b.5; covers OP_VECMAT). */

/* tensor_outer: moved to backend_tape/linear/linalg/outer.c (Phase 1b.4). */

/* tensor_mm: moved to backend_tape/linear/linalg/mm.c (Phase 1b.5). */

/* Batched matrix-matrix multiply: [B,m,n] x [n,k] -> [B,m,k]
   Weight matrix b is shared across all batch elements. */
/* tensor_bmm, tensor_bmm_3x3: moved to backend_tape/linear/linalg/ (Phase 1b.6). */

/* Softmax over the last dim of a [B,m,n] tensor — F32 + F64 unified path. */
/* tensor_softmax_3d: moved to backend_tape/nn/softmax/softmax_3d.c (Phase 1c.1). */

/* tensor_transpose_last2: moved to backend_tape/linear/linalg/transpose_last2.c (Phase 1b.6). */

/* tensor_reshape_4d, tensor_reshape_3d: moved to backend_tape/linear/shape/ (Phase 1b.9). */

/* tensor_expand_mask: moved to backend_tape/nn/mask/expand_mask.c (Phase 1c.3). */

/* tensor_tile_2d + Tile2dMeta: moved to backend_tape/linear/linalg/tile_2d.c (Phase 1b.6). */

/* Stack B tensors of shape [m, n] into [B, m, n].
   All tensors must have the same shape. No gradient tracking (data tensors). */
TensorHandle tensor_batch(TensorHandle* handles, int count) {
    Tensor* first = (Tensor*)handles[0];
    int elem_size = first->numel;
    int total = count * elem_size;
    int rank = first->rank + 1;
    int is_f32 = (first->dtype_tag == DT_F32);
    for (int i = 1; i < count; i++)
        if (((Tensor*)handles[i])->dtype_tag != first->dtype_tag)
            tape_abort_mixed_dtype("tensor_batch");
    int* shape = malloc(rank * sizeof(int));
    shape[0] = count;
    for (int i = 0; i < first->rank; i++) shape[i+1] = first->shape[i];
    Tensor* r;
    if (is_f32) {
        float* data = arena_alloc(total * sizeof(float));
        for (int i = 0; i < count; i++)
            memcpy(data + i * elem_size, ((Tensor*)handles[i])->data, elem_size * sizeof(float));
        r = make_tensor_arena_f32(data, total, shape, rank, 0);
    } else {
        double* data = malloc(total * sizeof(double));
        for (int i = 0; i < count; i++)
            memcpy(data + i * elem_size, ((Tensor*)handles[i])->data, elem_size * sizeof(double));
        r = make_tensor(data, shape, rank, 0);
        free(data);
    }
    free(shape);
    return r;
}

/* Split [B, ...] tensor into B tensors of shape [...].
   Returns array of B tensor handles (caller must free array). */
TensorHandle* tensor_unbatch(TensorHandle h, int* out_count) {
    Tensor* t = (Tensor*)h;
    int B = t->shape[0];
    *out_count = B;
    int elem_size = t->numel / B;
    int inner_rank = t->rank - 1;
    size_t es = tape_elem_size(t->dtype_tag);
    TensorHandle* handles = malloc(B * sizeof(TensorHandle));
    for (int i = 0; i < B; i++) {
        Tensor* r = arena_alloc(sizeof(Tensor));
        memset(r, 0, sizeof(Tensor));
        r->data = (char*)t->data + (size_t)(i * elem_size) * es;  /* byte-correct view */
        r->shape = arena_alloc(inner_rank * sizeof(int));
        for (int j = 0; j < inner_rank; j++) r->shape[j] = t->shape[j+1];
        r->rank = inner_rank;
        r->numel = elem_size;
        r->requires_grad = t->requires_grad;
        r->tape_idx = -1;
        r->persistent = 0;
        r->dtype_tag = t->dtype_tag;
        handles[i] = (TensorHandle)r;
    }
    return handles;
}

/* tensor_transpose_2d: moved to backend_tape/linear/linalg/transpose_2d.c (Phase 1b.6). */

/* Row-wise softmax on 2D tensor: [m,n] -> [m,n], each row sums to 1 */
/* tensor_softmax_2d, tensor_log_softmax_2d: moved to backend_tape/nn/softmax/ (Phase 1c.1). */

/* Element-wise multiply for multi-element tensors (same shape) */
TensorHandle tensor_mul_elementwise(TensorHandle ha, TensorHandle hb) {
    return tensor_mul(ha, hb);  /* tensor_mul already handles multi-element via binop_elementwise */
}

/* Sum all elements of a tensor (not just scalar) */
TensorHandle tensor_sum_all(TensorHandle h) {
    return tensor_sum(h);  /* tensor_sum already sums all elements */
}

/* Masked fill: replace positions where mask[i]=1 with value */
/* tensor_masked_fill: moved to backend_tape/nn/mask/masked_fill.c (Phase 1c.3). */


/* Row-wise layer normalization on 2D tensor: y[i,j] = gamma[j] * x_hat[i,j] + beta[j]
   where x_hat[i,j] = (x[i,j] - mean_i) / sqrt(var_i + eps) */
/* tensor_layer_norm_2d: moved to backend_tape/nn/norm/layer_norm_2d.c (Phase 1c.4). */

/* ================================================================
   Activation / normalization
   ================================================================ */

/* F32 stamping of tensor_softmax. Stable formulation (subtract max). */
/* tensor_softmax, tensor_log_softmax (+ _f32 helpers): moved to
 * backend_tape/nn/softmax/ (Phase 1c.1). */

/* ================================================================
   Loss functions
   ================================================================ */

/* Loss kernels — dtype-aware reads via tape_load_d so F32 inputs are
   honoured; output is an F32 scalar when inputs are F32, else F64. Mixed
   dtype between input and target is rejected. */
/* tensor_bce_with_logits: moved to backend_tape/nn/loss/bce_with_logits.c (Phase 1c.6). */

TensorHandle tensor_cross_entropy(TensorHandle hinput, TensorHandle htarget) {
    /* Simplified: compute -sum(target * log_softmax(input)) / n */
    Tensor* input = (Tensor*)hinput;
    Tensor* target = (Tensor*)htarget;
    if (input->dtype_tag != target->dtype_tag) tape_abort_mixed_dtype("tensor_cross_entropy");
    TensorHandle ls = tensor_log_softmax(hinput, 0);
    Tensor* lsT = (Tensor*)ls;
    double loss = 0;
    for (int i = 0; i < lsT->numel; i++) loss -= tape_load_d(target, i) * tape_load_d(lsT, i);
    loss /= lsT->numel;
    return (input->dtype_tag == DT_F32) ? make_scalar_f32(loss, 0) : make_scalar(loss, 0);
}

TensorHandle tensor_mse_loss(TensorHandle hinput, TensorHandle htarget) {
    Tensor* input = (Tensor*)hinput;
    Tensor* target = (Tensor*)htarget;
    if (input->dtype_tag != target->dtype_tag) tape_abort_mixed_dtype("tensor_mse_loss");
    double loss = 0;
    for (int i = 0; i < input->numel; i++) {
        double d = tape_load_d(input, i) - tape_load_d(target, i);
        loss += d * d;
    }
    double mean = loss / input->numel;
    return (input->dtype_tag == DT_F32) ? make_scalar_f32(mean, 0) : make_scalar(mean, 0);
}

/* ================================================================
   NTM-specific compositions
   ================================================================ */

/* tensor_cosine_similarity: moved to backend_tape/nn/attention/cosine_similarity.c (Phase 1c.5). */

TensorHandle tensor_conv1d_circular(TensorHandle hinput, TensorHandle hkernel) {
    Tensor* input = (Tensor*)hinput;
    Tensor* kernel = (Tensor*)hkernel;
    if (input->dtype_tag != kernel->dtype_tag) tape_abort_mixed_dtype("tensor_conv1d_circular");
    int n = input->numel, k = kernel->numel, pad = k / 2;
    int shape[] = {n};
    int rg = input->requires_grad || kernel->requires_grad;
    if (input->dtype_tag == DT_F32) {
        float* out = arena_alloc(n * sizeof(float));
        for (int i = 0; i < n; i++) {
            float s = 0;
            for (int j = 0; j < k; j++) {
                int idx = (i - pad + j + n) % n;
                s += ((float*)input->data)[idx] * ((float*)kernel->data)[k - 1 - j];
            }
            out[i] = s;
        }
        Tensor* r = make_tensor_arena_f32(out, n, shape, 1, rg);
        if (r->requires_grad) tape_append(OP_CONV1D_CIRC, r, input, kernel, 0);
        return r;
    }
    double* out = calloc(n, sizeof(double));
    for (int i = 0; i < n; i++) {
        double s = 0;
        for (int j = 0; j < k; j++) {
            int idx = (i - pad + j + n) % n;
            s += ((double*)input->data)[idx] * ((double*)kernel->data)[k - 1 - j];
        }
        out[i] = s;
    }
    Tensor* r = make_tensor(out, shape, 1, rg);
    free(out);
    if (r->requires_grad) tape_append(OP_CONV1D_CIRC, r, input, kernel, 0);
    return r;
}

/* ================================================================
   Cross-Attention: Q @ K^T * scale [+ mask] -> softmax -> @ V
   Q [B,seqQ,d], K [B,seqK,d], V [B,seqK,d] -> [B,seqQ,d]
   ================================================================ */

/* tensor_cross_attention: moved to backend_tape/nn/attention/cross_attention.c (Phase 1c.5). */

/* ================================================================
   Embedding: row gather from weight matrix
   weight [vocabSize, embedDim], indices [n] -> output [n * embedDim]
   ================================================================ */

/* tensor_embedding: moved to backend_tape/nn/attention/embedding.c (Phase 1c.5). */

/* ================================================================
   Batch Normalization: per-channel, across spatial dims
   Input treated as [C, spatial]. Normalizes each channel independently.
   ================================================================ */

/* tensor_batch_norm: moved to backend_tape/nn/norm/batch_norm.c (Phase 1c.4). */

/* ================================================================
   Group Normalization: normalize within channel groups
   ================================================================ */

TensorHandle tensor_group_norm(TensorHandle hinput, TensorHandle hgamma, TensorHandle hbeta,
                               int numGroups, int channels, int spatial, double eps) {
    Tensor* input = (Tensor*)hinput;
    Tensor* gamma = (Tensor*)hgamma;
    Tensor* beta = (Tensor*)hbeta;
    int n = channels * spatial;
    int chPerGroup = channels / numGroups;
    int groupSize = chPerGroup * spatial;

    if (input->dtype_tag != gamma->dtype_tag || input->dtype_tag != beta->dtype_tag)
        tape_abort_mixed_dtype("tensor_group_norm");
    int is_f32 = (input->dtype_tag == DT_F32);
    int out_shape[] = {n};
    void* out = is_f32 ? (void*)arena_alloc(n * sizeof(float))
                       : (void*)calloc(n, sizeof(double));
    for (int g = 0; g < numGroups; g++) {
        double mean = 0;
        int base = g * groupSize;
        for (int j = 0; j < groupSize; j++) mean += tape_load_d(input, base + j);
        mean /= groupSize;
        double var = 0;
        for (int j = 0; j < groupSize; j++) {
            double d = tape_load_d(input, base + j) - mean;
            var += d * d;
        }
        var /= groupSize;
        double rstd = 1.0 / sqrt(var + eps);
        for (int c = 0; c < chPerGroup; c++) {
            int absC = g * chPerGroup + c;
            double gc = tape_load_d(gamma, absC);
            double bc = tape_load_d(beta, absC);
            for (int s = 0; s < spatial; s++) {
                int idx = absC * spatial + s;
                double x_hat = (tape_load_d(input, idx) - mean) * rstd;
                double v = gc * x_hat + bc;
                if (is_f32) ((float*)out)[idx] = (float)v;
                else        ((double*)out)[idx] = v;
            }
        }
    }
    Tensor* r;
    if (is_f32) r = make_tensor_arena_f32((float*)out, n, out_shape, 1, input->requires_grad || gamma->requires_grad);
    else { r = make_tensor((double*)out, out_shape, 1, input->requires_grad || gamma->requires_grad); free(out); }
    /* No backward tape entry for now — torch/MLX handle it natively */
    return r;
}

/* ================================================================
   Dropout: inverted dropout with mask
   ================================================================ */

/* tensor_dropout: moved to backend_tape/nn/norm/dropout.c (Phase 1c.4). */

/* ================================================================
   Gather / Scatter
   ================================================================ */

/* tensor_gather: moved to backend_tape/linear/index/gather.c (Phase 1b.7). */

/* tensor_scatter_add: moved to backend_tape/linear/index/scatter_add.c (Phase 1b.7.b). */

/* ================================================================
   Sort / Scan
   ================================================================ */

/* tensor_argsort, tensor_cumprod (+ argsort comparators):
 * moved to backend_tape/linear/sort/ (Phase 1b.8). */

/* ================================================================
   Average Pooling
   ================================================================ */

TensorHandle tensor_avg_pool1d(TensorHandle hinput, int kL, int stride) {
    Tensor* input = (Tensor*)hinput;
    int C = input->shape[0], L = input->shape[1];
    int oL = (L - kL) / stride + 1;
    double scale = 1.0 / kL;
    int is_f32 = (input->dtype_tag == DT_F32);
    int out_shape[] = {C, oL};
    int numel = C * oL;
    void* out = is_f32 ? (void*)arena_alloc(numel * sizeof(float))
                       : (void*)calloc(numel, sizeof(double));
    for (int c = 0; c < C; c++)
        for (int ol = 0; ol < oL; ol++) {
            double s = 0;
            for (int kl = 0; kl < kL; kl++) s += tape_load_d(input, c*L + ol*stride + kl);
            double v = s * scale;
            if (is_f32) ((float*)out)[c*oL + ol] = (float)v;
            else        ((double*)out)[c*oL + ol] = v;
        }
    Tensor* r;
    if (is_f32) r = make_tensor_arena_f32((float*)out, numel, out_shape, 2, input->requires_grad);
    else { r = make_tensor((double*)out, out_shape, 2, input->requires_grad); free(out); }
    if (r->requires_grad) {
        TapeEntry* e = tape_append(OP_AVG_POOL1D, r, input, NULL, 0);
        AvgPool1DMeta* meta = arena_alloc(sizeof(AvgPool1DMeta));
        meta->C = C; meta->L = L; meta->kL = kL; meta->stride = stride; meta->oL = oL;
        e->op_meta = meta;
    }
    return r;
}

TensorHandle tensor_avg_pool2d(TensorHandle hinput, int kH, int kW, int strideH, int strideW) {
    Tensor* input = (Tensor*)hinput;
    int C = input->shape[0], H = input->shape[1], W = input->shape[2];
    int oH = (H - kH) / strideH + 1;
    int oW = (W - kW) / strideW + 1;
    double scale = 1.0 / (kH * kW);
    int is_f32 = (input->dtype_tag == DT_F32);
    int out_shape[] = {C, oH, oW};
    int numel = C * oH * oW;
    void* out = is_f32 ? (void*)arena_alloc(numel * sizeof(float))
                       : (void*)calloc(numel, sizeof(double));
    for (int c = 0; c < C; c++)
        for (int oh = 0; oh < oH; oh++)
            for (int ow = 0; ow < oW; ow++) {
                double s = 0;
                for (int kh = 0; kh < kH; kh++)
                    for (int kw = 0; kw < kW; kw++)
                        s += tape_load_d(input, c*H*W + (oh*strideH+kh)*W + ow*strideW+kw);
                double v = s * scale;
                if (is_f32) ((float*)out)[c*oH*oW + oh*oW + ow] = (float)v;
                else        ((double*)out)[c*oH*oW + oh*oW + ow] = v;
            }
    Tensor* r;
    if (is_f32) r = make_tensor_arena_f32((float*)out, numel, out_shape, 3, input->requires_grad);
    else { r = make_tensor((double*)out, out_shape, 3, input->requires_grad); free(out); }
    if (r->requires_grad) {
        TapeEntry* e = tape_append(OP_AVG_POOL2D, r, input, NULL, 0);
        AvgPool2DMeta* meta = arena_alloc(sizeof(AvgPool2DMeta));
        meta->C = C; meta->H = H; meta->W = W;
        meta->kH = kH; meta->kW = kW; meta->strH = strideH; meta->strW = strideW;
        meta->oH = oH; meta->oW = oW;
        e->op_meta = meta;
    }
    return r;
}

/* ================================================================
   Conv1D: input [inC, L], kernel [outC, inC, kL], bias [outC] or NULL
   Output: [outC, oL]
   ================================================================ */

TensorHandle tensor_conv1d(TensorHandle hinput, TensorHandle hkernel,
                           TensorHandle hbias, int pad, int stride) {
    Tensor* input = (Tensor*)hinput;
    Tensor* kernel = (Tensor*)hkernel;
    Tensor* bias = (Tensor*)hbias;
    if (input->dtype_tag != kernel->dtype_tag ||
        (bias && bias->dtype_tag != input->dtype_tag))
        tape_abort_mixed_dtype("tensor_conv1d");
    int inC = input->shape[0], L = input->shape[1];
    int outC = kernel->shape[0], kL = kernel->shape[2];
    int oL = (L + 2*pad - kL) / stride + 1;
    int is_f32 = (input->dtype_tag == DT_F32);
    int numel = outC * oL;
    int out_shape[] = {outC, oL};
    int rg = input->requires_grad || kernel->requires_grad || (bias && bias->requires_grad);
    void* out = is_f32 ? (void*)arena_alloc(numel * sizeof(float))
                       : (void*)calloc(numel, sizeof(double));
    for (int oc = 0; oc < outC; oc++) {
        for (int ol = 0; ol < oL; ol++) {
            double val = bias ? tape_load_d(bias, oc) : 0.0;
            for (int ic = 0; ic < inC; ic++)
                for (int kl = 0; kl < kL; kl++) {
                    int il = ol * stride - pad + kl;
                    if (il >= 0 && il < L)
                        val += tape_load_d(input, ic*L + il) * tape_load_d(kernel, oc*inC*kL + ic*kL + kl);
                }
            if (is_f32) ((float*)out)[oc*oL + ol] = (float)val;
            else        ((double*)out)[oc*oL + ol] = val;
        }
    }
    Tensor* r;
    if (is_f32) r = make_tensor_arena_f32((float*)out, numel, out_shape, 2, rg);
    else { r = make_tensor((double*)out, out_shape, 2, rg); free(out); }
    if (r->requires_grad) {
        TapeEntry* e = tape_append(OP_CONV1D, r, input, kernel, 0);
        Conv1DMeta* meta = arena_alloc(sizeof(Conv1DMeta));
        meta->inC = inC; meta->outC = outC; meta->L = L;
        meta->kL = kL; meta->pad = pad; meta->stride = stride; meta->oL = oL;
        e->op_meta = meta;
        e->inputs = (Tensor**)bias;
    }
    return r;
}

TensorHandle tensor_max_pool1d(TensorHandle hinput, int kL, int stride) {
    Tensor* input = (Tensor*)hinput;
    int C = input->shape[0], L = input->shape[1];
    int oL = (L - kL) / stride + 1;
    int is_f32 = (input->dtype_tag == DT_F32);
    int numel = C * oL;
    int out_shape[] = {C, oL};
    void* out = is_f32 ? (void*)arena_alloc(numel * sizeof(float))
                       : (void*)calloc(numel, sizeof(double));
    int* max_idx = malloc(numel * sizeof(int));
    for (int c = 0; c < C; c++)
        for (int ol = 0; ol < oL; ol++) {
            double best = -1e30;
            int best_idx = 0;
            for (int kl = 0; kl < kL; kl++) {
                int flat = c*L + (ol * stride + kl);
                double v = tape_load_d(input, flat);
                if (v > best) { best = v; best_idx = flat; }
            }
            if (is_f32) ((float*)out)[c*oL + ol] = (float)best;
            else        ((double*)out)[c*oL + ol] = best;
            max_idx[c*oL + ol] = best_idx;
        }
    Tensor* r;
    if (is_f32) r = make_tensor_arena_f32((float*)out, numel, out_shape, 2, input->requires_grad);
    else { r = make_tensor((double*)out, out_shape, 2, input->requires_grad); free(out); }
    if (r->requires_grad) {
        TapeEntry* e = tape_append(OP_MAX_POOL1D, r, input, NULL, 0);
        MaxPool1DMeta* meta = arena_alloc(sizeof(MaxPool1DMeta));
        meta->C = C; meta->L = L; meta->kL = kL; meta->stride = stride; meta->oL = oL;
        meta->max_indices = max_idx;
        e->op_meta = meta;
    } else {
        free(max_idx);
    }
    return r;
}

TensorHandle tensor_create_param_3d(int d0, int d1, int d2, double* data) {
    int numel = d0 * d1 * d2;
    Tensor* t = calloc(1, sizeof(Tensor));
    t->data = malloc(numel * sizeof(double));
    memcpy(t->data, data, numel * sizeof(double));
    free(data);
    t->shape = malloc(3 * sizeof(int));
    t->shape[0] = d0; t->shape[1] = d1; t->shape[2] = d2;
    t->rank = 3; t->numel = numel;
    t->requires_grad = 1;
    t->tape_idx = -1;
    t->persistent = 1;
    tape_append(OP_CONST, t, NULL, NULL, 0);
    return t;
}

/* ================================================================
   Transposed Convolution
   ConvTranspose1D: output[oc][ol] = sum over ic,kl of input[ic][il] * kernel[ic][oc][kl]
   where ol = il*stride - pad + kl
   ================================================================ */

TensorHandle tensor_conv_transpose1d(TensorHandle hinput, TensorHandle hkernel,
                                     TensorHandle hbias, int pad, int stride) {
    Tensor* input = (Tensor*)hinput;
    Tensor* kernel = (Tensor*)hkernel;
    Tensor* bias = (Tensor*)hbias;
    if (input->dtype_tag != kernel->dtype_tag ||
        (bias && bias->dtype_tag != input->dtype_tag))
        tape_abort_mixed_dtype("tensor_conv_transpose1d");
    int inC = input->shape[0], L = input->shape[1];
    int outC = kernel->shape[1], kL = kernel->shape[2];
    int oL = (L - 1) * stride - 2 * pad + kL;
    int is_f32 = (input->dtype_tag == DT_F32);
    int numel = outC * oL;
    int out_shape[] = {outC, oL};
    int rg = input->requires_grad || kernel->requires_grad;
    /* Compute in double for sum stability; narrow to float on store. */
    double* dbl = calloc(numel, sizeof(double));
    if (bias) for (int oc = 0; oc < outC; oc++)
        for (int ol = 0; ol < oL; ol++) dbl[oc*oL + ol] = tape_load_d(bias, oc);
    for (int ic = 0; ic < inC; ic++)
        for (int il = 0; il < L; il++)
            for (int oc = 0; oc < outC; oc++)
                for (int kl = 0; kl < kL; kl++) {
                    int ol = il * stride - pad + kl;
                    if (ol >= 0 && ol < oL)
                        dbl[oc*oL + ol] += tape_load_d(input, ic*L + il)
                                         * tape_load_d(kernel, ic*outC*kL + oc*kL + kl);
                }
    Tensor* r;
    if (is_f32) {
        float* out = arena_alloc(numel * sizeof(float));
        for (int i = 0; i < numel; i++) out[i] = (float)dbl[i];
        free(dbl);
        r = make_tensor_arena_f32(out, numel, out_shape, 2, rg);
    } else {
        r = make_tensor(dbl, out_shape, 2, rg);
        free(dbl);
    }
    return r;
}

TensorHandle tensor_conv_transpose2d(TensorHandle hinput, TensorHandle hkernel,
                                     TensorHandle hbias, int padH, int padW,
                                     int strideH, int strideW) {
    Tensor* input = (Tensor*)hinput;
    Tensor* kernel = (Tensor*)hkernel;
    Tensor* bias = (Tensor*)hbias;
    if (input->dtype_tag != kernel->dtype_tag ||
        (bias && bias->dtype_tag != input->dtype_tag))
        tape_abort_mixed_dtype("tensor_conv_transpose2d");
    int inC = input->shape[0], H = input->shape[1], W = input->shape[2];
    int outC = kernel->shape[1], kH = kernel->shape[2], kW = kernel->shape[3];
    int oH = (H - 1) * strideH - 2 * padH + kH;
    int oW = (W - 1) * strideW - 2 * padW + kW;
    int is_f32 = (input->dtype_tag == DT_F32);
    int numel = outC * oH * oW;
    int out_shape[] = {outC, oH, oW};
    int rg = input->requires_grad || kernel->requires_grad;
    double* dbl = calloc(numel, sizeof(double));
    if (bias) for (int oc = 0; oc < outC; oc++)
        for (int oh = 0; oh < oH; oh++)
            for (int ow = 0; ow < oW; ow++) dbl[oc*oH*oW + oh*oW + ow] = tape_load_d(bias, oc);
    for (int ic = 0; ic < inC; ic++)
        for (int ih = 0; ih < H; ih++)
            for (int iw = 0; iw < W; iw++)
                for (int oc = 0; oc < outC; oc++)
                    for (int kh = 0; kh < kH; kh++)
                        for (int kw = 0; kw < kW; kw++) {
                            int oh = ih*strideH - padH + kh;
                            int ow = iw*strideW - padW + kw;
                            if (oh >= 0 && oh < oH && ow >= 0 && ow < oW)
                                dbl[oc*oH*oW + oh*oW + ow] += tape_load_d(input, ic*H*W + ih*W + iw)
                                    * tape_load_d(kernel, ic*outC*kH*kW + oc*kH*kW + kh*kW + kw);
                        }
    Tensor* r;
    if (is_f32) {
        float* out = arena_alloc(numel * sizeof(float));
        for (int i = 0; i < numel; i++) out[i] = (float)dbl[i];
        free(dbl);
        r = make_tensor_arena_f32(out, numel, out_shape, 3, rg);
    } else {
        r = make_tensor(dbl, out_shape, 3, rg);
        free(dbl);
    }
    return r;
}

/* ================================================================
   Grouped Convolution: splits channels into groups, applies separate convs.
   For tape backend, we just call the ungrouped conv per-group.
   ================================================================ */

TensorHandle tensor_conv1d_grouped(TensorHandle hinput, TensorHandle hkernel,
                                   TensorHandle hbias, int pad, int stride, int groups) {
    if (groups == 1) return tensor_conv1d(hinput, hkernel, hbias, pad, stride);
    /* Decompose into per-group conv1d and concatenate */
    Tensor* input = (Tensor*)hinput;
    Tensor* kernel = (Tensor*)hkernel;
    Tensor* bias = (Tensor*)hbias;
    if (input->dtype_tag != kernel->dtype_tag ||
        (bias && bias->dtype_tag != input->dtype_tag))
        tape_abort_mixed_dtype("tensor_conv1d_grouped");
    int inC = input->shape[0], L = input->shape[1];
    int outC = kernel->shape[0];
    int inC_g = inC / groups;
    int outC_g = outC / groups;
    int kL = kernel->shape[2];
    int oL = (L + 2*pad - kL) / stride + 1;
    int total = outC * oL;
    int is_f32 = (input->dtype_tag == DT_F32);
    int out_shape[] = {outC, oL};
    int rg = input->requires_grad || kernel->requires_grad;
    void* out = is_f32 ? (void*)arena_alloc(total * sizeof(float))
                       : (void*)calloc(total, sizeof(double));
    for (int g = 0; g < groups; g++) {
        for (int oc = 0; oc < outC_g; oc++) {
            int abs_oc = g * outC_g + oc;
            for (int ol = 0; ol < oL; ol++) {
                double val = bias ? tape_load_d(bias, abs_oc) : 0.0;
                for (int ic = 0; ic < inC_g; ic++) {
                    int abs_ic = g * inC_g + ic;
                    for (int kl = 0; kl < kL; kl++) {
                        int il = ol * stride - pad + kl;
                        if (il >= 0 && il < L)
                            val += tape_load_d(input, abs_ic*L + il)
                                 * tape_load_d(kernel, abs_oc*inC_g*kL + ic*kL + kl);
                    }
                }
                if (is_f32) ((float*)out)[abs_oc*oL + ol] = (float)val;
                else        ((double*)out)[abs_oc*oL + ol] = val;
            }
        }
    }
    Tensor* r;
    if (is_f32) r = make_tensor_arena_f32((float*)out, total, out_shape, 2, rg);
    else { r = make_tensor((double*)out, out_shape, 2, rg); free(out); }
    /* No separate backward for grouped — reuse OP_CONV1D with groups=1 per group.
       For simplicity, grouped conv on tape backend doesn't support backward yet.
       Torch and MLX backends use native grouped conv with full autograd. */
    return r;
}

TensorHandle tensor_conv2d_grouped(TensorHandle hinput, TensorHandle hkernel,
                                   TensorHandle hbias, int padH, int padW,
                                   int strideH, int strideW, int groups) {
    if (groups == 1) return tensor_conv2d(hinput, hkernel, hbias, padH, padW, strideH, strideW);
    Tensor* input = (Tensor*)hinput;
    Tensor* kernel = (Tensor*)hkernel;
    Tensor* bias = (Tensor*)hbias;
    if (input->dtype_tag != kernel->dtype_tag ||
        (bias && bias->dtype_tag != input->dtype_tag))
        tape_abort_mixed_dtype("tensor_conv2d_grouped");
    int inC = input->shape[0], H = input->shape[1], W = input->shape[2];
    int outC = kernel->shape[0];
    int inC_g = inC / groups;
    int outC_g = outC / groups;
    int kH = kernel->shape[2], kW = kernel->shape[3];
    int oH = (H + 2*padH - kH) / strideH + 1;
    int oW = (W + 2*padW - kW) / strideW + 1;
    int numel = outC * oH * oW;
    int is_f32 = (input->dtype_tag == DT_F32);
    int out_shape[] = {outC, oH, oW};
    int rg = input->requires_grad || kernel->requires_grad;
    void* out = is_f32 ? (void*)arena_alloc(numel * sizeof(float))
                       : (void*)calloc(numel, sizeof(double));
    for (int g = 0; g < groups; g++) {
        for (int oc = 0; oc < outC_g; oc++) {
            int abs_oc = g * outC_g + oc;
            for (int oh = 0; oh < oH; oh++)
                for (int ow = 0; ow < oW; ow++) {
                    double val = bias ? tape_load_d(bias, abs_oc) : 0.0;
                    for (int ic = 0; ic < inC_g; ic++) {
                        int abs_ic = g * inC_g + ic;
                        for (int kh = 0; kh < kH; kh++)
                            for (int kw = 0; kw < kW; kw++) {
                                int ih = oh*strideH - padH + kh;
                                int iw = ow*strideW - padW + kw;
                                if (ih >= 0 && ih < H && iw >= 0 && iw < W)
                                    val += tape_load_d(input, abs_ic*H*W + ih*W + iw)
                                         * tape_load_d(kernel, abs_oc*inC_g*kH*kW + ic*kH*kW + kh*kW + kw);
                            }
                    }
                    if (is_f32) ((float*)out)[abs_oc*oH*oW + oh*oW + ow] = (float)val;
                    else        ((double*)out)[abs_oc*oH*oW + oh*oW + ow] = val;
                }
        }
    }
    Tensor* r;
    if (is_f32) r = make_tensor_arena_f32((float*)out, numel, out_shape, 3, rg);
    else { r = make_tensor((double*)out, out_shape, 3, rg); free(out); }
    return r;
}

/* ================================================================
   Conv2D: input [inC, H, W], kernel [outC, inC, kH, kW], bias [outC] or NULL
   Output: [outC, oH, oW]
   ================================================================ */

TensorHandle tensor_conv2d(TensorHandle hinput, TensorHandle hkernel,
                           TensorHandle hbias, int padH, int padW,
                           int strideH, int strideW) {
    Tensor* input = (Tensor*)hinput;
    Tensor* kernel = (Tensor*)hkernel;
    Tensor* bias = (Tensor*)hbias;
    if (input->dtype_tag != kernel->dtype_tag ||
        (bias && bias->dtype_tag != input->dtype_tag))
        tape_abort_mixed_dtype("tensor_conv2d");
    int inC = input->shape[0], H = input->shape[1], W = input->shape[2];
    int outC = kernel->shape[0], kH = kernel->shape[2], kW = kernel->shape[3];
    int oH = (H + 2*padH - kH) / strideH + 1;
    int oW = (W + 2*padW - kW) / strideW + 1;
    int out_numel = outC * oH * oW;
    int is_f32 = (input->dtype_tag == DT_F32);
    int out_shape[] = {outC, oH, oW};
    int rg = input->requires_grad || kernel->requires_grad || (bias && bias->requires_grad);

    void* out = is_f32 ? (void*)arena_alloc(out_numel * sizeof(float))
                       : (void*)calloc(out_numel, sizeof(double));
    for (int oc = 0; oc < outC; oc++) {
        for (int oh = 0; oh < oH; oh++) {
            for (int ow = 0; ow < oW; ow++) {
                double val = bias ? tape_load_d(bias, oc) : 0.0;
                for (int ic = 0; ic < inC; ic++) {
                    for (int kh = 0; kh < kH; kh++) {
                        for (int kw = 0; kw < kW; kw++) {
                            int ih = oh * strideH - padH + kh;
                            int iw = ow * strideW - padW + kw;
                            if (ih >= 0 && ih < H && iw >= 0 && iw < W) {
                                val += tape_load_d(input, ic*H*W + ih*W + iw)
                                     * tape_load_d(kernel, oc*inC*kH*kW + ic*kH*kW + kh*kW + kw);
                            }
                        }
                    }
                }
                if (is_f32) ((float*)out)[oc*oH*oW + oh*oW + ow] = (float)val;
                else        ((double*)out)[oc*oH*oW + oh*oW + ow] = val;
            }
        }
    }

    Tensor* r;
    if (is_f32) r = make_tensor_arena_f32((float*)out, out_numel, out_shape, 3, rg);
    else { r = make_tensor((double*)out, out_shape, 3, rg); free(out); }

    if (r->requires_grad) {
        TapeEntry* e = tape_append(OP_CONV2D, r, input, kernel, 0);
        Conv2DMeta* meta = arena_alloc(sizeof(Conv2DMeta));
        meta->inC = inC; meta->outC = outC;
        meta->H = H; meta->W = W;
        meta->kH = kH; meta->kW = kW;
        meta->padH = padH; meta->padW = padW;
        meta->strH = strideH; meta->strW = strideW;
        meta->oH = oH; meta->oW = oW;
        e->op_meta = meta;
        /* Store bias pointer in scalar_arg slot (cast) for backward */
        e->inputs = (Tensor**)bias;  /* reuse inputs field for bias ptr */
    }
    return r;
}

/* ================================================================
   Batched Conv2D: input [B, inC, H, W] x kernel [outC, inC, kH, kW]
                   + bias [outC] -> [B, outC, oH, oW]

   Forward uses the standard im2col + cblas_dgemm decomposition:
       X_col [M, K] where M = B*oH*oW, K = inC*kH*kW
       Y_unf [M, outC] = X_col @ W^T   (single dgemm)
       out   [B, outC, oH, oW] = permute(Y_unf, (0,2,1) on (B*oHoW, outC))
                                + bias broadcast.
   This is what PyTorch / cuDNN do at the unfused-conv path; the dgemm
   replaces an O(B·outC·inC·kH·kW·oH·oW) hand-rolled triple loop with
   Apple Accelerate's blocked sgemm.
   ================================================================ */

/* im2col: build X_col [M, K] where M = B*oH*oW, K = inC*kH*kW.
   Each row is one (batch, out-position)'s unfolded inC*kH*kW window. */
static void conv2d_im2col(const double* input, int B, int inC, int H, int W,
                          int kH, int kW, int padH, int padW,
                          int strideH, int strideW, int oH, int oW,
                          double* X_col) {
    int K = inC * kH * kW;
    int M = B * oH * oW;
    memset(X_col, 0, (size_t)M * K * sizeof(double));
    for (int b = 0; b < B; b++) {
        const double* inp_b = input + (size_t)b * inC * H * W;
        for (int oh = 0; oh < oH; oh++) {
            for (int ow = 0; ow < oW; ow++) {
                double* row = X_col + ((size_t)b * oH * oW + oh * oW + ow) * K;
                for (int ic = 0; ic < inC; ic++) {
                    for (int kh = 0; kh < kH; kh++) {
                        int ih = oh * strideH - padH + kh;
                        if (ih < 0 || ih >= H) continue;
                        for (int kw = 0; kw < kW; kw++) {
                            int iw = ow * strideW - padW + kw;
                            if (iw < 0 || iw >= W) continue;
                            row[ic * kH * kW + kh * kW + kw] =
                                inp_b[ic * H * W + ih * W + iw];
                        }
                    }
                }
            }
        }
    }
}

/* col2im (gradient accumulating version): scatter dX_col [M, K] back into
   dInput [B, inC, H, W]. Padding cells are dropped. */
static void conv2d_col2im_accumulate(const double* dX_col, int B, int inC,
                                      int H, int W, int kH, int kW,
                                      int padH, int padW, int strideH,
                                      int strideW, int oH, int oW,
                                      double* dInput) {
    int K = inC * kH * kW;
    for (int b = 0; b < B; b++) {
        double* din_b = dInput + (size_t)b * inC * H * W;
        for (int oh = 0; oh < oH; oh++) {
            for (int ow = 0; ow < oW; ow++) {
                const double* row = dX_col + ((size_t)b * oH * oW + oh * oW + ow) * K;
                for (int ic = 0; ic < inC; ic++) {
                    for (int kh = 0; kh < kH; kh++) {
                        int ih = oh * strideH - padH + kh;
                        if (ih < 0 || ih >= H) continue;
                        for (int kw = 0; kw < kW; kw++) {
                            int iw = ow * strideW - padW + kw;
                            if (iw < 0 || iw >= W) continue;
                            din_b[ic * H * W + ih * W + iw] +=
                                row[ic * kH * kW + kh * kW + kw];
                        }
                    }
                }
            }
        }
    }
}

TensorHandle tensor_conv2d_batched(TensorHandle hinput, TensorHandle hkernel,
                                    TensorHandle hbias, int padH, int padW,
                                    int strideH, int strideW) {
    Tensor* input = (Tensor*)hinput;
    Tensor* kernel = (Tensor*)hkernel;
    Tensor* bias = (Tensor*)hbias;
    if (input->dtype_tag != kernel->dtype_tag ||
        (bias && bias->dtype_tag != input->dtype_tag))
        tape_abort_mixed_dtype("tensor_conv2d_batched");

    int B = input->shape[0], inC = input->shape[1];
    int H = input->shape[2], W = input->shape[3];
    int outC = kernel->shape[0], kH = kernel->shape[2], kW = kernel->shape[3];
    int oH = (H + 2*padH - kH) / strideH + 1;
    int oW = (W + 2*padW - kW) / strideW + 1;
    int out_numel = B * outC * oH * oW;
    int is_f32 = (input->dtype_tag == DT_F32);
    int out_shape[] = {B, outC, oH, oW};
    int rg = input->requires_grad || kernel->requires_grad || (bias && bias->requires_grad);

    void* out_buf;
    if (is_f32) {
        /* F32 path: direct 6-loop computation; im2col + cblas_sgemm is
           future work. Loops do double arithmetic and narrow at the
           store (numerical-stability-friendly). */
        float* out = arena_alloc(out_numel * sizeof(float));
        for (int b = 0; b < B; b++)
            for (int oc = 0; oc < outC; oc++) {
                double b_val = bias ? tape_load_d(bias, oc) : 0.0;
                for (int oh = 0; oh < oH; oh++)
                    for (int ow = 0; ow < oW; ow++) {
                        double val = b_val;
                        for (int ic = 0; ic < inC; ic++)
                            for (int kh = 0; kh < kH; kh++)
                                for (int kw = 0; kw < kW; kw++) {
                                    int ih = oh*strideH - padH + kh;
                                    int iw = ow*strideW - padW + kw;
                                    if (ih >= 0 && ih < H && iw >= 0 && iw < W)
                                        val += tape_load_d(input, b*inC*H*W + ic*H*W + ih*W + iw)
                                             * tape_load_d(kernel, oc*inC*kH*kW + ic*kH*kW + kh*kW + kw);
                                }
                        out[((size_t)b * outC + oc) * oH * oW + oh*oW + ow] = (float)val;
                    }
            }
        out_buf = out;
    } else {
        int K = inC * kH * kW;
        int M = B * oH * oW;
        double* X_col = (double*)calloc((size_t)M * K, sizeof(double));
        conv2d_im2col(input->data, B, inC, H, W, kH, kW, padH, padW,
                      strideH, strideW, oH, oW, X_col);
        double* Y_unf = calloc((size_t)M * outC, sizeof(double));
#ifdef __APPLE__
        cblas_dgemm(CblasRowMajor, CblasNoTrans, CblasTrans,
                    M, outC, K, 1.0,
                    X_col, K,
                    kernel->data, K,
                    0.0, Y_unf, outC);
#else
        for (int m = 0; m < M; m++)
            for (int oc = 0; oc < outC; oc++) {
                double s = 0;
                for (int k = 0; k < K; k++)
                    s += X_col[m*K + k] * ((double*)kernel->data)[oc*K + k];
                Y_unf[m*outC + oc] = s;
            }
#endif
        double* out = calloc(out_numel, sizeof(double));
        for (int b = 0; b < B; b++) {
            for (int oc = 0; oc < outC; oc++) {
                double b_val = bias ? ((double*)bias->data)[oc] : 0.0;
                double* out_chan = out + ((size_t)b * outC + oc) * oH * oW;
                for (int oh = 0; oh < oH; oh++) {
                    for (int ow = 0; ow < oW; ow++) {
                        int row = b * oH * oW + oh * oW + ow;
                        out_chan[oh*oW + ow] = Y_unf[row * outC + oc] + b_val;
                    }
                }
            }
        }
        free(Y_unf);
        free(X_col);
        out_buf = out;
    }

    Tensor* r;
    if (is_f32) r = make_tensor_arena_f32((float*)out_buf, out_numel, out_shape, 4, rg);
    else { r = make_tensor((double*)out_buf, out_shape, 4, rg); free(out_buf); }

    if (r->requires_grad) {
        TapeEntry* e = tape_append(OP_CONV2D_BATCHED, r, input, kernel, 0);
        Conv2DBatchedMeta* meta = arena_alloc(sizeof(Conv2DBatchedMeta));
        meta->B = B; meta->inC = inC; meta->outC = outC;
        meta->H = H; meta->W = W;
        meta->kH = kH; meta->kW = kW;
        meta->padH = padH; meta->padW = padW;
        meta->strH = strideH; meta->strW = strideW;
        meta->oH = oH; meta->oW = oW;
        e->op_meta = meta;
        /* Store bias pointer in scalar_arg slot (cast) for backward */
        e->inputs = (Tensor**)bias;  /* reuse inputs field for bias ptr */
    }
    return r;
}

/* ================================================================
   MaxPool2D: input [C, H, W] -> [C, oH, oW]
   ================================================================ */

TensorHandle tensor_max_pool2d(TensorHandle hinput, int kH, int kW,
                               int strideH, int strideW) {
    Tensor* input = (Tensor*)hinput;
    int C = input->shape[0], H = input->shape[1], W = input->shape[2];
    int oH = (H - kH) / strideH + 1;
    int oW = (W - kW) / strideW + 1;
    int out_numel = C * oH * oW;
    int is_f32 = (input->dtype_tag == DT_F32);
    int out_shape[] = {C, oH, oW};

    void* out_buf = is_f32 ? (void*)arena_alloc(out_numel * sizeof(float))
                           : (void*)calloc(out_numel, sizeof(double));
    int* max_idx = malloc(out_numel * sizeof(int));

    for (int c = 0; c < C; c++) {
        for (int oh = 0; oh < oH; oh++) {
            for (int ow = 0; ow < oW; ow++) {
                double best = -1e30;
                int best_idx = 0;
                for (int kh = 0; kh < kH; kh++) {
                    for (int kw = 0; kw < kW; kw++) {
                        int ih = oh * strideH + kh;
                        int iw = ow * strideW + kw;
                        int flat = c*H*W + ih*W + iw;
                        double v = tape_load_d(input, flat);
                        if (v > best) { best = v; best_idx = flat; }
                    }
                }
                int out_idx = c*oH*oW + oh*oW + ow;
                if (is_f32) ((float*)out_buf)[out_idx] = (float)best;
                else        ((double*)out_buf)[out_idx] = best;
                max_idx[out_idx] = best_idx;
            }
        }
    }

    Tensor* r;
    if (is_f32) r = make_tensor_arena_f32((float*)out_buf, out_numel, out_shape, 3, input->requires_grad);
    else { r = make_tensor((double*)out_buf, out_shape, 3, input->requires_grad); free(out_buf); }

    if (r->requires_grad) {
        TapeEntry* e = tape_append(OP_MAX_POOL2D, r, input, NULL, 0);
        MaxPool2DMeta* meta = arena_alloc(sizeof(MaxPool2DMeta));
        meta->C = C; meta->H = H; meta->W = W;
        meta->kH = kH; meta->kW = kW;
        meta->strH = strideH; meta->strW = strideW;
        meta->oH = oH; meta->oW = oW;
        meta->max_indices = max_idx;  /* heap-allocated, freed in tape_reset */
        e->op_meta = meta;
    } else {
        free(max_idx);
    }
    return r;
}

/* ================================================================
   Batched MaxPool2D: input [B, C, H, W] -> [B, C, oH, oW]
   ================================================================ */

TensorHandle tensor_max_pool2d_batched(TensorHandle hinput, int kH, int kW,
                                        int strideH, int strideW) {
    Tensor* input = (Tensor*)hinput;
    int B = input->shape[0], C = input->shape[1];
    int H = input->shape[2], W = input->shape[3];
    int oH = (H - kH) / strideH + 1;
    int oW = (W - kW) / strideW + 1;
    int out_per_sample = C * oH * oW;
    int out_numel = B * out_per_sample;
    int is_f32 = (input->dtype_tag == DT_F32);
    int out_shape[] = {B, C, oH, oW};

    void* out_buf = is_f32 ? (void*)arena_alloc(out_numel * sizeof(float))
                           : (void*)calloc(out_numel, sizeof(double));
    int* max_idx = malloc(out_numel * sizeof(int));

    for (int b = 0; b < B; b++) {
        int base = b * C * H * W;
        int out_base = b * out_per_sample;
        for (int c = 0; c < C; c++) {
            for (int oh = 0; oh < oH; oh++) {
                for (int ow = 0; ow < oW; ow++) {
                    double best = -1e30;
                    int best_idx = 0;
                    for (int kh = 0; kh < kH; kh++) {
                        for (int kw = 0; kw < kW; kw++) {
                            int ih = oh * strideH + kh;
                            int iw = ow * strideW + kw;
                            int flat = c*H*W + ih*W + iw;
                            double v = tape_load_d(input, base + flat);
                            if (v > best) { best = v; best_idx = base + flat; }
                        }
                    }
                    int out_idx = c*oH*oW + oh*oW + ow;
                    if (is_f32) ((float*)out_buf)[out_base + out_idx] = (float)best;
                    else        ((double*)out_buf)[out_base + out_idx] = best;
                    max_idx[out_base + out_idx] = best_idx;
                }
            }
        }
    }

    Tensor* r;
    if (is_f32) r = make_tensor_arena_f32((float*)out_buf, out_numel, out_shape, 4, input->requires_grad);
    else { r = make_tensor((double*)out_buf, out_shape, 4, input->requires_grad); free(out_buf); }

    if (r->requires_grad) {
        TapeEntry* e = tape_append(OP_MAX_POOL2D_BATCHED, r, input, NULL, 0);
        MaxPool2DBatchedMeta* meta = arena_alloc(sizeof(MaxPool2DBatchedMeta));
        meta->B = B; meta->C = C; meta->H = H; meta->W = W;
        meta->kH = kH; meta->kW = kW;
        meta->strH = strideH; meta->strW = strideW;
        meta->oH = oH; meta->oW = oW;
        meta->max_indices = max_idx;
        e->op_meta = meta;
    } else {
        free(max_idx);
    }
    return r;
}

/* ================================================================
   Shape manipulation
   ================================================================ */

/* tensor_reshape: moved to backend_tape/linear/shape/reshape.c (Phase 1b.1.b). */

/* tensor_unsqueeze, tensor_squeeze: moved to backend_tape/linear/shape/ (Phase 1b.1.d). */

/* tensor_select: moved to backend_tape/linear/shape/select.c (Phase 1b.1). */

/* tensor_stack, tensor_cat: moved to backend_tape/linear/concat/ (Phase 1b.2.a). */

/* tensor_cat2: moved to backend_tape/linear/concat/cat2.c (Phase 1b.2.b). */

/* tensor_narrow: moved to backend_tape/linear/shape/narrow.c (Phase 1b.1.c). */

/* ================================================================
   Autograd — backward pass
   ================================================================ */

/* Forward declarations for profiling */
#include <sys/time.h>
double _wall_ms(void) {  /* non-static so tape.c can extern it */
    struct timeval tv;
    gettimeofday(&tv, NULL);
    return tv.tv_sec * 1000.0 + tv.tv_usec / 1000.0;
}
static double prof_forward_ms = 0, prof_backward_ms = 0, prof_optimizer_ms = 0;
static int prof_forward_ops = 0, prof_backward_ops = 0, prof_epochs = 0;
static double prof_epoch_start = 0; /* set by backend_epoch_begin() */
static int prof_backward_processed = 0, prof_backward_skipped = 0;
static double prof_backward_per_op[OP_COUNT] = {0};
static int prof_backward_count_per_op[OP_COUNT] = {0};
/* Per-op forward timing. Non-static so the forward declarations near
   tape_append (which is defined earlier in the file) can refer to them. */
double prof_forward_per_op[OP_COUNT] = {0};
int prof_forward_count_per_op[OP_COUNT] = {0};
/* Direct kernel-only timer — only OP_ADD/SUB/MUL/DIV's vDSP path
   populates it today, for diagnosing the "ADD bucket dominates
   forward" attribution. Compare prof_kernel_per_op[OP_ADD] vs
   prof_forward_per_op[OP_ADD] to see how much of the bucket is
   actual kernel time versus inter-op leakage. */
double prof_kernel_per_op[OP_COUNT] = {0};
int prof_kernel_count_per_op[OP_COUNT] = {0};
/* Path-classification counters for binop_elementwise — [fast vDSP,
   scalar bcast, general bcast]. Diagnostic for the
   "ADD bucket dominates forward" investigation. */
int prof_binop_path_count[3] = {0};
double prof_binop_general_ms = 0;
/* Full-function (entry-to-exit) timer for binop_elementwise. Compare
   with prof_forward_per_op to see how much of the attributed bucket
   is actually inside our function vs leaked from somewhere else. */
double prof_binop_inside_ms[OP_COUNT] = {0};
int prof_binop_inside_count[OP_COUNT] = {0};
/* Wall-time of the previous tape_append (or epoch_begin). The delta
   from that moment to the next tape_append is attributed to the op
   being recorded now — i.e. its compute + tape-append cost. Set to 0
   to disable accumulation (e.g. during backward). */
double prof_op_t_prev = 0;



void tensor_backward(TensorHandle h) {
    double t0 = _wall_ms();
    /* Attribute time since epoch_begin to forward */
    if (prof_epoch_start > 0) {
        prof_forward_ms += t0 - prof_epoch_start;
        prof_epoch_start = 0;
    }
    /* Stop per-op forward accumulation; the next epoch_begin will rearm. */
    prof_op_t_prev = 0;
    Tensor* loss = (Tensor*)h;
    if (loss->tape_idx < 0) return;

    /* Initialize loss gradient to 1.0 */
    ensure_grad(loss);
    ((double*)loss->grad)[0] = 1.0;

    int processed = 0, skipped = 0;

    /* Walk tape in reverse via chunk-array — same semantics as the old
       `for (int i = loss->tape_idx; i >= 0; i--) { TapeEntry* e = &tape[i]; }`
       but indexes the chunked tape directly so the cost stays O(N) total. */
    int _num_chunks_b = 0;
    for (TypedArenaChunk* _c = tape_arena.head; _c; _c = _c->next) _num_chunks_b++;
    TypedArenaChunk** _chunks_b = malloc(_num_chunks_b * sizeof(TypedArenaChunk*));
    { int _ci = 0; for (TypedArenaChunk* _c = tape_arena.head; _c; _c = _c->next) _chunks_b[_ci++] = _c; }
    int _start_cidx = loss->tape_idx / TAPE_CHUNK_SIZE;
    int _start_intra = loss->tape_idx % TAPE_CHUNK_SIZE;
    for (int _cidx = _start_cidx; _cidx >= 0; _cidx--) {
        TapeEntry* _entries_b = (TapeEntry*)_chunks_b[_cidx]->data;
        int _last_intra = (_cidx == _start_cidx) ? _start_intra : TAPE_CHUNK_SIZE - 1;
        for (int _j = _last_intra; _j >= 0; _j--) {
        TapeEntry* e = &_entries_b[_j];
        Tensor* r = e->result;
        if (!r->grad) { skipped++; continue; }
        processed++;
        double t_op = _wall_ms();

        Tensor* a = e->arg1;
        Tensor* b = e->arg2;

        /* Phase 1a.2+: try the per-op dispatch table first. Migrated
           ops register their backward via TAPE_REGISTER_OP at file scope;
           unmigrated ones fall through to the legacy switch below.
           The switch shrinks each commit as ops move to backend_tape/<slice>/. */
        TapeBackwardFn _fn = tape_dispatch_get(e->op);
        if (_fn) { _fn(e); goto after_backward; }

        switch (e->op) {
        case OP_CONST: break; /* leaf — grad already accumulated */

        /* OP_ADD, OP_SUB: moved to backend_tape/core/elementwise/ (Phase 1a.2/3).
           Migrated path: dispatch table at top of this loop. */

        /* Elementwise-binop backward (OP_MUL/DIV/POW) — handle three
           cases per side: same-shape (fast loop), scalar (sum-reduce), and
           general numpy-style broadcast (walk r-positions with broadcast
           strides, accumulating into the operand's flat index). */
        /* OP_MUL: moved to backend_tape/core/elementwise/mul.c (Phase 1a.4). */

        /* OP_DIV: moved to backend_tape/core/elementwise/div.c (Phase 1a.5). */

        /* OP_NEG/ABS/EXP/LOG/SQRT: moved to backend_tape/core/elementwise/ (Phase 1a.6). */

        /* OP_POW: moved to backend_tape/core/elementwise/pow.c (Phase 1a.7). */

        /* OP_SIGMOID/TANH/SOFTPLUS: moved to backend_tape/core/elementwise/ (Phase 1a.8). */

        /* OP_TILE_2D: moved to backend_tape/linear/linalg/tile_2d.c (Phase 1b.6). */

        case OP_GRU_CELL: {
            /* nn.GRU backward. arg1 = ih, arg2 = hh, prev in meta.
                 z = sigmoid(ih_z + hh_z),  r = sigmoid(ih_r + hh_r)
                 n = tanh(ih_n + r * hh_n)
                 h' = (1-z) * n + z * prev
               Gradient flows:
                 d_z       = dh' * (prev - n)
                 d_z_raw   = d_z * z * (1-z)        (goes to ih_z and hh_z)
                 d_n       = dh' * (1-z)
                 d_n_pre   = d_n * (1-n*n)          (where n_pre = ih_n + r*hh_n)
                 d_ih_n    = d_n_pre
                 d_(r*hh_n)= d_n_pre   →  d_r = d_n_pre * hh_n
                                          d_hh_n = d_n_pre * r
                 d_r_raw   = d_r * r * (1-r)        (goes to ih_r and hh_r)
                 d_prev    = dh' * z                                              */
            GruCellMeta* meta = (GruCellMeta*)e->op_meta;
            int oo = meta->o;
            Tensor* ih = a;
            Tensor* hh = b;
            Tensor* prev = meta->prev;
            ensure_grad(r);
            for (int i = 0; i < oo; i++) {
                double dh = ((double*)r->grad)[i];
                double zv = meta->zG[i];
                double rv = meta->rG[i];
                double nv = meta->nG[i];
                double hh_n_i = tape_load_d(hh, 2*oo + i);

                double d_z_raw = dh * (tape_load_d(prev, i) - nv) * zv * (1.0 - zv);
                double d_n_pre = dh * (1.0 - zv) * (1.0 - nv * nv);
                double d_r     = d_n_pre * hh_n_i;
                double d_r_raw = d_r * rv * (1.0 - rv);
                double d_hh_n  = d_n_pre * rv;

                if (ih && ih->requires_grad) {
                    ensure_grad(ih);
                    ((double*)ih->grad)[i]        += d_z_raw;
                    ((double*)ih->grad)[oo + i]   += d_r_raw;
                    ((double*)ih->grad)[2*oo + i] += d_n_pre;   /* d_ih_n = d_n_pre */
                }
                if (hh && hh->requires_grad) {
                    ensure_grad(hh);
                    ((double*)hh->grad)[i]        += d_z_raw;
                    ((double*)hh->grad)[oo + i]   += d_r_raw;
                    ((double*)hh->grad)[2*oo + i] += d_hh_n;
                }
                if (prev && prev->requires_grad) {
                    ensure_grad(prev);
                    ((double*)prev->grad)[i] += dh * zv;
                }
            }
            break;
        }

        /* OP_GELU: moved to backend_tape/nn/activation/gelu.c (Phase 1c.2). */

        /* OP_ADD_SCALAR/MUL_SCALAR/CLAMP_MIN: moved to backend_tape/core/scalar/ (Phase 1a.9). */

        /* OP_SELECT: moved to backend_tape/linear/shape/select.c (Phase 1b.1). */

        /* OP_RESHAPE: moved to backend_tape/linear/shape/reshape.c (Phase 1b.1.b). */

        case OP_STACK:
            /* Distribute gradient from stacked tensor back to constituent scalars */
            if (e->inputs) {
                for (int j = 0; j < e->input_count; j++) {
                    Tensor* inp = e->inputs[j];
                    if (inp->requires_grad) {
                        ensure_grad(inp);
                        ensure_grad(r);
                        ((double*)inp->grad)[0] += ((double*)r->grad)[j];
                    }
                }
            }
            break;

        /* OP_SUM, OP_MEAN: moved to backend_tape/linear/reduction/ (Phase 1b.3). */

        /* OP_DOT: moved to backend_tape/linear/linalg/dot.c (Phase 1b.4). */

        /* OP_VECMAT: moved to backend_tape/linear/linalg/matmul.c (Phase 1b.5). */

        /* OP_CAT: moved to backend_tape/linear/concat/cat2.c (Phase 1b.2.b). */

        /* OP_NARROW: moved to backend_tape/linear/shape/narrow.c (Phase 1b.1.c). */

        /* OP_MM: moved to backend_tape/linear/linalg/mm.c (Phase 1b.5). */

        /* OP_BMM, OP_BMM_3X3: moved to backend_tape/linear/linalg/ (Phase 1b.6). */

        /* OP_SOFTMAX_3D: moved to backend_tape/nn/softmax/softmax_3d.c (Phase 1c.1). */

        /* OP_TRANSPOSE_LAST2, OP_TRANSPOSE_2D: moved to backend_tape/linear/linalg/ (Phase 1b.6). */

        /* OP_SOFTMAX_2D, OP_LOG_SOFTMAX_2D: moved to backend_tape/nn/softmax/ (Phase 1c.1). */

        /* OP_MASKED_FILL: moved to backend_tape/nn/mask/masked_fill.c (Phase 1c.3). */

/* OP_LAYER_NORM_2D: moved to backend_tape/nn/norm/layer_norm_2d.c (Phase 1c.4). */

        /* OP_MV: moved to backend_tape/linear/linalg/mv.c (Phase 1b.4.b). */

        /* OP_CONCAT_2D_AXIS1: moved to backend_tape/linear/concat/concat_2d_axis1.c (Phase 1b.2.c). */

        /* OP_LINEAR_2D: moved to backend_tape/linear/linalg/linear_2d.c (Phase 1b.5). */

        /* OP_LINEAR: moved to backend_tape/linear/linalg/linear.c (Phase 1b.5). */

        /* OP_OUTER: moved to backend_tape/linear/linalg/outer.c (Phase 1b.4). */

        /* OP_SOFTMAX, OP_LOG_SOFTMAX: moved to backend_tape/nn/softmax/ (Phase 1c.1). */

        /* OP_BCE_WITH_LOGITS: moved to backend_tape/nn/loss/bce_with_logits.c (Phase 1c.6). */

        case OP_LSTM_GATES: {
            /* LSTM gates backward: propagate from hidden output to combined + prev_cell.
               hidden[j] = oG[j] * tanh(cell[j])
               cell[j] = fG[j] * prevCell[j] + iG[j] * gG[j]
               The cell output gradient needs to be collected separately.
               For now, this backward only handles the hidden output's gradient. */
            LstmGatesMeta* lm = (LstmGatesMeta*)e->op_meta;
            if (lm && a) {
                int o_lstm = lm->o;
                ensure_grad(a);  /* combined [4*o] */
                ensure_grad(r);  /* hidden [o] */
                if (b) ensure_grad(b);  /* prev_cell [o] */

                for (int j = 0; j < o_lstm; j++) {
                    double d_h = ((double*)r->grad)[j];
                    double tanhC = tanh(lm->new_cell[j]);

                    /* d_oGate = d_h * tanh(cell) */
                    double d_oG = d_h * tanhC;
                    /* d_cell from hidden path */
                    double d_cell = d_h * lm->oG[j] * (1.0 - tanhC * tanhC);

                    /* d_fGate = d_cell * prevCell */
                    double d_fG = d_cell * (b ? tape_load_d(b, j) : 0);
                    /* d_iGate = d_cell * gG */
                    double d_iG = d_cell * lm->gG[j];
                    /* d_gGate = d_cell * iG */
                    double d_gG = d_cell * lm->iG[j];
                    /* d_prevCell = d_cell * fG */
                    if (b) ((double*)b->grad)[j] += d_cell * lm->fG[j];

                    /* Activation derivatives → combined gradient */
                    ((double*)a->grad)[j]          += d_iG * lm->iG[j] * (1.0 - lm->iG[j]);  /* sigmoid' */
                    ((double*)a->grad)[o_lstm + j]  += d_fG * lm->fG[j] * (1.0 - lm->fG[j]);
                    ((double*)a->grad)[2*o_lstm + j] += d_gG * (1.0 - lm->gG[j] * lm->gG[j]);  /* tanh' */
                    ((double*)a->grad)[3*o_lstm + j] += d_oG * lm->oG[j] * (1.0 - lm->oG[j]);
                }
            }
            break;
        }

        case OP_LSTM_GATES_CELL: {
            /* Cell output backward: cell[j] = fG[j]*prevCell[j] + iG[j]*gG[j]
               d_cell comes directly from downstream (FC layers reading cell state,
               and next timestep's LSTM using it as prev_cell). */
            LstmGatesMeta* lm = (LstmGatesMeta*)e->op_meta;
            if (lm && a) {
                int o_lstm = lm->o;
                ensure_grad(a);  /* combined [4*o] */
                ensure_grad(r);  /* cell [o] */
                if (b) ensure_grad(b);  /* prev_cell [o] */

                for (int j = 0; j < o_lstm; j++) {
                    double d_cell = ((double*)r->grad)[j];

                    /* d_fGate = d_cell * prevCell */
                    double d_fG = d_cell * (b ? tape_load_d(b, j) : 0);
                    /* d_iGate = d_cell * gG */
                    double d_iG = d_cell * lm->gG[j];
                    /* d_gGate = d_cell * iG */
                    double d_gG = d_cell * lm->iG[j];
                    /* d_prevCell = d_cell * fG */
                    if (b) ((double*)b->grad)[j] += d_cell * lm->fG[j];

                    /* Activation derivatives → combined gradient (additive with OP_LSTM_GATES) */
                    ((double*)a->grad)[j]            += d_iG * lm->iG[j] * (1.0 - lm->iG[j]);
                    ((double*)a->grad)[o_lstm + j]    += d_fG * lm->fG[j] * (1.0 - lm->fG[j]);
                    ((double*)a->grad)[2*o_lstm + j]  += d_gG * (1.0 - lm->gG[j] * lm->gG[j]);
                    /* No output gate gradient from cell path (oG only affects hidden) */
                }
            }
            break;
        }

/* OP_COSINE_SIM: moved to backend_tape/nn/attention/cosine_similarity.c (Phase 1c.5). */

        case OP_CONV1D_CIRC: {
            /* Circular convolution backward — tape_load_d covers F64 + F32 reads. */
            int n_cv = a->numel, k_cv = b->numel, pad_cv = k_cv / 2;
            ensure_grad(r);
            if (a->requires_grad) {
                ensure_grad(a);
                for (int ii = 0; ii < n_cv; ii++) {
                    for (int j = 0; j < k_cv; j++) {
                        int idx = (ii - pad_cv + j + n_cv) % n_cv;
                        ((double*)a->grad)[idx] += ((double*)r->grad)[ii] * tape_load_d(b, k_cv - 1 - j);
                    }
                }
            }
            if (b->requires_grad) {
                ensure_grad(b);
                for (int ii = 0; ii < n_cv; ii++) {
                    for (int j = 0; j < k_cv; j++) {
                        int idx = (ii - pad_cv + j + n_cv) % n_cv;
                        ((double*)b->grad)[k_cv - 1 - j] += ((double*)r->grad)[ii] * tape_load_d(a, idx);
                    }
                }
            }
            break;
        }

/* OP_EMBEDDING: moved to backend_tape/nn/attention/embedding.c (Phase 1c.5). */

/* OP_BATCH_NORM: moved to backend_tape/nn/norm/batch_norm.c (Phase 1c.4). */

/* OP_DROPOUT: moved to backend_tape/nn/norm/dropout.c (Phase 1c.4). */

        case OP_AVG_POOL1D: {
            AvgPool1DMeta* meta = (AvgPool1DMeta*)e->op_meta;
            ensure_grad(r);
            if (a && a->requires_grad) {
                ensure_grad(a);
                double scale = 1.0 / meta->kL;
                for (int c = 0; c < meta->C; c++)
                    for (int ol = 0; ol < meta->oL; ol++)
                        for (int kl = 0; kl < meta->kL; kl++)
                            ((double*)a->grad)[c*meta->L + ol*meta->stride + kl] += ((double*)r->grad)[c*meta->oL + ol] * scale;
            }
            break;
        }

        case OP_AVG_POOL2D: {
            AvgPool2DMeta* meta = (AvgPool2DMeta*)e->op_meta;
            ensure_grad(r);
            if (a && a->requires_grad) {
                ensure_grad(a);
                double scale = 1.0 / (meta->kH * meta->kW);
                for (int c = 0; c < meta->C; c++)
                    for (int oh = 0; oh < meta->oH; oh++)
                        for (int ow = 0; ow < meta->oW; ow++)
                            for (int kh = 0; kh < meta->kH; kh++)
                                for (int kw = 0; kw < meta->kW; kw++)
                                    ((double*)a->grad)[c*meta->H*meta->W + (oh*meta->strH+kh)*meta->W + ow*meta->strW+kw]
                                        += ((double*)r->grad)[c*meta->oH*meta->oW + oh*meta->oW + ow] * scale;
            }
            break;
        }

        case OP_CONV1D: {
            Conv1DMeta* meta = (Conv1DMeta*)e->op_meta;
            int inC = meta->inC, outC = meta->outC, LL = meta->L;
            int kL = meta->kL, pad = meta->pad, str = meta->stride, oL = meta->oL;
            ensure_grad(r);
            if (a && a->requires_grad) {
                ensure_grad(a);
                for (int oc = 0; oc < outC; oc++)
                    for (int ol = 0; ol < oL; ol++) {
                        double dout = ((double*)r->grad)[oc*oL + ol];
                        for (int ic = 0; ic < inC; ic++)
                            for (int kl = 0; kl < kL; kl++) {
                                int il = ol * str - pad + kl;
                                if (il >= 0 && il < LL)
                                    ((double*)a->grad)[ic*LL + il] += dout * tape_load_d(b, oc*inC*kL + ic*kL + kl);
                            }
                    }
            }
            if (b && b->requires_grad) {
                ensure_grad(b);
                for (int oc = 0; oc < outC; oc++)
                    for (int ic = 0; ic < inC; ic++)
                        for (int kl = 0; kl < kL; kl++) {
                            double s = 0;
                            for (int ol = 0; ol < oL; ol++) {
                                int il = ol * str - pad + kl;
                                if (il >= 0 && il < LL)
                                    s += ((double*)r->grad)[oc*oL + ol] * tape_load_d(a, ic*LL + il);
                            }
                            ((double*)b->grad)[oc*inC*kL + ic*kL + kl] += s;
                        }
            }
            Tensor* bias_t = (Tensor*)e->inputs;
            if (bias_t && bias_t->requires_grad) {
                ensure_grad(bias_t);
                for (int oc = 0; oc < outC; oc++) {
                    double s = 0;
                    for (int ol = 0; ol < oL; ol++) s += ((double*)r->grad)[oc*oL + ol];
                    ((double*)bias_t->grad)[oc] += s;
                }
            }
            break;
        }

        case OP_MAX_POOL1D: {
            MaxPool1DMeta* meta = (MaxPool1DMeta*)e->op_meta;
            ensure_grad(r);
            if (a && a->requires_grad) {
                ensure_grad(a);
                int out_numel = meta->C * meta->oL;
                for (int i = 0; i < out_numel; i++)
                    ((double*)a->grad)[meta->max_indices[i]] += ((double*)r->grad)[i];
            }
            break;
        }

        case OP_CONV2D: {
            /* r = conv2d(a=input, b=kernel) + bias
               a=[inC,H,W], b=[outC,inC,kH,kW], r=[outC,oH,oW] */
            Conv2DMeta* meta = (Conv2DMeta*)e->op_meta;
            int inC = meta->inC, outC = meta->outC;
            int HH = meta->H, WW = meta->W, kH = meta->kH, kW = meta->kW;
            int padH = meta->padH, padW = meta->padW;
            int strideH = meta->strH, strideW = meta->strW;
            int oH = meta->oH, oW = meta->oW;
            ensure_grad(r);

            /* d_input — tape_load_d on b->data covers F32 kernels. */
            if (a && a->requires_grad) {
                ensure_grad(a);
                for (int oc = 0; oc < outC; oc++)
                    for (int oh = 0; oh < oH; oh++)
                        for (int ow = 0; ow < oW; ow++) {
                            double dout = ((double*)r->grad)[oc*oH*oW + oh*oW + ow];
                            for (int ic = 0; ic < inC; ic++)
                                for (int kh = 0; kh < kH; kh++)
                                    for (int kw = 0; kw < kW; kw++) {
                                        int ih = oh * strideH - padH + kh;
                                        int iw = ow * strideW - padW + kw;
                                        if (ih >= 0 && ih < HH && iw >= 0 && iw < WW)
                                            ((double*)a->grad)[ic*HH*WW + ih*WW + iw] +=
                                                dout * tape_load_d(b, oc*inC*kH*kW + ic*kH*kW + kh*kW + kw);
                                    }
                        }
            }

            /* d_kernel — tape_load_d on a->data covers F32 inputs. */
            if (b && b->requires_grad) {
                ensure_grad(b);
                for (int oc = 0; oc < outC; oc++)
                    for (int ic = 0; ic < inC; ic++)
                        for (int kh = 0; kh < kH; kh++)
                            for (int kw = 0; kw < kW; kw++) {
                                double s = 0;
                                for (int oh = 0; oh < oH; oh++)
                                    for (int ow = 0; ow < oW; ow++) {
                                        int ih = oh * strideH - padH + kh;
                                        int iw = ow * strideW - padW + kw;
                                        if (ih >= 0 && ih < HH && iw >= 0 && iw < WW)
                                            s += ((double*)r->grad)[oc*oH*oW + oh*oW + ow]
                                               * tape_load_d(a, ic*HH*WW + ih*WW + iw);
                                    }
                                ((double*)b->grad)[oc*inC*kH*kW + ic*kH*kW + kh*kW + kw] += s;
                            }
            }

            /* d_bias */
            Tensor* bias_t = (Tensor*)e->inputs;  /* stored in inputs field */
            if (bias_t && bias_t->requires_grad) {
                ensure_grad(bias_t);
                for (int oc = 0; oc < outC; oc++) {
                    double s = 0;
                    for (int oh = 0; oh < oH; oh++)
                        for (int ow = 0; ow < oW; ow++)
                            s += ((double*)r->grad)[oc*oH*oW + oh*oW + ow];
                    ((double*)bias_t->grad)[oc] += s;
                }
            }
            break;
        }

        case OP_MAX_POOL2D: {
            /* r = max_pool2d(a=input). Gradient flows only to max positions. */
            MaxPool2DMeta* meta = (MaxPool2DMeta*)e->op_meta;
            ensure_grad(r);
            if (a && a->requires_grad) {
                ensure_grad(a);
                int out_numel = meta->C * meta->oH * meta->oW;
                for (int i = 0; i < out_numel; i++)
                    ((double*)a->grad)[meta->max_indices[i]] += ((double*)r->grad)[i];
            }
            break;
        }

        case OP_CONV2D_BATCHED: {
            /* r = conv2d_batched(input [B,inC,H,W], kernel [outC,inC,kH,kW]) + bias
               r=[B,outC,oH,oW]. Backward via im2col + cblas_dgemm in F64; for F32
               inputs the existing F64 dgemm path is reused by widening input and
               kernel to temporary double buffers (grads are F64 anyway, so this
               keeps the case body untouched). */
            Conv2DBatchedMeta* meta = (Conv2DBatchedMeta*)e->op_meta;
            int B = meta->B;
            int inC = meta->inC, outC = meta->outC;
            int HH = meta->H, WW = meta->W, kH = meta->kH, kW = meta->kW;
            int padH = meta->padH, padW = meta->padW;
            int strideH = meta->strH, strideW = meta->strW;
            int oH = meta->oH, oW = meta->oW;
            int K_unf = inC * kH * kW;
            int M_unf = B * oH * oW;
            int out_per_sample = outC * oH * oW;
            ensure_grad(r);

            Tensor* bias_t = (Tensor*)e->inputs;
            int need_dW = b && b->requires_grad;
            int need_dX = a && a->requires_grad;
            int need_dB = bias_t && bias_t->requires_grad;

            /* For F32 inputs/kernel, widen to double buffers so the existing
               cblas_dgemm + conv2d_im2col paths work unchanged. */
            double* a_data_dbl = NULL;
            double* b_data_dbl = NULL;
            const void* a_data_ptr = a->data;
            const void* b_data_ptr = b->data;
            if (a->dtype_tag == DT_F32) {
                size_t a_n = (size_t)B * inC * HH * WW;
                a_data_dbl = (double*)malloc(a_n * sizeof(double));
                for (size_t i = 0; i < a_n; i++) a_data_dbl[i] = (double)((float*)a->data)[i];
                a_data_ptr = a_data_dbl;
            }
            if (b->dtype_tag == DT_F32) {
                size_t b_n = (size_t)outC * inC * kH * kW;
                b_data_dbl = (double*)malloc(b_n * sizeof(double));
                for (size_t i = 0; i < b_n; i++) b_data_dbl[i] = (double)((float*)b->data)[i];
                b_data_ptr = b_data_dbl;
            }

            /* Permute dY [B, outC, oH, oW] -> dY_unf [B*oH*oW, outC] */
            double* dY_unf = (need_dW || need_dX) ?
                (double*)calloc((size_t)M_unf * outC, sizeof(double)) : NULL;
            if (dY_unf) {
                for (int bb = 0; bb < B; bb++) {
                    const double* dout_b = ((double*)r->grad) + (size_t)bb * out_per_sample;
                    for (int oc = 0; oc < outC; oc++) {
                        for (int oh = 0; oh < oH; oh++) {
                            for (int ow = 0; ow < oW; ow++) {
                                int row = bb * oH * oW + oh * oW + ow;
                                dY_unf[row * outC + oc] = dout_b[oc*oH*oW + oh*oW + ow];
                            }
                        }
                    }
                }
            }

            /* d_kernel — single dgemm: dW[outC,K] = dY_unf^T[outC,M] @ X_col[M,K] */
            if (need_dW) {
                ensure_grad(b);
                double* X_col = (double*)calloc((size_t)M_unf * K_unf, sizeof(double));
                conv2d_im2col((const double*)a_data_ptr, B, inC, HH, WW, kH, kW, padH, padW,
                              strideH, strideW, oH, oW, X_col);
#ifdef __APPLE__
                cblas_dgemm(CblasRowMajor, CblasTrans, CblasNoTrans,
                            outC, K_unf, M_unf, 1.0,
                            dY_unf, outC,
                            X_col, K_unf,
                            1.0, b->grad, K_unf);
#else
                for (int oc = 0; oc < outC; oc++)
                    for (int kk = 0; kk < K_unf; kk++) {
                        double s = 0;
                        for (int m = 0; m < M_unf; m++)
                            s += dY_unf[m*outC + oc] * X_col[m*K_unf + kk];
                        ((double*)b->grad)[oc*K_unf + kk] += s;
                    }
#endif
                free(X_col);
            }

            /* d_input — dX_col[M,K] = dY_unf[M,outC] @ W[outC,K], then col2im */
            if (need_dX) {
                ensure_grad(a);
                double* dX_col = calloc((size_t)M_unf * K_unf, sizeof(double));
#ifdef __APPLE__
                cblas_dgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans,
                            M_unf, K_unf, outC, 1.0,
                            dY_unf, outC,
                            (const double*)b_data_ptr, K_unf,
                            0.0, dX_col, K_unf);
#else
                for (int m = 0; m < M_unf; m++)
                    for (int kk = 0; kk < K_unf; kk++) {
                        double s = 0;
                        for (int oc = 0; oc < outC; oc++)
                            s += dY_unf[m*outC + oc] * ((const double*)b_data_ptr)[oc*K_unf + kk];
                        dX_col[m*K_unf + kk] = s;
                    }
#endif
                conv2d_col2im_accumulate(dX_col, B, inC, HH, WW, kH, kW,
                                          padH, padW, strideH, strideW,
                                          oH, oW, a->grad);
                free(dX_col);
            }

            /* d_bias — sum across B and (oH, oW) per output channel */
            if (need_dB) {
                ensure_grad(bias_t);
                for (int oc = 0; oc < outC; oc++) {
                    double s = 0;
                    for (int bb = 0; bb < B; bb++) {
                        const double* dout_b = ((double*)r->grad) + (size_t)bb * out_per_sample;
                        for (int oh = 0; oh < oH; oh++)
                            for (int ow = 0; ow < oW; ow++)
                                s += dout_b[oc*oH*oW + oh*oW + ow];
                    }
                    ((double*)bias_t->grad)[oc] += s;
                }
            }
            if (dY_unf) free(dY_unf);
            if (a_data_dbl) free(a_data_dbl);
            if (b_data_dbl) free(b_data_dbl);
            break;
        }

        case OP_MAX_POOL2D_BATCHED: {
            /* r = max_pool2d_batched(a=input [B,C,H,W]). max_indices are absolute
               into a->data, so direct scatter works the same as the per-sample case. */
            MaxPool2DBatchedMeta* meta = (MaxPool2DBatchedMeta*)e->op_meta;
            ensure_grad(r);
            if (a && a->requires_grad) {
                ensure_grad(a);
                int out_numel = meta->B * meta->C * meta->oH * meta->oW;
                for (int i = 0; i < out_numel; i++)
                    ((double*)a->grad)[meta->max_indices[i]] += ((double*)r->grad)[i];
            }
            break;
        }

        /* OP_SCATTER_ADD: moved to backend_tape/linear/index/scatter_add.c (Phase 1b.7.b). */
        /* OP_GATHER: moved to backend_tape/linear/index/gather.c (Phase 1b.7). */
        /* OP_CUMPROD: moved to backend_tape/linear/sort/cumprod.c (Phase 1b.8.b). */

        /* OP_LEAKY_RELU, OP_SILU: moved to backend_tape/nn/activation/ (Phase 1c.2). */

        default: break; /* unimplemented backward */
        }
        after_backward:
        /* Accumulate per-op timing */
        if (e->op < OP_COUNT) {
            prof_backward_per_op[e->op] += _wall_ms() - t_op;
            prof_backward_count_per_op[e->op]++;
        }
        }  /* close inner _j loop */
    }      /* close outer _cidx loop */
    free(_chunks_b);
    prof_backward_processed += processed;
    prof_backward_skipped += skipped;
    prof_backward_ms += _wall_ms() - t0;
    prof_backward_ops += processed;

    /* Phase 1.5e diagnostic: when DEBUG_PARAM_GRADS is set, dump per-param
       gradient L2 norm to stderr. Use to identify zero/NaN/wrong-magnitude
       grads after a single backward pass. param_registry is declared
       further down in the file; defer to the inner function below. */
    extern void _dbg_dump_param_grads_if_enabled(void);
    _dbg_dump_param_grads_if_enabled();
}

TensorHandle tensor_grad(TensorHandle h) {
    Tensor* t = (Tensor*)h;
    if (!t->grad) return NULL;
    return make_scalar(((double*)t->grad)[0], 0);
}

void tensor_zero_grad(TensorHandle h) {
    Tensor* t = (Tensor*)h;
    if (t->grad) memset(t->grad, 0, t->numel * sizeof(double));
}

int tensor_requires_grad(TensorHandle h) {
    return ((Tensor*)h)->requires_grad;
}

TensorHandle tensor_detach(TensorHandle h) {
    return tensor_clone(h);
}

TensorHandle tensor_with_grad(TensorHandle h) {
    Tensor* t = (Tensor*)h;
    Tensor* r = make_scalar(((double*)t->data)[0], 1);
    tape_append(OP_CONST, r, NULL, NULL, 0);
    return r;
}

void tensor_set_requires_grad(TensorHandle h, int rg) {
    Tensor* t = (Tensor*)h;
    t->requires_grad = rg;
    if (rg && t->tape_idx < 0) {
        tape_append(OP_CONST, t, NULL, NULL, 0);
    }
}

void tensor_no_grad_begin(void) { no_grad_depth++; }
void tensor_no_grad_end(void)   { if (no_grad_depth > 0) no_grad_depth--; }
/* No buffer ceiling on tape; per-epoch generation free is a no-op. */
void tensor_epoch_begin(void) {}
void tensor_epoch_end(void) {}

/* ================================================================
   Device (CPU only)
   ================================================================ */

TensorHandle tensor_to_device(TensorHandle t, const char* device) { return t; }
const char* tensor_device(TensorHandle t) { return "cpu"; }

/* ================================================================
   LSTM
   ================================================================ */

/* tensor_lstm_cell: moved to backend_tape/nn/recurrent/lstm_cell.c (Phase 1c.7.a). */

void tensor_lstm_gates(TensorHandle combined_h, TensorHandle prev_cell_h, int o,
                       TensorHandle* out_h, TensorHandle* out_c)
{
    Tensor* combined = (Tensor*)combined_h;
    Tensor* prev_cell = (Tensor*)prev_cell_h;
    if (combined->dtype_tag != prev_cell->dtype_tag)
        tape_abort_mixed_dtype("tensor_lstm_gates");
    int rg = combined->requires_grad || prev_cell->requires_grad;
    int shape[] = {o};
    int is_f32 = (combined->dtype_tag == DT_F32);

    /* Save gate activations for backward — cache stays double* for both
       dtypes (the backward accumulates into F64 grads). */
    LstmGatesMeta* meta = NULL;
    if (rg) {
        meta = arena_alloc(sizeof(LstmGatesMeta));
        meta->o = o;
        meta->iG = arena_alloc(o * sizeof(double));
        meta->fG = arena_alloc(o * sizeof(double));
        meta->gG = arena_alloc(o * sizeof(double));
        meta->oG = arena_alloc(o * sizeof(double));
        meta->new_cell = arena_alloc(o * sizeof(double));
    }

    if (is_f32) {
        float* out_hidden = arena_alloc(o * sizeof(float));
        float* out_cell   = arena_alloc(o * sizeof(float));
        for (int j = 0; j < o; j++) {
            double ig = 1.0 / (1.0 + exp(-tape_load_d(combined, j)));
            double fg = 1.0 / (1.0 + exp(-tape_load_d(combined, o+j)));
            double gg = tanh(tape_load_d(combined, 2*o+j));
            double og = 1.0 / (1.0 + exp(-tape_load_d(combined, 3*o+j)));
            double cell_v = fg * tape_load_d(prev_cell, j) + ig * gg;
            out_cell[j] = (float)cell_v;
            out_hidden[j] = (float)(og * tanh(cell_v));
            if (meta) {
                meta->iG[j] = ig; meta->fG[j] = fg;
                meta->gG[j] = gg; meta->oG[j] = og;
                meta->new_cell[j] = cell_v;
            }
        }
        *out_h = make_tensor_arena_f32(out_hidden, o, shape, 1, rg);
        *out_c = make_tensor_arena_f32(out_cell,   o, shape, 1, rg);
    } else {
        double* out_hidden = calloc(o, sizeof(double));
        double* out_cell = calloc(o, sizeof(double));
        for (int j = 0; j < o; j++) {
            double ig = 1.0 / (1.0 + exp(-((double*)combined->data)[j]));
            double fg = 1.0 / (1.0 + exp(-((double*)combined->data)[o+j]));
            double gg = tanh(((double*)combined->data)[2*o+j]);
            double og = 1.0 / (1.0 + exp(-((double*)combined->data)[3*o+j]));
            out_cell[j] = fg * ((double*)prev_cell->data)[j] + ig * gg;
            out_hidden[j] = og * tanh(out_cell[j]);
            if (meta) {
                meta->iG[j] = ig; meta->fG[j] = fg;
                meta->gG[j] = gg; meta->oG[j] = og;
                meta->new_cell[j] = out_cell[j];
            }
        }
        *out_h = make_tensor(out_hidden, shape, 1, rg);
        *out_c = make_tensor(out_cell, shape, 1, rg);
        free(out_hidden);
        free(out_cell);
    }

    if (rg) {
        /* Record hidden output with OP_LSTM_GATES — backward propagates d_hidden */
        TapeEntry* e_h = tape_append(OP_LSTM_GATES, (Tensor*)*out_h, combined, prev_cell, (double)o);
        e_h->op_meta = meta;
        /* Record cell output with OP_LSTM_GATES_CELL — backward propagates d_cell.
           Both entries share the same metadata and accumulate gradients additively
           into combined->grad and prev_cell->grad. */
        TapeEntry* e_c = tape_append(OP_LSTM_GATES_CELL, (Tensor*)*out_c, combined, prev_cell, (double)o);
        e_c->op_meta = meta;
    }
}

TensorPair* tensor_lstm_gates_pair(TensorHandle combined, TensorHandle prev_cell, int o) {
    TensorPair* p = arena_alloc(sizeof(TensorPair));
    tensor_lstm_gates(combined, prev_cell, o, &p->first, &p->second);
    return p;
}

TensorHandle tensor_pair_first(TensorPair* p) { return p->first; }
TensorHandle tensor_pair_second(TensorPair* p) { return p->second; }
void tensor_pair_free(TensorPair* p) { free(p); }

/* ================================================================
   GRU Cell
   combined = [z_raw, r_raw, n_raw] each [o], total [3*o]
   z = sigmoid(z_raw), r = sigmoid(r_raw)
   n = tanh(n_raw) -- note: n_raw should already include r*h contribution
   h' = (1-z)*n + z*prev_hidden
   ================================================================ */

TensorHandle tensor_gru_cell(TensorHandle hih, TensorHandle hhh, TensorHandle hprev, int o) {
    /* Standard nn.GRU equation. Takes ih = W_ih @ x + b_ih and
       hh = W_hh @ h + b_hh as separate [3*o] vectors (caller's
       responsibility to compute the two halves).
         z = sigmoid(ih_z + hh_z)
         r = sigmoid(ih_r + hh_r)
         n = tanh(ih_n + r * hh_n)
         h' = (1 - z) * n + z * prev
       Pre-2026-05-09 this kernel took a single combined = ih + hh
       and ignored r (simplified GRU); aligned to the standard
       nn.GRU equation so the example matches what library users
       expect. */
    Tensor* ih = (Tensor*)hih;
    Tensor* hh = (Tensor*)hhh;
    Tensor* prev = (Tensor*)hprev;
    if (ih->dtype_tag != hh->dtype_tag || ih->dtype_tag != prev->dtype_tag)
        tape_abort_mixed_dtype("tensor_gru_cell");
    int shape[] = {o};
    int rg = ih->requires_grad || hh->requires_grad || prev->requires_grad;
    int is_f32 = (ih->dtype_tag == DT_F32);

    /* Meta caches (zG/rG/nG) stay double* — backward writes F64 grads. */
    double* zG = malloc(o * sizeof(double));
    double* rG = malloc(o * sizeof(double));
    double* nG = malloc(o * sizeof(double));

    Tensor* r;
    if (is_f32) {
        float* out = arena_alloc(o * sizeof(float));
        for (int i = 0; i < o; i++) {
            zG[i] = 1.0 / (1.0 + exp(-(tape_load_d(ih, i) + tape_load_d(hh, i))));
            rG[i] = 1.0 / (1.0 + exp(-(tape_load_d(ih, o+i) + tape_load_d(hh, o+i))));
            nG[i] = tanh(tape_load_d(ih, 2*o+i) + rG[i] * tape_load_d(hh, 2*o+i));
            double h = (1.0 - zG[i]) * nG[i] + zG[i] * tape_load_d(prev, i);
            out[i] = (float)h;
        }
        r = make_tensor_arena_f32(out, o, shape, 1, rg);
    } else {
        double* out = calloc(o, sizeof(double));
        for (int i = 0; i < o; i++) {
            zG[i] = 1.0 / (1.0 + exp(-(((double*)ih->data)[i] + ((double*)hh->data)[i])));
            rG[i] = 1.0 / (1.0 + exp(-(((double*)ih->data)[o + i] + ((double*)hh->data)[o + i])));
            nG[i] = tanh(((double*)ih->data)[2*o + i] + rG[i] * ((double*)hh->data)[2*o + i]);
            out[i] = (1.0 - zG[i]) * nG[i] + zG[i] * ((double*)prev->data)[i];
        }
        r = make_tensor(out, shape, 1, rg);
        free(out);
    }

    if (r->requires_grad) {
        TapeEntry* e = tape_append(OP_GRU_CELL, r, ih, hh, 0);
        GruCellMeta* meta = arena_alloc(sizeof(GruCellMeta));
        meta->o = o;
        meta->zG = zG; meta->rG = rG; meta->nG = nG;
        meta->prev = prev;
        e->op_meta = meta;
    } else {
        free(zG); free(rG); free(nG);
    }
    return r;
}

/* ================================================================
   Parameter Registry
   ================================================================ */

typedef struct {
    char name[256];
    Tensor* tensor;
} ParamEntry;

#define MAX_PARAMS 65536
static ParamEntry param_registry[MAX_PARAMS];
static int param_count_val = 0;

void param_register(const char* name, TensorHandle h) {
    Tensor* t = (Tensor*)h;
    /* Replace if exists */
    for (int i = 0; i < param_count_val; i++) {
        if (strcmp(param_registry[i].name, name) == 0) {
            param_registry[i].tensor = t;
            return;
        }
    }
    if (param_count_val < MAX_PARAMS) {
        strncpy(param_registry[param_count_val].name, name, 255);
        param_registry[param_count_val].tensor = t;
        param_count_val++;
    }
}

void param_clear(void) { param_count_val = 0; }
int param_count(void) { return param_count_val; }
const char* param_name(int idx) { return param_registry[idx].name; }

/* Phase 1.5e diagnostic: dump per-param gradient L2 norms after a backward
   pass. Enabled by setting DEBUG_PARAM_GRADS env var. Defined here (rather
   than inline in tensor_backward) so it can see the static param_registry. */
void _dbg_dump_param_grads_if_enabled(void) {
    if (!getenv("DEBUG_PARAM_GRADS")) return;
    fprintf(stderr, "=== param grads after backward ===\n");
    for (int i = 0; i < param_count_val; i++) {
        Tensor* t = param_registry[i].tensor;
        double l2 = 0.0;
        int has_nan = 0;
        if (t->grad) {
            for (int j = 0; j < t->numel; j++) {
                double g = ((double*)t->grad)[j];
                if (isnan(g) || isinf(g)) has_nan = 1;
                l2 += g * g;
            }
            l2 = sqrt(l2);
        }
        fprintf(stderr, "  %-40s numel=%-6d l2=%12.6e%s%s\n",
                param_registry[i].name, t->numel, l2,
                t->grad ? "" : " NO_GRAD",
                has_nan ? " NAN_OR_INF!" : "");
    }
}

/* Phase 1.5e diagnostic: dump h0/c0 param value trajectories + first 3
   element values. Set DEBUG_LSTM_TRAJ to print every N epochs. */
static int _dbg_traj_step = 0;
void _dbg_dump_lstm_traj_if_enabled(void) {
    if (!getenv("DEBUG_LSTM_TRAJ")) return;
    int every = 100;
    const char* every_s = getenv("DEBUG_LSTM_TRAJ_EVERY");
    if (every_s) every = atoi(every_s);
    _dbg_traj_step++;
    if (_dbg_traj_step % every != 0 && _dbg_traj_step != 1) return;
    for (int i = 0; i < param_count_val; i++) {
        const char* nm = param_registry[i].name;
        /* Match _h0 or _c0 (LSTM learned init) */
        size_t L = strlen(nm);
        if (L >= 3 && (strcmp(nm + L - 3, "_h0") == 0 || strcmp(nm + L - 3, "_c0") == 0)) {
            Tensor* t = param_registry[i].tensor;
            double l2 = 0.0, mn = 1e300, mx = -1e300;
            for (int j = 0; j < t->numel; j++) {
                double v = ((double*)t->data)[j];
                l2 += v*v;
                if (v < mn) mn = v;
                if (v > mx) mx = v;
            }
            l2 = sqrt(l2);
            fprintf(stderr, "[traj epoch %d] %s l2=%.10g min=%.10g max=%.10g | t[0..2]=%.10g, %.10g, %.10g\n",
                    _dbg_traj_step, nm, l2, mn, mx,
                    t->numel >= 1 ? ((double*)t->data)[0] : 0.0,
                    t->numel >= 2 ? ((double*)t->data)[1] : 0.0,
                    t->numel >= 3 ? ((double*)t->data)[2] : 0.0);
        }
    }
}

double param_grad_item(int idx) {
    Tensor* t = param_registry[idx].tensor;
    if (!t->grad) return 0.0;
    return ((double*)t->grad)[0];
}

double param_grad_item_at(int param_idx, int elem_idx) {
    Tensor* t = param_registry[param_idx].tensor;
    if (!t->grad || elem_idx >= t->numel) return 0.0;
    return ((double*)t->grad)[elem_idx];
}

double param_grad_item_and_zero(int idx) {
    Tensor* t = param_registry[idx].tensor;
    if (!t->grad) return 0.0;
    double v = ((double*)t->grad)[0];
    ((double*)t->grad)[0] = 0.0;
    return v;
}

TensorHandle param_tensor(int idx) { return param_registry[idx].tensor; }

void param_zero_all_grads(void) {
    for (int i = 0; i < param_count_val; i++) {
        Tensor* t = param_registry[i].tensor;
        if (t->grad) memset(t->grad, 0, t->numel * sizeof(double));
    }
}

void param_subtract_delta(int idx, double delta) {
    Tensor* t = param_registry[idx].tensor;
    ((double*)t->data)[0] -= delta;
}

void param_load_data(int idx, const double* data, int numel) {
    Tensor* t = param_registry[idx].tensor;
    if (t->numel != numel) {
        fprintf(stderr, "param_load_data: size mismatch for '%s': expected %d, got %d\n",
                param_registry[idx].name, t->numel, numel);
        return;
    }
    memcpy(t->data, data, numel * sizeof(double));
}

/* Byte-level I64 in-place loader — see backend.h. Tape's lingua-franca
   storage routes every int64 through `tape_store_d` (narrows to float
   on F32 storage, plain double-write otherwise). Values above 2^53
   lose precision at this conversion, matching the existing lingua-
   franca behaviour — no regression. */
void param_load_data_int64(int idx, const int64_t* data, int numel) {
    Tensor* t = param_registry[idx].tensor;
    if (t->numel != numel) {
        fprintf(stderr, "param_load_data_int64: size mismatch for '%s': expected %d, got %d\n",
                param_registry[idx].name, t->numel, numel);
        return;
    }
    for (int i = 0; i < numel; i++) {
        tape_store_d(t, i, (double)data[i]);
    }
}

TensorHandle tensor_subtract_scalar_inplace(TensorHandle h, double val) {
    Tensor* t = (Tensor*)h;
    if (t->dtype_tag == DT_F32) {
        float vf = (float)val;
        for (int i = 0; i < t->numel; i++) ((float*)t->data)[i] -= vf;
    } else {
        for (int i = 0; i < t->numel; i++) ((double*)t->data)[i] -= val;
    }
    return h;
}

/* ================================================================
   Convenience functions
   ================================================================ */

/* Create a one-hot encoded 1D tensor from token indices.
   tokens: array of token indices (int), n_tokens long
   vocab_size: number of classes per token
   Output: 1D tensor of length n_tokens * vocab_size */
TensorHandle tensor_one_hot(int* tokens, int n_tokens, int vocab_size, int dtag) {
    (void)dtag;  /* one-hot's 0/1 image fits losslessly in every dtype; today
                    the live callers (Mnist / Gpt / Transformer) hit the F64
                    path, so the dtag is currently a no-op here — when a
                    non-F64 caller appears, route through tape_round_to_dtype
                    + the dtag-keyed arena allocators added in the parity
                    work. */
    int total = n_tokens * vocab_size;
    double* data = calloc(total, sizeof(double));  /* zeros */
    for (int i = 0; i < n_tokens; i++) {
        int tok = tokens[i];
        if (tok >= 0 && tok < vocab_size)
            data[i * vocab_size + tok] = 1.0;
    }
    int shape[] = {total};
    Tensor* r = make_tensor(data, shape, 1, 0);
    free(data);
    free(tokens);
    return r;
}

TensorHandle tensor_create_1d(int n, double* data, int requires_grad) {
    int shape[] = {n};
    TensorHandle t = tensor_create(data, shape, 1, requires_grad);
    free(data);  /* tensor_create copies data into arena; free the original */
    return t;
}

TensorHandle tensor_create_2d(int rows, int cols, double* data, int requires_grad) {
    int shape[] = {rows, cols};
    TensorHandle t = tensor_create(data, shape, 2, requires_grad);
    free(data);
    return t;
}

TensorHandle tensor_reshape_2d(TensorHandle h, int rows, int cols) {
    int shape[] = {rows, cols};
    return tensor_reshape(h, shape, 2);
}

TensorHandle tensor_reshape_1d(TensorHandle h, int n) {
    int shape[] = {n};
    return tensor_reshape(h, shape, 1);
}

/* tensor_alloc_doubles / tensor_free_doubles / tensor_read_double /
 * tensor_ptr_array_alloc live in shared_utils.c (unified across all
 * backends; see packages/backends/shared_utils.{c,h}). */

TensorHandle tensor_stack_from_array(TensorHandle* arr, int count, int dim) {
    /* Fast path: if all inputs are consecutive selects from the same parent
       tensor (data pointers are contiguous), skip the copy and return a
       tensor that shares the parent's data. This eliminates the repack
       cost when tensorToScalars → vecStackTensor round-trips. */
    if (count > 0) {
        Tensor* first = (Tensor*)arr[0];
        double* base = first->data;
        int consecutive = 1;
        int rg_check = first->requires_grad;
        for (int i = 1; i < count; i++) {
            Tensor* t = (Tensor*)arr[i];
            if (t->data != base + i) { consecutive = 0; break; }
            if (t->requires_grad) rg_check = 1;
        }
        if (consecutive) { 
            /* Create a tensor that shares the parent's data buffer (no copy) */
            Tensor* r = arena_alloc(sizeof(Tensor));
            memset(r, 0, sizeof(Tensor));
            r->data = base;  /* shared with parent */
            r->shape = arena_alloc(sizeof(int));
            r->shape[0] = count;
            r->rank = 1;
            r->numel = count;
            r->requires_grad = rg_check;
            r->persistent = 0;
            /* Still record OP_STACK with input pointers for backward.
               STACK backward distributes ((double*)r->grad)[i] to inputs[i]->grad[0].
               The inputs are SELECT views, so their grad flows to the parent. */
            if (rg_check) {
                Tensor** inputs = malloc(count * sizeof(Tensor*));
                for (int i = 0; i < count; i++) inputs[i] = (Tensor*)arr[i];
                TapeEntry* e = tape_append(OP_STACK, r, NULL, NULL, 0);
                e->inputs = inputs;
                e->input_count = count;
            }
            free(arr);
            return r;
        }
    }

    /* Slow path: copy values and create new tensor */
    double* data = malloc(count * sizeof(double));
    int rg = 0;
    Tensor** inputs = malloc(count * sizeof(Tensor*));
    for (int i = 0; i < count; i++) {
        Tensor* t = (Tensor*)arr[i];
        data[i] = ((double*)t->data)[0];
        inputs[i] = t;
        if (t->requires_grad) rg = 1;
    }
    free(arr);
    int shape[] = {count};
    Tensor* r = make_tensor(data, shape, 1, rg);
    free(data);
    if (rg) {
        TapeEntry* e = tape_append(OP_STACK, r, NULL, NULL, 0);
        e->inputs = inputs;
        e->input_count = count;
    } else {
        free(inputs);
    }
    return r;
}

TensorHandle tensor_cat_from_array(TensorHandle* arr, int count, int dim) {
    return tensor_stack_from_array(arr, count, dim);
}

/* ================================================================
   Tensor-level parameter creation
   ================================================================ */

TensorHandle tensor_create_param_2d(int rows, int cols, double* data) {
    /* Param tensors use regular malloc — persist across arena resets */
    int numel = rows * cols;
    Tensor* t = calloc(1, sizeof(Tensor));
    t->data = malloc(numel * sizeof(double));
    memcpy(t->data, data, numel * sizeof(double));
    free(data);  /* free the input buffer (caller used tensor_alloc_doubles) */
    t->shape = malloc(2 * sizeof(int));
    t->shape[0] = rows; t->shape[1] = cols;
    t->rank = 2; t->numel = numel;
    t->requires_grad = 1;
    t->tape_idx = -1;
    t->persistent = 1;
    tape_append(OP_CONST, t, NULL, NULL, 0);
    return t;
}

TensorHandle tensor_create_param_4d(int d0, int d1, int d2, int d3, double* data) {
    int numel = d0 * d1 * d2 * d3;
    Tensor* t = calloc(1, sizeof(Tensor));
    t->data = malloc(numel * sizeof(double));
    memcpy(t->data, data, numel * sizeof(double));
    free(data);
    t->shape = malloc(4 * sizeof(int));
    t->shape[0] = d0; t->shape[1] = d1; t->shape[2] = d2; t->shape[3] = d3;
    t->rank = 4; t->numel = numel;
    t->requires_grad = 1;
    t->tape_idx = -1;
    t->persistent = 1;
    tape_append(OP_CONST, t, NULL, NULL, 0);
    return t;
}

TensorHandle tensor_create_param_1d(int n, double* data) {
    Tensor* t = calloc(1, sizeof(Tensor));
    t->data = malloc(n * sizeof(double));
    memcpy(t->data, data, n * sizeof(double));
    free(data);
    t->shape = malloc(sizeof(int));
    t->shape[0] = n;
    t->rank = 1; t->numel = n;
    t->requires_grad = 1;
    t->tape_idx = -1;
    t->persistent = 1;
    tape_append(OP_CONST, t, NULL, NULL, 0);
    return t;
}

/* Persistent tensors WITHOUT requires_grad — for non-learnable NTM state */
TensorHandle tensor_create_state_2d(int rows, int cols, double* data) {
    int numel = rows * cols;
    Tensor* t = calloc(1, sizeof(Tensor));
    t->data = malloc(numel * sizeof(double));
    memcpy(t->data, data, numel * sizeof(double));
    free(data);
    t->shape = malloc(2 * sizeof(int));
    t->shape[0] = rows; t->shape[1] = cols;
    t->rank = 2; t->numel = numel;
    t->requires_grad = 0;
    t->tape_idx = -1;
    t->persistent = 1;
    return t;
}

TensorHandle tensor_create_state_1d(int n, double* data) {
    Tensor* t = calloc(1, sizeof(Tensor));
    t->data = malloc(n * sizeof(double));
    memcpy(t->data, data, n * sizeof(double));
    free(data);
    t->shape = malloc(sizeof(int));
    t->shape[0] = n;
    t->rank = 1; t->numel = n;
    t->requires_grad = 0;
    t->tape_idx = -1;
    t->persistent = 1;
    return t;
}

/* ================================================================
   Per-dtype creation variants (legacy, pre-unified-dispatch)
   --------------------------------------------------------------
   These per-dtype `_f32` / `_f64` symbols predate the unified
   `tensor_create_<shape>_streamed(..., int dtag)` entry points that
   landed with the FFI tag-dispatch unification (2026-05-22). The
   typed Idris surface now routes through the unified streamed
   symbols exclusively; these per-dtype variants are kept only for
   `backend.h` ABI completeness across the multi-link dylib (every
   backend must export every prototype declared in backend.h).

   The _f64 variants delegate to the unsuffixed F64 creators —
   identity, F64 is the lingua-franca path. The _f32 variants are
   left as abort stubs here even though tape now has *real* F32
   storage via `tape_arena_f32_from_doubles` /
   `tape_persistent_f32_from_doubles` (used by the unified streamed
   path); no caller reaches the legacy `_f32` symbols any more
   (`grep tensor_create_.*_f32` in `packages/idris-ml*` /
   `packages/idris-ml-examples` finds zero hits), so the abort
   diagnostic is reachable only via direct C linkage. Cleaning them
   out is a separate ABI rewrite (paired with row 21 RuntimeDType
   reorder); the abort gives a clear diagnostic if anything ever
   does.
   ================================================================ */

#include <stdio.h>

static TensorHandle tape_f32_unsupported(const char* sym) {
    fprintf(stderr,
        "[tape backend] %s called but tape has no fp32 arena. "
        "Bind your code to F64 on tape, or build with BACKEND=mlx / torch.\n",
        sym);
    abort();
}

TensorHandle tensor_create_scalar_f64(double v, int rg)                                 { return tensor_create_scalar(v, rg); }
TensorHandle tensor_create_f64(double* d, int* s, int r, int rg)                        { return tensor_create(d, s, r, rg); }
TensorHandle tensor_create_1d_f64(int n, double* d, int rg)                             { return tensor_create_1d(n, d, rg); }
TensorHandle tensor_create_2d_f64(int rows, int cols, double* d, int rg)                { return tensor_create_2d(rows, cols, d, rg); }
TensorHandle tensor_create_param_1d_f64(int n, double* d)                               { return tensor_create_param_1d(n, d); }
TensorHandle tensor_create_param_2d_f64(int rows, int cols, double* d)                  { return tensor_create_param_2d(rows, cols, d); }
TensorHandle tensor_create_param_3d_f64(int d0, int d1, int d2, double* d)              { return tensor_create_param_3d(d0, d1, d2, d); }
TensorHandle tensor_create_param_4d_f64(int d0, int d1, int d2, int d3, double* d)      { return tensor_create_param_4d(d0, d1, d2, d3, d); }
TensorHandle tensor_create_state_1d_f64(int n, double* d)                               { return tensor_create_state_1d(n, d); }
TensorHandle tensor_create_state_2d_f64(int rows, int cols, double* d)                  { return tensor_create_state_2d(rows, cols, d); }

TensorHandle tensor_create_scalar_f32(double v, int rg)                                 { (void)v; (void)rg; return tape_f32_unsupported("tensor_create_scalar_f32"); }
TensorHandle tensor_create_f32(double* d, int* s, int r, int rg)                        { (void)d; (void)s; (void)r; (void)rg; return tape_f32_unsupported("tensor_create_f32"); }
TensorHandle tensor_create_1d_f32(int n, double* d, int rg)                             { (void)n; (void)d; (void)rg; return tape_f32_unsupported("tensor_create_1d_f32"); }
TensorHandle tensor_create_2d_f32(int rows, int cols, double* d, int rg)                { (void)rows; (void)cols; (void)d; (void)rg; return tape_f32_unsupported("tensor_create_2d_f32"); }
TensorHandle tensor_create_param_1d_f32(int n, double* d)                               { (void)n; (void)d; return tape_f32_unsupported("tensor_create_param_1d_f32"); }
TensorHandle tensor_create_param_2d_f32(int rows, int cols, double* d)                  { (void)rows; (void)cols; (void)d; return tape_f32_unsupported("tensor_create_param_2d_f32"); }
TensorHandle tensor_create_param_3d_f32(int d0, int d1, int d2, double* d)              { (void)d0; (void)d1; (void)d2; (void)d; return tape_f32_unsupported("tensor_create_param_3d_f32"); }
TensorHandle tensor_create_param_4d_f32(int d0, int d1, int d2, int d3, double* d)      { (void)d0; (void)d1; (void)d2; (void)d3; (void)d; return tape_f32_unsupported("tensor_create_param_4d_f32"); }
TensorHandle tensor_create_state_1d_f32(int n, double* d)                               { (void)n; (void)d; return tape_f32_unsupported("tensor_create_state_1d_f32"); }
TensorHandle tensor_create_state_2d_f32(int rows, int cols, double* d)                  { (void)rows; (void)cols; (void)d; return tape_f32_unsupported("tensor_create_state_2d_f32"); }

/* Per-dtype cast primitives (legacy, pre-unified-dispatch).
 * The live cast path is `tensor_cast_dtype_streamed(src, stream_tag, dtag)`
 * which handles every dtag with the matching real storage (F32 → real
 * `float*` arena; F64 identity; the 8 inference dtypes via
 * `tape_retag_round` + the lingua franca). These per-dtype `_f64` / `_f32`
 * symbols are kept only for backend.h ABI completeness in the multi-link
 * dylib; the typed Idris surface no longer reaches them. F64 is an alias
 * (the FFI wrapper retains the handle and Idris gets a fresh wrapper
 * around the same C handle; gradients flow through the source's tape entry
 * — no new tape op is appended since the operation is observationally
 * identity). F32 aborts here; reach the real F32 cast path via
 * `tensor_cast_dtype_streamed(src, _, 0)`. */
TensorHandle tensor_cast_dtype_f64(TensorHandle src)                                     { return src; }
TensorHandle tensor_cast_dtype_f32(TensorHandle src)                                     { (void)src; return tape_f32_unsupported("tensor_cast_dtype_f32"); }

/* tensor_view_2d, tensor_view_1d: moved to backend_tape/linear/shape/ (Phase 1b.1.d). */

double tensor_item_2d(TensorHandle h, int row, int col) {
    Tensor* t = (Tensor*)h;
    return tape_load_d(t, row * t->shape[1] + col);
}

/* tensor_item_1d: moved to backend_tape/core/lifecycle/item1d.c (Phase 1a.10). */

/* ================================================================
   Native Optimizer
   ================================================================ */

typedef struct {
    double lr;
    int type; /* 0=SGD, 1=RMSprop, 2=Adam, 3=AdamW */
    double alpha, eps, weight_decay, momentum;
    double beta1, beta2;
    double* v;  /* second moment (RMSprop/Adam) */
    double* m;  /* first moment (Adam) / momentum buffer (RMSprop) */
    int t;      /* step count */
    int allocated;
    double* param_lr;      /* per-param LR overrides (NULL = use opt->lr for all) */
    int param_lr_count;    /* number of entries in param_lr */
    char prefix[128];      /* param-name prefix filter (empty = manages all) */
} Optimizer;

/* Returns 1 if param[i]'s name starts with opt->prefix (or prefix is empty). */
static int opt_owns_param(Optimizer* opt, int i) {
    if (opt->prefix[0] == '\0') return 1;
    return strncmp(param_registry[i].name, opt->prefix, strlen(opt->prefix)) == 0;
}

/* Compute total number of elements across all params (for per-element optimizer buffers) */
static int param_total_elements(void) {
    int total = 0;
    for (int i = 0; i < param_count_val; i++)
        total += param_registry[i].tensor->numel;
    return total;
}

/* Offset into the flat per-element buffer for param i, element j */
static int param_element_offset(int param_idx) {
    int off = 0;
    for (int i = 0; i < param_idx; i++)
        off += param_registry[i].tensor->numel;
    return off;
}

static void optimizer_ensure_buffers(Optimizer* opt) {
    if (opt->allocated) return;
    int n = param_total_elements();
    opt->v = calloc(n, sizeof(double));
    opt->m = calloc(n, sizeof(double));
    opt->allocated = 1;
}

OptimizerHandle optimizer_create_sgd(double lr) {
    Optimizer* opt = calloc(1, sizeof(Optimizer));
    opt->lr = lr;
    opt->type = 0;
    return opt;
}

OptimizerHandle optimizer_create_rmsprop(double lr, double alpha, double eps,
                                          double weight_decay, double momentum) {
    Optimizer* opt = calloc(1, sizeof(Optimizer));
    opt->lr = lr; opt->type = 1;
    opt->alpha = alpha; opt->eps = eps;
    opt->weight_decay = weight_decay; opt->momentum = momentum;
    return opt;
}

OptimizerHandle optimizer_create_adam(double lr, double beta1, double beta2, double eps) {
    Optimizer* opt = calloc(1, sizeof(Optimizer));
    opt->lr = lr; opt->type = 2;
    opt->beta1 = beta1; opt->beta2 = beta2; opt->eps = eps;
    return opt;
}

/* Adam that only updates params whose registry name starts with `prefix`.
 * Empty prefix behaves like optimizer_create_adam (manages all params). */
OptimizerHandle optimizer_create_adam_group(double lr, double beta1, double beta2,
                                            double eps, const char* prefix) {
    Optimizer* opt = (Optimizer*)optimizer_create_adam(lr, beta1, beta2, eps);
    if (prefix) {
        strncpy(opt->prefix, prefix, sizeof(opt->prefix) - 1);
        opt->prefix[sizeof(opt->prefix) - 1] = '\0';
    }
    return opt;
}

/* Polyak soft update at the param-registry level.
 * For each param P whose name starts with `online_scope`, find the
 * corresponding param Q whose name is `target_scope ++ suffix(P)` and
 * blend Q's data in-place: Q.data ← (1−tau)·Q.data + tau·P.data.
 *
 * Used by SAC: actor / q1 / q2 network params are registered with
 * distinct scope prefixes; target-Q params are registered with
 * `q1_tgt_` / `q2_tgt_` prefixes. One call per target network per
 * training step moves the target toward the online at rate τ.
 * Returns number of param-pairs blended (for sanity checking). */
int polyak_blend(double tau, const char* online_scope, const char* target_scope) {
    if (!online_scope || !target_scope) return 0;
    size_t on_len = strlen(online_scope);
    size_t tg_len = strlen(target_scope);
    int blended = 0;
    double one_minus_tau = 1.0 - tau;
    for (int i = 0; i < param_count_val; i++) {
        const char* on_name = param_registry[i].name;
        if (strncmp(on_name, online_scope, on_len) != 0) continue;
        /* Build target name: target_scope ++ (on_name + on_len). */
        char tgt_name[256];
        size_t suffix_len = strlen(on_name + on_len);
        if (tg_len + suffix_len + 1 > sizeof(tgt_name)) continue;
        memcpy(tgt_name, target_scope, tg_len);
        memcpy(tgt_name + tg_len, on_name + on_len, suffix_len + 1);
        /* Find target param. */
        for (int j = 0; j < param_count_val; j++) {
            if (strcmp(param_registry[j].name, tgt_name) != 0) continue;
            Tensor* on_t = param_registry[i].tensor;
            Tensor* tg_t = param_registry[j].tensor;
            if (on_t->numel != tg_t->numel) break;
            for (int k = 0; k < on_t->numel; k++) {
                ((double*)tg_t->data)[k] = one_minus_tau * ((double*)tg_t->data)[k] + tau * ((double*)on_t->data)[k];
            }
            blended++;
            break;
        }
    }
    return blended;
}

OptimizerHandle optimizer_create_adamw(double lr, double beta1, double beta2, double eps,
                                       double weight_decay) {
    Optimizer* opt = calloc(1, sizeof(Optimizer));
    opt->lr = lr; opt->type = 3;
    opt->beta1 = beta1; opt->beta2 = beta2; opt->eps = eps;
    opt->weight_decay = weight_decay;
    return opt;
}

void optimizer_free(OptimizerHandle h) {
    Optimizer* opt = (Optimizer*)h;
    free(opt->v); free(opt->m); free(opt->param_lr); free(opt);
}

void optimizer_set_param_lr(OptimizerHandle h, const char* name, double lr) {
    Optimizer* opt = (Optimizer*)h;
    /* Ensure param_lr array is large enough */
    if (opt->param_lr == NULL || opt->param_lr_count < param_count_val) {
        int new_count = param_count_val > 64 ? param_count_val : 64;
        double* new_lr = realloc(opt->param_lr, new_count * sizeof(double));
        /* Initialize new entries to -1 (sentinel: use base LR) */
        for (int i = opt->param_lr_count; i < new_count; i++) new_lr[i] = -1.0;
        opt->param_lr = new_lr;
        opt->param_lr_count = new_count;
    }
    /* Find param by name and set its LR */
    for (int i = 0; i < param_count_val; i++) {
        if (strcmp(param_registry[i].name, name) == 0) {
            opt->param_lr[i] = lr;
            return;
        }
    }
}

void optimizer_set_lr(OptimizerHandle h, double lr) {
    Optimizer* opt = (Optimizer*)h;
    opt->lr = lr;
}

void optimizer_zero_grad(OptimizerHandle h) {
    (void)h;
    param_zero_all_grads();
}

void optimizer_step(OptimizerHandle h) {
    double t0_opt = _wall_ms();
    Optimizer* opt = (Optimizer*)h;
    optimizer_ensure_buffers(opt);
    opt->t++;

    for (int i = 0; i < param_count_val; i++) {
        if (!opt_owns_param(opt, i)) continue;
        /* Phase 1.5e DIAGNOSTIC: SKIP_LSTM_INIT skips updating params whose
           names end in _h0 or _c0 (LSTM learned initial state). Equivalent
           to keeping them as zero state tensors. Used to localize whether
           the convergence regression is in the gradient values being
           applied to h0/c0 vs being elsewhere. Remove once diagnosed. */
        if (getenv("SKIP_LSTM_INIT")) {
            const char* nm = param_registry[i].name;
            size_t L = strlen(nm);
            if (L >= 3 && (strcmp(nm + L - 3, "_h0") == 0 ||
                           strcmp(nm + L - 3, "_c0") == 0)) {
                continue;
            }
        }
        Tensor* t = param_registry[i].tensor;
        if (!t->grad) continue;
        int base = param_element_offset(i);

        /* Per-param LR: use override if set, otherwise base LR */
        double lr = opt->lr;
        if (opt->param_lr && i < opt->param_lr_count && opt->param_lr[i] >= 0)
            lr = opt->param_lr[i];

        for (int j = 0; j < t->numel; j++) {
            double g = ((double*)t->grad)[j];
            int idx = base + j;  /* per-element index into optimizer buffers */

            /* Dtype-aware reads + writes so F32 params take f32-precision
               updates (asserted by the rung-4 F32-exactness check). Moment
               buffers (opt->m / opt->v) stay F64 — standard mixed-precision
               practice and lets the F64 numerics path stay byte-identical. */
            double w = tape_load_d(t, j);
            switch (opt->type) {
            case 0: /* SGD */
                tape_store_d(t, j, w - lr * g);
                break;

            case 1: { /* RMSprop — keep lr OUTSIDE the momentum buffer to match
                         torch.optim.RMSprop. Folding lr into the buffer
                         (buf = m*buf + lr*g/avg) coincides with PyTorch only at
                         constant lr; under an LR schedule the buffer carries
                         stale rates and diverges. */
                opt->v[idx] = opt->alpha * opt->v[idx] + (1.0 - opt->alpha) * g * g;
                double avg = sqrt(opt->v[idx]) + opt->eps;
                if (opt->momentum > 0) {
                    opt->m[idx] = opt->momentum * opt->m[idx] + g / avg;
                    tape_store_d(t, j, w - lr * opt->m[idx]);
                } else {
                    tape_store_d(t, j, w - lr * g / avg);
                }
                break;
            }

            case 2: { /* Adam */
                opt->m[idx] = opt->beta1 * opt->m[idx] + (1.0 - opt->beta1) * g;
                opt->v[idx] = opt->beta2 * opt->v[idx] + (1.0 - opt->beta2) * g * g;
                double mhat = opt->m[idx] / (1.0 - pow(opt->beta1, opt->t));
                double vhat = opt->v[idx] / (1.0 - pow(opt->beta2, opt->t));
                tape_store_d(t, j, w - lr * mhat / (sqrt(vhat) + opt->eps));
                break;
            }

            case 3: { /* AdamW (decoupled weight decay) */
                opt->m[idx] = opt->beta1 * opt->m[idx] + (1.0 - opt->beta1) * g;
                opt->v[idx] = opt->beta2 * opt->v[idx] + (1.0 - opt->beta2) * g * g;
                double mhat = opt->m[idx] / (1.0 - pow(opt->beta1, opt->t));
                double vhat = opt->v[idx] / (1.0 - pow(opt->beta2, opt->t));
                double w1 = w - lr * mhat / (sqrt(vhat) + opt->eps);
                tape_store_d(t, j, w1 - lr * opt->weight_decay * w1);
                break;
            }
            }
        }
    }

    /* Phase 1.5e: dump h0/c0 trajectory if enabled */
    {
        extern void _dbg_dump_lstm_traj_if_enabled(void);
        _dbg_dump_lstm_traj_if_enabled();
    }

    /* Snapshot tape size before reset */
    prof_forward_ops = tape_size;

    /* Reset tape and re-register ONLY the param tensors (from param_registry).
       Ephemeral tensors (select results, intermediates) are not re-registered.
       They will be recreated in the next forward pass. */
    tape_reset();
    for (int j = 0; j < param_count_val; j++) {
        Tensor* t = param_registry[j].tensor;
        t->tape_idx = -1;
        if (t->grad) memset(t->grad, 0, t->numel * sizeof(double));
        tape_append(OP_CONST, t, NULL, NULL, 0);
    }
    prof_optimizer_ms += _wall_ms() - t0_opt;
    prof_epochs++;
    /* Auto-start timing for next epoch's forward pass + per-op accumulator */
    double t_next = _wall_ms();
    prof_epoch_start = t_next;
    prof_op_t_prev = t_next;
}

/* Internal clip-value helper scoped to params owned by `opt`. */
static void clip_grad_value_opt(Optimizer* opt, double max_val) {
    for (int i = 0; i < param_count_val; i++) {
        if (opt && !opt_owns_param(opt, i)) continue;
        Tensor* t = param_registry[i].tensor;
        if (!t->grad) continue;
        for (int j = 0; j < t->numel; j++) {
            if (((double*)t->grad)[j] > max_val) ((double*)t->grad)[j] = max_val;
            if (((double*)t->grad)[j] < -max_val) ((double*)t->grad)[j] = -max_val;
        }
    }
}

/* Internal clip-norm helper scoped to params owned by `opt`. */
static double clip_grad_norm_opt(Optimizer* opt, double max_norm) {
    double total = 0;
    for (int i = 0; i < param_count_val; i++) {
        if (opt && !opt_owns_param(opt, i)) continue;
        Tensor* t = param_registry[i].tensor;
        if (!t->grad) continue;
        for (int j = 0; j < t->numel; j++) total += ((double*)t->grad)[j] * ((double*)t->grad)[j];
    }
    double norm = sqrt(total);
    if (norm > max_norm) {
        double scale = max_norm / norm;
        for (int i = 0; i < param_count_val; i++) {
            if (opt && !opt_owns_param(opt, i)) continue;
            Tensor* t = param_registry[i].tensor;
            if (!t->grad) continue;
            for (int j = 0; j < t->numel; j++) ((double*)t->grad)[j] *= scale;
        }
    }
    return norm;
}

/* Public global clippers retained for direct-FFI callers (backward compat). */
void optimizer_clip_grad_value(double max_val) {
    clip_grad_value_opt(NULL, max_val);
}

double optimizer_clip_grad_norm(double max_norm) {
    return clip_grad_norm_opt(NULL, max_norm);
}

/* ================================================================
   Optimizer buffer accessors (for serialization)
   ================================================================ */

int optimizer_buf_count(OptimizerHandle h) {
    (void)h;
    return param_count_val;
}

void optimizer_get_m(OptimizerHandle h, int idx, double* out) {
    Optimizer* opt = (Optimizer*)h;
    if (!opt->allocated) { memset(out, 0, param_registry[idx].tensor->numel * sizeof(double)); return; }
    int offset = param_element_offset(idx);
    int numel = param_registry[idx].tensor->numel;
    memcpy(out, opt->m + offset, numel * sizeof(double));
}

void optimizer_get_v(OptimizerHandle h, int idx, double* out) {
    Optimizer* opt = (Optimizer*)h;
    if (!opt->allocated) { memset(out, 0, param_registry[idx].tensor->numel * sizeof(double)); return; }
    int offset = param_element_offset(idx);
    int numel = param_registry[idx].tensor->numel;
    memcpy(out, opt->v + offset, numel * sizeof(double));
}

void optimizer_set_m(OptimizerHandle h, int idx, const double* data) {
    Optimizer* opt = (Optimizer*)h;
    optimizer_ensure_buffers(opt);
    int offset = param_element_offset(idx);
    int numel = param_registry[idx].tensor->numel;
    memcpy(opt->m + offset, data, numel * sizeof(double));
}

void optimizer_set_v(OptimizerHandle h, int idx, const double* data) {
    Optimizer* opt = (Optimizer*)h;
    optimizer_ensure_buffers(opt);
    int offset = param_element_offset(idx);
    int numel = param_registry[idx].tensor->numel;
    memcpy(opt->v + offset, data, numel * sizeof(double));
}

void optimizer_get_meta(OptimizerHandle h, double* out9) {
    Optimizer* opt = (Optimizer*)h;
    out9[0] = (double)opt->type;
    out9[1] = opt->lr;
    out9[2] = opt->beta1;
    out9[3] = opt->beta2;
    out9[4] = opt->eps;
    out9[5] = opt->alpha;
    out9[6] = opt->weight_decay;
    out9[7] = opt->momentum;
    out9[8] = (double)opt->t;
}

void optimizer_set_meta(OptimizerHandle h, const double* in9) {
    Optimizer* opt = (Optimizer*)h;
    opt->type = (int)in9[0];
    opt->lr = in9[1];
    opt->beta1 = in9[2];
    opt->beta2 = in9[3];
    opt->eps = in9[4];
    opt->alpha = in9[5];
    opt->weight_decay = in9[6];
    opt->momentum = in9[7];
    opt->t = (int)in9[8];
}

/* ================================================================
   System
   ================================================================ */

/* get_rss_mb / get_current_rss_mb live in shared_utils.c (compiled
 * once, unified symbol). Local callers in this file resolve them
 * via the unsuffixed names because both symbols are in the rename
 * header's EXCLUDE set. */

void backend_reset_for_eval(void) {
    tape_reset();
    /* Re-register params so they have valid tape indices */
    for (int j = 0; j < param_count_val; j++) {
        Tensor* t = param_registry[j].tensor;
        t->tape_idx = -1;
        if (t->grad) memset(t->grad, 0, t->numel * sizeof(double));
        tape_append(OP_CONST, t, NULL, NULL, 0);
    }
}

int tensor_live_count(int dummy) { (void)dummy; return (int)tape_size; }
int tensor_peak_live_count(int dummy) { (void)dummy; return (int)g_tape_peak; }

/* backend_memory_report and backend_supports_tensor_params removed
 * (no Idris-side callers). */

/* ================================================================
   Profiling
   ================================================================ */

void backend_epoch_begin(void) {
    double t = _wall_ms();
    prof_epoch_start = t;
    prof_op_t_prev = t;
}

void backend_profile_reset(void) {
    prof_forward_ms = prof_backward_ms = prof_optimizer_ms = 0;
    prof_forward_ops = prof_backward_ops = prof_epochs = 0;
    prof_epoch_start = 0;
    prof_op_t_prev = 0;
    prof_backward_processed = prof_backward_skipped = 0;
    memset(prof_backward_per_op, 0, sizeof(prof_backward_per_op));
    memset(prof_backward_count_per_op, 0, sizeof(prof_backward_count_per_op));
    memset(prof_forward_per_op, 0, sizeof(prof_forward_per_op));
    memset(prof_forward_count_per_op, 0, sizeof(prof_forward_count_per_op));
    memset(prof_kernel_per_op, 0, sizeof(prof_kernel_per_op));
    memset(prof_kernel_count_per_op, 0, sizeof(prof_kernel_count_per_op));
    memset(prof_binop_inside_ms, 0, sizeof(prof_binop_inside_ms));
    memset(prof_binop_inside_count, 0, sizeof(prof_binop_inside_count));
    memset(prof_binop_path_count, 0, sizeof(prof_binop_path_count));
    prof_binop_general_ms = 0;
}

static const char* op_name(int op) {
    static const char* names[] = {
        "CONST", "ADD", "SUB", "MUL", "DIV",
        "NEG", "ABS", "EXP", "LOG", "SQRT", "POW",
        "SIGMOID", "TANH",
        "MV", "LINEAR", "DOT", "OUTER",
        "SOFTMAX", "LOG_SOFTMAX",
        "SUM", "MEAN",
        "BCE_LOGITS",
        "LSTM_GATES",
        "ADD_S", "MUL_S", "CLAMP",
        "COS_SIM", "CONV1D_CIRC",
        "LSTM_CELL",
        "STACK", "RESHAPE", "SELECT", "VECMAT", "CAT", "NARROW",
        "LOG_SM_2D",
        "MM", "TRANS_2D", "SM_2D", "MASK_FILL", "LN_2D",
        "BMM", "BMM_3X3", "SM_3D", "TRANS_L2",
        "GELU", "GRU", "EMBED", "BATCH_NORM", "DROPOUT",
        "AVGP1D", "AVGP2D", "CONV1D", "MAXP1D", "CONV2D", "CONV2D_B", "MAXP2D", "MAXP2D_B",
        "CUMPROD", "GATHER", "SCATTER_ADD", "LEAKY_RELU", "SILU",
        "LINEAR_2D", "CONCAT_2D", "SOFTPLUS", "TILE_2D"
    };
    /* Compile-time check: names[] must cover every op tag.
       Add to BOTH this list and the enum when introducing new ops. */
    _Static_assert(sizeof(names)/sizeof(names[0]) == OP_COUNT,
                   "op_name names[] out of sync with OP_COUNT — add new ops here");
    if (op >= 0 && op < OP_COUNT) return names[op];
    return "???";
}

void backend_profile_report(void) {
    fprintf(stderr, "=== Profile Report ===\n");
    fprintf(stderr, "  Epochs: %d\n", prof_epochs);
    fprintf(stderr, "  Tape entries (last fwd): %d\n", tape_size);
    fprintf(stderr, "  Params: %d tensors, %d elements\n",
            param_count_val, ({int n=0; for(int i=0;i<param_count_val;i++) n+=param_registry[i].tensor->numel; n;}));
    fprintf(stderr, "  Forward:   %.1fms total (%.1fms/epoch)\n",
            prof_forward_ms, prof_epochs > 0 ? prof_forward_ms / prof_epochs : 0);
    fprintf(stderr, "  Backward:  %.1fms total (%.1fms/epoch)\n",
            prof_backward_ms, prof_epochs > 0 ? prof_backward_ms / prof_epochs : 0);
    fprintf(stderr, "  Optimizer: %.1fms total (%.1fms/epoch)\n",
            prof_optimizer_ms, prof_epochs > 0 ? prof_optimizer_ms / prof_epochs : 0);
    double total = prof_forward_ms + prof_backward_ms + prof_optimizer_ms;
    fprintf(stderr, "  C total:   %.1fms total (%.1fms/epoch)\n",
            total, prof_epochs > 0 ? total / prof_epochs : 0);
    /* Tape walk stats */
    int total_visited = prof_backward_processed + prof_backward_skipped;
    if (total_visited > 0) {
        fprintf(stderr, "  Backward walk: %d processed, %d skipped (%.0f%% dead)\n",
                prof_backward_processed, prof_backward_skipped,
                100.0 * prof_backward_skipped / total_visited);
    }
    /* Top-5 ops by backward time */
    fprintf(stderr, "  Top backward ops:\n");
    for (int rank = 0; rank < 5; rank++) {
        int best = -1;
        double best_time = 0;
        for (int j = 0; j < OP_COUNT; j++) {
            if (prof_backward_per_op[j] > best_time) {
                /* Skip already printed */
                int already = 0;
                for (int k = 0; k < rank; k++) {
                    /* Find k-th best again to skip it */
                    double kt = 0; int ki = -1;
                    for (int m = 0; m < OP_COUNT; m++) {
                        if (prof_backward_per_op[m] > kt) { kt = prof_backward_per_op[m]; ki = m; }
                    }
                    /* This naive approach doesn't work for rank>0. Use simpler method. */
                    (void)kt; (void)ki;
                }
                (void)already;
                best = j; best_time = prof_backward_per_op[j];
            }
        }
        if (best < 0 || best_time < 0.001) break;
        fprintf(stderr, "    %-12s %.2fms (%d calls)\n",
                op_name(best), best_time, prof_backward_count_per_op[best]);
        prof_backward_per_op[best] = -1; /* mark as printed (will be reset on next profile_reset) */
    }
    /* Top-10 ops by forward time (broader than backward — more ops contribute) */
    fprintf(stderr, "  Top forward ops:\n");
    for (int rank = 0; rank < 10; rank++) {
        int best = -1;
        double best_time = 0;
        for (int j = 0; j < OP_COUNT; j++) {
            if (prof_forward_per_op[j] > best_time) {
                best = j; best_time = prof_forward_per_op[j];
            }
        }
        if (best < 0 || best_time < 0.001) break;
        int n = prof_forward_count_per_op[best];
        double per_call_us = n > 0 ? (best_time * 1000.0 / n) : 0.0;
        fprintf(stderr, "    %-12s %.2fms (%d calls, %.2f us/call)\n",
                op_name(best), best_time, n, per_call_us);
        prof_forward_per_op[best] = -1; /* mark as printed */
    }
    /* Kernel-only timer (elementwise vDSP path only, today). Shows the
       actual kernel time independent of the tape_append attribution. */
    int any_kernel = 0;
    for (int j = 0; j < OP_COUNT; j++) {
        if (prof_kernel_count_per_op[j] > 0) { any_kernel = 1; break; }
    }
    if (any_kernel) {
        fprintf(stderr, "  Direct-kernel timing (subset of ops):\n");
        for (int j = 0; j < OP_COUNT; j++) {
            int n = prof_kernel_count_per_op[j];
            if (n == 0) continue;
            double per_call_us = prof_kernel_per_op[j] * 1000.0 / n;
            fprintf(stderr, "    %-12s %.2fms (%d calls, %.2f us/call) [kernel only]\n",
                    op_name(j), prof_kernel_per_op[j], n, per_call_us);
        }
    }
    fprintf(stderr, "  binop_elementwise paths: fast=%d scalar_bcast=%d general_bcast=%d  general_bcast_total=%.2fms\n",
            prof_binop_path_count[0], prof_binop_path_count[1],
            prof_binop_path_count[2], prof_binop_general_ms);
    int any_inside = 0;
    for (int j = 0; j < OP_COUNT; j++) {
        if (prof_binop_inside_count[j] > 0) { any_inside = 1; break; }
    }
    if (any_inside) {
        fprintf(stderr, "  binop_elementwise inside (entry-to-exit):\n");
        for (int j = 0; j < OP_COUNT; j++) {
            int n = prof_binop_inside_count[j];
            if (n == 0) continue;
            double per_call_us = prof_binop_inside_ms[j] * 1000.0 / n;
            fprintf(stderr, "    %-12s %.2fms (%d calls, %.2f us/call) [in-function]\n",
                    op_name(j), prof_binop_inside_ms[j], n, per_call_us);
        }
    }
}

/* ================================================================
   Debug
   ================================================================ */

const char* backend_name(void) { return "tape"; }

/* Job 3 Phase B — mx::compile is mlx-only; tape backend always reports
   disabled regardless of MLX_COMPILE env var. */
int  tensor_mlx_compile_enabled(void) { return 0; }
int  tensor_mlx_compile_invocations(void) { return 0; }
void tensor_mlx_compile_reset_stats(void) { }

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

/* ================================================================
   Portable FFI helpers (for RefC compatibility)
   ================================================================ */

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

/* tensor_write_double_return / tensor_ptr_array_set_return /
 * tensor_alloc_ints / tensor_free_ints / tensor_write_int_return
 * live in shared_utils.c. */

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

double native_train_step(OptimizerHandle opt, int clip_mode, double clip_val,
                         TensorHandle loss_ptr, double loss_val) {
    Tensor* loss = (Tensor*)loss_ptr;
    optimizer_zero_grad(opt);
    if (loss->requires_grad) tensor_backward(loss_ptr);
    /* Scope grad-clipping to this optimizer's owned params, so multi-
     * optimizer setups (SAC actor/q1/q2) each clip their own norm. */
    if (clip_mode == 1) clip_grad_value_opt((Optimizer*)opt, clip_val);
    else if (clip_mode == 2) clip_grad_norm_opt((Optimizer*)opt, clip_val);
    optimizer_step(opt);
    return loss_val;
}

int optimizer_step_with_clip(OptimizerHandle opt, int clip_mode, double clip_val, int dummy) {
    (void)dummy;
    if (clip_mode == 1) clip_grad_value_opt((Optimizer*)opt, clip_val);
    else if (clip_mode == 2) clip_grad_norm_opt((Optimizer*)opt, clip_val);
    optimizer_step(opt);
    optimizer_zero_grad(opt);
    return 0;
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



/* ---- Unified dtag-dispatch create/cast entry points ----
   One symbol per shape, dtag-keyed, superseding the per-dtype
   *_f32_streamed / *_f64_streamed wrappers. dtag 1 (F64) is the hot path —
   plain F64 create, untouched. dtag 0 (F32) allocates real float storage
   so the F32 elementwise kernels can read `(float*)t->data` directly.
   Every other dtag (BF16, F16, I8/I16/I32/I64, U8, Bool) is inference-only: builds the
   tensor as F64 then rounds each value through the target dtype, leaving
   the storage as a double buffer (Phase 2 `tape_retag_round` lingua
   franca). The Idris `Compatible` gate still keeps tape at F64 + F32
   trainable; the inference dtypes wait for Phase 4 to open. */
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

/* Under the kind-major dtag layout (closed 2026-05-23): dtag 15 = F64 (the
   lingua franca, fast-path to the f64 creator), dtag 14 = F32 (real 4-byte
   float storage). All other valid dtags (Bool=1, U8=4, I8/I16/I32/I64,
   F16=13, BF16=17) route through tape_retag_round's lingua-franca path
   (store doubles in F64 layout, retag to the inference dtype). Invalid
   dtags abort via tape_tag_from_dtag's default arm. */
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
    /* F32 target: real F32 storage (4 bytes/elem), matching the Phase 3
       streamed-create path. Skipping this and falling through to
       tape_retag_round would produce a lingua-franca F32 (double storage,
       DT_F32 tag) — internally consistent for tensor_item_1d (reads as
       double*) but garbage for tensor_to_doubles / tape_load_d / the F32
       kernels (all assume 4-byte-per-elem float storage). tape_load_d on
       the read side normalizes real-F32 vs lingua-franca sources. */
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
