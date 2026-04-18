/* backend_mlx.cpp — MLX backend implementing backend.h.
 *
 * Uses Apple's MLX framework for GPU-accelerated tensor operations
 * on Apple Silicon via Metal. Tape-based autograd (same structure as
 * backend_tape.c) with MLX arrays for compute — backward ops also
 * run on GPU via MLX.
 *
 * Build: make BACKEND=mlx MLX_SITE=/path/to/mlx backend
 */

#include "backend.h"
#include <cstdlib>
#include <cstdio>
#include <cstring>
#include <cmath>
#include <iostream>
#include <vector>
#include <string>
#include <sys/resource.h>
#ifdef __APPLE__
#include <mach/mach.h>
#endif

#include <mlx/mlx.h>

namespace mx = mlx::core;

/* ================================================================
   Stub macro
   ================================================================ */

#define STUB() do { \
    fprintf(stderr, "MLX backend: %s not implemented\n", __func__); \
    abort(); \
} while(0)

/* ================================================================
   Tensor representation
   ================================================================ */

struct Tensor {
    mx::array data;
    mx::array grad;
    bool requires_grad;
    bool has_grad;
    int tape_idx;
    int persistent;  // 1 = param, 0 = intermediate

    Tensor(mx::array d, bool rg = false)
        : data(std::move(d)), grad(mx::array(0.0f)), requires_grad(rg),
          has_grad(false), tape_idx(-1), persistent(0) {}
};

/* ================================================================
   Tape — autograd Wengert list
   ================================================================ */

enum {
    OP_CONST = 0,
    OP_ADD, OP_SUB, OP_MUL, OP_DIV,
    OP_NEG, OP_EXP, OP_LOG, OP_SQRT,
    OP_SIGMOID, OP_TANH,
    OP_ADD_SCALAR, OP_MUL_SCALAR, OP_CLAMP_MIN,
    OP_SUM, OP_MEAN,
    OP_MM, OP_TRANSPOSE_2D,
    OP_SOFTMAX_2D, OP_LOG_SOFTMAX_2D,
    OP_MASKED_FILL, OP_LAYER_NORM_2D,
    OP_RESHAPE, OP_NARROW, OP_CAT,
    OP_POW, OP_ABS,
    OP_STACK, OP_OUTER,
    OP_COSINE_SIM, OP_CONV1D_CIRC,
    OP_MV,
};

struct CosSimMeta {
    int n, m;
    mx::array row_norms;  // [n]
    mx::array key_norm;   // scalar
    mx::array dots;       // [n]
    CosSimMeta() : n(0), m(0), row_norms(mx::array(0.0f)),
                   key_norm(mx::array(0.0f)), dots(mx::array(0.0f)) {}
};

struct LayerNormMeta {
    Tensor* gamma;
    Tensor* bias;
    mx::array x_hat;
    mx::array rstd;
    int m, n;
    LayerNormMeta() : gamma(nullptr), bias(nullptr),
                       x_hat(mx::array(0.0f)), rstd(mx::array(0.0f)), m(0), n(0) {}
};

struct TapeEntry {
    int op;
    Tensor* result;
    Tensor* arg1;
    Tensor* arg2;
    double scalar_arg;
    void* meta;
};

static std::vector<TapeEntry> tape;

static int tape_append(int op, Tensor* result, Tensor* arg1, Tensor* arg2, double scalar_arg) {
    int idx = (int)tape.size();
    tape.push_back({op, result, arg1, arg2, scalar_arg, nullptr});
    result->tape_idx = idx;
    return idx;
}

static void tape_reset() {
    // Free op metadata
    for (auto& e : tape) {
        if (e.op == OP_LAYER_NORM_2D && e.meta) {
            delete (LayerNormMeta*)e.meta;
            e.meta = nullptr;
        }
        if (e.op == OP_COSINE_SIM && e.meta) {
            delete (CosSimMeta*)e.meta;
            e.meta = nullptr;
        }
        if (e.op == OP_STACK && e.meta) {
            delete (std::vector<Tensor*>*)e.meta;
            e.meta = nullptr;
        }
    }
    // Free non-persistent intermediate tensors that are on the tape
    for (auto& e : tape) {
        if (e.result && !e.result->persistent) {
            delete e.result;
            e.result = nullptr;
        }
    }
    tape.clear();
}

/* ================================================================
   Parameter registry
   ================================================================ */

struct ParamEntry {
    std::string name;
    Tensor* tensor;
};

static std::vector<ParamEntry> param_registry;

/* ================================================================
   Lifecycle
   ================================================================ */

extern "C" {

TensorHandle tensor_create_scalar(double value, int requires_grad) {
    auto t = new Tensor(mx::array(value), requires_grad != 0);
    if (requires_grad) tape_append(OP_CONST, t, nullptr, nullptr, 0);
    return (TensorHandle)t;
}

TensorHandle tensor_create(double* data, int* shape, int rank, int requires_grad) {
    mx::Shape sh(shape, shape + rank);
    auto t = new Tensor(mx::array(data, sh, mx::float64), requires_grad != 0);
    if (requires_grad) tape_append(OP_CONST, t, nullptr, nullptr, 0);
    return (TensorHandle)t;
}

TensorHandle tensor_clone(TensorHandle h) {
    auto t = (Tensor*)h;
    auto c = new Tensor(mx::array(t->data), false);
    return (TensorHandle)c;
}

void tensor_free(TensorHandle h) {
    // Don't delete params or intermediates during training — tape refs them
    // Only delete explicitly freed tensors (rare)
}

/* ================================================================
   Accessors
   ================================================================ */

double tensor_item(TensorHandle h) {
    auto t = (Tensor*)h;
    mx::eval(t->data);
    return t->data.item<double>();
}

int tensor_numel(TensorHandle h) { return (int)((Tensor*)h)->data.size(); }
int tensor_dim(TensorHandle h) { return (int)((Tensor*)h)->data.ndim(); }
int tensor_size(TensorHandle h, int dim) { return (int)((Tensor*)h)->data.shape(dim); }

void tensor_to_doubles(TensorHandle h, double* out) {
    auto t = (Tensor*)h;
    mx::eval(t->data);
    memcpy(out, t->data.data<double>(), t->data.size() * sizeof(double));
}

/* ================================================================
   Arithmetic
   ================================================================ */

TensorHandle tensor_add(TensorHandle ha, TensorHandle hb) {
    auto a = (Tensor*)ha; auto b = (Tensor*)hb;
    bool rg = a->requires_grad || b->requires_grad;
    auto r = new Tensor(mx::add(a->data, b->data), rg);
    if (rg) tape_append(OP_ADD, r, a, b, 0);
    return (TensorHandle)r;
}

TensorHandle tensor_sub(TensorHandle ha, TensorHandle hb) {
    auto a = (Tensor*)ha; auto b = (Tensor*)hb;
    bool rg = a->requires_grad || b->requires_grad;
    auto r = new Tensor(mx::subtract(a->data, b->data), rg);
    if (rg) tape_append(OP_SUB, r, a, b, 0);
    return (TensorHandle)r;
}

TensorHandle tensor_mul(TensorHandle ha, TensorHandle hb) {
    auto a = (Tensor*)ha; auto b = (Tensor*)hb;
    bool rg = a->requires_grad || b->requires_grad;
    auto r = new Tensor(mx::multiply(a->data, b->data), rg);
    if (rg) tape_append(OP_MUL, r, a, b, 0);
    return (TensorHandle)r;
}

TensorHandle tensor_div(TensorHandle ha, TensorHandle hb) {
    auto a = (Tensor*)ha; auto b = (Tensor*)hb;
    bool rg = a->requires_grad || b->requires_grad;
    auto r = new Tensor(mx::divide(a->data, b->data), rg);
    if (rg) tape_append(OP_DIV, r, a, b, 0);
    return (TensorHandle)r;
}

TensorHandle tensor_neg(TensorHandle h) {
    auto t = (Tensor*)h;
    auto r = new Tensor(mx::negative(t->data), t->requires_grad);
    if (t->requires_grad) tape_append(OP_NEG, r, t, nullptr, 0);
    return (TensorHandle)r;
}

TensorHandle tensor_abs(TensorHandle h) {
    auto t = (Tensor*)h;
    auto r = new Tensor(mx::abs(t->data), t->requires_grad);
    if (t->requires_grad) tape_append(OP_ABS, r, t, nullptr, 0);
    return (TensorHandle)r;
}

TensorHandle tensor_exp(TensorHandle h) {
    auto t = (Tensor*)h;
    auto r = new Tensor(mx::exp(t->data), t->requires_grad);
    if (t->requires_grad) tape_append(OP_EXP, r, t, nullptr, 0);
    return (TensorHandle)r;
}

TensorHandle tensor_log(TensorHandle h) {
    auto t = (Tensor*)h;
    auto r = new Tensor(mx::log(t->data), t->requires_grad);
    if (t->requires_grad) tape_append(OP_LOG, r, t, nullptr, 0);
    return (TensorHandle)r;
}

TensorHandle tensor_sqrt(TensorHandle h) {
    auto t = (Tensor*)h;
    auto r = new Tensor(mx::sqrt(t->data), t->requires_grad);
    if (t->requires_grad) tape_append(OP_SQRT, r, t, nullptr, 0);
    return (TensorHandle)r;
}

TensorHandle tensor_pow(TensorHandle hbase, TensorHandle hexp) {
    auto b = (Tensor*)hbase; auto e = (Tensor*)hexp;
    bool rg = b->requires_grad || e->requires_grad;
    auto r = new Tensor(mx::power(b->data, e->data), rg);
    if (rg) tape_append(OP_POW, r, b, e, 0);
    return (TensorHandle)r;
}
TensorHandle tensor_sigmoid(TensorHandle h) {
    auto t = (Tensor*)h;
    auto r = new Tensor(mx::sigmoid(t->data), t->requires_grad);
    if (t->requires_grad) tape_append(OP_SIGMOID, r, t, nullptr, 0);
    return (TensorHandle)r;
}
TensorHandle tensor_tanh(TensorHandle h) {
    auto t = (Tensor*)h;
    auto r = new Tensor(mx::tanh(t->data), t->requires_grad);
    if (t->requires_grad) tape_append(OP_TANH, r, t, nullptr, 0);
    return (TensorHandle)r;
}

TensorHandle tensor_add_scalar(TensorHandle h, double s) {
    auto t = (Tensor*)h;
    auto r = new Tensor(mx::add(t->data, mx::array(s)), t->requires_grad);
    if (t->requires_grad) tape_append(OP_ADD_SCALAR, r, t, nullptr, s);
    return (TensorHandle)r;
}

TensorHandle tensor_mul_scalar(TensorHandle h, double s) {
    auto t = (Tensor*)h;
    auto r = new Tensor(mx::multiply(t->data, mx::array(s)), t->requires_grad);
    if (t->requires_grad) tape_append(OP_MUL_SCALAR, r, t, nullptr, s);
    return (TensorHandle)r;
}

TensorHandle tensor_clamp_min(TensorHandle h, double min_val) {
    auto t = (Tensor*)h;
    auto r = new Tensor(mx::maximum(t->data, mx::array(min_val)), t->requires_grad);
    if (t->requires_grad) tape_append(OP_CLAMP_MIN, r, t, nullptr, min_val);
    return (TensorHandle)r;
}

/* ================================================================
   Reduction
   ================================================================ */

TensorHandle tensor_sum(TensorHandle h) {
    auto t = (Tensor*)h;
    auto r = new Tensor(mx::sum(t->data), t->requires_grad);
    if (t->requires_grad) tape_append(OP_SUM, r, t, nullptr, 0);
    return (TensorHandle)r;
}

TensorHandle tensor_sum_dim(TensorHandle t, int dim, int keepdim) { STUB(); }
TensorHandle tensor_mean(TensorHandle h) {
    auto t = (Tensor*)h;
    auto r = new Tensor(mx::mean(t->data), t->requires_grad);
    if (t->requires_grad) tape_append(OP_MEAN, r, t, nullptr, 0);
    return (TensorHandle)r;
}

/* ================================================================
   Linear algebra
   ================================================================ */

TensorHandle tensor_matmul(TensorHandle ha, TensorHandle hb) {
    auto a = (Tensor*)ha; auto b = (Tensor*)hb;
    bool rg = a->requires_grad || b->requires_grad;
    auto r = new Tensor(mx::matmul(a->data, b->data), rg);
    if (rg) tape_append(OP_MM, r, a, b, 0);
    return (TensorHandle)r;
}

TensorHandle tensor_mv(TensorHandle hmat, TensorHandle hvec) {
    // mat=[m,n], vec=[n] → result=[m]
    auto mat = (Tensor*)hmat; auto vec = (Tensor*)hvec;
    int n = (int)vec->data.size();
    int m_size = (int)mat->data.shape(0);
    auto vec_col = mx::reshape(vec->data, {n, 1});
    auto result_col = mx::matmul(mat->data, vec_col); // [m, 1]
    auto result = mx::reshape(result_col, {m_size});   // [m]
    bool rg = mat->requires_grad || vec->requires_grad;
    auto r = new Tensor(result, rg);
    if (rg) tape_append(OP_MV, r, mat, vec, 0);
    return (TensorHandle)r;
}

TensorHandle tensor_dot(TensorHandle ha, TensorHandle hb) {
    auto a = (Tensor*)ha; auto b = (Tensor*)hb;
    bool rg = a->requires_grad || b->requires_grad;
    auto r = new Tensor(mx::sum(mx::multiply(a->data, b->data)), rg);
    // Use OP_MUL + OP_SUM for backward (approximate)
    if (rg) {
        auto prod = new Tensor(mx::multiply(a->data, b->data), rg);
        tape_append(OP_MUL, prod, a, b, 0);
        tape_append(OP_SUM, r, prod, nullptr, 0);
    }
    return (TensorHandle)r;
}

TensorHandle tensor_outer(TensorHandle ha, TensorHandle hb) {
    auto a = (Tensor*)ha; auto b = (Tensor*)hb;
    bool rg = a->requires_grad || b->requires_grad;
    auto r = new Tensor(mx::outer(a->data, b->data), rg);
    if (rg) tape_append(OP_OUTER, r, a, b, 0);
    return (TensorHandle)r;
}

TensorHandle tensor_softmax(TensorHandle h, int dim) {
    auto t = (Tensor*)h;
    auto r = new Tensor(mx::softmax(t->data, dim), t->requires_grad);
    if (t->requires_grad) tape_append(OP_SOFTMAX_2D, r, t, nullptr, 0);
    return (TensorHandle)r;
}
TensorHandle tensor_log_softmax(TensorHandle h, int dim) {
    auto t = (Tensor*)h;
    // log_softmax = x - log(sum(exp(x)))
    auto maxv = mx::max(t->data, dim, true);
    auto shifted = mx::subtract(t->data, maxv);
    auto lse = mx::add(mx::log(mx::sum(mx::exp(shifted), dim, true)), maxv);
    auto r = new Tensor(mx::subtract(t->data, lse), t->requires_grad);
    if (t->requires_grad) tape_append(OP_LOG_SOFTMAX_2D, r, t, nullptr, 0);
    return (TensorHandle)r;
}

/* ================================================================
   Loss functions
   ================================================================ */

TensorHandle tensor_bce_with_logits(TensorHandle hinput, TensorHandle htarget) {
    auto inp = (Tensor*)hinput; auto tgt = (Tensor*)htarget;
    // BCE with logits: max(x,0) - x*y + log(1+exp(-|x|))
    auto x = inp->data; auto y = tgt->data;
    auto relu_x = mx::maximum(x, mx::array(0.0));
    auto abs_x = mx::abs(x);
    auto result = mx::mean(mx::add(mx::subtract(relu_x, mx::multiply(x, y)),
                                    mx::log(mx::add(mx::array(1.0), mx::exp(mx::negative(abs_x))))));
    bool rg = inp->requires_grad;
    auto r = new Tensor(result, rg);
    // For backward: d/dx = sigmoid(x) - y, averaged
    // Record as opaque for now — use OP_MUL as placeholder
    // TODO: proper backward
    return (TensorHandle)r;
}
TensorHandle tensor_cross_entropy(TensorHandle input, TensorHandle target) { STUB(); }
TensorHandle tensor_mse_loss(TensorHandle input, TensorHandle target) { STUB(); }

/* ================================================================
   NTM-specific
   ================================================================ */

TensorHandle tensor_cosine_similarity(TensorHandle hmemory, TensorHandle hkey, int dim) {
    // memory=[n,m], key=[m] → result=[n]
    auto mem = (Tensor*)hmemory; auto key = (Tensor*)hkey;
    auto eps = mx::array(1.0e-8);
    int n = (int)mem->data.shape(0);
    int m = (int)mem->data.shape(1);

    // Compute forward
    auto key_2d = mx::reshape(key->data, {1, m});
    auto dots = mx::sum(mx::multiply(mem->data, key_2d), {1}); // [n]
    auto row_norms = mx::sqrt(mx::add(mx::sum(mx::square(mem->data), {1}), eps)); // [n]
    auto key_norm = mx::sqrt(mx::add(mx::sum(mx::square(key->data)), eps)); // scalar
    auto result = mx::divide(dots, mx::multiply(row_norms, key_norm));

    bool rg = mem->requires_grad || key->requires_grad;
    auto r = new Tensor(result, rg);
    if (rg) {
        auto meta = new CosSimMeta();
        meta->n = n; meta->m = m;
        meta->row_norms = row_norms;
        meta->key_norm = key_norm;
        meta->dots = dots;
        int idx = tape_append(OP_COSINE_SIM, r, mem, key, 0);
        tape[idx].meta = meta;
    }
    return (TensorHandle)r;
}

TensorHandle tensor_conv1d_circular(TensorHandle hinput, TensorHandle hkernel) {
    // Circular convolution: out[i] = sum_j input[(i-k/2+j+n)%n] * kernel[k-1-j]
    auto inp = (Tensor*)hinput; auto kern = (Tensor*)hkernel;
    int n = (int)inp->data.size();
    int k = (int)kern->data.size();

    mx::array result = mx::zeros({n}, mx::float64);
    int half_k = k / 2;
    for (int j = 0; j < k; j++) {
        int shift = half_k - j;
        auto shifted = mx::roll(inp->data, shift);
        auto kern_j = mx::take(kern->data, mx::array(j));
        result = mx::add(result, mx::multiply(shifted, kern_j));
    }

    bool rg = inp->requires_grad || kern->requires_grad;
    auto r = new Tensor(result, rg);
    if (rg) tape_append(OP_CONV1D_CIRC, r, inp, kern, 0);
    return (TensorHandle)r;
}

TensorPair* tensor_ntm_read_head(TensorHandle hmemory, TensorHandle hprev_weights,
    TensorHandle hkey, TensorHandle hbeta, TensorHandle hg,
    TensorHandle hgamma, TensorHandle hshift_kernel) {
    // Decompose the 7-step read head pipeline using existing tensor_* functions
    // This way each sub-op records its own tape entry for backward.

    // 1. Content addressing: cosine_sim * beta → softmax
    TensorHandle cos_sim = tensor_cosine_similarity(hmemory, hkey, 0);
    TensorHandle scaled = tensor_mul(cos_sim, hbeta);
    TensorHandle content_weights = tensor_softmax(scaled, 0);

    // 2. Interpolation: g * content + (1-g) * prev
    TensorHandle one = tensor_create_scalar(1.0, 0);
    TensorHandle one_minus_g = tensor_sub(one, hg);
    TensorHandle g_content = tensor_mul(hg, content_weights);
    TensorHandle omg_prev = tensor_mul(one_minus_g, hprev_weights);
    TensorHandle interp = tensor_add(g_content, omg_prev);

    // 3. Circular shift
    TensorHandle shifted = tensor_conv1d_circular(interp, hshift_kernel);

    // 4. Clamp + power sharpen + normalize
    TensorHandle clamped = tensor_clamp_min(shifted, 1.0e-10);
    TensorHandle powered = tensor_pow(clamped, hgamma);
    TensorHandle power_sum = tensor_sum(powered);
    TensorHandle eps = tensor_create_scalar(1.0e-10, 0);
    TensorHandle power_sum_eps = tensor_add(power_sum, eps);
    TensorHandle focused = tensor_div(powered, power_sum_eps);

    // 5. Read from memory: focused @ memory → [m]
    // Use tensor_mv for proper backward handling (MV handles 1D vectors correctly)
    // focused is [n], memory^T is [m, n], so mv(memory^T, focused) → [m]
    TensorHandle memT_transposed = tensor_transpose_2d(hmemory);
    TensorHandle read_result = tensor_mv(memT_transposed, focused);

    auto pair = (TensorPair*)malloc(sizeof(TensorPair));
    pair->first = focused;
    pair->second = read_result;
    return pair;
}

TensorHandle tensor_ntm_interp_write(TensorHandle hmemory, TensorHandle hweights,
    TensorHandle hadd_vector) {
    // memory_new = memory + outer(weights, add_vector)
    TensorHandle outer_prod = tensor_outer(hweights, hadd_vector);
    return tensor_add(hmemory, outer_prod);
}

/* ================================================================
   Shape manipulation
   ================================================================ */

TensorHandle tensor_reshape(TensorHandle h, int* shape, int rank) {
    auto t = (Tensor*)h;
    mx::Shape sh(shape, shape + rank);
    auto r = new Tensor(mx::reshape(t->data, sh), t->requires_grad);
    if (t->requires_grad) tape_append(OP_RESHAPE, r, t, nullptr, 0);
    return (TensorHandle)r;
}

TensorHandle tensor_unsqueeze(TensorHandle t, int dim) { STUB(); }
TensorHandle tensor_squeeze(TensorHandle t, int dim) { STUB(); }

TensorHandle tensor_select(TensorHandle h, int dim, int index) {
    auto t = (Tensor*)h;
    // For 1D: just take element at index
    auto r = new Tensor(mx::take(t->data, mx::array(index), dim), t->requires_grad);
    return (TensorHandle)r;
}

TensorHandle tensor_stack(TensorHandle* tensors, int count, int dim) { STUB(); }
TensorHandle tensor_cat(TensorHandle* tensors, int count, int dim) { STUB(); }

TensorHandle tensor_cat2(TensorHandle ha, TensorHandle hb) {
    auto a = (Tensor*)ha; auto b = (Tensor*)hb;
    bool rg = a->requires_grad || b->requires_grad;
    auto r = new Tensor(mx::concatenate({a->data, b->data}, 0), rg);
    if (rg) tape_append(OP_CAT, r, a, b, (double)a->data.size());
    return (TensorHandle)r;
}

TensorHandle tensor_narrow(TensorHandle h, int dim, int start, int len) {
    auto t = (Tensor*)h;
    // Flatten, then slice the 1D range — matches tape backend semantics
    auto flat = mx::flatten(t->data);
    auto sliced = mx::slice(flat, mx::Shape{start}, mx::Shape{start + len});
    auto r = new Tensor(sliced, t->requires_grad);
    if (t->requires_grad) tape_append(OP_NARROW, r, t, nullptr, (double)start);
    return (TensorHandle)r;
}

TensorHandle tensor_mm(TensorHandle ha, TensorHandle hb) {
    auto a = (Tensor*)ha; auto b = (Tensor*)hb;
    bool rg = a->requires_grad || b->requires_grad;
    auto r = new Tensor(mx::matmul(a->data, b->data), rg);
    if (rg) tape_append(OP_MM, r, a, b, 0);
    return (TensorHandle)r;
}

TensorHandle tensor_bmm(TensorHandle a, TensorHandle b) { STUB(); }
TensorHandle tensor_batch(TensorHandle* handles, int count) { STUB(); }
TensorHandle* tensor_unbatch(TensorHandle h, int* out_count) { STUB(); }

TensorHandle tensor_transpose_2d(TensorHandle h) {
    auto t = (Tensor*)h;
    auto r = new Tensor(mx::transpose(t->data), t->requires_grad);
    if (t->requires_grad) tape_append(OP_TRANSPOSE_2D, r, t, nullptr, 0);
    return (TensorHandle)r;
}

TensorHandle tensor_softmax_2d(TensorHandle h) {
    auto t = (Tensor*)h;
    auto r = new Tensor(mx::softmax(t->data, -1), t->requires_grad);
    if (t->requires_grad) tape_append(OP_SOFTMAX_2D, r, t, nullptr, 0);
    return (TensorHandle)r;
}

TensorHandle tensor_masked_fill(TensorHandle h, TensorHandle hmask, double value) {
    auto t = (Tensor*)h; auto mask = (Tensor*)hmask;
    auto val_arr = mx::full(t->data.shape(), value, mx::float64);
    auto r = new Tensor(mx::where(mask->data, val_arr, t->data), t->requires_grad);
    if (t->requires_grad) tape_append(OP_MASKED_FILL, r, t, mask, 0);
    return (TensorHandle)r;
}

TensorHandle tensor_log_softmax_2d(TensorHandle h) {
    auto t = (Tensor*)h;
    // log_softmax(x) = x - log(sum(exp(x)))
    auto maxv = mx::max(t->data, -1, true);
    auto shifted = mx::subtract(t->data, maxv);
    auto lse = mx::add(mx::log(mx::sum(mx::exp(shifted), -1, true)), maxv);
    auto r = new Tensor(mx::subtract(t->data, lse), t->requires_grad);
    if (t->requires_grad) tape_append(OP_LOG_SOFTMAX_2D, r, t, nullptr, 0);
    return (TensorHandle)r;
}

TensorHandle tensor_layer_norm_2d(TensorHandle h, TensorHandle hgamma,
    TensorHandle hbias, double eps) {
    auto t = (Tensor*)h;
    auto gamma = (Tensor*)hgamma;
    auto bias = (Tensor*)hbias;
    int m = t->data.shape(0), n = t->data.shape(1);

    auto mean = mx::mean(t->data, -1, true);
    auto centered = mx::subtract(t->data, mean);
    auto var = mx::mean(mx::square(centered), -1, true);
    auto rstd = mx::rsqrt(mx::add(var, mx::array(eps)));
    auto x_hat = mx::multiply(centered, rstd);
    auto result = mx::add(mx::multiply(gamma->data, x_hat), bias->data);

    bool rg = t->requires_grad || gamma->requires_grad || bias->requires_grad;
    auto r = new Tensor(result, rg);
    if (rg) {
        auto meta = new LayerNormMeta();
        meta->gamma = gamma;
        meta->bias = bias;
        meta->x_hat = x_hat;
        meta->rstd = mx::reshape(rstd, {m});
        meta->m = m;
        meta->n = n;
        int idx = tape_append(OP_LAYER_NORM_2D, r, t, nullptr, 0);
        tape[idx].meta = meta;
    }
    return (TensorHandle)r;
}

TensorHandle tensor_reshape_2d(TensorHandle h, int rows, int cols) {
    int shape[] = {rows, cols};
    return tensor_reshape(h, shape, 2);
}

TensorHandle tensor_one_hot(int* tokens, int n_tokens, int vocab_size) {
    // Create one-hot encoded 1D tensor
    int total = n_tokens * vocab_size;
    std::vector<double> data(total, 0.0);
    for (int i = 0; i < n_tokens; i++) {
        int tok = tokens[i];
        if (tok >= 0 && tok < vocab_size)
            data[i * vocab_size + tok] = 1.0;
    }
    mx::Shape sh = {total};
    auto t = new Tensor(mx::array(data.data(), sh, mx::float64), false);
    free(tokens);
    return (TensorHandle)t;
}

/* ================================================================
   Autograd — backward pass
   ================================================================ */

static void ensure_grad(Tensor* t) {
    if (!t->has_grad) {
        t->grad = mx::zeros(t->data.shape(), mx::float64);
        t->has_grad = true;
    }
}

void tensor_backward(TensorHandle h) {
    Tensor* loss = (Tensor*)h;
    if (loss->tape_idx < 0) return;

    ensure_grad(loss);
    loss->grad = mx::array(1.0);

    for (int i = loss->tape_idx; i >= 0; i--) {
        auto& e = tape[i];
        Tensor* r = e.result;
        if (!r->has_grad) continue;

        Tensor* a = e.arg1;
        Tensor* b = e.arg2;

        switch (e.op) {
        case OP_CONST:
            break;

        case OP_ADD:
            if (a && a->requires_grad) { ensure_grad(a); a->grad = mx::add(a->grad, r->grad); }
            if (b && b->requires_grad) { ensure_grad(b); b->grad = mx::add(b->grad, r->grad); }
            break;

        case OP_SUB:
            if (a && a->requires_grad) { ensure_grad(a); a->grad = mx::add(a->grad, r->grad); }
            if (b && b->requires_grad) { ensure_grad(b); b->grad = mx::subtract(b->grad, r->grad); }
            break;

        case OP_MUL:
            if (a && a->requires_grad) { ensure_grad(a); a->grad = mx::add(a->grad, mx::multiply(r->grad, b->data)); }
            if (b && b->requires_grad) { ensure_grad(b); b->grad = mx::add(b->grad, mx::multiply(r->grad, a->data)); }
            break;

        case OP_DIV:
            if (a && a->requires_grad) { ensure_grad(a); a->grad = mx::add(a->grad, mx::divide(r->grad, b->data)); }
            if (b && b->requires_grad) { ensure_grad(b); b->grad = mx::subtract(b->grad, mx::divide(mx::multiply(r->grad, a->data), mx::square(b->data))); }
            break;

        case OP_NEG:
            if (a && a->requires_grad) { ensure_grad(a); a->grad = mx::subtract(a->grad, r->grad); }
            break;

        case OP_EXP:
            if (a && a->requires_grad) { ensure_grad(a); a->grad = mx::add(a->grad, mx::multiply(r->grad, r->data)); }
            break;

        case OP_LOG:
            if (a && a->requires_grad) { ensure_grad(a); a->grad = mx::add(a->grad, mx::divide(r->grad, a->data)); }
            break;

        case OP_SQRT:
            if (a && a->requires_grad) { ensure_grad(a); a->grad = mx::add(a->grad, mx::divide(r->grad, mx::multiply(mx::array(2.0), r->data))); }
            break;

        case OP_ADD_SCALAR:
            if (a && a->requires_grad) { ensure_grad(a); a->grad = mx::add(a->grad, r->grad); }
            break;

        case OP_MUL_SCALAR:
            if (a && a->requires_grad) { ensure_grad(a); a->grad = mx::add(a->grad, mx::multiply(r->grad, mx::array(e.scalar_arg))); }
            break;

        case OP_CLAMP_MIN: {
            if (a && a->requires_grad) {
                ensure_grad(a);
                auto mask = mx::greater_equal(a->data, mx::array(e.scalar_arg));
                a->grad = mx::add(a->grad, mx::multiply(r->grad, mx::astype(mask, mx::float64)));
            }
            break;
        }

        case OP_SIGMOID:
            if (a && a->requires_grad) {
                ensure_grad(a);
                // d/dx sigmoid(x) = sigmoid(x) * (1 - sigmoid(x))
                a->grad = mx::add(a->grad, mx::multiply(r->grad,
                    mx::multiply(r->data, mx::subtract(mx::array(1.0), r->data))));
            }
            break;

        case OP_TANH:
            if (a && a->requires_grad) {
                ensure_grad(a);
                // d/dx tanh(x) = 1 - tanh(x)^2
                a->grad = mx::add(a->grad, mx::multiply(r->grad,
                    mx::subtract(mx::array(1.0), mx::square(r->data))));
            }
            break;

        case OP_POW: {
            // d/db (b^e) = e * b^(e-1) * grad
            if (a && a->requires_grad) {
                ensure_grad(a);
                a->grad = mx::add(a->grad, mx::multiply(r->grad,
                    mx::multiply(b->data, mx::power(a->data, mx::subtract(b->data, mx::array(1.0))))));
            }
            break;
        }

        case OP_ABS:
            if (a && a->requires_grad) {
                ensure_grad(a);
                // d/dx |x| = sign(x)
                a->grad = mx::add(a->grad, mx::multiply(r->grad, mx::sign(a->data)));
            }
            break;

        case OP_SUM:
            if (a && a->requires_grad) {
                ensure_grad(a);
                a->grad = mx::add(a->grad, mx::broadcast_to(r->grad, a->data.shape()));
            }
            break;

        case OP_MEAN:
            if (a && a->requires_grad) {
                ensure_grad(a);
                double n = (double)a->data.size();
                a->grad = mx::add(a->grad, mx::multiply(
                    mx::broadcast_to(r->grad, a->data.shape()), mx::array(1.0 / n)));
            }
            break;

        case OP_MM: {
            // r = a @ b, d_a = grad @ b^T, d_b = a^T @ grad
            if (a && a->requires_grad) { ensure_grad(a); a->grad = mx::add(a->grad, mx::matmul(r->grad, mx::transpose(b->data))); }
            if (b && b->requires_grad) { ensure_grad(b); b->grad = mx::add(b->grad, mx::matmul(mx::transpose(a->data), r->grad)); }
            break;
        }

        case OP_TRANSPOSE_2D:
            if (a && a->requires_grad) { ensure_grad(a); a->grad = mx::add(a->grad, mx::transpose(r->grad)); }
            break;

        case OP_SOFTMAX_2D: {
            if (a && a->requires_grad) {
                ensure_grad(a);
                auto dot = mx::sum(mx::multiply(r->grad, r->data), -1, true);
                a->grad = mx::add(a->grad, mx::multiply(r->data, mx::subtract(r->grad, dot)));
            }
            break;
        }

        case OP_LOG_SOFTMAX_2D: {
            if (a && a->requires_grad) {
                ensure_grad(a);
                auto sum_grad = mx::sum(r->grad, -1, true);
                a->grad = mx::add(a->grad, mx::subtract(r->grad, mx::multiply(mx::exp(r->data), sum_grad)));
            }
            break;
        }

        case OP_MASKED_FILL: {
            if (a && a->requires_grad) {
                ensure_grad(a);
                auto pass = mx::where(b->data, mx::zeros(r->grad.shape(), mx::float64), r->grad);
                a->grad = mx::add(a->grad, pass);
            }
            break;
        }

        case OP_LAYER_NORM_2D: {
            auto meta = (LayerNormMeta*)e.meta;
            int mm = meta->m, nn = meta->n;
            // d_gamma, d_bias
            if (meta->gamma && meta->gamma->requires_grad) {
                ensure_grad(meta->gamma);
                meta->gamma->grad = mx::add(meta->gamma->grad, mx::sum(mx::multiply(r->grad, meta->x_hat), 0));
            }
            if (meta->bias && meta->bias->requires_grad) {
                ensure_grad(meta->bias);
                meta->bias->grad = mx::add(meta->bias->grad, mx::sum(r->grad, 0));
            }
            // d_input
            if (a && a->requires_grad) {
                ensure_grad(a);
                auto dx_hat = mx::multiply(r->grad, meta->gamma->data);
                auto mean_dxhat = mx::mean(dx_hat, -1, true);
                auto mean_dxhat_xhat = mx::mean(mx::multiply(dx_hat, meta->x_hat), -1, true);
                auto rstd_2d = mx::reshape(meta->rstd, {mm, 1});
                a->grad = mx::add(a->grad, mx::multiply(rstd_2d,
                    mx::subtract(dx_hat, mx::add(mean_dxhat, mx::multiply(meta->x_hat, mean_dxhat_xhat)))));
            }
            break;
        }

        case OP_RESHAPE:
            if (a && a->requires_grad) { ensure_grad(a); a->grad = mx::add(a->grad, mx::reshape(r->grad, a->data.shape())); }
            break;

        case OP_NARROW: {
            int start = (int)e.scalar_arg;
            if (a && a->requires_grad) {
                ensure_grad(a);
                // Scatter gradient back — work on flattened arrays
                auto flat_grad = mx::flatten(a->grad);
                flat_grad = mx::slice_update(flat_grad, r->grad,
                    mx::Shape{start}, mx::Shape{start + (int)r->data.size()});
                a->grad = mx::reshape(flat_grad, a->data.shape());
            }
            break;
        }

        case OP_CAT: {
            int split = (int)e.scalar_arg;
            if (a && a->requires_grad) { ensure_grad(a); a->grad = mx::add(a->grad, mx::slice(r->grad, {0}, {split})); }
            if (b && b->requires_grad) { ensure_grad(b); b->grad = mx::add(b->grad, mx::slice(r->grad, {split}, {(int)r->data.size()})); }
            break;
        }

        case OP_MV: {
            // r = a @ b where a=[m,n], b=[n], r=[m]
            // d_a[i,j] = grad[i] * b[j]  (outer product)
            // d_b[j] = sum_i(a[i,j] * grad[i])
            if (a && a->requires_grad) {
                ensure_grad(a);
                // grad=[m], b=[n] → outer product [m,n]
                a->grad = mx::add(a->grad, mx::outer(r->grad, b->data));
            }
            if (b && b->requires_grad) {
                ensure_grad(b);
                // a=[m,n]^T @ grad=[m] → [n]
                auto aT = mx::transpose(a->data);
                auto grad_col = mx::reshape(r->grad, {(int)r->grad.size(), 1});
                auto result = mx::reshape(mx::matmul(aT, grad_col), {(int)b->data.size()});
                b->grad = mx::add(b->grad, result);
            }
            break;
        }

        case OP_COSINE_SIM: {
            // r = cosine(a, b) where a=[n,m], b=[m], r=[n]
            // d_a[i,j] = grad[i] * (b[j] / (norm_a[i] * norm_b) - cos[i] * a[i,j] / norm_a[i]^2)
            // d_b[j] = sum_i grad[i] * (a[i,j] / (norm_a[i] * norm_b) - cos[i] * b[j] / norm_b^2)
            auto meta = (CosSimMeta*)e.meta;
            if (a && a->requires_grad) {
                ensure_grad(a);
                int nn = meta->n, mm = meta->m;
                // grad_expanded = grad[:, None] → [n, 1]
                auto g_2d = mx::reshape(r->grad, {nn, 1});
                auto b_2d = mx::reshape(b->data, {1, mm});
                auto nab = mx::multiply(mx::reshape(meta->row_norms, {nn, 1}), meta->key_norm);
                auto cos_2d = mx::reshape(r->data, {nn, 1});
                auto rn2 = mx::reshape(mx::square(meta->row_norms), {nn, 1});
                // d_a = grad * (b / (na*nb) - cos * a / na^2)
                auto term1 = mx::divide(b_2d, nab);
                auto term2 = mx::divide(mx::multiply(cos_2d, a->data), rn2);
                a->grad = mx::add(a->grad, mx::multiply(g_2d, mx::subtract(term1, term2)));
            }
            if (b && b->requires_grad) {
                ensure_grad(b);
                int nn = meta->n, mm = meta->m;
                auto g_2d = mx::reshape(r->grad, {nn, 1});
                auto nab = mx::multiply(mx::reshape(meta->row_norms, {nn, 1}), meta->key_norm);
                auto cos_2d = mx::reshape(r->data, {nn, 1});
                auto bn2 = mx::square(meta->key_norm);
                auto b_2d = mx::reshape(b->data, {1, mm});
                auto term1 = mx::divide(a->data, nab);
                auto term2 = mx::divide(mx::multiply(cos_2d, b_2d), bn2);
                auto per_row = mx::multiply(g_2d, mx::subtract(term1, term2));
                b->grad = mx::add(b->grad, mx::sum(per_row, {0}));
            }
            break;
        }

        case OP_CONV1D_CIRC: {
            // r = conv1d_circular(a, b) where a=[n], b=[k]
            // Backward: reverse the convolution
            int nn = (int)a->data.size();
            int kk = (int)b->data.size();
            int half_k = kk / 2;
            if (a && a->requires_grad) {
                ensure_grad(a);
                // d_input[idx] += grad[i] * kernel[k-1-j] for each (i,j) pair
                // Equivalent: convolve grad with flipped kernel
                for (int j = 0; j < kk; j++) {
                    int shift = -(half_k - j);  // reverse shift
                    auto shifted_grad = mx::roll(r->grad, shift);
                    auto kern_j = mx::take(b->data, mx::array(j));
                    a->grad = mx::add(a->grad, mx::multiply(shifted_grad, kern_j));
                }
            }
            if (b && b->requires_grad) {
                ensure_grad(b);
                // d_kernel[j] = sum_i grad[i] * input[shifted_idx]
                mx::eval(r->grad);
                mx::eval(a->data);
                auto grad_data = r->grad.data<double>();
                auto inp_data = a->data.data<double>();
                std::vector<double> dk(kk, 0.0);
                for (int j = 0; j < kk; j++) {
                    int shift = half_k - j;
                    for (int i = 0; i < nn; i++) {
                        int idx = ((i + shift) % nn + nn) % nn;
                        dk[j] += grad_data[i] * inp_data[idx];
                    }
                }
                b->grad = mx::add(b->grad, mx::array(dk.data(), {kk}, mx::float64));
            }
            break;
        }

        case OP_OUTER: {
            // r = outer(a, b) where a=[n], b=[m], r=[n,m]
            // d_a[i] = sum_j(grad[i,j] * b[j])
            // d_b[j] = sum_i(grad[i,j] * a[i])
            if (a && a->requires_grad) {
                ensure_grad(a);
                a->grad = mx::add(a->grad, mx::sum(mx::multiply(r->grad, b->data), {1}));
            }
            if (b && b->requires_grad) {
                ensure_grad(b);
                b->grad = mx::add(b->grad, mx::sum(mx::multiply(r->grad, mx::reshape(a->data, {(int)a->data.size(), 1})), {0}));
            }
            break;
        }

        case OP_STACK: {
            // Distribute gradient from stacked tensor back to constituent scalars
            auto inputs = (std::vector<Tensor*>*)e.meta;
            if (inputs) {
                mx::eval(r->grad);
                auto grad_data = r->grad.data<double>();
                for (int j = 0; j < (int)inputs->size(); j++) {
                    auto inp = (*inputs)[j];
                    if (inp->requires_grad) {
                        ensure_grad(inp);
                        inp->grad = mx::add(inp->grad, mx::array(grad_data[j]));
                    }
                }
            }
            break;
        }

        default:
            break;
        }
    }
}

TensorHandle tensor_grad(TensorHandle h) { STUB(); }

void tensor_zero_grad(TensorHandle h) {
    auto t = (Tensor*)h;
    if (t->has_grad) {
        t->grad = mx::zeros(t->data.shape(), mx::float64);
    }
}

int tensor_requires_grad(TensorHandle h) { return ((Tensor*)h)->requires_grad ? 1 : 0; }
TensorHandle tensor_detach(TensorHandle h) { STUB(); }
TensorHandle tensor_with_grad(TensorHandle h) { STUB(); }

void tensor_set_requires_grad(TensorHandle h, int rg) {
    ((Tensor*)h)->requires_grad = (rg != 0);
}

void tensor_no_grad_begin(void) {}
void tensor_no_grad_end(void) {}

/* ================================================================
   Device
   ================================================================ */

TensorHandle tensor_to_device(TensorHandle t, const char* device) { return t; }
const char* tensor_device(TensorHandle t) { return "gpu"; }

/* ================================================================
   LSTM (stubs)
   ================================================================ */

void tensor_lstm_cell(TensorHandle input, TensorHandle hx, TensorHandle cx,
    TensorHandle w_ih, TensorHandle w_hh, TensorHandle b_ih, TensorHandle b_hh,
    TensorHandle* out_h, TensorHandle* out_c) { STUB(); }
void tensor_lstm_gates(TensorHandle combined, TensorHandle prev_cell, int o,
    TensorHandle* out_h, TensorHandle* out_c) { STUB(); }
TensorPair* tensor_lstm_gates_pair(TensorHandle hcombined, TensorHandle hprev_cell, int o) {
    // Decompose into primitives — each records its own tape entry
    // Split combined [4*o] into 4 gates
    TensorHandle ig_raw = tensor_narrow(hcombined, 0, 0, o);
    TensorHandle fg_raw = tensor_narrow(hcombined, 0, o, o);
    TensorHandle gg_raw = tensor_narrow(hcombined, 0, 2*o, o);
    TensorHandle og_raw = tensor_narrow(hcombined, 0, 3*o, o);
    // Apply activations
    TensorHandle ig = tensor_sigmoid(ig_raw);
    TensorHandle fg = tensor_sigmoid(fg_raw);
    TensorHandle gg = tensor_tanh(gg_raw);
    TensorHandle og = tensor_sigmoid(og_raw);
    // c_t = f_t ⊙ c_{t-1} + i_t ⊙ g_t
    TensorHandle fc = tensor_mul(fg, hprev_cell);
    TensorHandle ig_gg = tensor_mul(ig, gg);
    TensorHandle new_cell = tensor_add(fc, ig_gg);
    // h_t = o_t ⊙ tanh(c_t)
    TensorHandle tanh_cell = tensor_tanh(new_cell);
    TensorHandle new_hidden = tensor_mul(og, tanh_cell);
    // Return pair
    auto pair = (TensorPair*)malloc(sizeof(TensorPair));
    pair->first = new_hidden;
    pair->second = new_cell;
    return pair;
}

TensorHandle tensor_pair_first(TensorPair* p) { return p->first; }
TensorHandle tensor_pair_second(TensorPair* p) { return p->second; }
void tensor_pair_free(TensorPair* p) { if (p) free(p); }

/* ================================================================
   Parameter registry
   ================================================================ */

void param_register(const char* name, TensorHandle h) {
    auto t = (Tensor*)h;
    t->persistent = 1;
    t->requires_grad = true;
    param_registry.push_back({std::string(name), t});
}

void param_clear(void) { param_registry.clear(); }
int param_count(void) { return (int)param_registry.size(); }
const char* param_name(int idx) { return param_registry[idx].name.c_str(); }

double param_grad_item(int idx) {
    auto t = param_registry[idx].tensor;
    if (!t->has_grad) return 0.0;
    mx::eval(t->grad);
    return t->grad.item<double>();
}

double param_grad_item_at(int param_idx, int elem_idx) {
    auto t = param_registry[param_idx].tensor;
    if (!t->has_grad) return 0.0;
    mx::eval(t->grad);
    return t->grad.data<double>()[elem_idx];
}

double param_grad_item_and_zero(int idx) {
    double g = param_grad_item(idx);
    param_registry[idx].tensor->grad = mx::zeros(param_registry[idx].tensor->data.shape(), mx::float64);
    return g;
}

TensorHandle param_tensor(int idx) { return (TensorHandle)param_registry[idx].tensor; }

void param_zero_all_grads(void) {
    for (auto& p : param_registry) {
        if (p.tensor->has_grad) {
            p.tensor->grad = mx::zeros(p.tensor->data.shape(), mx::float64);
        }
    }
}

void param_subtract_delta(int idx, double delta) {
    auto t = param_registry[idx].tensor;
    t->data = mx::subtract(t->data, mx::array(delta));
}

TensorHandle tensor_subtract_scalar_inplace(TensorHandle h, double val) {
    auto t = (Tensor*)h;
    t->data = mx::subtract(t->data, mx::array(val));
    return h;
}

/* ================================================================
   Convenience functions
   ================================================================ */

TensorHandle tensor_create_1d(int n, double* data, int requires_grad) {
    int shape[] = {n};
    auto t = tensor_create(data, shape, 1, requires_grad);
    free(data);
    return t;
}

TensorHandle tensor_create_2d(int rows, int cols, double* data, int requires_grad) {
    int shape[] = {rows, cols};
    auto t = tensor_create(data, shape, 2, requires_grad);
    free(data);
    return t;
}

double* tensor_alloc_doubles(int n) { return (double*)calloc(n, sizeof(double)); }
void tensor_free_doubles(double* buf) { free(buf); }
double tensor_read_double(double* buf, int idx) { return buf[idx]; }
void tensor_write_double(double* buf, int idx, double val) { buf[idx] = val; }

TensorHandle* tensor_ptr_array_alloc(int n) {
    return (TensorHandle*)calloc(n, sizeof(TensorHandle));
}
void tensor_ptr_array_set(TensorHandle* arr, int idx, TensorHandle t) { arr[idx] = t; }

TensorHandle tensor_stack_from_array(TensorHandle* arr, int count, int dim) {
    std::vector<mx::array> arrs;
    bool rg = false;
    for (int i = 0; i < count; i++) {
        auto t = (Tensor*)arr[i];
        arrs.push_back(t->data);
        if (t->requires_grad) rg = true;
    }
    auto r = new Tensor(mx::stack(arrs, dim), rg);
    // Record OP_STACK so backward can distribute gradients
    if (rg) {
        int idx = tape_append(OP_STACK, r, nullptr, nullptr, (double)count);
        tape[idx].meta = (void*)(new std::vector<Tensor*>());
        for (int i = 0; i < count; i++)
            ((std::vector<Tensor*>*)tape[idx].meta)->push_back((Tensor*)arr[i]);
    }
    return (TensorHandle)r;
}

TensorHandle tensor_cat_from_array(TensorHandle* arr, int count, int dim) { STUB(); }

/* ================================================================
   Tensor-level parameter creation
   ================================================================ */

TensorHandle tensor_create_param_2d(int rows, int cols, double* data) {
    int shape[] = {rows, cols};
    auto t = tensor_create(data, shape, 2, 1);
    free(data);
    return t;
}

TensorHandle tensor_create_param_1d(int n, double* data) {
    int shape[] = {n};
    auto t = tensor_create(data, shape, 1, 1);
    free(data);
    return t;
}

TensorHandle tensor_create_state_2d(int rows, int cols, double* data) {
    int shape[] = {rows, cols};
    auto t = tensor_create(data, shape, 2, 0);
    free(data);
    return t;
}

TensorHandle tensor_create_state_1d(int n, double* data) {
    int shape[] = {n};
    auto t = tensor_create(data, shape, 1, 0);
    free(data);
    return t;
}

TensorHandle tensor_view_2d(TensorHandle mat, int row, int col) {
    auto t = (Tensor*)mat;
    // Return a scalar tensor sharing the value
    int cols = t->data.shape(1);
    auto val = mx::take(mx::flatten(t->data), mx::array(row * cols + col));
    auto r = new Tensor(val, t->requires_grad);
    r->persistent = 1;
    return (TensorHandle)r;
}

TensorHandle tensor_view_1d(TensorHandle vec, int idx) {
    auto t = (Tensor*)vec;
    auto val = mx::take(t->data, mx::array(idx));
    auto r = new Tensor(val, t->requires_grad);
    r->persistent = 1;
    return (TensorHandle)r;
}

double tensor_item_2d(TensorHandle mat, int row, int col) {
    auto t = (Tensor*)mat;
    mx::eval(t->data);
    int cols = t->data.shape(1);
    return t->data.data<double>()[row * cols + col];
}

double tensor_item_1d(TensorHandle vec, int idx) {
    auto t = (Tensor*)vec;
    mx::eval(t->data);
    return t->data.data<double>()[idx];
}

/* ================================================================
   Causal mask
   ================================================================ */

TensorHandle tensor_causal_mask(int n) {
    // Upper triangular: 1.0 above diagonal, 0.0 on/below
    std::vector<double> data(n * n, 0.0);
    for (int i = 0; i < n; i++)
        for (int j = i + 1; j < n; j++)
            data[i * n + j] = 1.0;
    mx::Shape sh = {n, n};
    auto t = new Tensor(mx::array(data.data(), sh, mx::float64), false);
    return (TensorHandle)t;
}

/* ================================================================
   Optimizer
   ================================================================ */

struct Optimizer {
    int type; // 0=sgd, 1=rmsprop, 2=adam
    double lr, beta1, beta2, eps;
    double alpha, weight_decay, momentum;
    int t;
    // Per-parameter buffers
    std::vector<mx::array> m_bufs, v_bufs;
};

OptimizerHandle optimizer_create_sgd(double lr) {
    auto opt = new Optimizer();
    opt->type = 0; opt->lr = lr; opt->t = 0;
    return (OptimizerHandle)opt;
}

OptimizerHandle optimizer_create_rmsprop(double lr, double alpha, double eps,
    double weight_decay, double momentum) {
    auto opt = new Optimizer();
    opt->type = 1; opt->lr = lr; opt->alpha = alpha; opt->eps = eps;
    opt->weight_decay = weight_decay; opt->momentum = momentum; opt->t = 0;
    return (OptimizerHandle)opt;
}

OptimizerHandle optimizer_create_adam(double lr, double beta1, double beta2, double eps) {
    auto opt = new Optimizer();
    opt->type = 2; opt->lr = lr; opt->beta1 = beta1; opt->beta2 = beta2; opt->eps = eps; opt->t = 0;
    return (OptimizerHandle)opt;
}

void optimizer_free(OptimizerHandle h) { delete (Optimizer*)h; }
void optimizer_zero_grad(OptimizerHandle h) { param_zero_all_grads(); }

void optimizer_step(OptimizerHandle h) {
    auto opt = (Optimizer*)h;
    opt->t++;
    int np = (int)param_registry.size();

    // Ensure optimizer buffers
    if ((int)opt->m_bufs.size() != np) {
        opt->m_bufs.clear();
        opt->v_bufs.clear();
        for (auto& p : param_registry) {
            opt->m_bufs.push_back(mx::zeros(p.tensor->data.shape(), mx::float64));
            opt->v_bufs.push_back(mx::zeros(p.tensor->data.shape(), mx::float64));
        }
    }

    for (int i = 0; i < np; i++) {
        auto t = param_registry[i].tensor;
        if (!t->has_grad) continue;

        mx::eval(t->grad);
        auto g = t->grad;

        switch (opt->type) {
        case 0: // SGD
            t->data = mx::subtract(t->data, mx::multiply(mx::array(opt->lr), g));
            break;
        case 2: { // Adam
            opt->m_bufs[i] = mx::add(mx::multiply(mx::array(opt->beta1), opt->m_bufs[i]),
                                      mx::multiply(mx::array(1.0 - opt->beta1), g));
            opt->v_bufs[i] = mx::add(mx::multiply(mx::array(opt->beta2), opt->v_bufs[i]),
                                      mx::multiply(mx::array(1.0 - opt->beta2), mx::square(g)));
            auto mhat = mx::divide(opt->m_bufs[i], mx::array(1.0 - std::pow(opt->beta1, opt->t)));
            auto vhat = mx::divide(opt->v_bufs[i], mx::array(1.0 - std::pow(opt->beta2, opt->t)));
            t->data = mx::subtract(t->data,
                mx::divide(mx::multiply(mx::array(opt->lr), mhat),
                            mx::add(mx::sqrt(vhat), mx::array(opt->eps))));
            break;
        }
        default:
            break;
        }
    }

    // Eval all updated params
    std::vector<mx::array> to_eval;
    for (auto& p : param_registry) to_eval.push_back(p.tensor->data);
    mx::eval(to_eval);

    // Reset tape
    tape_reset();
    for (auto& p : param_registry) {
        p.tensor->tape_idx = -1;
        p.tensor->has_grad = false;
        tape_append(OP_CONST, p.tensor, nullptr, nullptr, 0);
    }
}

void optimizer_clip_grad_value(double max_val) {
    for (auto& p : param_registry) {
        if (p.tensor->has_grad) {
            p.tensor->grad = mx::clip(p.tensor->grad, mx::array(-max_val), mx::array(max_val));
        }
    }
}

double optimizer_clip_grad_norm(double max_norm) {
    // Compute total norm
    mx::array total = mx::array(0.0);
    for (auto& p : param_registry) {
        if (p.tensor->has_grad) {
            total = mx::add(total, mx::sum(mx::square(p.tensor->grad)));
        }
    }
    mx::eval(total);
    double norm = std::sqrt(total.item<double>());
    if (norm > max_norm) {
        double scale = max_norm / norm;
        for (auto& p : param_registry) {
            if (p.tensor->has_grad) {
                p.tensor->grad = mx::multiply(p.tensor->grad, mx::array(scale));
            }
        }
    }
    return norm;
}

/* ================================================================
   Backend capabilities
   ================================================================ */

int backend_supports_tensor_params(void) { return 1; }

/* ================================================================
   System
   ================================================================ */

int get_rss_mb(void) {
    struct rusage ru;
    getrusage(RUSAGE_SELF, &ru);
#ifdef __APPLE__
    return (int)(ru.ru_maxrss / (1024 * 1024));
#else
    return (int)(ru.ru_maxrss / 1024);
#endif
}

int get_current_rss_mb(void) {
#ifdef __APPLE__
    mach_task_basic_info_data_t info;
    mach_msg_type_number_t count = MACH_TASK_BASIC_INFO_COUNT;
    if (task_info(mach_task_self(), MACH_TASK_BASIC_INFO,
                  (task_info_t)&info, &count) == KERN_SUCCESS)
        return (int)(info.resident_size / (1024 * 1024));
#endif
    return get_rss_mb();
}
void backend_memory_report(void) { fprintf(stderr, "MLX backend: memory report not implemented\n"); }
void backend_reset_for_eval(void) {
    tape_reset();
    for (auto& p : param_registry) {
        p.tensor->tape_idx = -1;
        p.tensor->has_grad = false;
        tape_append(OP_CONST, p.tensor, nullptr, nullptr, 0);
    }
}
void backend_profile_reset(void) {}
void backend_profile_report(void) {}

/* ================================================================
   Debug
   ================================================================ */

const char* backend_name(void) { return "mlx"; }

void tensor_print(TensorHandle h) {
    auto t = (Tensor*)h;
    mx::eval(t->data);
    std::cout << t->data << std::endl;
}

} // extern "C"
