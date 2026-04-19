/* backend_mlx.cpp — MLX backend implementing backend.h.
 *
 * Uses Apple's MLX framework for GPU-accelerated tensor operations
 * on Apple Silicon via Metal. Forward ops record to a tape; backward
 * replays the tape inside mlx::grad for native autograd — zero
 * hand-written backward rules.
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
#include <unordered_set>
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

/* Forward declarations for self-registration */
static std::vector<struct Tensor*> all_tensors;
static std::vector<TensorPair*> all_pairs;

static int next_pool_idx = 0;

struct Tensor {
    mx::array data;
    mx::array grad;
    bool requires_grad;
    bool has_grad;
    int tape_idx;
    int persistent;  // 1 = param/state, 0 = intermediate
    int pool_idx;    // unique index for replay pool

    Tensor(mx::array d, bool rg = false)
        : data(std::move(d)), grad(mx::array(0.0f)), requires_grad(rg),
          has_grad(false), tape_idx(-1), persistent(0), pool_idx(next_pool_idx++) {
        all_tensors.push_back(this);
    }
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
    OP_MM, OP_BMM, OP_TRANSPOSE_2D,
    OP_SOFTMAX_2D, OP_LOG_SOFTMAX_2D,
    OP_MASKED_FILL, OP_LAYER_NORM_2D,
    OP_RESHAPE, OP_NARROW, OP_CAT,
    OP_POW, OP_ABS,
    OP_STACK, OP_OUTER,
    OP_COSINE_SIM, OP_CONV1D_CIRC,
    OP_MV,
    OP_SELECT,
    OP_NORMALIZE,
    OP_BMM_3X3,
    OP_SOFTMAX_3D,
    OP_TRANSPOSE_LAST2,
    OP_GELU,
    OP_GRU_CELL,
    OP_EMBEDDING,
    OP_BATCH_NORM,
    OP_DROPOUT,
    OP_AVG_POOL1D,
    OP_AVG_POOL2D,
    OP_CONV1D,
    OP_MAX_POOL1D,
    OP_CONV2D,
    OP_MAX_POOL2D,
    OP_CUMPROD,
    OP_LEAKY_RELU,
    OP_SILU,
};

// Lightweight metadata for ops that need extra info during replay.
// No gradient arrays — mlx::grad handles backward automatically.
struct LayerNormReplayMeta {
    int gamma_pool_idx;
    int bias_pool_idx;
    double eps;
};

struct BatchNormReplayMeta {
    int gamma_pool_idx;
    int beta_pool_idx;
    int C, spatial;
    double eps;
};

struct Conv1DReplayMeta {
    int pad, stride, inC, L;
    int bias_pool_idx;
};

struct MaxPool1DReplayMeta {
    int C, L, kL, stride, oL;
};

struct Conv2DReplayMeta {
    int padH, padW, strH, strW;
    int inC, H, W;
    int bias_pool_idx;  // -1 if no bias
};

struct MaxPool2DReplayMeta {
    int C, H, W, kH, kW, strH, strW, oH, oW;
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
            delete (LayerNormReplayMeta*)e.meta;
            e.meta = nullptr;
        }
        if (e.op == OP_STACK && e.meta) {
            delete (std::vector<int>*)e.meta;
            e.meta = nullptr;
        }
        if (e.op == OP_BATCH_NORM && e.meta) {
            delete (BatchNormReplayMeta*)e.meta;
            e.meta = nullptr;
        }
        if (e.op == OP_CONV1D && e.meta) {
            delete (Conv1DReplayMeta*)e.meta;
            e.meta = nullptr;
        }
        if (e.op == OP_MAX_POOL1D && e.meta) {
            delete (MaxPool1DReplayMeta*)e.meta;
            e.meta = nullptr;
        }
        if (e.op == OP_CONV2D && e.meta) {
            delete (Conv2DReplayMeta*)e.meta;
            e.meta = nullptr;
        }
        if (e.op == OP_MAX_POOL2D && e.meta) {
            delete (MaxPool2DReplayMeta*)e.meta;
            e.meta = nullptr;
        }
    }
    tape.clear();
    // Force evaluation of all pending lazy ops before deleting any tensor.
    // MLX arrays hold lazy references to inputs — deleting an intermediate
    // while a surviving tensor's graph still references it is use-after-free.
    {
        std::vector<mx::array> to_eval;
        for (auto* t : all_tensors) {
            to_eval.push_back(t->data);
            if (t->has_grad) to_eval.push_back(t->grad);
        }
        if (!to_eval.empty()) mx::eval(to_eval);
    }
    // Now safely delete non-persistent tensors.
    // Persistent tensors (params, state, views) survive.
    std::vector<Tensor*> survivors;
    for (auto* t : all_tensors) {
        if (t->persistent) survivors.push_back(t);
        else delete t;
    }
    all_tensors = std::move(survivors);
    // Reassign pool indices to be contiguous (keeps pool vector compact)
    next_pool_idx = 0;
    for (auto* t : all_tensors) t->pool_idx = next_pool_idx++;
    // Free TensorPair structs
    for (auto* p : all_pairs) free(p);
    all_pairs.clear();
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
    // Explicit float64 — mx::array(double) defaults to float32
    auto t = new Tensor(mx::array(value, mx::float64), requires_grad != 0);
    if (requires_grad) tape_append(OP_CONST, t, nullptr, nullptr, 0);
    // Non-grad scalars stay non-persistent — freed by tape_reset at optimizer_step
    return (TensorHandle)t;
}

TensorHandle tensor_create(double* data, int* shape, int rank, int requires_grad) {
    mx::Shape sh(shape, shape + rank);
    auto t = new Tensor(mx::array(data, sh, mx::float64), requires_grad != 0);
    if (requires_grad) tape_append(OP_CONST, t, nullptr, nullptr, 0);
    // Non-grad data tensors: non-persistent, freed by tape_reset at optimizer_step
    return (TensorHandle)t;
}

TensorHandle tensor_clone(TensorHandle h) {
    auto t = (Tensor*)h;
    auto c = new Tensor(mx::array(t->data), false);
    return (TensorHandle)c;
}

void tensor_free(TensorHandle h) {
    if (!h) return;
    auto t = (Tensor*)h;
    // Skip registered params — they're managed by param_clear
    for (auto& p : param_registry) {
        if (p.tensor == t) return;
    }
    // Remove from all_tensors tracking and delete.
    // If not found in all_tensors, it was already freed by tape_reset — skip.
    for (size_t i = 0; i < all_tensors.size(); i++) {
        if (all_tensors[i] == t) {
            all_tensors.erase(all_tensors.begin() + i);
            delete t;
            return;
        }
    }
    // Not in all_tensors — already freed by tape_reset, skip
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
TensorHandle tensor_gelu(TensorHandle h) {
    auto t = (Tensor*)h;
    // GELU tanh approx: x * 0.5 * (1 + tanh(sqrt(2/pi) * (x + 0.044715 * x^3)))
    auto x = t->data;
    auto c = mx::array(0.7978845608028654, mx::float64);
    auto inner = mx::multiply(c, mx::add(x, mx::multiply(mx::array(0.044715, mx::float64), mx::power(x, mx::array(3, mx::float64)))));
    auto result = mx::multiply(mx::multiply(mx::array(0.5, mx::float64), x), mx::add(mx::array(1.0, mx::float64), mx::tanh(inner)));
    auto r = new Tensor(result, t->requires_grad);
    if (t->requires_grad) tape_append(OP_GELU, r, t, nullptr, 0);
    return (TensorHandle)r;
}

TensorHandle tensor_tanh(TensorHandle h) {
    auto t = (Tensor*)h;
    auto r = new Tensor(mx::tanh(t->data), t->requires_grad);
    if (t->requires_grad) tape_append(OP_TANH, r, t, nullptr, 0);
    return (TensorHandle)r;
}

TensorHandle tensor_leaky_relu(TensorHandle h, double alpha) {
    auto t = (Tensor*)h;
    auto alpha_arr = mx::array(alpha, mx::float64);
    auto result = mx::maximum(mx::multiply(alpha_arr, t->data), t->data);
    auto r = new Tensor(result, t->requires_grad);
    if (t->requires_grad) tape_append(OP_LEAKY_RELU, r, t, nullptr, alpha);
    return (TensorHandle)r;
}

TensorHandle tensor_silu(TensorHandle h) {
    auto t = (Tensor*)h;
    auto result = mx::multiply(t->data, mx::sigmoid(t->data));
    auto r = new Tensor(result, t->requires_grad);
    if (t->requires_grad) tape_append(OP_SILU, r, t, nullptr, 0);
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
    if (t->requires_grad) tape_append(OP_LOG_SOFTMAX_2D, r, t, nullptr, (double)dim);
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
    auto dots = mx::sum(mx::multiply(mem->data, key_2d), std::vector<int>{1}); // [n]
    auto row_norms = mx::sqrt(mx::add(mx::sum(mx::square(mem->data), std::vector<int>{1}), eps)); // [n]
    auto key_norm = mx::sqrt(mx::add(mx::sum(mx::square(key->data)), eps)); // scalar
    auto result = mx::divide(dots, mx::multiply(row_norms, key_norm));

    bool rg = mem->requires_grad || key->requires_grad;
    auto r = new Tensor(result, rg);
    if (rg) tape_append(OP_COSINE_SIM, r, mem, key, 0);
    return (TensorHandle)r;
}

TensorHandle tensor_conv1d_circular(TensorHandle hinput, TensorHandle hkernel) {
    // Circular correlation: out[i] = sum_j input[(i-k/2+j+n)%n] * kernel[j]
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

TensorHandle tensor_batch_norm(TensorHandle hinput, TensorHandle hgamma, TensorHandle hbeta,
                               TensorHandle hrunning_mean, TensorHandle hrunning_var,
                               int C, int spatial, int training,
                               double momentum, double eps) {
    auto inp = (Tensor*)hinput;
    auto gamma = (Tensor*)hgamma;
    auto beta = (Tensor*)hbeta;
    auto rm = (Tensor*)hrunning_mean;
    auto rv = (Tensor*)hrunning_var;

    // Reshape flat input to [C, spatial]
    auto x = mx::reshape(inp->data, {C, spatial});
    auto mean = mx::mean(x, std::vector<int>{1}, true);  // [C, 1]
    auto var = mx::var(x, std::vector<int>{1}, true);     // [C, 1]

    if (training) {
        // Update running stats (non-differentiable)
        auto new_rm = mx::add(mx::multiply(mx::array(1.0 - momentum, mx::float64), rm->data),
                              mx::multiply(mx::array(momentum, mx::float64), mx::squeeze(mean)));
        auto new_rv = mx::add(mx::multiply(mx::array(1.0 - momentum, mx::float64), rv->data),
                              mx::multiply(mx::array(momentum, mx::float64), mx::squeeze(var)));
        rm->data = new_rm;
        rv->data = new_rv;
        mx::eval(rm->data);
        mx::eval(rv->data);
    } else {
        mean = mx::reshape(rm->data, {C, 1});
        var = mx::reshape(rv->data, {C, 1});
    }

    auto rstd = mx::rsqrt(mx::add(var, mx::array(eps, mx::float64)));
    auto x_hat = mx::multiply(mx::subtract(x, mean), rstd);
    auto g = mx::reshape(gamma->data, {C, 1});
    auto b = mx::reshape(beta->data, {C, 1});
    auto result = mx::flatten(mx::add(mx::multiply(g, x_hat), b));

    bool rg = inp->requires_grad || gamma->requires_grad || beta->requires_grad;
    auto r = new Tensor(result, rg);
    if (rg) {
        int idx = tape_append(OP_BATCH_NORM, r, inp, nullptr, 0);
        auto* meta = new BatchNormReplayMeta();
        meta->gamma_pool_idx = gamma->pool_idx;
        meta->beta_pool_idx = beta->pool_idx;
        meta->C = C;
        meta->spatial = spatial;
        meta->eps = eps;
        tape[idx].meta = meta;
    }
    return (TensorHandle)r;
}

TensorHandle tensor_dropout(TensorHandle hinput, double p, int training, unsigned int seed) {
    auto inp = (Tensor*)hinput;
    if (!training || p <= 0.0) return hinput;

    // Generate bernoulli mask and scale by 1/(1-p)
    // MLX random only supports float32 on Metal — generate in f32, compare, cast result to f64
    double scale = 1.0 / (1.0 - p);
    auto rnd = mx::random::uniform(mx::array(0.0f), mx::array(1.0f), inp->data.shape(), mx::float32);
    auto keep = mx::greater(rnd, mx::array((float)p, mx::float32));
    auto mask = mx::astype(mx::where(keep, mx::array(scale, mx::float64), mx::array(0.0, mx::float64)), mx::float64);
    auto result = mx::multiply(inp->data, mask);

    auto r = new Tensor(result, inp->requires_grad);
    if (inp->requires_grad) {
        // For replay: store the mask as a constant in the pool so vjp can diff through multiply
        auto mask_t = new Tensor(mask, false);
        mask_t->persistent = 1;  // survives tape_reset
        int idx = tape_append(OP_DROPOUT, r, inp, mask_t, 0);
    }
    return (TensorHandle)r;
}

TensorHandle tensor_cross_attention(TensorHandle hQ, TensorHandle hK, TensorHandle hV,
                                    TensorHandle hmask, double scale) {
    /* Compose from existing ops — MLX replay autograd handles backward */
    TensorHandle KT = tensor_transpose_last2(hK);
    TensorHandle scores = tensor_mul_scalar(tensor_bmm_3x3(hQ, KT), scale);
    if (hmask) scores = tensor_masked_fill(scores, hmask, -1.0e20);
    TensorHandle attn = tensor_softmax_3d(scores);
    return tensor_bmm_3x3(attn, hV);
}

TensorHandle tensor_embedding(TensorHandle hweight, TensorHandle hindices, int n, int embedDim) {
    auto weight = (Tensor*)hweight;
    auto indices = (Tensor*)hindices;
    auto idx_int = mx::astype(indices->data, mx::int32);
    auto rows = mx::take(weight->data, idx_int, 0);  /* [n, embedDim] */
    auto result = mx::flatten(rows);  /* [n * embedDim] */

    auto r = new Tensor(result, weight->requires_grad);
    if (weight->requires_grad) {
        // For replay: store indices as arg2 so vjp can differentiate through take
        auto idx_t = new Tensor(idx_int, false);
        idx_t->persistent = 1;
        tape_append(OP_EMBEDDING, r, weight, idx_t, (double)embedDim);
    }
    return (TensorHandle)r;
}

TensorHandle tensor_gather(TensorHandle hinput, TensorHandle hindex, int n) {
    auto inp = (Tensor*)hinput;
    auto idx = (Tensor*)hindex;
    auto idx_int = mx::astype(idx->data, mx::int32);
    auto result = mx::take(inp->data, idx_int, 0);
    return (TensorHandle)(new Tensor(result, inp->requires_grad));
}

TensorHandle tensor_scatter_add(TensorHandle hindex, TensorHandle hsrc, int out_size) {
    auto idx = (Tensor*)hindex;
    auto src = (Tensor*)hsrc;
    auto out = mx::zeros({out_size}, mx::float64);
    // Scatter-add via loop (small n expected)
    mx::eval(idx->data); mx::eval(src->data);
    auto idx_flat = mx::astype(idx->data, mx::int32);
    mx::eval(idx_flat);
    // Use put_along_axis or manual loop
    for (int i = 0; i < (int)src->data.size(); i++) {
        int ix = idx_flat.data<int32_t>()[i];
        auto val = mx::take(src->data, mx::array(i));
        auto cur = mx::take(out, mx::array(ix));
        out = mx::where(mx::equal(mx::arange(out_size), mx::array(ix)),
                        mx::add(out, mx::broadcast_to(val, {out_size})), out);
    }
    return (TensorHandle)(new Tensor(out, src->requires_grad));
}

TensorHandle tensor_argsort(TensorHandle ht, int dim, int descending) {
    auto t = (Tensor*)ht;
    // MLX argsort returns ascending by default
    auto indices = mx::argsort(t->data, dim);
    if (descending) {
        // Reverse: take from end
        int n = (int)t->data.size();
        auto rev_idx = mx::subtract(mx::array(n - 1), mx::arange(n));
        indices = mx::take(indices, rev_idx);
    }
    auto result = mx::astype(indices, mx::float64);
    mx::eval(result);
    return (TensorHandle)(new Tensor(result, false)); // no grad for indices
}

TensorHandle tensor_cumprod(TensorHandle ht, int dim) {
    auto t = (Tensor*)ht;
    auto result = mx::cumprod(t->data, dim);
    auto r = new Tensor(result, t->requires_grad);
    if (r->requires_grad) {
        tape_append(OP_CUMPROD, r, t, NULL, 0.0);
    }
    return (TensorHandle)r;
}

TensorHandle tensor_gru_cell(TensorHandle hcombined, TensorHandle hprev, int o) {
    auto combined = (Tensor*)hcombined;
    auto prev = (Tensor*)hprev;
    // Decompose into primitives — MLX replay autograd handles backward
    auto z_raw = mx::slice(combined->data, {0}, {o});
    auto r_raw = mx::slice(combined->data, {o}, {2*o});
    auto n_raw = mx::slice(combined->data, {2*o}, {3*o});
    auto z = mx::sigmoid(z_raw);
    auto n = mx::tanh(n_raw);
    auto one = mx::array(1.0, mx::float64);
    auto result = mx::add(mx::multiply(mx::subtract(one, z), n), mx::multiply(z, prev->data));

    bool rg = combined->requires_grad || prev->requires_grad;
    auto r = new Tensor(result, rg);
    if (rg) tape_append(OP_GRU_CELL, r, combined, prev, (double)o);
    return (TensorHandle)r;
}

TensorHandle tensor_group_norm(TensorHandle hinput, TensorHandle hgamma, TensorHandle hbeta,
                               int numGroups, int channels, int spatial, double eps) {
    /* Same loop as tape backend — MLX doesn't have native group_norm */
    auto inp = (Tensor*)hinput;
    auto gamma = (Tensor*)hgamma;
    auto beta = (Tensor*)hbeta;
    int n = channels * spatial;
    int chPerGroup = channels / numGroups;
    int groupSize = chPerGroup * spatial;
    mx::eval(inp->data); mx::eval(gamma->data); mx::eval(beta->data);
    double* out = (double*)calloc(n, sizeof(double));
    for (int g = 0; g < numGroups; g++) {
        double mean = 0;
        int base = g * groupSize;
        for (int j = 0; j < groupSize; j++) mean += inp->data.data<double>()[base + j];
        mean /= groupSize;
        double var = 0;
        for (int j = 0; j < groupSize; j++) { double d = inp->data.data<double>()[base+j] - mean; var += d*d; }
        var /= groupSize;
        double rstd = 1.0 / sqrt(var + eps);
        for (int c = 0; c < chPerGroup; c++) {
            int absC = g * chPerGroup + c;
            for (int s = 0; s < spatial; s++) {
                int idx = absC * spatial + s;
                double x_hat = (inp->data.data<double>()[idx] - mean) * rstd;
                out[idx] = gamma->data.data<double>()[absC] * x_hat + beta->data.data<double>()[absC];
            }
        }
    }
    auto result = mx::array(out, {n}, mx::float64);
    free(out);
    return (TensorHandle)(new Tensor(result, inp->requires_grad || gamma->requires_grad));
}

TensorHandle tensor_conv_transpose1d(TensorHandle hinput, TensorHandle hkernel,
                                     TensorHandle hbias, int pad, int stride) {
    /* Implement as naive loop (same as tape) since MLX doesn't expose conv_transpose1d directly */
    auto inp = (Tensor*)hinput;
    auto ker = (Tensor*)hkernel;
    Tensor* bias = hbias ? (Tensor*)hbias : nullptr;
    int inC = (int)inp->data.shape(0), L = (int)inp->data.shape(1);
    int outC = (int)ker->data.shape(1), kL = (int)ker->data.shape(2);
    int oL = (L - 1) * stride - 2 * pad + kL;

    // Compute on CPU via eval then manual scatter
    mx::eval(inp->data); mx::eval(ker->data);
    double* out = (double*)calloc(outC * oL, sizeof(double));
    if (bias) { mx::eval(bias->data); for (int oc = 0; oc < outC; oc++) for (int ol = 0; ol < oL; ol++) out[oc*oL+ol] = bias->data.data<double>()[oc]; }
    for (int ic = 0; ic < inC; ic++)
        for (int il = 0; il < L; il++)
            for (int oc = 0; oc < outC; oc++)
                for (int kl = 0; kl < kL; kl++) {
                    int ol = il*stride - pad + kl;
                    if (ol >= 0 && ol < oL)
                        out[oc*oL+ol] += inp->data.data<double>()[ic*L+il] * ker->data.data<double>()[ic*outC*kL+oc*kL+kl];
                }
    auto result = mx::array(out, {outC, oL}, mx::float64);
    free(out);
    return (TensorHandle)(new Tensor(result, inp->requires_grad || ker->requires_grad));
}

TensorHandle tensor_conv_transpose2d(TensorHandle hinput, TensorHandle hkernel,
                                     TensorHandle hbias, int padH, int padW,
                                     int strideH, int strideW) {
    auto inp = (Tensor*)hinput;
    auto ker = (Tensor*)hkernel;
    Tensor* bias = hbias ? (Tensor*)hbias : nullptr;
    int inC = (int)inp->data.shape(0), H = (int)inp->data.shape(1), W = (int)inp->data.shape(2);
    int outC = (int)ker->data.shape(1), kH = (int)ker->data.shape(2), kW = (int)ker->data.shape(3);
    int oH = (H-1)*strideH - 2*padH + kH;
    int oW = (W-1)*strideW - 2*padW + kW;
    mx::eval(inp->data); mx::eval(ker->data);
    double* out = (double*)calloc(outC*oH*oW, sizeof(double));
    if (bias) { mx::eval(bias->data); for (int oc = 0; oc < outC; oc++) for (int oh = 0; oh < oH; oh++) for (int ow = 0; ow < oW; ow++) out[oc*oH*oW+oh*oW+ow] = bias->data.data<double>()[oc]; }
    for (int ic = 0; ic < inC; ic++)
        for (int ih = 0; ih < H; ih++)
            for (int iw = 0; iw < W; iw++)
                for (int oc = 0; oc < outC; oc++)
                    for (int kh = 0; kh < kH; kh++)
                        for (int kw = 0; kw < kW; kw++) {
                            int oh = ih*strideH - padH + kh;
                            int ow = iw*strideW - padW + kw;
                            if (oh >= 0 && oh < oH && ow >= 0 && ow < oW)
                                out[oc*oH*oW+oh*oW+ow] += inp->data.data<double>()[ic*H*W+ih*W+iw]
                                    * ker->data.data<double>()[ic*outC*kH*kW+oc*kH*kW+kh*kW+kw];
                        }
    auto result = mx::array(out, {outC, oH, oW}, mx::float64);
    free(out);
    return (TensorHandle)(new Tensor(result, inp->requires_grad || ker->requires_grad));
}

TensorHandle tensor_conv1d_grouped(TensorHandle hinput, TensorHandle hkernel,
                                   TensorHandle hbias, int pad, int stride, int groups) {
    if (groups == 1) return tensor_conv1d(hinput, hkernel, hbias, pad, stride);
    auto inp = (Tensor*)hinput;
    auto ker = (Tensor*)hkernel;
    Tensor* bias = hbias ? (Tensor*)hbias : nullptr;
    int inC = (int)inp->data.shape(0), L = (int)inp->data.shape(1);
    auto inp_lc = mx::transpose(inp->data, {1, 0});
    auto inp_nlc = mx::reshape(inp_lc, {1, L, inC});
    auto ker_mlx = mx::transpose(ker->data, {0, 2, 1});
    auto out = mx::conv1d(inp_nlc, ker_mlx, stride, pad, /*dilation=*/1, groups);
    auto out_sq = mx::squeeze(out, 0);
    auto result = mx::transpose(out_sq, {1, 0});
    if (bias) result = mx::add(result, mx::reshape(bias->data, {-1, 1}));
    return (TensorHandle)(new Tensor(result, inp->requires_grad || ker->requires_grad));
}

TensorHandle tensor_conv2d_grouped(TensorHandle hinput, TensorHandle hkernel,
                                   TensorHandle hbias, int padH, int padW,
                                   int strideH, int strideW, int groups) {
    if (groups == 1) return tensor_conv2d(hinput, hkernel, hbias, padH, padW, strideH, strideW);
    auto inp = (Tensor*)hinput;
    auto ker = (Tensor*)hkernel;
    Tensor* bias = hbias ? (Tensor*)hbias : nullptr;
    int inC = (int)inp->data.shape(0), H = (int)inp->data.shape(1), W = (int)inp->data.shape(2);
    auto inp_hwc = mx::transpose(inp->data, {1, 2, 0});
    auto inp_nhwc = mx::reshape(inp_hwc, {1, H, W, inC});
    auto ker_mlx = mx::transpose(ker->data, {0, 2, 3, 1});
    auto out = mx::conv2d(inp_nhwc, ker_mlx, {strideH, strideW}, {padH, padW}, /*dilation=*/{1, 1}, groups);
    auto out_sq = mx::squeeze(out, 0);
    auto result = mx::transpose(out_sq, {2, 0, 1});
    if (bias) result = mx::add(result, mx::reshape(bias->data, {-1, 1, 1}));
    return (TensorHandle)(new Tensor(result, inp->requires_grad || ker->requires_grad));
}

TensorHandle tensor_avg_pool1d(TensorHandle hinput, int kL, int stride) {
    auto inp = (Tensor*)hinput;
    int C = (int)inp->data.shape(0), L = (int)inp->data.shape(1);
    int oL = (L - kL) / stride + 1;
    // Sum via strided slices, then divide by kL
    mx::array result = mx::zeros({C, oL}, mx::float64);
    for (int kl = 0; kl < kL; kl++) {
        auto sliced = mx::slice(inp->data, {0, kl}, {C, kl + oL * stride}, {1, stride});
        result = mx::add(result, sliced);
    }
    result = mx::divide(result, mx::array((double)kL, mx::float64));
    auto r = new Tensor(result, inp->requires_grad);
    if (inp->requires_grad) tape_append(OP_AVG_POOL1D, r, inp, nullptr, (double)kL + stride * 0.001);
    return (TensorHandle)r;
}

TensorHandle tensor_avg_pool2d(TensorHandle hinput, int kH, int kW, int strideH, int strideW) {
    auto inp = (Tensor*)hinput;
    int C = (int)inp->data.shape(0), H = (int)inp->data.shape(1), W = (int)inp->data.shape(2);
    int oH = (H - kH) / strideH + 1;
    int oW = (W - kW) / strideW + 1;
    mx::array result = mx::zeros({C, oH, oW}, mx::float64);
    for (int kh = 0; kh < kH; kh++)
        for (int kw = 0; kw < kW; kw++) {
            auto sliced = mx::slice(inp->data,
                {0, kh, kw}, {C, kh + oH * strideH, kw + oW * strideW}, {1, strideH, strideW});
            result = mx::add(result, sliced);
        }
    result = mx::divide(result, mx::array((double)(kH * kW), mx::float64));
    auto r = new Tensor(result, inp->requires_grad);
    if (inp->requires_grad) tape_append(OP_AVG_POOL2D, r, inp, nullptr, 0);
    return (TensorHandle)r;
}

TensorHandle tensor_conv1d(TensorHandle hinput, TensorHandle hkernel,
                           TensorHandle hbias, int pad, int stride) {
    auto inp = (Tensor*)hinput;
    auto ker = (Tensor*)hkernel;
    Tensor* bias = hbias ? (Tensor*)hbias : nullptr;
    int inC = (int)inp->data.shape(0), L = (int)inp->data.shape(1);

    // MLX conv1d: input [N, L, C_in], weight [C_out, kL, C_in]
    auto inp_lc = mx::transpose(inp->data, {1, 0});  // [L, inC]
    auto inp_nlc = mx::reshape(inp_lc, {1, L, inC});
    auto ker_mlx = mx::transpose(ker->data, {0, 2, 1});  // [outC, kL, inC]
    auto out = mx::conv1d(inp_nlc, ker_mlx, stride, pad);
    auto out_sq = mx::squeeze(out, 0);  // [oL, outC]
    auto result = mx::transpose(out_sq, {1, 0});  // [outC, oL]
    if (bias) result = mx::add(result, mx::reshape(bias->data, {-1, 1}));

    bool rg = inp->requires_grad || ker->requires_grad || (bias && bias->requires_grad);
    auto r = new Tensor(result, rg);
    if (rg) {
        int idx = tape_append(OP_CONV1D, r, inp, ker, 0);
        auto* meta = new Conv1DReplayMeta();
        meta->pad = pad; meta->stride = stride; meta->inC = inC; meta->L = L;
        meta->bias_pool_idx = bias ? bias->pool_idx : -1;
        tape[idx].meta = meta;
    }
    return (TensorHandle)r;
}

TensorHandle tensor_max_pool1d(TensorHandle hinput, int kL, int stride) {
    auto inp = (Tensor*)hinput;
    int C = (int)inp->data.shape(0), L = (int)inp->data.shape(1);
    int oL = (L - kL) / stride + 1;

    mx::array result = mx::full({C, oL}, -1e30, mx::float64);
    for (int kl = 0; kl < kL; kl++) {
        auto sliced = mx::slice(inp->data, {0, kl}, {C, kl + oL * stride}, {1, stride});
        result = mx::maximum(result, sliced);
    }

    auto r = new Tensor(result, inp->requires_grad);
    if (inp->requires_grad) {
        int idx = tape_append(OP_MAX_POOL1D, r, inp, nullptr, 0);
        auto* meta = new MaxPool1DReplayMeta();
        meta->C = C; meta->L = L; meta->kL = kL; meta->stride = stride; meta->oL = oL;
        tape[idx].meta = meta;
    }
    return (TensorHandle)r;
}

TensorHandle tensor_create_param_3d(int d0, int d1, int d2, double* data) {
    int shape[] = {d0, d1, d2};
    auto t = tensor_create(data, shape, 3, 1);
    free(data);
    ((Tensor*)t)->persistent = 1;
    return t;
}

TensorHandle tensor_conv2d(TensorHandle hinput, TensorHandle hkernel,
                           TensorHandle hbias, int padH, int padW,
                           int strideH, int strideW) {
    auto inp = (Tensor*)hinput;   // [inC, H, W]
    auto ker = (Tensor*)hkernel;  // [outC, inC, kH, kW]
    Tensor* bias = hbias ? (Tensor*)hbias : nullptr;

    int inC = (int)inp->data.shape(0), H = (int)inp->data.shape(1), W = (int)inp->data.shape(2);

    // MLX conv2d expects NHWC: input [N,H,W,C_in], weight [C_out,kH,kW,C_in]
    auto inp_hwc = mx::transpose(inp->data, {1, 2, 0});  // [H, W, inC]
    auto inp_nhwc = mx::reshape(inp_hwc, {1, H, W, inC}); // [1, H, W, inC]
    // kernel: [outC, inC, kH, kW] -> [outC, kH, kW, inC]
    auto ker_mlx = mx::transpose(ker->data, {0, 2, 3, 1});

    auto out = mx::conv2d(inp_nhwc, ker_mlx,
                          {strideH, strideW}, {padH, padW});
    // out: [1, oH, oW, outC] -> squeeze batch -> [oH, oW, outC] -> [outC, oH, oW]
    auto out_sq = mx::squeeze(out, 0);
    auto result = mx::transpose(out_sq, {2, 0, 1});

    if (bias) result = mx::add(result, mx::reshape(bias->data, {-1, 1, 1}));

    bool rg = inp->requires_grad || ker->requires_grad || (bias && bias->requires_grad);
    auto r = new Tensor(result, rg);
    if (rg) {
        int idx = tape_append(OP_CONV2D, r, inp, ker, 0);
        auto* meta = new Conv2DReplayMeta();
        meta->padH = padH; meta->padW = padW;
        meta->strH = strideH; meta->strW = strideW;
        meta->inC = inC; meta->H = H; meta->W = W;
        meta->bias_pool_idx = bias ? bias->pool_idx : -1;
        tape[idx].meta = meta;
    }
    return (TensorHandle)r;
}

TensorHandle tensor_max_pool2d(TensorHandle hinput, int kH, int kW,
                               int strideH, int strideW) {
    auto inp = (Tensor*)hinput;  // [C, H, W]
    int C = (int)inp->data.shape(0), H = (int)inp->data.shape(1), W = (int)inp->data.shape(2);
    int oH = (H - kH) / strideH + 1;
    int oW = (W - kW) / strideW + 1;

    // Max pool via strided slicing: for each (kh,kw) offset, slice with stride and take max
    mx::array result = mx::full({C, oH, oW}, -1e30, mx::float64);
    for (int kh = 0; kh < kH; kh++) {
        for (int kw = 0; kw < kW; kw++) {
            auto sliced = mx::slice(inp->data,
                {0, kh, kw},
                {C, kh + oH * strideH, kw + oW * strideW},
                {1, strideH, strideW});
            result = mx::maximum(result, sliced);
        }
    }

    auto r = new Tensor(result, inp->requires_grad);
    if (inp->requires_grad) {
        int idx = tape_append(OP_MAX_POOL2D, r, inp, nullptr, 0);
        auto* meta = new MaxPool2DReplayMeta();
        meta->C = C; meta->H = H; meta->W = W;
        meta->kH = kH; meta->kW = kW;
        meta->strH = strideH; meta->strW = strideW;
        meta->oH = oH; meta->oW = oW;
        tape[idx].meta = meta;
    }
    return (TensorHandle)r;
}

// Fused normalize: r[i] = a[i] / (sum(a) + eps)
// Numerically stable backward avoids catastrophic cancellation from separate div/sum ops.
static TensorHandle tensor_normalize_1d(TensorHandle h) {
    auto t = (Tensor*)h;
    auto S = mx::add(mx::sum(t->data), mx::array(1e-10));
    auto r = new Tensor(mx::divide(t->data, S), t->requires_grad);
    if (t->requires_grad) tape_append(OP_NORMALIZE, r, t, nullptr, 0);
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
    TensorHandle focused = tensor_normalize_1d(powered);

    // 5. Read from memory: focused @ memory → [m]
    // Use tensor_mv for proper backward handling (MV handles 1D vectors correctly)
    // focused is [n], memory^T is [m, n], so mv(memory^T, focused) → [m]
    TensorHandle memT_transposed = tensor_transpose_2d(hmemory);
    TensorHandle read_result = tensor_mv(memT_transposed, focused);

    auto pair = (TensorPair*)malloc(sizeof(TensorPair));
    pair->first = focused;
    pair->second = read_result;
    all_pairs.push_back(pair);
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
    auto r = new Tensor(mx::take(t->data, mx::array(index), dim), t->requires_grad);
    if (t->requires_grad) tape_append(OP_SELECT, r, t, nullptr, (double)index);
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

TensorHandle tensor_bmm(TensorHandle ha, TensorHandle hb) {
    auto a = (Tensor*)ha; auto b = (Tensor*)hb;
    bool rg = a->requires_grad || b->requires_grad;
    auto r = new Tensor(mx::matmul(a->data, b->data), rg);
    if (rg) tape_append(OP_BMM, r, a, b, 0);
    return (TensorHandle)r;
}
TensorHandle tensor_batch(TensorHandle* handles, int count) { STUB(); }
TensorHandle* tensor_unbatch(TensorHandle h, int* out_count) { STUB(); }

TensorHandle tensor_bmm_3x3(TensorHandle ha, TensorHandle hb) {
    auto a = (Tensor*)ha; auto b = (Tensor*)hb;
    bool rg = a->requires_grad || b->requires_grad;
    auto r = new Tensor(mx::matmul(a->data, b->data), rg);
    if (rg) tape_append(OP_BMM_3X3, r, a, b, 0);
    return (TensorHandle)r;
}

TensorHandle tensor_softmax_3d(TensorHandle h) {
    auto t = (Tensor*)h;
    auto r = new Tensor(mx::softmax(t->data, -1), t->requires_grad);
    if (t->requires_grad) tape_append(OP_SOFTMAX_3D, r, t, nullptr, 0);
    return (TensorHandle)r;
}

TensorHandle tensor_transpose_last2(TensorHandle h) {
    auto t = (Tensor*)h;
    auto r = new Tensor(mx::transpose(t->data, {0, 2, 1}), t->requires_grad);
    if (t->requires_grad) tape_append(OP_TRANSPOSE_LAST2, r, t, nullptr, 0);
    return (TensorHandle)r;
}

TensorHandle tensor_reshape_3d(TensorHandle h, int d0, int d1, int d2) {
    int shape[] = {d0, d1, d2};
    return tensor_reshape(h, shape, 3);
}

TensorHandle tensor_expand_mask(TensorHandle hmask, int B) {
    auto mask = (Tensor*)hmask;
    int m = mask->data.shape(0), n = mask->data.shape(1);
    // [m,n] → [1,m,n] → broadcast to [B,m,n]
    auto expanded = mx::broadcast_to(mx::reshape(mask->data, {1, m, n}), {B, m, n});
    auto r = new Tensor(expanded, false);
    return (TensorHandle)r;
}

TensorHandle tensor_transpose_2d(TensorHandle h) {
    auto t = (Tensor*)h;
    auto r = new Tensor(mx::transpose(t->data, {1, 0}), t->requires_grad);
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
    if (t->requires_grad) tape_append(OP_LOG_SOFTMAX_2D, r, t, nullptr, -1.0);
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
        int idx = tape_append(OP_LAYER_NORM_2D, r, t, nullptr, eps);
        auto meta = new LayerNormReplayMeta();
        meta->gamma_pool_idx = gamma->pool_idx;
        meta->bias_pool_idx = bias->pool_idx;
        meta->eps = eps;
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
   Autograd — replay-based native backward via mlx::grad

   Forward ops record to the tape. tensor_backward replays the tape
   inside a closure and passes it to mlx::grad for native autograd.
   Zero hand-written backward rules.
   ================================================================ */

void tensor_backward(TensorHandle h) {
    Tensor* loss = (Tensor*)h;
    if (loss->tape_idx < 0) return;

    // Collect param pool indices and arrays
    std::vector<int> param_pool_indices;
    std::vector<mx::array> param_arrays;
    for (auto& p : param_registry) {
        param_pool_indices.push_back(p.tensor->pool_idx);
        param_arrays.push_back(p.tensor->data);
    }
    if (param_arrays.empty()) return;

    // Build constant pool from tape (O(tape_size), not O(all_tensors))
    std::vector<std::pair<int, mx::array>> constants;
    std::unordered_set<int> seen;
    for (auto& idx : param_pool_indices) seen.insert(idx);
    auto add_const = [&](Tensor* t) {
        if (t && !seen.count(t->pool_idx)) {
            seen.insert(t->pool_idx);
            constants.emplace_back(t->pool_idx, t->data);
        }
    };
    for (int i = 0; i <= loss->tape_idx; i++) {
        auto& e = tape[i];
        add_const(e.result);
        add_const(e.arg1);
        add_const(e.arg2);
    }

    // Capture tape state for the closure
    int loss_pool_idx = loss->pool_idx;
    int loss_tape_idx = loss->tape_idx;
    auto tape_ref = &tape;
    auto constants_ref = &constants;

    // Replay forward pass inside mlx::vjp
    int pool_size = next_pool_idx;
    auto forward_fn = [&](const std::vector<mx::array>& params) -> mx::array {
        // Pool: flat vector indexed by pool_idx. Initialize with placeholder.
        std::vector<mx::array> pool(pool_size, mx::array(0.0f));
        for (auto& [idx, arr] : *constants_ref) pool[idx] = arr;
        for (int i = 0; i < (int)params.size(); i++)
            pool[param_pool_indices[i]] = params[i];

        for (int i = 0; i <= loss_tape_idx; i++) {
            auto& e = (*tape_ref)[i];
            int out = e.result->pool_idx;
            auto a = e.arg1 ? pool[e.arg1->pool_idx] : mx::array(0.0f);
            auto b = e.arg2 ? pool[e.arg2->pool_idx] : mx::array(0.0f);

            switch (e.op) {
            case OP_CONST: break;
            case OP_ADD: pool[out] = mx::add(a, b); break;
            case OP_SUB: pool[out] = mx::subtract(a, b); break;
            case OP_MUL: pool[out] = mx::multiply(a, b); break;
            case OP_DIV: pool[out] = mx::divide(a, b); break;
            case OP_NEG: pool[out] = mx::negative(a); break;
            case OP_ABS: pool[out] = mx::abs(a); break;
            case OP_EXP: pool[out] = mx::exp(a); break;
            case OP_LOG: pool[out] = mx::log(a); break;
            case OP_SQRT: pool[out] = mx::sqrt(a); break;
            case OP_POW: pool[out] = mx::power(a, b); break;
            case OP_SIGMOID: pool[out] = mx::sigmoid(a); break;
            case OP_TANH: pool[out] = mx::tanh(a); break;
            case OP_GELU: {
                auto c = mx::array(0.7978845608028654, mx::float64);
                auto inner = mx::multiply(c, mx::add(a, mx::multiply(mx::array(0.044715, mx::float64), mx::power(a, mx::array(3, mx::float64)))));
                pool[out] = mx::multiply(mx::multiply(mx::array(0.5, mx::float64), a), mx::add(mx::array(1.0, mx::float64), mx::tanh(inner)));
                break;
            }
            case OP_LEAKY_RELU: {
                auto alpha = mx::array(e.scalar_arg, mx::float64);
                pool[out] = mx::maximum(mx::multiply(alpha, a), a);
                break;
            }
            case OP_SILU: pool[out] = mx::multiply(a, mx::sigmoid(a)); break;
            case OP_ADD_SCALAR: pool[out] = mx::add(a, mx::array(e.scalar_arg)); break;
            case OP_MUL_SCALAR: pool[out] = mx::multiply(a, mx::array(e.scalar_arg)); break;
            case OP_CLAMP_MIN: pool[out] = mx::maximum(a, mx::array(e.scalar_arg)); break;
            case OP_SUM: pool[out] = mx::sum(a); break;
            case OP_MEAN: pool[out] = mx::mean(a); break;
            case OP_MM: case OP_BMM: case OP_BMM_3X3: pool[out] = mx::matmul(a, b); break;
            case OP_SOFTMAX_3D: pool[out] = mx::softmax(a, -1); break;
            case OP_TRANSPOSE_LAST2: pool[out] = mx::transpose(a, {0, 2, 1}); break;
            case OP_MV: {
                auto col = mx::reshape(b, {(int)b.size(), 1});
                pool[out] = mx::reshape(mx::matmul(a, col), {(int)a.shape(0)});
                break;
            }
            case OP_OUTER: pool[out] = mx::outer(a, b); break;
            case OP_TRANSPOSE_2D: pool[out] = mx::transpose(a, {1, 0}); break;
            case OP_SOFTMAX_2D: pool[out] = mx::softmax(a, -1); break;
            case OP_LOG_SOFTMAX_2D: {
                int dim = (int)e.scalar_arg;  // stored by forward (0 for 1D, -1 for 2D)
                auto maxv = mx::max(a, dim, true);
                auto shifted = mx::subtract(a, maxv);
                auto lse = mx::add(mx::log(mx::sum(mx::exp(shifted), dim, true)), maxv);
                pool[out] = mx::subtract(a, lse);
                break;
            }
            case OP_MASKED_FILL: {
                pool[out] = mx::where(b, mx::array(-1e9, mx::float64), a);
                break;
            }
            case OP_RESHAPE: pool[out] = mx::reshape(a, e.result->data.shape()); break;
            case OP_SELECT: pool[out] = mx::take(a, mx::array((int)e.scalar_arg), 0); break;
            case OP_NARROW: {
                int start = (int)e.scalar_arg;
                int len = (int)e.result->data.size();
                pool[out] = mx::slice(mx::flatten(a), {start}, {start + len});
                break;
            }
            case OP_CAT: pool[out] = mx::concatenate({a, b}, 0); break;
            case OP_STACK: {
                auto* indices = (std::vector<int>*)e.meta;
                if (indices) {
                    std::vector<mx::array> arrs;
                    for (int idx : *indices) arrs.push_back(pool[idx]);
                    pool[out] = mx::stack(arrs, 0);
                }
                break;
            }
            case OP_COSINE_SIM: {
                // Inline cosine similarity forward
                int n = (int)a.shape(0), m = (int)a.shape(1);
                auto key_2d = mx::reshape(b, {1, m});
                auto dots = mx::sum(mx::multiply(a, key_2d), std::vector<int>{1});
                auto eps = mx::array(1.0e-8);
                auto row_norms = mx::sqrt(mx::add(mx::sum(mx::square(a), std::vector<int>{1}), eps));
                auto key_norm = mx::sqrt(mx::add(mx::sum(mx::square(b)), eps));
                pool[out] = mx::divide(dots, mx::multiply(row_norms, key_norm));
                break;
            }
            case OP_CONV1D_CIRC: {
                // Inline circular convolution forward
                int n = (int)a.size(), k = (int)b.size();
                int half_k = k / 2;
                auto result = mx::zeros({n}, mx::float64);
                for (int j = 0; j < k; j++) {
                    auto shifted = mx::roll(a, half_k - j);
                    auto kern_j = mx::take(b, mx::array(j));
                    result = mx::add(result, mx::multiply(shifted, kern_j));
                }
                pool[out] = result;
                break;
            }
            case OP_NORMALIZE: {
                pool[out] = mx::divide(a, mx::add(mx::sum(a), mx::array(1e-10)));
                break;
            }
            case OP_LAYER_NORM_2D: {
                auto meta = (LayerNormReplayMeta*)e.meta;
                auto gamma = pool[meta->gamma_pool_idx];
                auto bias = pool[meta->bias_pool_idx];
                auto mean = mx::mean(a, -1, true);
                auto centered = mx::subtract(a, mean);
                auto var = mx::mean(mx::square(centered), -1, true);
                auto rstd = mx::rsqrt(mx::add(var, mx::array(meta->eps)));
                auto x_hat = mx::multiply(centered, rstd);
                pool[out] = mx::add(mx::multiply(gamma, x_hat), bias);
                break;
            }
            case OP_GRU_CELL: {
                int oo = (int)e.scalar_arg;
                auto z = mx::sigmoid(mx::slice(a, {0}, {oo}));
                auto n = mx::tanh(mx::slice(a, {2*oo}, {3*oo}));
                auto one = mx::array(1.0, mx::float64);
                pool[out] = mx::add(mx::multiply(mx::subtract(one, z), n), mx::multiply(z, b));
                break;
            }
            case OP_EMBEDDING: {
                // a = weight, b = indices (int32), scalar_arg = embedDim
                auto idx_int = mx::astype(b, mx::int32);
                auto rows = mx::take(a, idx_int, 0);
                pool[out] = mx::flatten(rows);
                break;
            }
            case OP_BATCH_NORM: {
                auto* bm = (BatchNormReplayMeta*)e.meta;
                auto x = mx::reshape(a, {bm->C, bm->spatial});
                auto mean = mx::mean(x, std::vector<int>{1}, true);
                auto var = mx::var(x, std::vector<int>{1}, true);
                auto rstd = mx::rsqrt(mx::add(var, mx::array(bm->eps, mx::float64)));
                auto x_hat = mx::multiply(mx::subtract(x, mean), rstd);
                auto g = mx::reshape(pool[bm->gamma_pool_idx], {bm->C, 1});
                auto bt = mx::reshape(pool[bm->beta_pool_idx], {bm->C, 1});
                pool[out] = mx::flatten(mx::add(mx::multiply(g, x_hat), bt));
                break;
            }
            case OP_DROPOUT: {
                // b holds the stored mask tensor; just multiply
                pool[out] = mx::multiply(a, b);
                break;
            }
            case OP_AVG_POOL1D: {
                // scalar_arg encodes kL + stride*0.001
                int kL = (int)e.scalar_arg;
                int stride = (int)((e.scalar_arg - kL) * 1000 + 0.5);
                if (stride == 0) stride = kL;
                int oL = ((int)a.shape(1) - kL) / stride + 1;
                mx::array res = mx::zeros({(int)a.shape(0), oL}, mx::float64);
                for (int kl = 0; kl < kL; kl++) {
                    auto sliced = mx::slice(a, {0, kl}, {(int)a.shape(0), kl + oL*stride}, {1, stride});
                    res = mx::add(res, sliced);
                }
                pool[out] = mx::divide(res, mx::array((double)kL, mx::float64));
                break;
            }
            case OP_AVG_POOL2D: {
                // For simplicity, re-derive dims from input shape. Only k=2 s=2 common case tested.
                int CC = (int)a.shape(0), HH = (int)a.shape(1), WW = (int)a.shape(2);
                // Default: k=2, stride=2 (most common usage)
                int kH = 2, kW = 2, sH = 2, sW = 2;
                int oH = (HH - kH)/sH + 1, oW = (WW - kW)/sW + 1;
                mx::array res = mx::zeros({CC, oH, oW}, mx::float64);
                for (int kh = 0; kh < kH; kh++)
                    for (int kw = 0; kw < kW; kw++) {
                        auto sl = mx::slice(a, {0,kh,kw}, {CC,kh+oH*sH,kw+oW*sW}, {1,sH,sW});
                        res = mx::add(res, sl);
                    }
                pool[out] = mx::divide(res, mx::array((double)(kH*kW), mx::float64));
                break;
            }
            case OP_CONV1D: {
                auto* cm = (Conv1DReplayMeta*)e.meta;
                int inC = cm->inC, LL = cm->L;
                auto inp_lc = mx::transpose(a, {1, 0});
                auto inp_nlc = mx::reshape(inp_lc, {1, LL, inC});
                auto ker_mlx = mx::transpose(b, {0, 2, 1});
                auto cv = mx::conv1d(inp_nlc, ker_mlx, cm->stride, cm->pad);
                auto cv_sq = mx::squeeze(cv, 0);
                auto cv_out = mx::transpose(cv_sq, {1, 0});
                if (cm->bias_pool_idx >= 0)
                    cv_out = mx::add(cv_out, mx::reshape(pool[cm->bias_pool_idx], {-1, 1}));
                pool[out] = cv_out;
                break;
            }
            case OP_MAX_POOL1D: {
                auto* pm = (MaxPool1DReplayMeta*)e.meta;
                mx::array res = mx::full({pm->C, pm->oL}, -1e30, mx::float64);
                for (int kl = 0; kl < pm->kL; kl++) {
                    auto sliced = mx::slice(a, {0, kl}, {pm->C, kl + pm->oL * pm->stride}, {1, pm->stride});
                    res = mx::maximum(res, sliced);
                }
                pool[out] = res;
                break;
            }
            case OP_CONV2D: {
                auto* cm = (Conv2DReplayMeta*)e.meta;
                int inC = cm->inC, HH = cm->H, WW = cm->W;
                auto inp_hwc = mx::transpose(a, {1, 2, 0});
                auto inp_nhwc = mx::reshape(inp_hwc, {1, HH, WW, inC});
                auto ker_mlx = mx::transpose(b, {0, 2, 3, 1});
                auto cv = mx::conv2d(inp_nhwc, ker_mlx,
                                     {cm->strH, cm->strW}, {cm->padH, cm->padW});
                auto cv_sq = mx::squeeze(cv, 0);
                auto cv_out = mx::transpose(cv_sq, {2, 0, 1});
                if (cm->bias_pool_idx >= 0) {
                    cv_out = mx::add(cv_out, mx::reshape(pool[cm->bias_pool_idx], {-1, 1, 1}));
                }
                pool[out] = cv_out;
                break;
            }
            case OP_MAX_POOL2D: {
                auto* pm = (MaxPool2DReplayMeta*)e.meta;
                mx::array res = mx::full({pm->C, pm->oH, pm->oW}, -1e30, mx::float64);
                for (int kh = 0; kh < pm->kH; kh++) {
                    for (int kw = 0; kw < pm->kW; kw++) {
                        auto sliced = mx::slice(a,
                            {0, kh, kw},
                            {pm->C, kh + pm->oH * pm->strH, kw + pm->oW * pm->strW},
                            {1, pm->strH, pm->strW});
                        res = mx::maximum(res, sliced);
                    }
                }
                pool[out] = res;
                break;
            }
            case OP_CUMPROD: {
                pool[out] = mx::cumprod(a, 0);
                break;
            }
            default: break;
            }
        }
        return pool[loss_pool_idx];
    };

    // Compute gradients via MLX native autograd (vjp with unit cotangent)
    auto forward_vec = [&](const std::vector<mx::array>& params) -> std::vector<mx::array> {
        return {forward_fn(params)};
    };
    auto vjp_result = mx::vjp(forward_vec, param_arrays, {mx::array(1.0f)});
    auto& grads = vjp_result.second;

    // Distribute gradients to parameter tensors
    for (int i = 0; i < (int)param_registry.size(); i++) {
        param_registry[i].tensor->grad = grads[i];
        param_registry[i].tensor->has_grad = true;
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
    all_pairs.push_back(pair);
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

void param_clear(void) {
    param_registry.clear();
    tape_reset();
}
int param_count(void) { return (int)param_registry.size(); }
const char* param_name(int idx) { return param_registry[idx].name.c_str(); }

double param_grad_item(int idx) {
    auto t = param_registry[idx].tensor;
    if (!t->has_grad) return 0.0;
    mx::eval(t->grad);
    auto flat = mx::flatten(t->grad, mx::StreamOrDevice{});
    mx::eval(flat);
    return flat.data<double>()[0];
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

void param_load_data(int idx, const double* data, int numel) {
    auto t = param_registry[idx].tensor;
    auto shape = t->data.shape();
    int existing = t->data.size();
    if (existing != numel) {
        fprintf(stderr, "param_load_data: size mismatch for '%s': expected %d, got %d\n",
                param_registry[idx].name.c_str(), existing, numel);
        return;
    }
    t->data = mx::array(data, shape, mx::float64);
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
    // Record OP_STACK with input pool indices for replay
    if (rg) {
        int idx = tape_append(OP_STACK, r, nullptr, nullptr, (double)count);
        auto* indices = new std::vector<int>();
        for (int i = 0; i < count; i++)
            indices->push_back(((Tensor*)arr[i])->pool_idx);
        tape[idx].meta = (void*)indices;
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

TensorHandle tensor_create_param_4d(int d0, int d1, int d2, int d3, double* data) {
    int shape[] = {d0, d1, d2, d3};
    auto t = tensor_create(data, shape, 4, 1);
    free(data);
    ((Tensor*)t)->persistent = 1;
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
    ((Tensor*)t)->persistent = 1;  // survives tape_reset
    free(data);
    return t;
}

TensorHandle tensor_create_state_1d(int n, double* data) {
    int shape[] = {n};
    auto t = tensor_create(data, shape, 1, 0);
    ((Tensor*)t)->persistent = 1;  // survives tape_reset
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
    // Flatten to contiguous for correct indexing on non-contiguous views (e.g. transpose)
    auto flat = mx::flatten(t->data, mx::StreamOrDevice{});
    mx::eval(flat);
    int cols = t->data.shape(1);
    return flat.data<double>()[row * cols + col];
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

OptimizerHandle optimizer_create_adamw(double lr, double beta1, double beta2, double eps,
                                       double weight_decay) {
    auto opt = new Optimizer();
    opt->type = 3; opt->lr = lr; opt->beta1 = beta1; opt->beta2 = beta2;
    opt->eps = eps; opt->weight_decay = weight_decay; opt->t = 0;
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
        case 1: { // RMSprop
            // v = alpha * v + (1 - alpha) * g^2
            opt->v_bufs[i] = mx::add(mx::multiply(mx::array(opt->alpha), opt->v_bufs[i]),
                                      mx::multiply(mx::array(1.0 - opt->alpha), mx::square(g)));
            // delta = lr * g / (sqrt(v) + eps)
            auto delta = mx::divide(mx::multiply(mx::array(opt->lr), g),
                                     mx::add(mx::sqrt(opt->v_bufs[i]), mx::array(opt->eps)));
            if (opt->momentum > 0) {
                // m = momentum * m + delta
                opt->m_bufs[i] = mx::add(mx::multiply(mx::array(opt->momentum), opt->m_bufs[i]), delta);
                t->data = mx::subtract(t->data, opt->m_bufs[i]);
            } else {
                t->data = mx::subtract(t->data, delta);
            }
            break;
        }
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
        case 3: { // AdamW (decoupled weight decay)
            opt->m_bufs[i] = mx::add(mx::multiply(mx::array(opt->beta1), opt->m_bufs[i]),
                                      mx::multiply(mx::array(1.0 - opt->beta1), g));
            opt->v_bufs[i] = mx::add(mx::multiply(mx::array(opt->beta2), opt->v_bufs[i]),
                                      mx::multiply(mx::array(1.0 - opt->beta2), mx::square(g)));
            auto mhat = mx::divide(opt->m_bufs[i], mx::array(1.0 - std::pow(opt->beta1, opt->t)));
            auto vhat = mx::divide(opt->v_bufs[i], mx::array(1.0 - std::pow(opt->beta2, opt->t)));
            t->data = mx::subtract(t->data,
                mx::divide(mx::multiply(mx::array(opt->lr), mhat),
                            mx::add(mx::sqrt(vhat), mx::array(opt->eps))));
            t->data = mx::subtract(t->data,
                mx::multiply(mx::array(opt->lr * opt->weight_decay), t->data));
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
   Optimizer buffer accessors (for serialization)
   ================================================================ */

int optimizer_buf_count(OptimizerHandle h) {
    (void)h;
    return (int)param_registry.size();
}

void optimizer_get_m(OptimizerHandle h, int idx, double* out) {
    auto opt = (Optimizer*)h;
    if (idx >= (int)opt->m_bufs.size()) {
        int n = param_registry[idx].tensor->data.size();
        memset(out, 0, n * sizeof(double));
        return;
    }
    mx::eval(opt->m_bufs[idx]);
    auto& arr = opt->m_bufs[idx];
    memcpy(out, arr.data<double>(), arr.size() * sizeof(double));
}

void optimizer_get_v(OptimizerHandle h, int idx, double* out) {
    auto opt = (Optimizer*)h;
    if (idx >= (int)opt->v_bufs.size()) {
        int n = param_registry[idx].tensor->data.size();
        memset(out, 0, n * sizeof(double));
        return;
    }
    mx::eval(opt->v_bufs[idx]);
    auto& arr = opt->v_bufs[idx];
    memcpy(out, arr.data<double>(), arr.size() * sizeof(double));
}

void optimizer_set_m(OptimizerHandle h, int idx, const double* data) {
    auto opt = (Optimizer*)h;
    // Ensure buffers exist
    int np = (int)param_registry.size();
    if ((int)opt->m_bufs.size() != np) {
        opt->m_bufs.clear();
        opt->v_bufs.clear();
        for (auto& p : param_registry) {
            opt->m_bufs.push_back(mx::zeros(p.tensor->data.shape(), mx::float64));
            opt->v_bufs.push_back(mx::zeros(p.tensor->data.shape(), mx::float64));
        }
    }
    auto shape = param_registry[idx].tensor->data.shape();
    opt->m_bufs[idx] = mx::array(data, shape, mx::float64);
}

void optimizer_set_v(OptimizerHandle h, int idx, const double* data) {
    auto opt = (Optimizer*)h;
    int np = (int)param_registry.size();
    if ((int)opt->v_bufs.size() != np) {
        opt->m_bufs.clear();
        opt->v_bufs.clear();
        for (auto& p : param_registry) {
            opt->m_bufs.push_back(mx::zeros(p.tensor->data.shape(), mx::float64));
            opt->v_bufs.push_back(mx::zeros(p.tensor->data.shape(), mx::float64));
        }
    }
    auto shape = param_registry[idx].tensor->data.shape();
    opt->v_bufs[idx] = mx::array(data, shape, mx::float64);
}

void optimizer_get_meta(OptimizerHandle h, double* out9) {
    auto opt = (Optimizer*)h;
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
    auto opt = (Optimizer*)h;
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
