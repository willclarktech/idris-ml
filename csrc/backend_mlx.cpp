/* backend_mlx.cpp — MLX backend implementing backend.h.
 *
 * Uses Apple's MLX framework for GPU-accelerated tensor operations
 * on Apple Silicon via Metal. Custom tape-based autograd (same structure
 * as backend_tape.c) with MLX arrays for compute.
 *
 * Status: SKELETON — only stubs. Not yet functional.
 */

#include "backend.h"
#include <cstdlib>
#include <cstdio>
#include <cstring>
#include <cmath>
#include <iostream>

// MLX C++ API
#include <mlx/mlx.h>

namespace mx = mlx::core;
using Shape = mx::Shape;  // SmallVector<int>

/* ================================================================
   Stub macro — prints function name and aborts
   ================================================================ */

#define STUB() do { \
    fprintf(stderr, "MLX backend: %s not implemented\n", __func__); \
    abort(); \
} while(0)

/* ================================================================
   Tensor representation
   ================================================================ */

struct Tensor {
    mx::array arr;          // MLX array (reference-counted)
    mx::array grad;         // gradient (empty if not computed)
    bool requires_grad;
    bool has_grad;
    char* param_name;       // NULL for intermediates
    int persistent;         // 1 = parameter, 0 = intermediate

    Tensor(mx::array a, bool rg = false)
        : arr(std::move(a)), grad(mx::array(0.0f)), requires_grad(rg),
          has_grad(false), param_name(nullptr), persistent(0) {}
};

/* ================================================================
   Lifecycle
   ================================================================ */

extern "C" {

TensorHandle tensor_create_scalar(double value, int requires_grad) {
    auto t = new Tensor(mx::array(value), requires_grad != 0);
    return (TensorHandle)t;
}

TensorHandle tensor_create(double* data, int* shape, int rank, int requires_grad) {
    Shape sh(shape, shape + rank);
    int numel = 1;
    for (int i = 0; i < rank; i++) numel *= shape[i];
    auto t = new Tensor(
        mx::array(data, sh, mx::float64),
        requires_grad != 0
    );
    return (TensorHandle)t;
}

TensorHandle tensor_clone(TensorHandle h) {
    auto t = (Tensor*)h;
    auto c = new Tensor(mx::array(t->arr), false);
    return (TensorHandle)c;
}

void tensor_free(TensorHandle h) {
    if (h) delete (Tensor*)h;
}

/* ================================================================
   Accessors
   ================================================================ */

double tensor_item(TensorHandle h) {
    auto t = (Tensor*)h;
    mx::eval(t->arr);
    return t->arr.item<double>();
}

int tensor_numel(TensorHandle h) {
    return (int)((Tensor*)h)->arr.size();
}

int tensor_dim(TensorHandle h) {
    return (int)((Tensor*)h)->arr.ndim();
}

int tensor_size(TensorHandle h, int dim) {
    return (int)((Tensor*)h)->arr.shape(dim);
}

void tensor_to_doubles(TensorHandle h, double* out) {
    auto t = (Tensor*)h;
    mx::eval(t->arr);
    auto data = t->arr.data<double>();
    memcpy(out, data, t->arr.size() * sizeof(double));
}

/* ================================================================
   Stubs for everything else
   ================================================================ */

TensorHandle tensor_add(TensorHandle a, TensorHandle b) {
    auto ta = (Tensor*)a;
    auto tb = (Tensor*)b;
    auto r = new Tensor(mx::add(ta->arr, tb->arr),
                        ta->requires_grad || tb->requires_grad);
    return (TensorHandle)r;
}

TensorHandle tensor_sub(TensorHandle a, TensorHandle b) { STUB(); }
TensorHandle tensor_mul(TensorHandle a, TensorHandle b) {
    auto ta = (Tensor*)a;
    auto tb = (Tensor*)b;
    auto r = new Tensor(mx::multiply(ta->arr, tb->arr),
                        ta->requires_grad || tb->requires_grad);
    return (TensorHandle)r;
}
TensorHandle tensor_div(TensorHandle a, TensorHandle b) { STUB(); }
TensorHandle tensor_neg(TensorHandle t) { STUB(); }
TensorHandle tensor_abs(TensorHandle t) { STUB(); }
TensorHandle tensor_exp(TensorHandle t) { STUB(); }
TensorHandle tensor_log(TensorHandle t) { STUB(); }
TensorHandle tensor_sqrt(TensorHandle t) { STUB(); }
TensorHandle tensor_pow(TensorHandle base, TensorHandle exp) { STUB(); }
TensorHandle tensor_sigmoid(TensorHandle t) { STUB(); }
TensorHandle tensor_tanh(TensorHandle t) { STUB(); }
TensorHandle tensor_add_scalar(TensorHandle t, double s) { STUB(); }
TensorHandle tensor_mul_scalar(TensorHandle t, double s) {
    auto tt = (Tensor*)t;
    auto r = new Tensor(mx::multiply(tt->arr, mx::array(s)),
                        tt->requires_grad);
    return (TensorHandle)r;
}
TensorHandle tensor_clamp_min(TensorHandle t, double min_val) {
    auto tt = (Tensor*)t;
    auto r = new Tensor(mx::maximum(tt->arr, mx::array(min_val)),
                        tt->requires_grad);
    return (TensorHandle)r;
}

TensorHandle tensor_sum(TensorHandle t) {
    auto tt = (Tensor*)t;
    auto r = new Tensor(mx::sum(tt->arr), tt->requires_grad);
    return (TensorHandle)r;
}
TensorHandle tensor_sum_dim(TensorHandle t, int dim, int keepdim) { STUB(); }
TensorHandle tensor_mean(TensorHandle t) { STUB(); }

TensorHandle tensor_matmul(TensorHandle a, TensorHandle b) { STUB(); }
TensorHandle tensor_mv(TensorHandle mat, TensorHandle vec) { STUB(); }
TensorHandle tensor_dot(TensorHandle a, TensorHandle b) { STUB(); }
TensorHandle tensor_outer(TensorHandle a, TensorHandle b) { STUB(); }

TensorHandle tensor_softmax(TensorHandle t, int dim) { STUB(); }
TensorHandle tensor_log_softmax(TensorHandle t, int dim) { STUB(); }

TensorHandle tensor_bce_with_logits(TensorHandle input, TensorHandle target) { STUB(); }
TensorHandle tensor_cross_entropy(TensorHandle input, TensorHandle target) { STUB(); }
TensorHandle tensor_mse_loss(TensorHandle input, TensorHandle target) { STUB(); }

TensorHandle tensor_cosine_similarity(TensorHandle a, TensorHandle b, int dim) { STUB(); }
TensorHandle tensor_conv1d_circular(TensorHandle input, TensorHandle kernel) { STUB(); }
TensorPair* tensor_ntm_read_head(TensorHandle memory, TensorHandle prev_weights,
    TensorHandle key, TensorHandle beta, TensorHandle g,
    TensorHandle gamma, TensorHandle shift_kernel) { STUB(); }
TensorHandle tensor_ntm_interp_write(TensorHandle memory, TensorHandle weights,
    TensorHandle add_vector) { STUB(); }

TensorHandle tensor_reshape(TensorHandle t, int* shape, int rank) {
    auto tt = (Tensor*)t;
    Shape sh(shape, shape + rank);
    auto r = new Tensor(mx::reshape(tt->arr, sh), tt->requires_grad);
    return (TensorHandle)r;
}
TensorHandle tensor_unsqueeze(TensorHandle t, int dim) { STUB(); }
TensorHandle tensor_squeeze(TensorHandle t, int dim) { STUB(); }
TensorHandle tensor_select(TensorHandle t, int dim, int index) { STUB(); }
TensorHandle tensor_stack(TensorHandle* tensors, int count, int dim) { STUB(); }
TensorHandle tensor_cat(TensorHandle* tensors, int count, int dim) { STUB(); }
TensorHandle tensor_cat2(TensorHandle a, TensorHandle b) {
    auto ta = (Tensor*)a;
    auto tb = (Tensor*)b;
    auto r = new Tensor(mx::concatenate({ta->arr, tb->arr}, 0),
                        ta->requires_grad || tb->requires_grad);
    return (TensorHandle)r;
}
TensorHandle tensor_narrow(TensorHandle t, int dim, int start, int len) {
    auto tt = (Tensor*)t;
    auto r = new Tensor(mx::slice(tt->arr, {start}, {start + len}),
                        tt->requires_grad);
    return (TensorHandle)r;
}
TensorHandle tensor_mm(TensorHandle a, TensorHandle b) {
    auto ta = (Tensor*)a;
    auto tb = (Tensor*)b;
    auto r = new Tensor(mx::matmul(ta->arr, tb->arr),
                        ta->requires_grad || tb->requires_grad);
    return (TensorHandle)r;
}
TensorHandle tensor_bmm(TensorHandle a, TensorHandle b) { STUB(); }
TensorHandle tensor_batch(TensorHandle* handles, int count) { STUB(); }
TensorHandle* tensor_unbatch(TensorHandle h, int* out_count) { STUB(); }
TensorHandle tensor_transpose_2d(TensorHandle t) {
    auto tt = (Tensor*)t;
    auto r = new Tensor(mx::transpose(tt->arr), tt->requires_grad);
    return (TensorHandle)r;
}
TensorHandle tensor_softmax_2d(TensorHandle t) {
    auto tt = (Tensor*)t;
    auto r = new Tensor(mx::softmax(tt->arr, -1), tt->requires_grad);
    return (TensorHandle)r;
}
TensorHandle tensor_masked_fill(TensorHandle t, TensorHandle mask, double value) {
    auto tt = (Tensor*)t;
    auto tm = (Tensor*)mask;
    auto val = mx::full(tt->arr.shape(), value);
    auto r = new Tensor(mx::where(tm->arr, val, tt->arr), tt->requires_grad);
    return (TensorHandle)r;
}
TensorHandle tensor_log_softmax_2d(TensorHandle t) { STUB(); }
TensorHandle tensor_layer_norm_2d(TensorHandle input, TensorHandle gamma,
    TensorHandle bias, double eps) { STUB(); }
TensorHandle tensor_one_hot(int* tokens, int n_tokens, int vocab_size) { STUB(); }

void tensor_backward(TensorHandle loss) { STUB(); }
TensorHandle tensor_grad(TensorHandle t) { STUB(); }
void tensor_zero_grad(TensorHandle t) { STUB(); }
int tensor_requires_grad(TensorHandle t) {
    return ((Tensor*)t)->requires_grad ? 1 : 0;
}
TensorHandle tensor_detach(TensorHandle t) { STUB(); }
TensorHandle tensor_with_grad(TensorHandle t) { STUB(); }
void tensor_set_requires_grad(TensorHandle t, int rg) {
    ((Tensor*)t)->requires_grad = (rg != 0);
}
void tensor_no_grad_begin(void) { /* no-op for now */ }
void tensor_no_grad_end(void) { /* no-op for now */ }

TensorHandle tensor_to_device(TensorHandle t, const char* device) { STUB(); }
const char* tensor_device(TensorHandle t) { return "gpu"; }

void tensor_lstm_cell(TensorHandle input, TensorHandle hx, TensorHandle cx,
    TensorHandle w_ih, TensorHandle w_hh, TensorHandle b_ih, TensorHandle b_hh,
    TensorHandle* out_h, TensorHandle* out_c) { STUB(); }
void tensor_lstm_gates(TensorHandle combined, TensorHandle prev_cell, int o,
    TensorHandle* out_h, TensorHandle* out_c) { STUB(); }
TensorPair* tensor_lstm_gates_pair(TensorHandle combined, TensorHandle prev_cell, int o) { STUB(); }
TensorHandle tensor_pair_first(TensorPair* p) { STUB(); }
TensorHandle tensor_pair_second(TensorPair* p) { STUB(); }
void tensor_pair_free(TensorPair* p) { STUB(); }

void param_register(const char* name, TensorHandle t) { STUB(); }
void param_clear(void) { STUB(); }
int param_count(void) { STUB(); }
const char* param_name(int idx) { STUB(); }
double param_grad_item(int idx) { STUB(); }
double param_grad_item_at(int param_idx, int elem_idx) { STUB(); }
double param_grad_item_and_zero(int idx) { STUB(); }
TensorHandle param_tensor(int idx) { STUB(); }
void param_zero_all_grads(void) { STUB(); }
void param_subtract_delta(int idx, double delta) { STUB(); }
TensorHandle tensor_subtract_scalar_inplace(TensorHandle t, double val) { STUB(); }

TensorHandle tensor_create_1d(int n, double* data, int requires_grad) {
    int shape[] = {n};
    return tensor_create(data, shape, 1, requires_grad);
}
TensorHandle tensor_create_2d(int rows, int cols, double* data, int requires_grad) {
    int shape[] = {rows, cols};
    return tensor_create(data, shape, 2, requires_grad);
}

double* tensor_alloc_doubles(int n) { return (double*)calloc(n, sizeof(double)); }
double tensor_read_double(double* buf, int idx) { return buf[idx]; }
void tensor_write_double(double* buf, int idx, double val) { buf[idx] = val; }

TensorHandle* tensor_ptr_array_alloc(int n) {
    return (TensorHandle*)calloc(n, sizeof(TensorHandle));
}
void tensor_ptr_array_set(TensorHandle* arr, int idx, TensorHandle t) { arr[idx] = t; }
TensorHandle tensor_stack_from_array(TensorHandle* arr, int count, int dim) { STUB(); }
TensorHandle tensor_cat_from_array(TensorHandle* arr, int count, int dim) { STUB(); }

TensorHandle tensor_create_param_2d(int rows, int cols, double* data) {
    int shape[] = {rows, cols};
    return tensor_create(data, shape, 2, 1);
}
TensorHandle tensor_create_param_1d(int n, double* data) {
    int shape[] = {n};
    return tensor_create(data, shape, 1, 1);
}
TensorHandle tensor_create_state_2d(int rows, int cols, double* data) {
    int shape[] = {rows, cols};
    return tensor_create(data, shape, 2, 0);
}
TensorHandle tensor_create_state_1d(int n, double* data) {
    int shape[] = {n};
    return tensor_create(data, shape, 1, 0);
}
TensorHandle tensor_view_2d(TensorHandle mat, int row, int col) { STUB(); }
TensorHandle tensor_view_1d(TensorHandle vec, int idx) { STUB(); }
double tensor_item_2d(TensorHandle mat, int row, int col) { STUB(); }
double tensor_item_1d(TensorHandle vec, int idx) { STUB(); }

OptimizerHandle optimizer_create_sgd(double lr) { STUB(); }
OptimizerHandle optimizer_create_rmsprop(double lr, double alpha, double eps,
    double weight_decay, double momentum) { STUB(); }
OptimizerHandle optimizer_create_adam(double lr, double beta1, double beta2, double eps) { STUB(); }
void optimizer_free(OptimizerHandle opt) { STUB(); }
void optimizer_step(OptimizerHandle opt) { STUB(); }
void optimizer_zero_grad(OptimizerHandle opt) { STUB(); }
void optimizer_clip_grad_value(double max_val) { STUB(); }
double optimizer_clip_grad_norm(double max_norm) { STUB(); }

int backend_supports_tensor_params(void) { return 1; }

int get_rss_mb(void) { return 0; }
int get_current_rss_mb(void) { return 0; }
void backend_memory_report(void) {
    fprintf(stderr, "MLX backend: memory report not implemented\n");
}
void backend_reset_for_eval(void) {
    fprintf(stderr, "MLX backend: reset_for_eval not implemented\n");
}
void backend_profile_reset(void) {}
void backend_profile_report(void) {
    fprintf(stderr, "MLX backend: profiling not implemented\n");
}

void tensor_print(TensorHandle h) {
    auto t = (Tensor*)h;
    mx::eval(t->arr);
    std::cout << t->arr << std::endl;
}

} // extern "C"
