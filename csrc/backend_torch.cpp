#include "backend.h"

#include <ATen/ATen.h>
#include <torch/torch.h>

#include <cstring>
#include <string>
#include <vector>
#include <unordered_set>
#include <sys/resource.h>
#ifdef __APPLE__
#include <mach/mach.h>
#endif

/* ---------- Intermediate tensor tracking ---------- */

// Track all non-persistent tensors so we can free them at optimizer_step
static std::vector<at::Tensor*> intermediates;
static std::vector<TensorPair*> all_pairs;
static bool tracking_enabled = true;

/* ---------- Helpers ---------- */

static inline at::Tensor* to_tensor(TensorHandle h) {
    return static_cast<at::Tensor*>(h);
}

static inline TensorHandle from_tensor(at::Tensor t) {
    auto* p = new at::Tensor(std::move(t));
    if (tracking_enabled) intermediates.push_back(p);
    return static_cast<TensorHandle>(p);
}

// Persistent variant: not tracked for cleanup (survives optimizer_step)
static inline TensorHandle from_tensor_persistent(at::Tensor t) {
    auto* p = new at::Tensor(std::move(t));
    return static_cast<TensorHandle>(p);
}

static void free_intermediates(); // defined after param_registry

/* ---------- Lifecycle ---------- */

TensorHandle tensor_create_scalar(double value, int requires_grad) {
    auto t = torch::tensor(value, torch::dtype(torch::kFloat64));
    if (requires_grad) t.requires_grad_(true);
    // User-created tensors are not tracked — caller manages lifetime via
    // tensor_free or param_register. Only computation intermediates (from
    // tensor_add, tensor_mul, etc.) are tracked for free_intermediates().
    return from_tensor_persistent(std::move(t));
}

TensorHandle tensor_create(double* data, int* shape, int rank, int requires_grad) {
    std::vector<int64_t> dims(rank);
    for (int i = 0; i < rank; i++) dims[i] = shape[i];
    auto opts = torch::TensorOptions().dtype(torch::kFloat64);
    auto t = torch::from_blob(data, dims, opts).clone();
    if (requires_grad) t.requires_grad_(true);
    return from_tensor_persistent(std::move(t));
}

TensorHandle tensor_clone(TensorHandle h) {
    return from_tensor(to_tensor(h)->clone());
}

// Track pointers freed by free_intermediates so tensor_free skips them
static std::unordered_set<void*> freed_by_cleanup;

void tensor_free(TensorHandle h) {
    // Torch tensors participate in autograd graphs — explicit deletion can
    // corrupt torch's internal bookkeeping. Let free_intermediates (called
    // by optimizer_step) handle bulk cleanup of computation intermediates.
    // Persistent user-created tensors leak slightly — acceptable for tests.
    (void)h;
}

/* ---------- Accessors ---------- */

double tensor_item(TensorHandle h) {
    return to_tensor(h)->item<double>();
}

int tensor_numel(TensorHandle h) {
    return static_cast<int>(to_tensor(h)->numel());
}

int tensor_dim(TensorHandle h) {
    return static_cast<int>(to_tensor(h)->dim());
}

int tensor_size(TensorHandle h, int dim) {
    return static_cast<int>(to_tensor(h)->size(dim));
}

void tensor_to_doubles(TensorHandle h, double* out) {
    auto t = to_tensor(h)->to(torch::kFloat64).contiguous();
    std::memcpy(out, t.data_ptr<double>(), t.numel() * sizeof(double));
}

/* ---------- Arithmetic ---------- */

TensorHandle tensor_add(TensorHandle a, TensorHandle b) {
    return from_tensor(torch::add(*to_tensor(a), *to_tensor(b)));
}

TensorHandle tensor_sub(TensorHandle a, TensorHandle b) {
    return from_tensor(torch::sub(*to_tensor(a), *to_tensor(b)));
}

TensorHandle tensor_mul(TensorHandle a, TensorHandle b) {
    return from_tensor(torch::mul(*to_tensor(a), *to_tensor(b)));
}

TensorHandle tensor_div(TensorHandle a, TensorHandle b) {
    return from_tensor(torch::div(*to_tensor(a), *to_tensor(b)));
}

TensorHandle tensor_neg(TensorHandle h) {
    return from_tensor(torch::neg(*to_tensor(h)));
}

TensorHandle tensor_abs(TensorHandle h) {
    return from_tensor(torch::abs(*to_tensor(h)));
}

TensorHandle tensor_exp(TensorHandle h) {
    return from_tensor(torch::exp(*to_tensor(h)));
}

TensorHandle tensor_log(TensorHandle h) {
    return from_tensor(torch::log(*to_tensor(h)));
}

TensorHandle tensor_sqrt(TensorHandle h) {
    return from_tensor(torch::sqrt(*to_tensor(h)));
}

TensorHandle tensor_pow(TensorHandle base, TensorHandle exp) {
    return from_tensor(torch::pow(*to_tensor(base), *to_tensor(exp)));
}

TensorHandle tensor_sigmoid(TensorHandle h) {
    return from_tensor(torch::sigmoid(*to_tensor(h)));
}

TensorHandle tensor_tanh(TensorHandle h) {
    return from_tensor(torch::tanh(*to_tensor(h)));
}

TensorHandle tensor_add_scalar(TensorHandle h, double s) {
    return from_tensor(*to_tensor(h) + s);
}

TensorHandle tensor_mul_scalar(TensorHandle h, double s) {
    return from_tensor(*to_tensor(h) * s);
}

TensorHandle tensor_clamp_min(TensorHandle h, double min_val) {
    return from_tensor(torch::clamp_min(*to_tensor(h), min_val));
}

/* ---------- Reduction ---------- */

TensorHandle tensor_sum(TensorHandle h) {
    return from_tensor(to_tensor(h)->sum());
}

TensorHandle tensor_sum_dim(TensorHandle h, int dim, int keepdim) {
    return from_tensor(to_tensor(h)->sum(dim, keepdim != 0));
}

TensorHandle tensor_mean(TensorHandle h) {
    return from_tensor(to_tensor(h)->mean());
}

/* ---------- Linear algebra ---------- */

TensorHandle tensor_matmul(TensorHandle a, TensorHandle b) {
    return from_tensor(torch::matmul(*to_tensor(a), *to_tensor(b)));
}

TensorHandle tensor_mv(TensorHandle mat, TensorHandle vec) {
    return from_tensor(torch::mv(*to_tensor(mat), *to_tensor(vec)));
}

TensorHandle tensor_dot(TensorHandle a, TensorHandle b) {
    return from_tensor(torch::dot(*to_tensor(a), *to_tensor(b)));
}

TensorHandle tensor_outer(TensorHandle a, TensorHandle b) {
    return from_tensor(torch::outer(*to_tensor(a), *to_tensor(b)));
}

/* ---------- Activation / normalization ---------- */

TensorHandle tensor_softmax(TensorHandle h, int dim) {
    return from_tensor(torch::softmax(*to_tensor(h), dim));
}

TensorHandle tensor_log_softmax(TensorHandle h, int dim) {
    return from_tensor(torch::log_softmax(*to_tensor(h), dim));
}

/* ---------- Loss ---------- */

TensorHandle tensor_bce_with_logits(TensorHandle input, TensorHandle target) {
    return from_tensor(torch::nn::functional::binary_cross_entropy_with_logits(
        *to_tensor(input), *to_tensor(target)));
}

TensorHandle tensor_cross_entropy(TensorHandle input, TensorHandle target) {
    return from_tensor(torch::nn::functional::cross_entropy(
        *to_tensor(input), *to_tensor(target)));
}

TensorHandle tensor_mse_loss(TensorHandle input, TensorHandle target) {
    return from_tensor(torch::mse_loss(*to_tensor(input), *to_tensor(target)));
}

/* ---------- NTM-specific compositions ---------- */

TensorHandle tensor_cosine_similarity(TensorHandle a, TensorHandle b, int dim) {
    return from_tensor(torch::nn::functional::cosine_similarity(
        *to_tensor(a), *to_tensor(b),
        torch::nn::functional::CosineSimilarityFuncOptions().dim(dim)));
}

TensorHandle tensor_conv1d_circular(TensorHandle input, TensorHandle kernel) {
    /* Circular convolution for NTM shift operation.
       input: [N], kernel: [K]
       Pad input circularly, then do 1D convolution. */
    auto& inp = *to_tensor(input);
    auto& ker = *to_tensor(kernel);

    int64_t n = inp.size(0);
    int64_t k = ker.size(0);
    int64_t pad = k / 2;

    /* Circular padding: concat [tail, input, head] */
    auto padded = torch::cat({inp.slice(0, n - pad, n), inp, inp.slice(0, 0, pad)});

    /* Reshape for conv1d: [batch=1, channels=1, length] */
    auto inp_3d = padded.reshape({1, 1, -1});
    auto ker_3d = ker.flip(0).reshape({1, 1, -1});

    auto out = torch::conv1d(inp_3d, ker_3d);
    return from_tensor(out.reshape({n}));
}

TensorHandle tensor_batch_norm(TensorHandle hinput, TensorHandle hgamma, TensorHandle hbeta,
                               TensorHandle hrunning_mean, TensorHandle hrunning_var,
                               int channels, int spatial, int training,
                               double momentum, double eps) {
    auto& inp = *to_tensor(hinput);
    auto& gamma = *to_tensor(hgamma);
    auto& beta = *to_tensor(hbeta);
    auto& rm = *to_tensor(hrunning_mean);
    auto& rv = *to_tensor(hrunning_var);

    /* Reshape to [1, C, spatial] for torch::batch_norm (expects [N,C,...]) */
    auto inp_3d = inp.reshape({1, (int64_t)channels, (int64_t)spatial});
    auto out = torch::batch_norm(inp_3d, gamma, beta, rm, rv,
                                 /*training=*/training, momentum, eps,
                                 /*cudnn_enabled=*/false);
    return from_tensor(out.reshape({-1}));  /* flatten back to [C*spatial] */
}

TensorHandle tensor_dropout(TensorHandle hinput, double p, int training, unsigned int seed) {
    (void)seed;  /* torch uses its own RNG */
    auto& inp = *to_tensor(hinput);
    if (!training || p <= 0.0) return hinput;
    auto out = torch::dropout(inp, p, /*train=*/true);
    return from_tensor(out);
}

TensorHandle tensor_conv2d(TensorHandle hinput, TensorHandle hkernel,
                           TensorHandle hbias, int padH, int padW,
                           int strideH, int strideW) {
    auto& inp = *to_tensor(hinput);   /* [inC, H, W] */
    auto& ker = *to_tensor(hkernel);  /* [outC, inC, kH, kW] */

    /* torch::conv2d expects [N, C, H, W] — add batch dim */
    auto inp_4d = inp.unsqueeze(0);
    at::Tensor bias_t;
    if (hbias) bias_t = *to_tensor(hbias);

    auto out = hbias
        ? torch::conv2d(inp_4d, ker, bias_t,
              /*stride=*/{strideH, strideW}, /*padding=*/{padH, padW})
        : torch::conv2d(inp_4d, ker, /*bias=*/{},
              /*stride=*/{strideH, strideW}, /*padding=*/{padH, padW});

    return from_tensor(out.squeeze(0));  /* remove batch dim */
}

TensorHandle tensor_max_pool2d(TensorHandle hinput, int kH, int kW,
                               int strideH, int strideW) {
    auto& inp = *to_tensor(hinput);  /* [C, H, W] */
    auto inp_4d = inp.unsqueeze(0);  /* [1, C, H, W] */
    auto out = torch::max_pool2d(inp_4d, {kH, kW}, {strideH, strideW});
    return from_tensor(out.squeeze(0));
}

TensorPair* tensor_ntm_read_head(
    TensorHandle memory_h, TensorHandle prev_weights_h,
    TensorHandle key_h, TensorHandle beta_h, TensorHandle g_h,
    TensorHandle gamma_h, TensorHandle shift_kernel_h)
{
    auto& memory = *to_tensor(memory_h);           /* [n, w] */
    auto& prev_weights = *to_tensor(prev_weights_h); /* [n] */
    auto& key = *to_tensor(key_h);                 /* [w] */
    auto& beta = *to_tensor(beta_h);               /* scalar */
    auto& g = *to_tensor(g_h);                     /* scalar */
    auto& gamma = *to_tensor(gamma_h);             /* scalar */
    auto& shift_kernel = *to_tensor(shift_kernel_h); /* [3] or [K] */

    /* 1. Content addressing: softmax(beta * cosine_similarity(key, memory)) */
    auto key_expanded = key.unsqueeze(0);           /* [1, w] */
    auto cos_sim = torch::nn::functional::cosine_similarity(
        memory, key_expanded,
        torch::nn::functional::CosineSimilarityFuncOptions().dim(1));
    auto content_weights = torch::softmax(beta * cos_sim, 0);

    /* 2. Interpolation: g * content + (1-g) * prev */
    auto interpolated = g * content_weights + (1.0 - g) * prev_weights;

    /* 3. Circular shift convolution */
    int64_t n = interpolated.size(0);
    int64_t k = shift_kernel.size(0);
    int64_t pad = k / 2;
    auto padded = torch::cat({interpolated.slice(0, n - pad, n), interpolated, interpolated.slice(0, 0, pad)});
    auto inp_3d = padded.reshape({1, 1, -1});
    auto ker_3d = shift_kernel.flip(0).reshape({1, 1, -1});
    auto shifted = torch::conv1d(inp_3d, ker_3d).reshape({n});

    /* 4. Sharpening: w^gamma / sum(w^gamma) */
    auto powered = torch::pow(torch::clamp(shifted, 1e-10), gamma);
    auto focused = powered / (powered.sum() + 1e-10);

    /* 5. Read: matmul(weights, memory) → [w] */
    auto read_output = torch::matmul(focused, memory);

    auto* p = new TensorPair;
    p->first = from_tensor(std::move(focused));
    p->second = from_tensor(std::move(read_output));
    all_pairs.push_back(p);
    return p;
}

TensorHandle tensor_ntm_interp_write(
    TensorHandle memory_h, TensorHandle weights_h, TensorHandle add_vector_h)
{
    auto& memory = *to_tensor(memory_h);
    auto& weights = *to_tensor(weights_h);
    auto& add_vector = *to_tensor(add_vector_h);
    /* memory' = memory + outer(weights, add_vector) */
    return from_tensor(memory + torch::outer(weights, add_vector));
}

/* ---------- Shape manipulation ---------- */

TensorHandle tensor_reshape(TensorHandle h, int* shape, int rank) {
    std::vector<int64_t> dims(rank);
    for (int i = 0; i < rank; i++) dims[i] = shape[i];
    return from_tensor(to_tensor(h)->reshape(dims));
}

TensorHandle tensor_unsqueeze(TensorHandle h, int dim) {
    return from_tensor(to_tensor(h)->unsqueeze(dim));
}

TensorHandle tensor_squeeze(TensorHandle h, int dim) {
    return from_tensor(to_tensor(h)->squeeze(dim));
}

TensorHandle tensor_select(TensorHandle h, int dim, int index) {
    return from_tensor(to_tensor(h)->select(dim, index));
}

TensorHandle tensor_stack(TensorHandle* tensors, int count, int dim) {
    std::vector<at::Tensor> vec(count);
    for (int i = 0; i < count; i++) vec[i] = *to_tensor(tensors[i]);
    return from_tensor(torch::stack(vec, dim));
}

TensorHandle tensor_cat(TensorHandle* tensors, int count, int dim) {
    std::vector<at::Tensor> vec(count);
    for (int i = 0; i < count; i++) vec[i] = *to_tensor(tensors[i]);
    return from_tensor(torch::cat(vec, dim));
}

/* ---------- Autograd ---------- */

void tensor_backward(TensorHandle h) {
    to_tensor(h)->backward();
}

TensorHandle tensor_grad(TensorHandle h) {
    auto& g = to_tensor(h)->grad();
    if (!g.defined()) return nullptr;
    return from_tensor(g);
}

void tensor_zero_grad(TensorHandle h) {
    auto& t = *to_tensor(h);
    if (t.grad().defined()) {
        t.grad().zero_();
    }
}

int tensor_requires_grad(TensorHandle h) {
    return to_tensor(h)->requires_grad() ? 1 : 0;
}

TensorHandle tensor_detach(TensorHandle h) {
    return from_tensor(to_tensor(h)->detach());
}

TensorHandle tensor_with_grad(TensorHandle h) {
    auto t = to_tensor(h)->detach().clone();
    t.requires_grad_(true);
    return from_tensor(std::move(t));
}

void tensor_set_requires_grad(TensorHandle h, int requires_grad) {
    to_tensor(h)->requires_grad_(requires_grad != 0);
}

/* No-grad scope */
static thread_local bool no_grad_active = false;
static thread_local std::unique_ptr<torch::NoGradGuard> no_grad_guard;

void tensor_no_grad_begin(void) {
    if (!no_grad_active) {
        no_grad_guard = std::make_unique<torch::NoGradGuard>();
        no_grad_active = true;
    }
}

void tensor_no_grad_end(void) {
    if (no_grad_active) {
        no_grad_guard.reset();
        no_grad_active = false;
    }
}

/* ---------- Device ---------- */

TensorHandle tensor_to_device(TensorHandle h, const char* device) {
    return from_tensor(to_tensor(h)->to(std::string(device)));
}

static thread_local std::string device_str;

const char* tensor_device(TensorHandle h) {
    auto d = to_tensor(h)->device();
    device_str = d.str();
    return device_str.c_str();
}

/* ---------- LSTM ---------- */

void tensor_lstm_cell(
    TensorHandle input, TensorHandle hx, TensorHandle cx,
    TensorHandle w_ih, TensorHandle w_hh,
    TensorHandle b_ih, TensorHandle b_hh,
    TensorHandle* out_h, TensorHandle* out_c)
{
    auto result = torch::lstm_cell(
        *to_tensor(input),
        {*to_tensor(hx), *to_tensor(cx)},
        *to_tensor(w_ih), *to_tensor(w_hh),
        *to_tensor(b_ih), *to_tensor(b_hh));
    *out_h = from_tensor(std::get<0>(result));
    *out_c = from_tensor(std::get<1>(result));
}

/* ---------- Parameter Registry ---------- */

struct ParamEntry {
    std::string name;
    at::Tensor* tensor;   /* non-owning: the Variable still owns the at::Tensor */
};

static std::vector<ParamEntry> param_registry;

void param_register(const char* name, TensorHandle h) {
    /* Replace if already registered under this name */
    for (auto& entry : param_registry) {
        if (entry.name == name) {
            entry.tensor = to_tensor(h);
            return;
        }
    }
    param_registry.push_back({name, to_tensor(h)});
}

void param_clear(void) {
    param_registry.clear();
    intermediates.clear();
    for (auto* p : all_pairs) delete p;
    all_pairs.clear();
    freed_by_cleanup.clear();
}

static void free_intermediates() {
    std::unordered_set<at::Tensor*> param_set;
    for (auto& entry : param_registry) param_set.insert(entry.tensor);
    freed_by_cleanup.clear();
    for (auto* p : intermediates) {
        if (p && param_set.find(p) == param_set.end()) {
            freed_by_cleanup.insert(p);
            delete p;
        }
    }
    intermediates.clear();
    // Free TensorPair structs
    for (auto* p : all_pairs) delete p;
    all_pairs.clear();
}

int param_count(void) {
    return static_cast<int>(param_registry.size());
}

static thread_local std::string param_name_buf;

const char* param_name(int idx) {
    param_name_buf = param_registry[idx].name;
    return param_name_buf.c_str();
}

double param_grad_item(int idx) {
    auto& g = param_registry[idx].tensor->grad();
    if (!g.defined()) return 0.0;
    return g.flatten().data_ptr<double>()[0];
}

double param_grad_item_and_zero(int idx) {
    auto* t = param_registry[idx].tensor;
    auto& g = t->grad();
    if (!g.defined()) return 0.0;
    double val = g.item<double>();
    g.zero_();
    return val;
}

TensorHandle param_tensor(int idx) {
    return static_cast<TensorHandle>(param_registry[idx].tensor);
}

void param_zero_all_grads(void) {
    for (auto& entry : param_registry) {
        if (entry.tensor->grad().defined()) {
            entry.tensor->grad().zero_();
        }
    }
}

void param_subtract_delta(int idx, double delta) {
    torch::NoGradGuard no_grad;
    auto& entry = param_registry[idx];
    entry.tensor->sub_(delta);
}

void param_load_data(int idx, const double* data, int numel) {
    torch::NoGradGuard no_grad;
    auto& entry = param_registry[idx];
    int existing = entry.tensor->numel();
    if (existing != numel) {
        fprintf(stderr, "param_load_data: size mismatch for '%s': expected %d, got %d\n",
                entry.name.c_str(), existing, numel);
        return;
    }
    memcpy(entry.tensor->data_ptr<double>(), data, numel * sizeof(double));
}

TensorHandle tensor_subtract_scalar_inplace(TensorHandle h, double val) {
    torch::NoGradGuard no_grad;
    to_tensor(h)->sub_(val);
    return h;
}

/* ---------- Convenience ---------- */

TensorHandle tensor_create_1d(int n, double* data, int requires_grad) {
    auto t = torch::from_blob(data, {(int64_t)n}, torch::kFloat64).clone();
    free(data);
    if (requires_grad) t.requires_grad_(true);
    return from_tensor(std::move(t));
}

TensorHandle tensor_create_2d(int rows, int cols, double* data, int requires_grad) {
    auto t = torch::from_blob(data, {(int64_t)rows, (int64_t)cols}, torch::kFloat64).clone();
    free(data);
    if (requires_grad) t.requires_grad_(true);
    return from_tensor(std::move(t));
}

double* tensor_alloc_doubles(int n) {
    return (double*)calloc(n, sizeof(double));
}
void tensor_free_doubles(double* buf) { free(buf); }

double tensor_read_double(double* buf, int idx) {
    return buf[idx];
}

void tensor_write_double(double* buf, int idx, double val) {
    buf[idx] = val;
}

/* ---------- Tensor pointer array ---------- */

TensorHandle* tensor_ptr_array_alloc(int n) {
    return (TensorHandle*)calloc(n, sizeof(TensorHandle));
}

void tensor_ptr_array_set(TensorHandle* arr, int idx, TensorHandle t) {
    arr[idx] = t;
}

TensorHandle tensor_stack_from_array(TensorHandle* arr, int count, int dim) {
    std::vector<at::Tensor> vec(count);
    for (int i = 0; i < count; i++) vec[i] = *to_tensor(arr[i]);
    free(arr);
    return from_tensor(torch::stack(vec, dim));
}

TensorHandle tensor_cat_from_array(TensorHandle* arr, int count, int dim) {
    std::vector<at::Tensor> vec(count);
    for (int i = 0; i < count; i++) vec[i] = *to_tensor(arr[i]);
    free(arr);
    return from_tensor(torch::cat(vec, dim));
}

/* ---------- Convenience shape ops (added for tensor path) ---------- */

TensorHandle tensor_mm(TensorHandle a, TensorHandle b) {
    return from_tensor(torch::mm(*to_tensor(a), *to_tensor(b)));
}

TensorHandle tensor_bmm(TensorHandle a, TensorHandle b) {
    // a=[B,m,n], b=[n,k] (shared weight) → [B,m,k]
    auto& ta = *to_tensor(a);
    auto& tb = *to_tensor(b);
    int B = ta.size(0);
    std::vector<at::Tensor> results;
    for (int i = 0; i < B; i++)
        results.push_back(torch::mm(ta[i], tb));
    return from_tensor(torch::stack(results));
}

TensorHandle tensor_bmm_3x3(TensorHandle a, TensorHandle b) {
    return from_tensor(torch::bmm(*to_tensor(a), *to_tensor(b)));
}

TensorHandle tensor_softmax_3d(TensorHandle h) {
    return from_tensor(torch::softmax(*to_tensor(h), -1));
}

TensorHandle tensor_transpose_last2(TensorHandle h) {
    return from_tensor(to_tensor(h)->transpose(-2, -1).contiguous());
}

TensorHandle tensor_reshape_3d(TensorHandle h, int d0, int d1, int d2) {
    return from_tensor(to_tensor(h)->reshape({(int64_t)d0, (int64_t)d1, (int64_t)d2}));
}

TensorHandle tensor_expand_mask(TensorHandle hmask, int B) {
    return from_tensor(to_tensor(hmask)->unsqueeze(0).expand({(int64_t)B, -1, -1}).contiguous());
}

TensorHandle tensor_transpose_2d(TensorHandle h) {
    return from_tensor(to_tensor(h)->t().contiguous());
}

TensorHandle tensor_softmax_2d(TensorHandle h) {
    return from_tensor(torch::softmax(*to_tensor(h), -1));
}

TensorHandle tensor_log_softmax_2d(TensorHandle h) {
    return from_tensor(torch::log_softmax(*to_tensor(h), -1));
}

TensorHandle tensor_masked_fill(TensorHandle h, TensorHandle mask, double value) {
    return from_tensor(to_tensor(h)->masked_fill(to_tensor(mask)->to(torch::kBool), value));
}

TensorHandle tensor_layer_norm_2d(TensorHandle input, TensorHandle gamma,
                                   TensorHandle bias, double eps) {
    auto& t = *to_tensor(input);
    int64_t n = t.size(-1);
    return from_tensor(torch::layer_norm(t, {n}, *to_tensor(gamma), *to_tensor(bias), eps));
}

TensorHandle tensor_cat2(TensorHandle a, TensorHandle b) {
    return from_tensor(torch::cat({*to_tensor(a), *to_tensor(b)}, 0));
}

TensorHandle tensor_narrow(TensorHandle h, int dim, int start, int len) {
    // Match tape backend: always returns 1D (flattened)
    auto t = to_tensor(h)->flatten().narrow(0, start, len).contiguous();
    return from_tensor(std::move(t));
}

TensorHandle tensor_reshape_2d(TensorHandle h, int rows, int cols) {
    return from_tensor(to_tensor(h)->reshape({(int64_t)rows, (int64_t)cols}));
}

TensorHandle tensor_causal_mask(int n) {
    auto t = torch::triu(torch::ones({(int64_t)n, (int64_t)n}, torch::kFloat64), 1);
    return from_tensor(std::move(t));
}

TensorHandle tensor_one_hot(int* tokens, int n_tokens, int vocab_size) {
    int total = n_tokens * vocab_size;
    auto t = torch::zeros({(int64_t)total}, torch::kFloat64);
    auto acc = t.accessor<double, 1>();
    for (int i = 0; i < n_tokens; i++) {
        int tok = tokens[i];
        if (tok >= 0 && tok < vocab_size)
            acc[i * vocab_size + tok] = 1.0;
    }
    return from_tensor(std::move(t));
}

TensorHandle tensor_batch(TensorHandle* handles, int count) {
    std::vector<at::Tensor> vec(count);
    for (int i = 0; i < count; i++) vec[i] = *to_tensor(handles[i]);
    return from_tensor(torch::stack(vec));
}

TensorHandle* tensor_unbatch(TensorHandle h, int* out_count) {
    auto tensors = to_tensor(h)->unbind(0);
    *out_count = (int)tensors.size();
    auto* arr = (TensorHandle*)malloc(*out_count * sizeof(TensorHandle));
    for (int i = 0; i < *out_count; i++)
        arr[i] = from_tensor(tensors[i].contiguous());
    return arr;
}

/* ---------- Tensor-level parameter creation ---------- */

TensorHandle tensor_create_param_2d(int rows, int cols, double* data) {
    auto t = torch::from_blob(data, {(int64_t)rows, (int64_t)cols}, torch::kFloat64).clone();
    t.requires_grad_(true);
    return from_tensor_persistent(std::move(t));
}

TensorHandle tensor_create_param_4d(int d0, int d1, int d2, int d3, double* data) {
    auto t = torch::from_blob(data, {(int64_t)d0, (int64_t)d1, (int64_t)d2, (int64_t)d3}, torch::kFloat64).clone();
    t.requires_grad_(true);
    return from_tensor_persistent(std::move(t));
}

TensorHandle tensor_create_param_1d(int n, double* data) {
    auto t = torch::from_blob(data, {(int64_t)n}, torch::kFloat64).clone();
    t.requires_grad_(true);
    return from_tensor_persistent(std::move(t));
}

TensorHandle tensor_create_state_2d(int rows, int cols, double* data) {
    auto t = torch::from_blob(data, {(int64_t)rows, (int64_t)cols}, torch::kFloat64).clone();
    return from_tensor_persistent(std::move(t));
}

TensorHandle tensor_create_state_1d(int n, double* data) {
    auto t = torch::from_blob(data, {(int64_t)n}, torch::kFloat64).clone();
    return from_tensor_persistent(std::move(t));
}

TensorHandle tensor_view_2d(TensorHandle h, int row, int col) {
    /* Returns a 0-dim view that shares storage with the parent tensor.
       Must be persistent — views into param tensors survive free_intermediates. */
    return from_tensor_persistent(to_tensor(h)->select(0, row).select(0, col));
}

TensorHandle tensor_view_1d(TensorHandle h, int idx) {
    return from_tensor_persistent(to_tensor(h)->select(0, idx));
}

double tensor_item_2d(TensorHandle h, int row, int col) {
    return to_tensor(h)->index({row, col}).item<double>();
}

double tensor_item_1d(TensorHandle h, int idx) {
    return (*to_tensor(h))[idx].item<double>();
}

/* ---------- Native Optimizer ---------- */

/* Helper: collect all param_registry tensors into a vector */
static std::vector<at::Tensor> collect_param_tensors() {
    std::vector<at::Tensor> params;
    params.reserve(param_registry.size());
    for (auto& entry : param_registry) {
        params.push_back(*entry.tensor);
    }
    return params;
}

/* Wrapper to track optimizer type alongside PyTorch optimizer */
struct OptWrapper {
    int type; // 0=sgd, 1=rmsprop, 2=adam
    double lr, beta1, beta2, eps, alpha, weight_decay, momentum;
    torch::optim::Optimizer* opt;
};

OptimizerHandle optimizer_create_sgd(double lr) {
    auto params = collect_param_tensors();
    auto* w = new OptWrapper();
    w->type = 0; w->lr = lr;
    w->opt = new torch::optim::SGD(params, torch::optim::SGDOptions(lr));
    return static_cast<OptimizerHandle>(w);
}

OptimizerHandle optimizer_create_rmsprop(double lr, double alpha, double eps,
                                          double weight_decay, double momentum) {
    auto params = collect_param_tensors();
    auto* w = new OptWrapper();
    w->type = 1; w->lr = lr; w->alpha = alpha; w->eps = eps;
    w->weight_decay = weight_decay; w->momentum = momentum;
    w->opt = new torch::optim::RMSprop(params,
        torch::optim::RMSpropOptions(lr).alpha(alpha).eps(eps)
            .weight_decay(weight_decay).momentum(momentum));
    return static_cast<OptimizerHandle>(w);
}

OptimizerHandle optimizer_create_adam(double lr, double beta1, double beta2, double eps) {
    auto params = collect_param_tensors();
    auto* w = new OptWrapper();
    w->type = 2; w->lr = lr; w->beta1 = beta1; w->beta2 = beta2; w->eps = eps;
    w->opt = new torch::optim::Adam(params,
        torch::optim::AdamOptions(lr).betas(std::make_tuple(beta1, beta2)).eps(eps));
    return static_cast<OptimizerHandle>(w);
}

void optimizer_free(OptimizerHandle h) {
    auto* w = static_cast<OptWrapper*>(h);
    delete w->opt;
    delete w;
}

void optimizer_step(OptimizerHandle h) {
    auto* w = static_cast<OptWrapper*>(h);
    auto* opt = w->opt;
    /* Re-sync param list from registry (handles late registration via autoName) */
    auto& param_groups = opt->param_groups();
    if (!param_groups.empty()) {
        auto& params = param_groups[0].params();
        auto current = collect_param_tensors();
        if (params.size() != current.size()) {
            params.clear();
            for (auto& t : current) params.push_back(t);
        }
    }
    opt->step();
    // Free intermediate tensors from this epoch's forward/backward
    free_intermediates();
}

void optimizer_zero_grad(OptimizerHandle h) {
    static_cast<OptWrapper*>(h)->opt->zero_grad();
}

void optimizer_clip_grad_value(double max_val) {
    auto params = collect_param_tensors();
    torch::nn::utils::clip_grad_value_(params, max_val);
}

double optimizer_clip_grad_norm(double max_norm) {
    auto params = collect_param_tensors();
    auto norm = torch::nn::utils::clip_grad_norm_(params, max_norm);
    return norm;
}

/* ================================================================
   Optimizer buffer accessors (for serialization)
   ================================================================ */

/* Helper: get the i-th param tensor's data key for state lookup */
static void* param_state_key(torch::optim::Optimizer* opt, int idx) {
    auto& params = opt->param_groups()[0].params();
    if (idx >= (int)params.size()) return nullptr;
    return params[idx].unsafeGetTensorImpl();
}

int optimizer_buf_count(OptimizerHandle h) {
    (void)h;
    return (int)param_registry.size();
}

void optimizer_get_m(OptimizerHandle h, int idx, double* out) {
    auto* w = static_cast<OptWrapper*>(h);
    int numel = (int)param_registry[idx].tensor->numel();
    auto key = param_state_key(w->opt, idx);
    if (!key || w->opt->state().count(key) == 0) {
        memset(out, 0, numel * sizeof(double));
        return;
    }
    auto& state = *w->opt->state().at(key);
    at::Tensor buf;
    if (w->type == 2) { /* Adam */
        buf = static_cast<torch::optim::AdamParamState&>(state).exp_avg();
    } else if (w->type == 1) { /* RMSprop */
        auto& rms = static_cast<torch::optim::RMSpropParamState&>(state);
        buf = rms.momentum_buffer().defined() ? rms.momentum_buffer() : at::zeros_like(*param_registry[idx].tensor);
    } else {
        memset(out, 0, numel * sizeof(double));
        return;
    }
    buf = buf.contiguous().to(torch::kFloat64);
    memcpy(out, buf.data_ptr<double>(), numel * sizeof(double));
}

void optimizer_get_v(OptimizerHandle h, int idx, double* out) {
    auto* w = static_cast<OptWrapper*>(h);
    int numel = (int)param_registry[idx].tensor->numel();
    auto key = param_state_key(w->opt, idx);
    if (!key || w->opt->state().count(key) == 0) {
        memset(out, 0, numel * sizeof(double));
        return;
    }
    auto& state = *w->opt->state().at(key);
    at::Tensor buf;
    if (w->type == 2) { /* Adam */
        buf = static_cast<torch::optim::AdamParamState&>(state).exp_avg_sq();
    } else if (w->type == 1) { /* RMSprop */
        buf = static_cast<torch::optim::RMSpropParamState&>(state).square_avg();
    } else {
        memset(out, 0, numel * sizeof(double));
        return;
    }
    buf = buf.contiguous().to(torch::kFloat64);
    memcpy(out, buf.data_ptr<double>(), numel * sizeof(double));
}

void optimizer_set_m(OptimizerHandle h, int idx, const double* data) {
    auto* w = static_cast<OptWrapper*>(h);
    int numel = (int)param_registry[idx].tensor->numel();
    auto key = param_state_key(w->opt, idx);
    if (!key) return;
    auto tensor = torch::from_blob((void*)data, {(int64_t)numel}, torch::kFloat64).clone();
    tensor = tensor.reshape(param_registry[idx].tensor->sizes());
    /* Ensure state entry exists */
    if (w->opt->state().count(key) == 0) {
        if (w->type == 2) w->opt->state()[key] = std::make_unique<torch::optim::AdamParamState>();
        else if (w->type == 1) w->opt->state()[key] = std::make_unique<torch::optim::RMSpropParamState>();
        else return;
    }
    auto& state = *w->opt->state().at(key);
    if (w->type == 2) {
        static_cast<torch::optim::AdamParamState&>(state).exp_avg(tensor);
    } else if (w->type == 1) {
        static_cast<torch::optim::RMSpropParamState&>(state).momentum_buffer(tensor);
    }
}

void optimizer_set_v(OptimizerHandle h, int idx, const double* data) {
    auto* w = static_cast<OptWrapper*>(h);
    int numel = (int)param_registry[idx].tensor->numel();
    auto key = param_state_key(w->opt, idx);
    if (!key) return;
    auto tensor = torch::from_blob((void*)data, {(int64_t)numel}, torch::kFloat64).clone();
    tensor = tensor.reshape(param_registry[idx].tensor->sizes());
    if (w->opt->state().count(key) == 0) {
        if (w->type == 2) w->opt->state()[key] = std::make_unique<torch::optim::AdamParamState>();
        else if (w->type == 1) w->opt->state()[key] = std::make_unique<torch::optim::RMSpropParamState>();
        else return;
    }
    auto& state = *w->opt->state().at(key);
    if (w->type == 2) {
        static_cast<torch::optim::AdamParamState&>(state).exp_avg_sq(tensor);
    } else if (w->type == 1) {
        static_cast<torch::optim::RMSpropParamState&>(state).square_avg(tensor);
    }
}

void optimizer_get_meta(OptimizerHandle h, double* out9) {
    auto* w = static_cast<OptWrapper*>(h);
    out9[0] = (double)w->type;
    out9[1] = w->lr;
    out9[2] = w->beta1;
    out9[3] = w->beta2;
    out9[4] = w->eps;
    out9[5] = w->alpha;
    out9[6] = w->weight_decay;
    out9[7] = w->momentum;
    /* Get step count from first param's state if available */
    int64_t step = 0;
    if (!w->opt->param_groups().empty()) {
        auto& params = w->opt->param_groups()[0].params();
        if (!params.empty()) {
            auto key = params[0].unsafeGetTensorImpl();
            if (w->opt->state().count(key)) {
                auto& state = *w->opt->state().at(key);
                if (w->type == 2) step = static_cast<torch::optim::AdamParamState&>(state).step();
                else if (w->type == 1) step = static_cast<torch::optim::RMSpropParamState&>(state).step();
            }
        }
    }
    out9[8] = (double)step;
}

void optimizer_set_meta(OptimizerHandle h, const double* in9) {
    auto* w = static_cast<OptWrapper*>(h);
    w->type = (int)in9[0];
    w->lr = in9[1];
    w->beta1 = in9[2];
    w->beta2 = in9[3];
    w->eps = in9[4];
    w->alpha = in9[5];
    w->weight_decay = in9[6];
    w->momentum = in9[7];
    /* Step count: set on all param states */
    int64_t step = (int64_t)in9[8];
    if (!w->opt->param_groups().empty()) {
        for (auto& p : w->opt->param_groups()[0].params()) {
            auto key = p.unsafeGetTensorImpl();
            if (w->opt->state().count(key)) {
                auto& state = *w->opt->state().at(key);
                if (w->type == 2) static_cast<torch::optim::AdamParamState&>(state).step(step);
                else if (w->type == 1) static_cast<torch::optim::RMSpropParamState&>(state).step(step);
            }
        }
    }
}

void tensor_lstm_gates(
    TensorHandle combined_h, TensorHandle prev_cell_h, int o,
    TensorHandle* out_h, TensorHandle* out_c)
{
    auto& combined = *to_tensor(combined_h);
    auto& prev_cell = *to_tensor(prev_cell_h);

    /* Split combined [4*o] into 4 gates of [o] */
    auto chunks = combined.split(o);
    auto i_gate = torch::sigmoid(chunks[0]);
    auto f_gate = torch::sigmoid(chunks[1]);
    auto g_gate = torch::tanh(chunks[2]);
    auto o_gate = torch::sigmoid(chunks[3]);

    auto new_cell = f_gate * prev_cell + i_gate * g_gate;
    auto new_hidden = o_gate * torch::tanh(new_cell);

    *out_h = from_tensor(std::move(new_hidden));
    *out_c = from_tensor(std::move(new_cell));
}

TensorPair* tensor_lstm_gates_pair(TensorHandle combined_h, TensorHandle prev_cell_h, int o) {
    auto& combined = *to_tensor(combined_h);
    auto& prev_cell = *to_tensor(prev_cell_h);
    auto chunks = combined.split(o);
    auto i_gate = torch::sigmoid(chunks[0]);
    auto f_gate = torch::sigmoid(chunks[1]);
    auto g_gate = torch::tanh(chunks[2]);
    auto o_gate = torch::sigmoid(chunks[3]);
    auto new_cell = f_gate * prev_cell + i_gate * g_gate;
    auto new_hidden = o_gate * torch::tanh(new_cell);
    auto* p = new TensorPair;
    p->first = from_tensor(std::move(new_hidden));
    p->second = from_tensor(std::move(new_cell));
    all_pairs.push_back(p);
    return p;
}

TensorHandle tensor_pair_first(TensorPair* p) { return p->first; }
TensorHandle tensor_pair_second(TensorPair* p) { return p->second; }
void tensor_pair_free(TensorPair* p) { delete p; }

/* ---------- Backend Capabilities ---------- */

int backend_supports_tensor_params(void) { return 1; }

/* ---------- System ---------- */

int get_rss_mb(void) {
    struct rusage usage;
    getrusage(RUSAGE_SELF, &usage);
#ifdef __APPLE__
    return (int)(usage.ru_maxrss / (1024 * 1024)); /* bytes on macOS */
#else
    return (int)(usage.ru_maxrss / 1024);           /* KB on Linux */
#endif
}

int get_current_rss_mb(void) {
#ifdef __APPLE__
    mach_task_basic_info_data_t info;
    mach_msg_type_number_t count = MACH_TASK_BASIC_INFO_COUNT;
    if (task_info(mach_task_self(), MACH_TASK_BASIC_INFO,
                  (task_info_t)&info, &count) == KERN_SUCCESS) {
        return (int)(info.resident_size / (1024 * 1024));
    }
#endif
    return get_rss_mb(); /* fallback to peak */
}

/* ---------- Backend Info ---------- */

const char* backend_name(void) { return "torch"; }

void backend_memory_report(void) {
    fprintf(stderr, "Torch backend: peak RSS = %d MB, current RSS = %d MB\n",
            get_rss_mb(), get_current_rss_mb());
}

void backend_reset_for_eval(void) {
    free_intermediates();
    for (auto& entry : param_registry) {
        if (entry.tensor->grad().defined())
            entry.tensor->grad().zero_();
    }
}

void backend_profile_reset(void) {}
void backend_profile_report(void) {}

double param_grad_item_at(int param_idx, int elem_idx) {
    auto& t = *param_registry[param_idx].tensor;
    if (!t.grad().defined()) return 0.0;
    return t.grad().data_ptr<double>()[elem_idx];
}

/* ---------- Debug ---------- */

void tensor_print(TensorHandle h) {
    std::cout << *to_tensor(h) << std::endl;
}
