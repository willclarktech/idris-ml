#include "backend.h"

#include <ATen/ATen.h>
#include <torch/torch.h>

#include <cstring>
#include <string>

/* ---------- Helpers ---------- */

static inline at::Tensor* to_tensor(TensorHandle h) {
    return static_cast<at::Tensor*>(h);
}

static inline TensorHandle from_tensor(at::Tensor t) {
    return static_cast<TensorHandle>(new at::Tensor(std::move(t)));
}

/* ---------- Lifecycle ---------- */

TensorHandle tensor_create_scalar(double value, int requires_grad) {
    auto t = torch::tensor(value, torch::dtype(torch::kFloat64));
    if (requires_grad) t.requires_grad_(true);
    return from_tensor(std::move(t));
}

TensorHandle tensor_create(double* data, int* shape, int rank, int requires_grad) {
    std::vector<int64_t> dims(rank);
    for (int i = 0; i < rank; i++) dims[i] = shape[i];
    auto opts = torch::TensorOptions().dtype(torch::kFloat64);
    auto t = torch::from_blob(data, dims, opts).clone();
    if (requires_grad) t.requires_grad_(true);
    return from_tensor(std::move(t));
}

TensorHandle tensor_clone(TensorHandle h) {
    return from_tensor(to_tensor(h)->clone());
}

void tensor_free(TensorHandle h) {
    delete to_tensor(h);
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
    return g.item<double>();
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

/* ---------- Convenience ---------- */

TensorHandle tensor_create_1d(int n, double* data, int requires_grad) {
    auto t = torch::from_blob(data, {(int64_t)n}, torch::kFloat64).clone();
    if (requires_grad) t.requires_grad_(true);
    return from_tensor(std::move(t));
}

TensorHandle tensor_create_2d(int rows, int cols, double* data, int requires_grad) {
    auto t = torch::from_blob(data, {(int64_t)rows, (int64_t)cols}, torch::kFloat64).clone();
    if (requires_grad) t.requires_grad_(true);
    return from_tensor(std::move(t));
}

double* tensor_alloc_doubles(int n) {
    return (double*)calloc(n, sizeof(double));
}

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

/* ---------- Debug ---------- */

void tensor_print(TensorHandle h) {
    std::cout << *to_tensor(h) << std::endl;
}
