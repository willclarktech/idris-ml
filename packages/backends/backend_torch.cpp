#include "backend.h"
#include "backend_torch/tensor.h"

#include <ATen/ATen.h>
#include <torch/torch.h>

#include <cstring>
#include <string>
#include <vector>
#include <unordered_set>
#include <sys/resource.h>
#include <sys/time.h>
#ifdef __APPLE__
#include <mach/mach.h>
#endif

/* ---------- MPS eager-init ----------
 * libtorch lazily initializes its MPS allocator + Metal command queue on
 * the first tensor that touches MPS. In multi-backend builds where cross-
 * backend tensor transfers run early (test-multi's Transfer suite), that
 * lazy init races macOS work-queue threads (MTLDevice setup, MPS
 * allocator pool ramp-up) and sporadically aborts the process with
 * SIGSEGV inside libtorch's `at::native::mps::*` paths. We saw this
 * empirically: re-ordering tests so an intra-torch CPU→MPS migration
 * runs FIRST dropped the crash rate from 100% to ~3%. Forcing the
 * init at dylib-load time (here) closes the window entirely — by the
 * time any Idris-side code calls `tensor_to_device_torch(h, "mps")`,
 * the MPS subsystem is already warm.
 *
 * Cost: one MPS tensor alloc+dealloc at process start (microseconds).
 * Skipped if torch wasn't built with MPS support or if no MPS device
 * is available on this host (Linux CI, non-Apple-Silicon Mac). */
__attribute__((constructor))
static void torch_mps_eager_init(void) {
    if (!at::hasMPS()) return;
    try {
        auto opts = torch::TensorOptions()
            .dtype(torch::kFloat32)
            .device(torch::Device(at::DeviceType::MPS));
        auto warm = torch::zeros({1}, opts);
        // Touch the data once to force the Metal command buffer to drain
        // (synchronize). The `.cpu()` round-trip is what
        // tensor_to_doubles_torch does at runtime — same code path.
        (void)warm.cpu();
        // `warm` falls out of scope; its storage refcount hits zero
        // and libtorch returns the MPS buffer to the allocator pool.
    } catch (...) {
        // First-touch failures (paravirt-MPS quirks on Tart VMs etc.)
        // shouldn't prevent dylib load. Subsequent Idris-side MPS use
        // will surface the real error.
    }
}

/* ---------- Profiling ---------- */
static double prof_backward_ms = 0, prof_optimizer_ms = 0;
static double prof_optimizer_math_ms = 0;  /* Just opt->step() / adam_step_foreach */
static int prof_epochs = 0;

static double _wall_ms_torch(void) {
    struct timeval tv;
    gettimeofday(&tv, NULL);
    return tv.tv_sec * 1000.0 + tv.tv_usec / 1000.0;
}

/* ---------- Intermediate tensor tracking ---------- */

// Track all non-persistent tensors so we can free them at optimizer_step.
// Reserved up-front to a typical-DNC working-set size to avoid per-epoch
// vector-grow churn (DNC-class workloads push ~3K intermediates/epoch).
// Param tensors go via from_tensor_persistent and are never tracked here,
// so free_intermediates can bulk-delete without filtering.
static std::vector<at::Tensor*> intermediates;
static std::vector<TensorPair*> all_pairs;
static bool tracking_enabled = true;
static long g_torch_peak_live = 0;  // high-water mark of intermediates.size()
struct _ReserveIntermediates {
    _ReserveIntermediates() { intermediates.reserve(4096); all_pairs.reserve(256); }
};
static _ReserveIntermediates _reserve_intermediates_instance;

/* ---------- Helpers ---------- */

// to_tensor lives inline in backend_torch/tensor.h (the modular tree
// header). from_tensor / from_tensor_persistent are declared there too,
// but defined here so they retain visibility over the monolith-private
// `intermediates` / `tracking_enabled` / `g_torch_peak_live` globals.
TensorHandle from_tensor(at::Tensor t) {
    auto* p = new at::Tensor(std::move(t));
    if (tracking_enabled) {
        intermediates.push_back(p);
        if ((long)intermediates.size() > g_torch_peak_live) g_torch_peak_live = (long)intermediates.size();
    }
    return static_cast<TensorHandle>(p);
}

// Persistent variant: not tracked for cleanup (survives optimizer_step)
TensorHandle from_tensor_persistent(at::Tensor t) {
    auto* p = new at::Tensor(std::move(t));
    return static_cast<TensorHandle>(p);
}

static void free_intermediates(); // defined after param_registry
static TensorHandle make_param_leaf(double* data, c10::IntArrayRef dims, torch::ScalarType dt); // defined near the F32 param creators

/* ---------- Lifecycle ----------
   tensor_create_scalar* / tensor_create* / tensor_cast_dtype_* extracted
   to backend_torch/core/lifecycle/{create_scalar,create,cast}.cpp. */

/* tensor_clone / tensor_free / tensor_retain_handle / tensor_release_handle
   extracted to backend_torch/core/lifecycle/. The `freed_by_cleanup`
   set + free_intermediates() impl stays here. */

// Track pointers freed by free_intermediates so tensor_free skips them
static std::unordered_set<void*> freed_by_cleanup;

/* ---------- Accessors ---------- */

/* tensor_item / tensor_item_1d / tensor_item_2d extracted to
   backend_torch/core/lifecycle/. tensor_item_1d / item_2d definitions
   that lived below are also gone. */

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
    // .cpu() before .data_ptr<>() — readback to host memory needs the
    // tensor on CPU. F64 on MPS isn't supported at construction so the
    // .to(kFloat64) for an MPS source goes through .cpu() first.
    auto t = to_tensor(h)->cpu().to(torch::kFloat64).contiguous();
    std::memcpy(out, t.data_ptr<double>(), t.numel() * sizeof(double));
}

void tensor_to_floats(TensorHandle h, float* out) {
    auto t = to_tensor(h)->cpu().to(torch::kFloat32).contiguous();
    std::memcpy(out, t.data_ptr<float>(), t.numel() * sizeof(float));
}

const char* tensor_dtype_name(TensorHandle h) {
    switch (to_tensor(h)->scalar_type()) {
        case torch::kFloat32:  return "F32";
        case torch::kFloat64:  return "F64";
        case torch::kBFloat16: return "BF16";
        case torch::kHalf:     return "F16";
        case torch::kChar:     return "I8";
        case torch::kShort:    return "I16";
        case torch::kInt:      return "I32";
        case torch::kLong:     return "I64";
        case torch::kByte:     return "U8";
        case torch::kBool:     return "BOOL";
        default:               return "F64";
    }
}

/* ---------- Arithmetic ---------- */

/* tensor_add / tensor_sub / tensor_mul / tensor_div extracted to
   backend_torch/core/elementwise/{add,sub,mul,div}.cpp. */

/* tensor_neg / tensor_abs / tensor_exp / tensor_log / tensor_sqrt /
   tensor_pow extracted to backend_torch/core/elementwise/. */

/* tensor_sigmoid / tensor_tanh / tensor_softplus extracted to
   backend_torch/core/elementwise/. gelu / leaky_relu / silu remain
   below pending 6c (nn/activation/). */

TensorHandle tensor_gelu(TensorHandle h) {
    return from_tensor(torch::gelu(*to_tensor(h)));
}

TensorHandle tensor_leaky_relu(TensorHandle h, double alpha) {
    return from_tensor(torch::leaky_relu(*to_tensor(h), alpha));
}

TensorHandle tensor_silu(TensorHandle h) {
    return from_tensor(torch::silu(*to_tensor(h)));
}

/* tensor_add_scalar / tensor_mul_scalar / tensor_clamp_min extracted
   to backend_torch/core/scalar/. */

/* Reduction ops live in backend_torch/linear/reduction/ */

/* ---------- Linear algebra ---------- */

TensorHandle tensor_matmul(TensorHandle a, TensorHandle b) {
    return from_tensor(torch::matmul(*to_tensor(a), *to_tensor(b)));
}

TensorHandle tensor_mv(TensorHandle mat, TensorHandle vec) {
    return from_tensor(torch::mv(*to_tensor(mat), *to_tensor(vec)));
}

TensorHandle tensor_linear(TensorHandle W, TensorHandle x, TensorHandle bias) {
    auto result = torch::mv(*to_tensor(W), *to_tensor(x));
    if (bias) result = result + *to_tensor(bias);
    return from_tensor(result);
}

TensorHandle tensor_linear_2d(TensorHandle W, TensorHandle X, TensorHandle bias) {
    /* X: [B, i], W: [o, i], bias: [o] -> Y: [B, o] = X @ W^T + bias */
    auto result = torch::nn::functional::linear(*to_tensor(X), *to_tensor(W),
                                                bias ? *to_tensor(bias) : torch::Tensor{});
    return from_tensor(result);
}

TensorHandle tensor_concat_2d_axis1(TensorHandle A, TensorHandle B) {
    /* A: [m, n], B: [m, k] -> [m, n+k] along axis 1 */
    return from_tensor(torch::cat({*to_tensor(A), *to_tensor(B)}, 1));
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
    /* Match tape's convention: -sum(target * log_softmax(input)) / numel.
       Differs from torch::nn::functional::cross_entropy, which expects
       target as class indices and means over the batch dim. The Idris
       side has no caller for this symbol today (the TODO row "Match
       PyTorch's catalogue of fused ops" tracks the future fused
       softmax_cross_entropy_with_logits landing); this impl just keeps
       the two backends agreeing for test_backend.c. */
    auto& in = *to_tensor(input);
    auto& tg = *to_tensor(target);
    auto ls = at::log_softmax(in, 0);
    auto loss = -(tg * ls).sum() / static_cast<double>(ls.numel());
    return from_tensor(loss);
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

TensorHandle tensor_cross_attention(TensorHandle hQ, TensorHandle hK, TensorHandle hV,
                                    TensorHandle hmask, double scale) {
    auto& Q = *to_tensor(hQ);
    auto& K = *to_tensor(hK);
    auto& V = *to_tensor(hV);
    auto scores = torch::bmm(Q, K.transpose(-2, -1)) * scale;
    if (hmask) scores = scores.masked_fill(to_tensor(hmask)->to(torch::kBool), -1.0e20);
    auto attn = torch::softmax(scores, -1);
    return from_tensor(torch::bmm(attn, V));
}

TensorHandle tensor_embedding(TensorHandle hweight, TensorHandle hindices, int n, int embedDim) {
    auto& weight = *to_tensor(hweight);  /* [vocabSize, embedDim] */
    auto& indices = *to_tensor(hindices);
    auto idx_long = indices.to(torch::kLong);
    auto out = torch::embedding(weight, idx_long);  /* [n, embedDim] */
    return from_tensor(out.reshape({-1}));  /* flatten to [n * embedDim] */
}

TensorHandle tensor_gather(TensorHandle hinput, TensorHandle hindex, int n) {
    auto& inp = *to_tensor(hinput);
    auto& idx = *to_tensor(hindex);
    auto idx_long = idx.to(torch::kLong);
    return from_tensor(torch::index_select(inp, 0, idx_long));
}

TensorHandle tensor_scatter_add(TensorHandle hindex, TensorHandle hsrc, int out_size) {
    auto& idx = *to_tensor(hindex);
    auto& src = *to_tensor(hsrc);
    // Match src's dtype + device — a hardcoded kFloat64 accumulator trips
    // "scatter(): Expected self.dtype to be equal to src.dtype" against an
    // F32 src (e.g. DNC allocation weighting on an F32 build).
    auto out = torch::zeros({(int64_t)out_size}, src.options());
    auto idx_long = idx.to(torch::kLong);
    out.scatter_add_(0, idx_long, src);
    return from_tensor(out);
}

TensorHandle tensor_argsort(TensorHandle ht, int dim, int descending) {
    auto& t = *to_tensor(ht);
    /* argsort produces indices — keep them in their natural integer dtype
       (kLong/I64), not a float. This is what the type-safe `targsort` Idris
       surface returns (Tensor _ d I64), and it sidesteps two latent bugs the
       old `.to(kFloat64)` had: precision loss above 2^53 indices, and an
       MPS abort (Metal has no F64). gather/scatter_add coerce the index to
       kLong internally, so the untyped DNC path is unaffected. */
    auto result = torch::argsort(t, dim, (bool)descending).to(torch::kLong);
    return from_tensor(result);
}

TensorHandle tensor_cumprod(TensorHandle ht, int dim) {
    auto& t = *to_tensor(ht);
    auto result = torch::cumprod(t, dim);
    return from_tensor(result);
}

TensorHandle tensor_gru_cell(TensorHandle hih, TensorHandle hhh,
                              TensorHandle hprev, int o) {
    /* nn.GRU equation. ih = W_ih @ x + b_ih, hh = W_hh @ h + b_hh.
       libtorch's autograd handles backward via the graph. */
    auto& ih = *to_tensor(hih);
    auto& hh = *to_tensor(hhh);
    auto& prev = *to_tensor(hprev);
    auto z = torch::sigmoid(ih.slice(0, 0, o) + hh.slice(0, 0, o));
    auto r = torch::sigmoid(ih.slice(0, o, 2*o) + hh.slice(0, o, 2*o));
    auto n = torch::tanh(ih.slice(0, 2*o, 3*o) + r * hh.slice(0, 2*o, 3*o));
    auto h_new = (1.0 - z) * n + z * prev;
    return from_tensor(h_new);
}

TensorHandle tensor_group_norm(TensorHandle hinput, TensorHandle hgamma, TensorHandle hbeta,
                               int numGroups, int channels, int spatial, double eps) {
    auto& inp = *to_tensor(hinput);
    auto& gamma = *to_tensor(hgamma);
    auto& beta = *to_tensor(hbeta);
    auto inp_3d = inp.reshape({1, (int64_t)channels, (int64_t)spatial});
    auto out = torch::group_norm(inp_3d, numGroups, gamma, beta, eps);
    return from_tensor(out.reshape({-1}));
}

TensorHandle tensor_conv_transpose1d(TensorHandle hinput, TensorHandle hkernel,
                                     TensorHandle hbias, int pad, int stride) {
    auto& inp = *to_tensor(hinput);
    auto& ker = *to_tensor(hkernel);
    auto inp_3d = inp.unsqueeze(0);
    at::Tensor bias_t;
    if (hbias) bias_t = *to_tensor(hbias);
    auto out = hbias
        ? torch::conv_transpose1d(inp_3d, ker, bias_t, {stride}, {pad})
        : torch::conv_transpose1d(inp_3d, ker, {}, {stride}, {pad});
    return from_tensor(out.squeeze(0));
}

TensorHandle tensor_conv_transpose2d(TensorHandle hinput, TensorHandle hkernel,
                                     TensorHandle hbias, int padH, int padW,
                                     int strideH, int strideW) {
    auto& inp = *to_tensor(hinput);
    auto& ker = *to_tensor(hkernel);
    auto inp_4d = inp.unsqueeze(0);
    at::Tensor bias_t;
    if (hbias) bias_t = *to_tensor(hbias);
    auto out = hbias
        ? torch::conv_transpose2d(inp_4d, ker, bias_t, {strideH, strideW}, {padH, padW})
        : torch::conv_transpose2d(inp_4d, ker, {}, {strideH, strideW}, {padH, padW});
    return from_tensor(out.squeeze(0));
}

TensorHandle tensor_conv1d_grouped(TensorHandle hinput, TensorHandle hkernel,
                                   TensorHandle hbias, int pad, int stride, int groups) {
    auto& inp = *to_tensor(hinput);
    auto& ker = *to_tensor(hkernel);
    auto inp_3d = inp.unsqueeze(0);
    at::Tensor bias_t;
    if (hbias) bias_t = *to_tensor(hbias);
    auto out = hbias
        ? torch::conv1d(inp_3d, ker, bias_t, {stride}, {pad}, /*dilation=*/{1}, groups)
        : torch::conv1d(inp_3d, ker, {}, {stride}, {pad}, {1}, groups);
    return from_tensor(out.squeeze(0));
}

TensorHandle tensor_conv2d_grouped(TensorHandle hinput, TensorHandle hkernel,
                                   TensorHandle hbias, int padH, int padW,
                                   int strideH, int strideW, int groups) {
    auto& inp = *to_tensor(hinput);
    auto& ker = *to_tensor(hkernel);
    auto inp_4d = inp.unsqueeze(0);
    at::Tensor bias_t;
    if (hbias) bias_t = *to_tensor(hbias);
    auto out = hbias
        ? torch::conv2d(inp_4d, ker, bias_t, {strideH, strideW}, {padH, padW}, {1, 1}, groups)
        : torch::conv2d(inp_4d, ker, {}, {strideH, strideW}, {padH, padW}, {1, 1}, groups);
    return from_tensor(out.squeeze(0));
}

TensorHandle tensor_avg_pool1d(TensorHandle hinput, int kL, int stride) {
    auto& inp = *to_tensor(hinput);
    auto inp_3d = inp.unsqueeze(0);
    auto out = torch::avg_pool1d(inp_3d, {kL}, {stride});
    return from_tensor(out.squeeze(0));
}

TensorHandle tensor_avg_pool2d(TensorHandle hinput, int kH, int kW, int strideH, int strideW) {
    auto& inp = *to_tensor(hinput);
    auto inp_4d = inp.unsqueeze(0);
    auto out = torch::avg_pool2d(inp_4d, {kH, kW}, {strideH, strideW});
    return from_tensor(out.squeeze(0));
}

TensorHandle tensor_conv1d(TensorHandle hinput, TensorHandle hkernel,
                           TensorHandle hbias, int pad, int stride) {
    auto& inp = *to_tensor(hinput);
    auto& ker = *to_tensor(hkernel);
    auto inp_3d = inp.unsqueeze(0);
    at::Tensor bias_t;
    if (hbias) bias_t = *to_tensor(hbias);
    auto out = hbias
        ? torch::conv1d(inp_3d, ker, bias_t, /*stride=*/{stride}, /*padding=*/{pad})
        : torch::conv1d(inp_3d, ker, /*bias=*/{}, /*stride=*/{stride}, /*padding=*/{pad});
    return from_tensor(out.squeeze(0));
}

TensorHandle tensor_max_pool1d(TensorHandle hinput, int kL, int stride) {
    auto& inp = *to_tensor(hinput);
    auto inp_3d = inp.unsqueeze(0);
    auto out = torch::max_pool1d(inp_3d, {kL}, {stride});
    return from_tensor(out.squeeze(0));
}

TensorHandle tensor_create_param_3d(int d0, int d1, int d2, double* data) {
    return make_param_leaf(data, {(int64_t)d0, (int64_t)d1, (int64_t)d2}, torch::kFloat64);
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

/* Batched conv2d — takes [B, inC, H, W] directly, returns [B, outC, oH, oW].
   torch::conv2d is already batch-native; the unsqueeze/squeeze dance in
   tensor_conv2d above is just for the per-sample API. */
TensorHandle tensor_conv2d_batched(TensorHandle hinput, TensorHandle hkernel,
                                    TensorHandle hbias, int padH, int padW,
                                    int strideH, int strideW) {
    auto& inp = *to_tensor(hinput);   /* [B, inC, H, W] */
    auto& ker = *to_tensor(hkernel);  /* [outC, inC, kH, kW] */
    at::Tensor bias_t;
    if (hbias) bias_t = *to_tensor(hbias);
    auto out = hbias
        ? torch::conv2d(inp, ker, bias_t,
              /*stride=*/{strideH, strideW}, /*padding=*/{padH, padW})
        : torch::conv2d(inp, ker, /*bias=*/{},
              /*stride=*/{strideH, strideW}, /*padding=*/{padH, padW});
    return from_tensor(out);
}

/* Batched max-pool2d — takes [B, C, H, W] directly, returns [B, C, oH, oW]. */
TensorHandle tensor_max_pool2d_batched(TensorHandle hinput, int kH, int kW,
                                        int strideH, int strideW) {
    auto& inp = *to_tensor(hinput);  /* [B, C, H, W] */
    auto out = torch::max_pool2d(inp, {kH, kW}, {strideH, strideW});
    return from_tensor(out);
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

extern "C" void _dbg_dump_param_grads_if_enabled_torch(void);

void tensor_backward(TensorHandle h) {
    double t0 = _wall_ms_torch();
    to_tensor(h)->backward();
    prof_backward_ms += _wall_ms_torch() - t0;
    /* Phase 1.5e diagnostic: dump per-param gradient L2 norms after backward.
       Implementation lives below the param_registry declaration. */
    _dbg_dump_param_grads_if_enabled_torch();
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

static inline bool idrisml_is_floating_st(torch::ScalarType dt);  /* defined below */

void tensor_set_requires_grad(TensorHandle h, int requires_grad) {
    auto t = to_tensor(h);
    /* torch throws if you request grad on a non-floating tensor. Inference
       dtypes (int/bool) can be registered (e.g. via registerParam, for
       serialization) but can't carry gradients — silently leave grad off
       rather than abort. */
    if (requires_grad && !idrisml_is_floating_st(t->scalar_type())) return;
    t->requires_grad_(requires_grad != 0);
}

/* No-grad scope. Counter (not bool) so nested withNoGrad scopes
   nest correctly — only the outermost begin creates the guard,
   only the outermost end releases it. */
static thread_local int no_grad_depth = 0;
static thread_local std::unique_ptr<torch::NoGradGuard> no_grad_guard;

void tensor_no_grad_begin(void) {
    if (no_grad_depth == 0) {
        no_grad_guard = std::make_unique<torch::NoGradGuard>();
    }
    no_grad_depth++;
}

void tensor_no_grad_end(void) {
    if (no_grad_depth > 0) {
        no_grad_depth--;
        if (no_grad_depth == 0) {
            no_grad_guard.reset();
        }
    }
}
/* No buffer ceiling on torch; per-epoch generation free is a no-op. */
void tensor_epoch_begin(void) {}
void tensor_epoch_end(void) {}

/* ---------- Device ---------- */

/* EAFP availability gate: a device-pin to absent/invalid hardware
 * (e.g. "cuda:1" on a 1-GPU box, or MPS on a non-Apple host) makes
 * libtorch's `.to()` throw a c10::Error. Unguarded, that exception
 * crosses the C->Chez FFI boundary and becomes std::terminate/SIGABRT.
 * Catch it here and return a NULL handle; the Idris side lifts NULL ->
 * Left DeviceError. This is the one source of truth for availability —
 * no separate is_available probe to drift. All torch device-pinning
 * (primCreateFromHost, primIntraMigrate, primCreate's post-create
 * migration) routes through here, so this single guard covers them. */
TensorHandle tensor_to_device(TensorHandle h, const char* device) {
    try {
        return from_tensor(to_tensor(h)->to(std::string(device)));
    } catch (const std::exception& e) {
        fprintf(stderr, "[torch] tensor_to_device(%s) failed: %s\n",
                device, e.what());
        return nullptr;
    } catch (...) {
        fprintf(stderr, "[torch] tensor_to_device(%s) failed: unknown\n",
                device);
        return nullptr;
    }
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
    /* torch::lstm_cell expects 2D input/hx/cx ([batch, *]). Mirror Python's
       nn.LSTMCell which auto-unsqueezes 1D inputs so unbatched callers
       (including the C-side gradient test harness) work. */
    auto in1d = *to_tensor(input);
    auto hx1d = *to_tensor(hx);
    auto cx1d = *to_tensor(cx);
    bool unbatched = (in1d.dim() == 1);
    auto in2d = unbatched ? in1d.unsqueeze(0) : in1d;
    auto hx2d = unbatched ? hx1d.unsqueeze(0) : hx1d;
    auto cx2d = unbatched ? cx1d.unsqueeze(0) : cx1d;
    auto result = torch::lstm_cell(
        in2d, {hx2d, cx2d},
        *to_tensor(w_ih), *to_tensor(w_hh),
        *to_tensor(b_ih), *to_tensor(b_hh));
    auto new_h = std::get<0>(result);
    auto new_c = std::get<1>(result);
    if (unbatched) { new_h = new_h.squeeze(0); new_c = new_c.squeeze(0); }
    *out_h = from_tensor(new_h);
    *out_c = from_tensor(new_c);
}

/* ---------- Diagnostics ---------- */

/* DEBUG_LSTM_TRAJ: dump h0/c0 param value trajectories. Mirrors the
   tape backend's diagnostic so a cross-backend convergence regression
   on RNN init can be localized. Walks the shared param registry. */
static int _dbg_traj_step_torch = 0;
extern "C" void _dbg_dump_lstm_traj_if_enabled_torch(void) {
    if (!getenv("DEBUG_LSTM_TRAJ")) return;
    int every = 100;
    const char* every_s = getenv("DEBUG_LSTM_TRAJ_EVERY");
    if (every_s) every = atoi(every_s);
    _dbg_traj_step_torch++;
    if (_dbg_traj_step_torch % every != 0 && _dbg_traj_step_torch != 1) return;
    for (int i = 0; i < param_count(); i++) {
        std::string nm(param_name(i));
        if (nm.size() >= 3 &&
            (nm.substr(nm.size()-3) == "_h0" || nm.substr(nm.size()-3) == "_c0")) {
            auto& t = *(at::Tensor*)param_tensor(i);
            auto t_cpu = t.detach().cpu().to(torch::kFloat64).contiguous();
            const double* d = t_cpu.data_ptr<double>();
            int numel = (int)t.numel();
            double l2 = 0.0, mn = 1e300, mx = -1e300;
            for (int j = 0; j < numel; j++) {
                double v = d[j];
                l2 += v*v;
                if (v < mn) mn = v;
                if (v > mx) mx = v;
            }
            l2 = std::sqrt(l2);
            fprintf(stderr, "[traj epoch %d] %s l2=%.10g min=%.10g max=%.10g | t[0..2]=%.10g, %.10g, %.10g\n",
                    _dbg_traj_step_torch, nm.c_str(), l2, mn, mx,
                    numel >= 1 ? d[0] : 0.0,
                    numel >= 2 ? d[1] : 0.0,
                    numel >= 3 ? d[2] : 0.0);
        }
    }
}

/* DEBUG_PARAM_GRADS: dump per-param gradient L2 norms after a backward
   pass. Mirrors backend_tape.c's diagnostic for cross-backend comparison. */
extern "C" void _dbg_dump_param_grads_if_enabled_torch(void) {
    if (!getenv("DEBUG_PARAM_GRADS")) return;
    fprintf(stderr, "=== param grads after backward (torch) ===\n");
    for (int i = 0; i < param_count(); i++) {
        std::string name(param_name(i));
        auto* tensor = (at::Tensor*)param_tensor(i);
        double l2 = 0.0;
        int has_nan = 0;
        int numel = (int)tensor->numel();
        if (tensor->grad().defined()) {
            auto g_cpu = tensor->grad().cpu().to(torch::kFloat64).contiguous();
            const double* g = g_cpu.data_ptr<double>();
            for (int j = 0; j < numel; j++) {
                double v = g[j];
                if (std::isnan(v) || std::isinf(v)) has_nan = 1;
                l2 += v * v;
            }
            l2 = std::sqrt(l2);
            fprintf(stderr, "  %-40s numel=%-6d l2=%12.6e%s\n",
                    name.c_str(), numel, l2,
                    has_nan ? " NAN_OR_INF!" : "");
        } else {
            fprintf(stderr, "  %-40s numel=%-6d NO_GRAD\n",
                    name.c_str(), numel);
        }
    }
}

/* The param-registry surface (param_register / param_count / param_name
   / param_tensor / param_grad_item* / param_zero_all_grads /
   param_subtract_delta / param_load_data / param_load_data_int64) lives
   in shared/training/param_registry.c and is opted in via the
   SHARED_BACKENDS_param_registry Makefile list. Torch supplies the
   per-tensor accessors (tensor_numel / tensor_has_grad / grad_read /
   grad_write / zero_grad / data_read / data_write / load_doubles /
   load_int64) through its port adapter at the bottom of this file. */

static void free_intermediates() {
    // Params are always created via from_tensor_persistent and are never
    // tracked in `intermediates`, so we can bulk-delete here without
    // filtering. (Previous version built an unordered_set<at::Tensor*>
    // from param_registry per call as a safety net — that was a hot-path
    // hash build for ~thousands of intermediates on DNC-class workloads.)
    freed_by_cleanup.clear();
    freed_by_cleanup.reserve(intermediates.size());
    for (auto* p : intermediates) {
        if (p) {
            freed_by_cleanup.insert(p);
            delete p;
        }
    }
    intermediates.clear();
    // Free TensorPair structs
    for (auto* p : all_pairs) delete p;
    all_pairs.clear();
}

// Byte-exact I64 readout — bypasses the double pivot so values
// above 2^53 survive. The `.to(kInt64)` is a no-op when the source
// is already I64; if a caller asks for int64 readout on a non-I64
// tensor we still narrow through torch's standard truncating cast
// (matches the implicit cast in `tensor_to_doubles` for the same
// case). `.cpu()` mirrors the device handling in `tensor_to_doubles`.
void tensor_to_int64(TensorHandle h, int64_t* out) {
    auto t = to_tensor(h)->cpu().to(torch::kInt64).contiguous();
    std::memcpy(out, t.data_ptr<int64_t>(), t.numel() * sizeof(int64_t));
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

/* tensor_alloc_doubles / tensor_free_doubles / tensor_read_double /
 * tensor_ptr_array_alloc live in shared_utils.c. */

/* ---------- Tensor pointer array ---------- */

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

TensorHandle tensor_reshape_4d(TensorHandle h, int d0, int d1, int d2, int d3) {
    return from_tensor(to_tensor(h)->reshape(
        {(int64_t)d0, (int64_t)d1, (int64_t)d2, (int64_t)d3}));
}

TensorHandle tensor_expand_mask(TensorHandle hmask, int B) {
    return from_tensor(to_tensor(hmask)->unsqueeze(0).expand({(int64_t)B, -1, -1}).contiguous());
}

TensorHandle tensor_tile_2d(TensorHandle h, int rep0, int rep1) {
    return from_tensor(to_tensor(h)->repeat({(int64_t)rep0, (int64_t)rep1}));
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

TensorHandle tensor_reshape_1d(TensorHandle h, int n) {
    return from_tensor(to_tensor(h)->reshape({(int64_t)n}));
}


/* Forward decl — st_for_dtag is defined further down with the unified
   create/cast block; tensor_one_hot below calls it. */
static torch::ScalarType st_for_dtag(int dtag);

TensorHandle tensor_one_hot(int* tokens, int n_tokens, int vocab_size, int dtag) {
    int total = n_tokens * vocab_size;
    // Build the 0/1 pattern in F64, then cast to the requested dtype so the
    // result honestly matches the Idris `dt` (0/1 is exact in every dtype —
    // float or int — so the cast is lossless). An F32 model gets a real F32
    // one-hot, an F64 model a real F64 one — no silent dtype divergence.
    auto t = torch::zeros({(int64_t)total}, torch::kFloat64);
    auto acc = t.accessor<double, 1>();
    for (int i = 0; i < n_tokens; i++) {
        int tok = tokens[i];
        if (tok >= 0 && tok < vocab_size)
            acc[i * vocab_size + tok] = 1.0;
    }
    /* Delegate to st_for_dtag for the kind-major dtag layout; invalid
       dtags abort there. F64 is the build dtype above, so skip the cast
       only when the requested output dtype is already F64. */
    torch::ScalarType st = st_for_dtag(dtag);
    if (st != torch::kFloat64) t = t.to(st);
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
    return make_param_leaf(data, {(int64_t)rows, (int64_t)cols}, torch::kFloat64);
}

TensorHandle tensor_create_param_4d(int d0, int d1, int d2, int d3, double* data) {
    return make_param_leaf(data, {(int64_t)d0, (int64_t)d1, (int64_t)d2, (int64_t)d3}, torch::kFloat64);
}

TensorHandle tensor_create_param_1d(int n, double* data) {
    return make_param_leaf(data, {(int64_t)n}, torch::kFloat64);
}

TensorHandle tensor_create_state_2d(int rows, int cols, double* data) {
    auto t = torch::from_blob(data, {(int64_t)rows, (int64_t)cols}, torch::kFloat64).clone();
    return from_tensor_persistent(std::move(t));
}

TensorHandle tensor_create_state_1d(int n, double* data) {
    auto t = torch::from_blob(data, {(int64_t)n}, torch::kFloat64).clone();
    return from_tensor_persistent(std::move(t));
}

/* ================================================================
   Per-dtype creation variants
   --------------------------------------------------------------
   Torch supports F32 and F64 natively (both first-class). The _f64
   variants are functionally identical to the existing unsuffixed
   creators; the _f32 variants build with kFloat32 and cast the
   double-typed input buffer down.
   ================================================================ */

static TensorHandle torch_cast_to(TensorHandle h, torch::ScalarType dt) {
    auto t = *to_tensor(h);
    return from_tensor_persistent(t.dtype() == dt ? t : t.to(dt));
}

/* Build a persistent grad-tracking leaf at dtype dt from a fp64 host buffer.
   Cast-before-requires_grad is load-bearing: `.to(dt)` applied to an
   already-requires_grad tensor yields a NON-LEAF (the ToCopy output), whose
   .grad never populates during backward — the optimizer then reads a zero
   gradient and silently no-ops, freezing F32 training at the init loss.
   The F64 param creators set requires_grad on the un-cast leaf (no cast), so
   they were unaffected; only the F32 path went cast-after-grad. */
static TensorHandle make_param_leaf(double* data, c10::IntArrayRef dims, torch::ScalarType dt) {
    auto t = torch::from_blob(data, dims, torch::kFloat64).clone();
    if (dt != torch::kFloat64) t = t.to(dt);
    t.requires_grad_(true);
    // A parameter must be an autograd leaf or its .grad never populates and
    // the optimizer silently no-ops (frozen training, no error). This fires
    // immediately at the construction site if a future change reorders the
    // cast/move after requires_grad_ on any backend build. See gotchas.md
    // "A parameter must be cast/moved before requires_grad_".
    TORCH_CHECK(t.is_leaf(),
        "parameter tensor is not an autograd leaf: cast/move (.to(dtype/device)) "
        "must precede requires_grad_, otherwise .grad never populates and the "
        "optimizer silently freezes training");
    return from_tensor_persistent(std::move(t));
}

/* F64 — aliases to existing unsuffixed implementations.
   tensor_create_scalar_f64 and tensor_create_f64 are already defined
   above via the _impl refactor, so they're omitted here. */
TensorHandle tensor_create_1d_f64(int n, double* d, int rg)                             { return tensor_create_1d(n, d, rg); }
TensorHandle tensor_create_2d_f64(int rows, int cols, double* d, int rg)                { return tensor_create_2d(rows, cols, d, rg); }
TensorHandle tensor_create_param_1d_f64(int n, double* d)                               { return tensor_create_param_1d(n, d); }
TensorHandle tensor_create_param_2d_f64(int rows, int cols, double* d)                  { return tensor_create_param_2d(rows, cols, d); }
TensorHandle tensor_create_param_3d_f64(int d0, int d1, int d2, double* d)              { return tensor_create_param_3d(d0, d1, d2, d); }
TensorHandle tensor_create_param_4d_f64(int d0, int d1, int d2, int d3, double* d)      { return tensor_create_param_4d(d0, d1, d2, d3, d); }
TensorHandle tensor_create_state_1d_f64(int n, double* d)                               { return tensor_create_state_1d(n, d); }
TensorHandle tensor_create_state_2d_f64(int rows, int cols, double* d)                  { return tensor_create_state_2d(rows, cols, d); }

/* F32 — build at fp64 then cast down. Note: tensor_create_scalar_f32 and
   tensor_create_f32 already exist (refactored in their original location
   with _impl helpers); these wrappers cover the remaining 8 cases. */
TensorHandle tensor_create_1d_f32(int n, double* d, int rg) {
    auto t = torch::from_blob(d, {(int64_t)n}, torch::kFloat64).clone();
    free(d);
    t = t.to(torch::kFloat32);
    if (rg) t.requires_grad_(true);   // cast-before-grad: keep the F32 tensor a leaf
    return from_tensor(std::move(t));
}
TensorHandle tensor_create_2d_f32(int rows, int cols, double* d, int rg) {
    auto t = torch::from_blob(d, {(int64_t)rows, (int64_t)cols}, torch::kFloat64).clone();
    free(d);
    t = t.to(torch::kFloat32);
    if (rg) t.requires_grad_(true);   // cast-before-grad: keep the F32 tensor a leaf
    return from_tensor(std::move(t));
}
TensorHandle tensor_create_param_1d_f32(int n, double* d) {
    return make_param_leaf(d, {(int64_t)n}, torch::kFloat32);
}
TensorHandle tensor_create_param_2d_f32(int rows, int cols, double* d) {
    return make_param_leaf(d, {(int64_t)rows, (int64_t)cols}, torch::kFloat32);
}
TensorHandle tensor_create_param_3d_f32(int d0, int d1, int d2, double* d) {
    return make_param_leaf(d, {(int64_t)d0, (int64_t)d1, (int64_t)d2}, torch::kFloat32);
}
TensorHandle tensor_create_param_4d_f32(int d0, int d1, int d2, int d3, double* d) {
    return make_param_leaf(d, {(int64_t)d0, (int64_t)d1, (int64_t)d2, (int64_t)d3}, torch::kFloat32);
}
TensorHandle tensor_create_state_1d_f32(int n, double* d) {
    auto h = tensor_create_state_1d(n, d);
    return torch_cast_to(h, torch::kFloat32);
}
TensorHandle tensor_create_state_2d_f32(int rows, int cols, double* d) {
    auto h = tensor_create_state_2d(rows, cols, d);
    return torch_cast_to(h, torch::kFloat32);
}

TensorHandle tensor_view_2d(TensorHandle h, int row, int col) {
    /* Returns a 0-dim view that shares storage with the parent tensor.
       Must be persistent — views into param tensors survive free_intermediates. */
    return from_tensor_persistent(to_tensor(h)->select(0, row).select(0, col));
}

TensorHandle tensor_view_1d(TensorHandle h, int idx) {
    return from_tensor_persistent(to_tensor(h)->select(0, idx));
}

/* tensor_item_1d / tensor_item_2d extracted to
   backend_torch/core/lifecycle/. */

/* ---------- Native Optimizer ---------- */

/* Helper: collect all param_registry tensors into a vector */
static std::vector<at::Tensor> collect_param_tensors() {
    std::vector<at::Tensor> params;
    params.reserve((size_t)param_count());
    for (int i_ = 0; i_ < param_count(); i_++) {
        auto* tensor = (at::Tensor*)param_tensor(i_);
        params.push_back(*tensor);
    }
    return params;
}

/* Wrapper to track optimizer type alongside PyTorch optimizer */
struct OptWrapper {
    int type; // 0=sgd, 1=rmsprop, 2=adam
    double lr, beta1, beta2, eps, alpha, weight_decay, momentum;
    torch::optim::Optimizer* opt;
    std::string prefix;  // empty = manages all params; else only params whose
                          // registry name starts with `prefix` (SAC multi-opt)
    int64_t pending_step = 0;  // step count restored by optimizer_set_meta,
                                // stamped onto per-param state when it is first
                                // created (lazily, in optimizer_set_m/_v) — the
                                // step lives inside Adam/RMSprop ParamState,
                                // which doesn't exist on a freshly-loaded opt.
};

static std::vector<at::Tensor> collect_param_tensors_filtered(const std::string& prefix) {
    std::vector<at::Tensor> params;
    params.reserve((size_t)param_count());
    for (int i_ = 0; i_ < param_count(); i_++) {
        auto* tensor = (at::Tensor*)param_tensor(i_);
        if (prefix.empty()) {
            params.push_back(*tensor);
        } else {
            std::string name(param_name(i_));
            if (name.rfind(prefix, 0) == 0) {
                params.push_back(*tensor);
            }
        }
    }
    return params;
}

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

OptimizerHandle optimizer_create_adam_group(double lr, double beta1, double beta2,
                                            double eps, const char* prefix) {
    std::string pfx = prefix ? prefix : "";
    auto params = collect_param_tensors_filtered(pfx);
    auto* w = new OptWrapper();
    w->type = 2; w->lr = lr; w->beta1 = beta1; w->beta2 = beta2; w->eps = eps;
    w->prefix = pfx;
    w->opt = new torch::optim::Adam(params,
        torch::optim::AdamOptions(lr).betas(std::make_tuple(beta1, beta2)).eps(eps));
    return static_cast<OptimizerHandle>(w);
}

OptimizerHandle optimizer_create_adamw(double lr, double beta1, double beta2, double eps,
                                       double weight_decay) {
    auto params = collect_param_tensors();
    auto* w = new OptWrapper();
    w->type = 3; w->lr = lr; w->beta1 = beta1; w->beta2 = beta2; w->eps = eps;
    w->opt = new torch::optim::AdamW(params,
        torch::optim::AdamWOptions(lr).betas(std::make_tuple(beta1, beta2))
                                      .eps(eps).weight_decay(weight_decay));
    return static_cast<OptimizerHandle>(w);
}

void optimizer_free(OptimizerHandle h) {
    auto* w = static_cast<OptWrapper*>(h);
    delete w->opt;
    delete w;
}

/* Fused multi-tensor Adam step using at::_foreach_*. Replaces the per-param
   loop in torch::optim::Adam::step() with batched MultiTensorApply kernels.
   Numerics are identical to the standard formulation: m and v live in the
   AdamParamState slots so save/load still works through libtorch's serializer.
   Params with undefined grad are skipped (matches libtorch behaviour). */
/* Core Adam foreach math: assumes caller has gathered lists, materialised
   state, bumped step, and entered a NoGradGuard. Shared by adam_step_foreach
   and adamw_step_foreach (AdamW adds decoupled weight-decay before the
   call but uses the same math thereafter). */
static void adam_core_foreach(double lr, double beta1, double beta2, double eps,
                              int64_t new_step,
                              std::vector<at::Tensor>& params,
                              std::vector<at::Tensor>& m_list,
                              std::vector<at::Tensor>& v_list,
                              std::vector<at::Tensor>& g_list) {
    double bc1 = 1.0 - std::pow(beta1, (double)new_step);
    double bc2 = 1.0 - std::pow(beta2, (double)new_step);
    double bc2_sqrt = std::sqrt(bc2);
    double step_size = lr / bc1;

    /* m = β1·m + (1-β1)·g — matches libtorch's mul_().add_(g, 1-β1) order. */
    at::_foreach_mul_(m_list, beta1);
    at::_foreach_add_(m_list, g_list, 1.0 - beta1);

    /* v = β2·v + (1-β2)·g² */
    at::_foreach_mul_(v_list, beta2);
    at::_foreach_addcmul_(v_list, g_list, g_list, 1.0 - beta2);

    /* denom = sqrt(v) / sqrt(bc2) + eps */
    auto denom = at::_foreach_sqrt(v_list);
    at::_foreach_div_(denom, bc2_sqrt);
    at::_foreach_add_(denom, eps);

    /* p -= step_size · m / denom */
    at::_foreach_addcdiv_(params, m_list, denom, -step_size);
}

static void adam_step_foreach(OptWrapper* w,
                               const std::vector<at::Tensor>& params) {
    auto& opt = *w->opt;
    auto& state = opt.state();

    std::vector<at::Tensor> active_params, m_list, v_list, g_list;
    active_params.reserve(params.size());
    m_list.reserve(params.size());
    v_list.reserve(params.size());
    g_list.reserve(params.size());

    int64_t new_step = 0;
    for (const auto& p : params) {
        if (!p.grad().defined()) continue;
        auto key = p.unsafeGetTensorImpl();
        if (state.count(key) == 0) {
            state[key] = std::make_unique<torch::optim::AdamParamState>();
            auto& s0 = static_cast<torch::optim::AdamParamState&>(*state[key]);
            s0.exp_avg(at::zeros_like(p));
            s0.exp_avg_sq(at::zeros_like(p));
            s0.step(0);
        }
        auto& s = static_cast<torch::optim::AdamParamState&>(*state.at(key));
        s.step(s.step() + 1);
        new_step = s.step();
        active_params.push_back(p);
        m_list.push_back(s.exp_avg());
        v_list.push_back(s.exp_avg_sq());
        g_list.push_back(p.grad());
    }

    if (active_params.empty()) return;

    /* In-place updates on leaf params with requires_grad=true would trip
       autograd's check_inplace. Same wrap as torch::optim::Adam::step(). */
    torch::NoGradGuard no_grad;
    adam_core_foreach(w->lr, w->beta1, w->beta2, w->eps, new_step,
                      active_params, m_list, v_list, g_list);
}

/* Fused multi-tensor AdamW step. Mirrors libtorch's AdamW::step(): decoupled
   weight-decay applied to params as `p *= 1 - lr*wd` BEFORE the Adam math
   (distinct from Adam, which folds weight_decay into the gradient if any).
   AdamWParamState is a separate type from AdamParamState in libtorch but
   carries the same field accessors (step / exp_avg / exp_avg_sq), so the
   shared adam_core_foreach math is reusable. */
static void adamw_step_foreach(OptWrapper* w,
                                const std::vector<at::Tensor>& params) {
    auto& opt = *w->opt;
    auto& state = opt.state();

    std::vector<at::Tensor> active_params, m_list, v_list, g_list;
    active_params.reserve(params.size());
    m_list.reserve(params.size());
    v_list.reserve(params.size());
    g_list.reserve(params.size());

    int64_t new_step = 0;
    for (const auto& p : params) {
        if (!p.grad().defined()) continue;
        auto key = p.unsafeGetTensorImpl();
        if (state.count(key) == 0) {
            state[key] = std::make_unique<torch::optim::AdamWParamState>();
            auto& s0 = static_cast<torch::optim::AdamWParamState&>(*state[key]);
            s0.exp_avg(at::zeros_like(p));
            s0.exp_avg_sq(at::zeros_like(p));
            s0.step(0);
        }
        auto& s = static_cast<torch::optim::AdamWParamState&>(*state.at(key));
        s.step(s.step() + 1);
        new_step = s.step();
        active_params.push_back(p);
        m_list.push_back(s.exp_avg());
        v_list.push_back(s.exp_avg_sq());
        g_list.push_back(p.grad());
    }

    if (active_params.empty()) return;

    torch::NoGradGuard no_grad;

    /* Decoupled weight decay: p *= 1 - lr*wd  (skip when wd == 0).
       Numerically equivalent to libtorch's AdamW::step() but not bit-identical
       on CPU — the at::_foreach_* code paths use slightly different SIMD /
       FMA ordering than chained per-tensor methods, producing ~1e-5 relative
       drift over a few epochs. Convergence trajectory matches (verified on
       Gpt). The structural benefit is GPU-shaped where the kernel-launch
       savings dominate any per-op fp noise. */
    if (w->weight_decay != 0.0) {
        at::_foreach_mul_(active_params, 1.0 - w->lr * w->weight_decay);
    }

    adam_core_foreach(w->lr, w->beta1, w->beta2, w->eps, new_step,
                      active_params, m_list, v_list, g_list);
}

/* Fused multi-tensor RMSprop step (non-centered). Matches libtorch's
   `torch::optim::RMSprop::step()` op order:
     (optional)  g_eff = g + weight_decay * p     (fresh clone, don't mutate real grad)
                 v.mul_(α).addcmul_(g, g, 1 - α)
                 avg = sqrt(v) + eps               (fresh tensor; preserves v for next step)
     (momentum)  buf.mul_(m).addcdiv_(g, avg);     p -= lr * buf
     (no momentum)                                 p -= lr * g / avg
   v / buf live in RMSpropParamState so libtorch's serializer keeps working.
   Params with undefined grad are skipped. */
static void rmsprop_step_foreach(OptWrapper* w,
                                  const std::vector<at::Tensor>& params) {
    auto& opt = *w->opt;
    auto& state = opt.state();
    const bool use_momentum = (w->momentum > 0.0);
    const bool use_wd = (w->weight_decay != 0.0);

    std::vector<at::Tensor> active_params, v_list, g_list, buf_list;
    active_params.reserve(params.size());
    v_list.reserve(params.size());
    g_list.reserve(params.size());
    if (use_momentum) buf_list.reserve(params.size());

    for (const auto& p : params) {
        if (!p.grad().defined()) continue;
        auto key = p.unsafeGetTensorImpl();
        if (state.count(key) == 0) {
            state[key] = std::make_unique<torch::optim::RMSpropParamState>();
            auto& s0 = static_cast<torch::optim::RMSpropParamState&>(*state[key]);
            s0.square_avg(at::zeros_like(p));
            s0.step(0);
        }
        auto& s = static_cast<torch::optim::RMSpropParamState&>(*state.at(key));
        s.step(s.step() + 1);
        if (use_momentum && !s.momentum_buffer().defined()) {
            s.momentum_buffer(at::zeros_like(p));
        }
        active_params.push_back(p);
        v_list.push_back(s.square_avg());
        g_list.push_back(p.grad());
        if (use_momentum) buf_list.push_back(s.momentum_buffer());
    }

    if (active_params.empty()) return;

    torch::NoGradGuard no_grad;

    /* g_eff = grads with weight_decay folded in, if any. Fresh-allocate so
       we don't mutate the real .grad() — matches per-param behaviour where
       libtorch does `grad = grad.add(p, alpha=wd)` and uses the result. */
    std::vector<at::Tensor> g_eff;
    if (use_wd) {
        g_eff.reserve(g_list.size());
        for (auto& g : g_list) g_eff.push_back(g.clone());
        at::_foreach_add_(g_eff, active_params, w->weight_decay);
    } else {
        g_eff = g_list;  /* alias — only read downstream, not mutated */
    }

    double alpha = w->alpha, lr = w->lr, eps = w->eps;

    /* v = α·v + (1-α)·g² */
    at::_foreach_mul_(v_list, alpha);
    at::_foreach_addcmul_(v_list, g_eff, g_eff, 1.0 - alpha);

    /* avg = sqrt(v) + ε  (fresh tensor list; v stays intact for next step) */
    auto avg = at::_foreach_sqrt(v_list);
    at::_foreach_add_(avg, eps);

    if (use_momentum) {
        /* buf = momentum·buf + g/avg ; p -= lr·buf */
        at::_foreach_mul_(buf_list, w->momentum);
        at::_foreach_addcdiv_(buf_list, g_eff, avg, 1.0);
        at::_foreach_add_(active_params, buf_list, -lr);
    } else {
        /* p -= lr · g / avg */
        at::_foreach_addcdiv_(active_params, g_eff, avg, -lr);
    }
}

/* Fused multi-tensor SGD step. Our wrapper exposes only `lr` (no momentum,
   no weight_decay, no nesterov), so the math collapses to a single
   _foreach_add_ call. Skips params with undefined grad. */
static void sgd_step_foreach(OptWrapper* w,
                              const std::vector<at::Tensor>& params) {
    std::vector<at::Tensor> active, grads;
    active.reserve(params.size());
    grads.reserve(params.size());
    for (const auto& p : params) {
        if (!p.grad().defined()) continue;
        active.push_back(p);
        grads.push_back(p.grad());
    }
    if (active.empty()) return;
    torch::NoGradGuard no_grad;
    at::_foreach_add_(active, grads, -w->lr);  /* p -= lr * g */
}

void optimizer_step(OptimizerHandle h) {
    double t0 = _wall_ms_torch();
    auto* w = static_cast<OptWrapper*>(h);
    auto* opt = w->opt;
    /* Re-sync param list from registry (handles late registration via autoName).
     * For group-scoped optimizers, only sync params whose name starts with w->prefix. */
    auto& param_groups = opt->param_groups();
    if (!param_groups.empty()) {
        auto& params_ref = param_groups[0].params();
        auto current = collect_param_tensors_filtered(w->prefix);
        if (params_ref.size() != current.size()) {
            params_ref.clear();
            for (auto& t : current) params_ref.push_back(t);
        }
        /* Fused multi-tensor foreach paths for the full optimizer family:
           SGD (type=0), RMSprop (type=1), Adam (type=2), AdamW (type=3). */
        double tm0 = _wall_ms_torch();
        /* TORCH_FOREACH=0 disables every fused multi-tensor path for A/B
           perf comparison. Defaults to on. */
        static const bool foreach_enabled = []() {
            const char* e = std::getenv("TORCH_FOREACH");
            return !(e && (e[0] == '0'));
        }();
        if (foreach_enabled) {
            switch (w->type) {
                case 0: sgd_step_foreach(w, params_ref); break;
                case 1: rmsprop_step_foreach(w, params_ref); break;
                case 2: adam_step_foreach(w, params_ref); break;
                case 3: adamw_step_foreach(w, params_ref); break;
                default: opt->step();
            }
        } else {
            opt->step();
        }
        prof_optimizer_math_ms += _wall_ms_torch() - tm0;
    } else {
        double tm0 = _wall_ms_torch();
        opt->step();
        prof_optimizer_math_ms += _wall_ms_torch() - tm0;
    }
    /* Phase 1.5e: dump h0/c0 trajectory if enabled */
    {
        extern void _dbg_dump_lstm_traj_if_enabled_torch(void);
        _dbg_dump_lstm_traj_if_enabled_torch();
    }
    // Free intermediate tensors from this epoch's forward/backward
    free_intermediates();
    prof_optimizer_ms += _wall_ms_torch() - t0;
    prof_epochs++;
}

void optimizer_zero_grad(OptimizerHandle h) {
    static_cast<OptWrapper*>(h)->opt->zero_grad();
}

void optimizer_set_param_lr(OptimizerHandle h, const char* name, double lr) {
    /* TODO: libtorch uses native param groups — per-param LR overrides
       would require rebuilding groups. Not yet implemented. */
    (void)h; (void)name; (void)lr;
}

void optimizer_set_lr(OptimizerHandle h, double lr) {
    auto* w = static_cast<OptWrapper*>(h);
    w->lr = lr;
    /* Update the LR on each param group's options. The typed options
       (SGDOptions / RMSpropOptions / AdamOptions / AdamWOptions) all
       provide an lr() setter. Dispatch by w->type so we cast to the
       right derived type. */
    for (auto& g : w->opt->param_groups()) {
        switch (w->type) {
            case 0:
                static_cast<torch::optim::SGDOptions&>(g.options()).lr(lr);
                break;
            case 1:
                static_cast<torch::optim::RMSpropOptions&>(g.options()).lr(lr);
                break;
            case 2:
                static_cast<torch::optim::AdamOptions&>(g.options()).lr(lr);
                break;
            case 3:
                static_cast<torch::optim::AdamWOptions&>(g.options()).lr(lr);
                break;
        }
    }
}

static void clip_grad_value_filtered(const std::string& prefix, double max_val) {
    auto params = collect_param_tensors_filtered(prefix);
    torch::nn::utils::clip_grad_value_(params, max_val);
}

static double clip_grad_norm_filtered(const std::string& prefix, double max_norm) {
    auto params = collect_param_tensors_filtered(prefix);
    return torch::nn::utils::clip_grad_norm_(params, max_norm);
}

void optimizer_clip_grad_value(double max_val) {
    clip_grad_value_filtered("", max_val);
}

double optimizer_clip_grad_norm(double max_norm) {
    return clip_grad_norm_filtered("", max_norm);
}

/* Polyak soft update: mirror of the tape-backend implementation. */
int polyak_blend(double tau, const char* online_scope, const char* target_scope) {
    if (!online_scope || !target_scope) return 0;
    std::string on_s(online_scope), tg_s(target_scope);
    int blended = 0;
    torch::NoGradGuard no_grad;
    for (int i = 0; i < param_count(); i++) {
        std::string on_name(param_name(i));
        if (on_name.rfind(on_s, 0) != 0) continue;
        std::string tgt_name = tg_s + on_name.substr(on_s.size());
        for (int j = 0; j < param_count(); j++) {
            if (std::string(param_name(j)) != tgt_name) continue;
            at::Tensor& on_t = *(at::Tensor*)param_tensor(i);
            at::Tensor& tg_t = *(at::Tensor*)param_tensor(j);
            if (!on_t.sizes().equals(tg_t.sizes())) break;
            tg_t.mul_(1.0 - tau).add_(on_t, tau);
            blended++;
            break;
        }
    }
    return blended;
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
    return (int)param_count();
}

void optimizer_get_m(OptimizerHandle h, int idx, double* out) {
    auto* w = static_cast<OptWrapper*>(h);
    int numel = (int)((at::Tensor*)param_tensor(idx))->numel();
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
        buf = rms.momentum_buffer().defined() ? rms.momentum_buffer() : at::zeros_like(*(at::Tensor*)param_tensor(idx));
    } else {
        memset(out, 0, numel * sizeof(double));
        return;
    }
    buf = buf.cpu().contiguous().to(torch::kFloat64);
    memcpy(out, buf.data_ptr<double>(), numel * sizeof(double));
}

void optimizer_get_v(OptimizerHandle h, int idx, double* out) {
    auto* w = static_cast<OptWrapper*>(h);
    int numel = (int)((at::Tensor*)param_tensor(idx))->numel();
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
    buf = buf.cpu().contiguous().to(torch::kFloat64);
    memcpy(out, buf.data_ptr<double>(), numel * sizeof(double));
}

void optimizer_set_m(OptimizerHandle h, int idx, const double* data) {
    auto* w = static_cast<OptWrapper*>(h);
    auto* param_t = (at::Tensor*)param_tensor(idx);
    int numel = (int)param_t->numel();
    auto key = param_state_key(w->opt, idx);
    if (!key) return;
    auto tensor = torch::from_blob((void*)data, {(int64_t)numel}, torch::kFloat64).clone();
    tensor = tensor.reshape(param_t->sizes());
    /* Ensure state entry exists, stamping the restored step on creation. */
    if (w->opt->state().count(key) == 0) {
        if (w->type == 2) {
            auto st = std::make_unique<torch::optim::AdamParamState>();
            st->step(w->pending_step);
            w->opt->state()[key] = std::move(st);
        } else if (w->type == 1) {
            auto st = std::make_unique<torch::optim::RMSpropParamState>();
            st->step(w->pending_step);
            w->opt->state()[key] = std::move(st);
        } else return;
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
    auto* param_t = (at::Tensor*)param_tensor(idx);
    int numel = (int)param_t->numel();
    auto key = param_state_key(w->opt, idx);
    if (!key) return;
    auto tensor = torch::from_blob((void*)data, {(int64_t)numel}, torch::kFloat64).clone();
    tensor = tensor.reshape(param_t->sizes());
    if (w->opt->state().count(key) == 0) {
        if (w->type == 2) {
            auto st = std::make_unique<torch::optim::AdamParamState>();
            st->step(w->pending_step);
            w->opt->state()[key] = std::move(st);
        } else if (w->type == 1) {
            auto st = std::make_unique<torch::optim::RMSpropParamState>();
            st->step(w->pending_step);
            w->opt->state()[key] = std::move(st);
        } else return;
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
    /* Step count: set on all existing param states, and stash for any
       states created later (optimizer_set_m/_v during load run after this). */
    int64_t step = (int64_t)in9[8];
    w->pending_step = step;
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

/* ---------- System ---------- */

/* get_rss_mb / get_current_rss_mb live in shared_utils.c (compiled
 * once, unified symbol). Both are in the rename header's EXCLUDE
 * set so internal callers below resolve to the shared definitions. */

/* ---------- Backend Info ---------- */

const char* backend_name(void) { return "torch"; }

/* backend_memory_report removed (no Idris-side callers). */

void backend_reset_for_eval(void) {
    free_intermediates();
    for (int i_ = 0; i_ < param_count(); i_++) {
        auto* tensor = (at::Tensor*)param_tensor(i_);
        if (tensor->grad().defined())
            tensor->grad().zero_();
    }
}

void backend_epoch_begin(void) { /* no-op for torch: profiling is backward+optimizer only */ }

void backend_profile_reset(void) {
    prof_backward_ms = prof_optimizer_ms = prof_optimizer_math_ms = 0;
    prof_epochs = 0;
}

void backend_profile_report(void) {
    fprintf(stderr, "=== Profile Report (torch backend) ===\n");
    fprintf(stderr, "  Epochs: %d\n", prof_epochs);
    fprintf(stderr, "  Params: %d tensors\n", (int)param_count());
    fprintf(stderr, "  Backward:  %.1fms total (%.1fms/epoch)\n",
            prof_backward_ms, prof_epochs > 0 ? prof_backward_ms / prof_epochs : 0);
    fprintf(stderr, "  Optimizer: %.1fms total (%.1fms/epoch)\n",
            prof_optimizer_ms, prof_epochs > 0 ? prof_optimizer_ms / prof_epochs : 0);
    fprintf(stderr, "    of which math: %.1fms total (%.2fms/epoch)\n",
            prof_optimizer_math_ms, prof_epochs > 0 ? prof_optimizer_math_ms / prof_epochs : 0);
    double total = prof_backward_ms + prof_optimizer_ms;
    fprintf(stderr, "  C total:   %.1fms total (%.1fms/epoch)\n",
            total, prof_epochs > 0 ? total / prof_epochs : 0);
}

/* param_grad_item_at lives in shared/training/param_registry.c. */

/* ---------- Debug ---------- */

void tensor_print(TensorHandle h) {
    // std::cout << at::Tensor requires the tensor to live on CPU.
    std::cout << to_tensor(h)->cpu() << std::endl;
}

/* Job 3 Phase B — mx::compile is mlx-only; torch backend always reports
   disabled regardless of MLX_COMPILE env var. */
int  tensor_mlx_compile_enabled(void) { return 0; }
int  tensor_mlx_compile_invocations(void) { return 0; }
void tensor_mlx_compile_reset_stats(void) { }

/* ---------- Portable FFI helpers ---------- */

/* The 10 *_return helpers (tensor_backward_return, param_register_return,
   param_zero_all_grads_return, tensor_to_doubles_return,
   tensor_backward_conditional, tensor_backward_return_loss, idrisml_seq,
   backend_reset_for_eval_return, backend_profile_reset_return,
   backend_profile_report_return) live in shared/training/ffi_shims.c,
   compiled once with rename_torch.h to produce torch-suffixed symbols.
   Each shim only calls into FFI-exported functions declared in backend.h
   (every backend supplies its own implementation under those names), so
   no port methods are needed.

   native_train_step + optimizer_step_with_clip stay torch-local: they
   reach into OptWrapper internals (the prefix-scoped clip variants) that
   the shared trampolines don't carry. They'll lift when torch opts into
   the shared optimizer surface. */

double native_train_step(OptimizerHandle opt, int clip_mode, double clip_val,
                         TensorHandle loss_ptr, double loss_val) {
    auto* w = static_cast<OptWrapper*>(opt);
    optimizer_zero_grad(opt);
    if (tensor_requires_grad(loss_ptr)) tensor_backward(loss_ptr);
    /* Scope grad-clipping to this optimizer's owned params (matches tape backend). */
    if (clip_mode == 1) clip_grad_value_filtered(w->prefix, clip_val);
    else if (clip_mode == 2) clip_grad_norm_filtered(w->prefix, clip_val);
    optimizer_step(opt);
    return loss_val;
}
int optimizer_step_with_clip(OptimizerHandle opt, int clip_mode, double clip_val, int dummy) {
    (void)dummy;
    auto* w = static_cast<OptWrapper*>(opt);
    if (clip_mode == 1) clip_grad_value_filtered(w->prefix, clip_val);
    else if (clip_mode == 2) clip_grad_norm_filtered(w->prefix, clip_val);
    optimizer_step(opt); optimizer_zero_grad(opt);
    return 0;
}
int tensor_live_count(int dummy) { (void)dummy; return (int)intermediates.size(); }
int tensor_peak_live_count(int dummy) { (void)dummy; return (int)g_torch_peak_live; }
/* dropout_random_seed lives in shared_utils.c. */



/* ---- Inference-only dtype scaffolding (BF16, F16, Int, Bool) ----
   Generic dtype-parameterised create/cast over the lean non-grad set
   (scalar/create/1d/2d/cast). requires_grad is honored only for floating
   dtypes; torch rejects autograd on integer/bool. Grad param/state
   variants for these dtypes are deferred to the mixed-precision-training
   row; the f32/f64 param/state wrappers above are unchanged. */
static inline bool idrisml_is_floating_st(torch::ScalarType dt) {
    return dt == torch::kFloat32 || dt == torch::kFloat64 ||
           dt == torch::kBFloat16 || dt == torch::kHalf;
}
static TensorHandle create_scalar_dt(double v, int rg, torch::ScalarType dt) {
    auto t = torch::tensor(v, torch::dtype(dt));
    if (rg && idrisml_is_floating_st(dt)) t.requires_grad_(true);
    return from_tensor_persistent(std::move(t));
}
static TensorHandle create_nd_dt(double* data, int* shape, int rank, int rg, torch::ScalarType dt) {
    std::vector<int64_t> dims(rank);
    for (int i = 0; i < rank; i++) dims[i] = shape[i];
    auto t = torch::from_blob(data, dims, torch::kFloat64).clone();
    if (dt != torch::kFloat64) t = t.to(dt);
    if (rg && idrisml_is_floating_st(dt)) t.requires_grad_(true);
    return from_tensor_persistent(std::move(t));
}
static TensorHandle create_1d_dt(int n, double* d, int rg, torch::ScalarType dt) {
    auto t = torch::from_blob(d, {(int64_t)n}, torch::kFloat64).clone();
    free(d);
    if (dt != torch::kFloat64) t = t.to(dt);
    if (rg && idrisml_is_floating_st(dt)) t.requires_grad_(true);
    return from_tensor(std::move(t));
}
static TensorHandle create_2d_dt(int rows, int cols, double* d, int rg, torch::ScalarType dt) {
    auto t = torch::from_blob(d, {(int64_t)rows, (int64_t)cols}, torch::kFloat64).clone();
    free(d);
    if (dt != torch::kFloat64) t = t.to(dt);
    if (rg && idrisml_is_floating_st(dt)) t.requires_grad_(true);
    return from_tensor(std::move(t));
}


/* ---- Unified dtag-dispatch create/cast entry points ----
   One symbol per shape, dtag-keyed, superseding the per-dtype
   *_f32_streamed / *_f64_streamed / inference-dtype wrappers above. The
   Idris-side Scheme wrappers pass the RuntimeDType tag as the trailing
   `dtag` arg; the body switches internally. F32/F64 route to the existing
   dedicated creators (byte-identical behavior); other dtags route through
   the generic create_*_dt / make_param_leaf path. */
static torch::ScalarType st_for_dtag(int dtag) {
    switch (dtag) {
        case 1:  return torch::kBool;       /* Bool */
        case 4:  return torch::kByte;       /* U8 */
        case 8:  return torch::kChar;       /* I8 */
        case 9:  return torch::kShort;      /* I16 */
        case 10: return torch::kInt;        /* I32 */
        case 11: return torch::kLong;       /* I64 */
        case 13: return torch::kHalf;       /* F16 */
        case 14: return torch::kFloat32;    /* F32 */
        case 15: return torch::kFloat64;    /* F64 */
        case 17: return torch::kBFloat16;   /* BF16 */
        default:
            std::fprintf(stderr,
                "invalid dtag %d: expected one of {1=Bool, 4=U8, 8-11=I8/I16/I32/I64, "
                "13-15=F16/F32/F64, 17=BF16}\n", dtag);
            std::abort();
    }
}

/* Per-shape dtag dispatchers bound into the port struct below. Renamed
   from `tensor_create_*_streamed` (FFI-named) — the shared
   shared/training/dtype_streamed.c TU now provides the FFI surface
   via port trampolines that strip the (mlx-only) stream_tag and call
   into these. F32/F64 route to the dedicated dtype creators
   (byte-identical with the previous in-file path); other dtags route
   through the generic create_*_dt / make_param_leaf / torch_cast_to
   path. */
static TensorHandle torch_create_scalar_dtag(double v, int rg, int dtag) {
    switch (dtag) {
        case 14: return tensor_create_scalar_f32(v, rg);
        case 15: return tensor_create_scalar_f64(v, rg);
        default: return create_scalar_dt(v, rg, st_for_dtag(dtag));
    }
}
static TensorHandle torch_create_dtag(double* data, int* shape, int rank, int rg, int dtag) {
    switch (dtag) {
        case 14: return tensor_create_f32(data, shape, rank, rg);
        case 15: return tensor_create_f64(data, shape, rank, rg);
        default: return create_nd_dt(data, shape, rank, rg, st_for_dtag(dtag));
    }
}
static TensorHandle torch_create_1d_dtag(int n, double* data, int rg, int dtag) {
    switch (dtag) {
        case 14: return tensor_create_1d_f32(n, data, rg);
        case 15: return tensor_create_1d_f64(n, data, rg);
        default: return create_1d_dt(n, data, rg, st_for_dtag(dtag));
    }
}
static TensorHandle torch_create_2d_dtag(int rows, int cols, double* data, int rg, int dtag) {
    switch (dtag) {
        case 14: return tensor_create_2d_f32(rows, cols, data, rg);
        case 15: return tensor_create_2d_f64(rows, cols, data, rg);
        default: return create_2d_dt(rows, cols, data, rg, st_for_dtag(dtag));
    }
}
static TensorHandle torch_create_param_1d_dtag(int n, double* data, int dtag) {
    switch (dtag) {
        case 14: return tensor_create_param_1d_f32(n, data);
        case 15: return tensor_create_param_1d_f64(n, data);
        default: return make_param_leaf(data, {(int64_t)n}, st_for_dtag(dtag));
    }
}
static TensorHandle torch_create_param_2d_dtag(int rows, int cols, double* data, int dtag) {
    switch (dtag) {
        case 14: return tensor_create_param_2d_f32(rows, cols, data);
        case 15: return tensor_create_param_2d_f64(rows, cols, data);
        default: return make_param_leaf(data, {(int64_t)rows, (int64_t)cols}, st_for_dtag(dtag));
    }
}
static TensorHandle torch_create_param_3d_dtag(int d0, int d1, int d2, double* data, int dtag) {
    switch (dtag) {
        case 14: return tensor_create_param_3d_f32(d0, d1, d2, data);
        case 15: return tensor_create_param_3d_f64(d0, d1, d2, data);
        default: return make_param_leaf(data, {(int64_t)d0, (int64_t)d1, (int64_t)d2}, st_for_dtag(dtag));
    }
}
static TensorHandle torch_create_param_4d_dtag(int d0, int d1, int d2, int d3, double* data, int dtag) {
    switch (dtag) {
        case 14: return tensor_create_param_4d_f32(d0, d1, d2, d3, data);
        case 15: return tensor_create_param_4d_f64(d0, d1, d2, d3, data);
        default: return make_param_leaf(data, {(int64_t)d0, (int64_t)d1, (int64_t)d2, (int64_t)d3}, st_for_dtag(dtag));
    }
}
static TensorHandle torch_create_state_1d_dtag(int n, double* data, int dtag) {
    switch (dtag) {
        case 14: return tensor_create_state_1d_f32(n, data);
        case 15: return tensor_create_state_1d_f64(n, data);
        default: return torch_cast_to(tensor_create_state_1d(n, data), st_for_dtag(dtag));
    }
}
static TensorHandle torch_create_state_2d_dtag(int rows, int cols, double* data, int dtag) {
    switch (dtag) {
        case 14: return tensor_create_state_2d_f32(rows, cols, data);
        case 15: return tensor_create_state_2d_f64(rows, cols, data);
        default: return torch_cast_to(tensor_create_state_2d(rows, cols, data), st_for_dtag(dtag));
    }
}
static TensorHandle torch_cast_dtype_dtag(TensorHandle src, int dtag) {
    return from_tensor(to_tensor(src)->to(st_for_dtag(dtag)));
}

/* ---------- Shared training port adapter ----------
 *
 * Provides the per-tensor accessors that shared/training/param_registry.c
 * uses to talk to libtorch (numel / has_grad / grad read+write / zero /
 * data read+write / bulk doubles+int64 loaders). Other port slots stay
 * NULL until torch opts into the matching shared TUs (optimizer_*,
 * dtag-streamed creators, ffi_shims) — the shared trampolines never call
 * them because torch is excluded from those SHARED_BACKENDS_<tu> lists.
 *
 * Hot-path note: for F64/F32 contiguous CPU tensors, the element
 * accessors hit `data_ptr<>()` directly (one load); other dtype / device
 * combos route through the slow `.flatten().index({i}).cpu().item<>()`
 * path. Param storage is always contiguous + same device throughout a
 * run, so the fast path covers ~all live use. */

#include "shared/training/port.h"

/* Port-typed (void*) trampolines for the dtag creators. The internal
   torch_create_*_dtag helpers above return TensorHandle (which is
   void* at the C level — but C++ enforces the cast at the function-
   pointer-init site). */
static void* torch_port_create_scalar(double v, int rg, int dtag)                                 { return torch_create_scalar_dtag(v, rg, dtag); }
static void* torch_port_create(double* d, int* s, int r, int rg, int dtag)                        { return torch_create_dtag(d, s, r, rg, dtag); }
static void* torch_port_create_1d(int n, double* d, int rg, int dtag)                             { return torch_create_1d_dtag(n, d, rg, dtag); }
static void* torch_port_create_2d(int rows, int cols, double* d, int rg, int dtag)                { return torch_create_2d_dtag(rows, cols, d, rg, dtag); }
static void* torch_port_create_param_1d(int n, double* d, int dtag)                               { return torch_create_param_1d_dtag(n, d, dtag); }
static void* torch_port_create_param_2d(int rows, int cols, double* d, int dtag)                  { return torch_create_param_2d_dtag(rows, cols, d, dtag); }
static void* torch_port_create_param_3d(int d0, int d1, int d2, double* d, int dtag)              { return torch_create_param_3d_dtag(d0, d1, d2, d, dtag); }
static void* torch_port_create_param_4d(int d0, int d1, int d2, int d3, double* d, int dtag)      { return torch_create_param_4d_dtag(d0, d1, d2, d3, d, dtag); }
static void* torch_port_create_state_1d(int n, double* d, int dtag)                               { return torch_create_state_1d_dtag(n, d, dtag); }
static void* torch_port_create_state_2d(int rows, int cols, double* d, int dtag)                  { return torch_create_state_2d_dtag(rows, cols, d, dtag); }
static void* torch_port_cast_dtype(void* src, int dtag)                                            { return torch_cast_dtype_dtag((TensorHandle)src, dtag); }

static int torch_port_tensor_numel(void* h) {
    return (int)to_tensor(h)->numel();
}

static int torch_port_tensor_requires_grad(void* h) {
    return to_tensor(h)->requires_grad() ? 1 : 0;
}

static int torch_port_tensor_has_grad(void* h) {
    return to_tensor(h)->mutable_grad().defined() ? 1 : 0;
}

static double torch_port_grad_read(void* h, int i) {
    auto* t = to_tensor(h);
    auto& g = t->mutable_grad();
    if (!g.defined()) return 0.0;
    if (g.is_cpu() && g.is_contiguous()) {
        if (g.dtype() == torch::kFloat64) return ((double*)g.data_ptr())[i];
        if (g.dtype() == torch::kFloat32) return (double)((float*)g.data_ptr())[i];
    }
    return g.flatten().index({i}).cpu().item<double>();
}

static void torch_port_grad_write(void* h, int i, double v) {
    auto* t = to_tensor(h);
    auto& g = t->mutable_grad();
    if (!g.defined()) return;
    if (g.is_cpu() && g.is_contiguous()) {
        if (g.dtype() == torch::kFloat64) { ((double*)g.data_ptr())[i] = v; return; }
        if (g.dtype() == torch::kFloat32) { ((float*)g.data_ptr())[i] = (float)v; return; }
    }
    g.flatten().index_put_({i}, v);
}

static void torch_port_zero_grad(void* h) {
    auto& g = to_tensor(h)->mutable_grad();
    if (g.defined()) g.zero_();
}

static double torch_port_data_read(void* h, int i) {
    auto* t = to_tensor(h);
    if (t->is_cpu() && t->is_contiguous()) {
        if (t->dtype() == torch::kFloat64) return ((double*)t->data_ptr())[i];
        if (t->dtype() == torch::kFloat32) return (double)((float*)t->data_ptr())[i];
    }
    return t->flatten().index({i}).cpu().item<double>();
}

static void torch_port_data_write(void* h, int i, double v) {
    auto* t = to_tensor(h);
    if (t->is_cpu() && t->is_contiguous()) {
        if (t->dtype() == torch::kFloat64) { ((double*)t->data_ptr())[i] = v; return; }
        if (t->dtype() == torch::kFloat32) { ((float*)t->data_ptr())[i] = (float)v; return; }
    }
    torch::NoGradGuard no_grad;
    t->flatten().index_put_({i}, v);
}

static void torch_port_load_doubles(void* h, const double* src, int n) {
    torch::NoGradGuard no_grad;
    auto* t = to_tensor(h);
    auto staging = torch::from_blob(const_cast<double*>(src), {(int64_t)n}, torch::kFloat64);
    t->view({n}).copy_(staging);
}

static void torch_port_load_int64(void* h, const int64_t* src, int n) {
    torch::NoGradGuard no_grad;
    auto* t = to_tensor(h);
    auto staging = torch::from_blob(const_cast<int64_t*>(src), {(int64_t)n}, torch::kInt64);
    t->view({n}).copy_(staging);
}

const BackendPort g_active_port = {
    /* Tensor introspection + per-element access + bulk grad/data ops:
       supplied by the torch port shims above. */
    .tensor_numel              = torch_port_tensor_numel,
    .tensor_requires_grad      = torch_port_tensor_requires_grad,
    .tensor_has_grad           = torch_port_tensor_has_grad,
    .data_read                 = torch_port_data_read,
    .data_write                = torch_port_data_write,
    .grad_read                 = torch_port_grad_read,
    .grad_write                = torch_port_grad_write,
    .zero_grad                 = torch_port_zero_grad,
    .load_doubles              = torch_port_load_doubles,
    .load_int64                = torch_port_load_int64,
    /* Slots whose shared TUs torch hasn't opted into yet — see the
       SHARED_BACKENDS_<tu> lists in the Makefile. These stay nullptr
       until torch's adapter ships their bindings AND the shared TU
       gets compiled for torch. Ordering matches port.h's struct
       declaration order (C++ ISO-required for designated init). */
    .backward                  = nullptr,
    .optimizer_create_sgd      = nullptr,
    .optimizer_create_rmsprop  = nullptr,
    .optimizer_create_adam     = nullptr,
    .optimizer_create_adam_group = nullptr,
    .optimizer_create_adamw    = nullptr,
    .optimizer_free            = nullptr,
    .optimizer_set_lr          = nullptr,
    .optimizer_set_param_lr    = nullptr,
    .optimizer_step            = nullptr,
    .optimizer_buf_count       = nullptr,
    .optimizer_get_m           = nullptr,
    .optimizer_get_v           = nullptr,
    .optimizer_set_m           = nullptr,
    .optimizer_set_v           = nullptr,
    .optimizer_get_meta        = nullptr,
    .optimizer_set_meta        = nullptr,
    .wall_ms                   = nullptr,
    /* Dtag-streamed creators: torch supplies its libtorch-backed
       dtag dispatchers via the torch_port_create_* shims above. */
    .create_scalar             = torch_port_create_scalar,
    .create                    = torch_port_create,
    .create_1d                 = torch_port_create_1d,
    .create_2d                 = torch_port_create_2d,
    .create_param_1d           = torch_port_create_param_1d,
    .create_param_2d           = torch_port_create_param_2d,
    .create_param_3d           = torch_port_create_param_3d,
    .create_param_4d           = torch_port_create_param_4d,
    .create_state_1d           = torch_port_create_state_1d,
    .create_state_2d           = torch_port_create_state_2d,
    .cast_dtype                = torch_port_cast_dtype,
};
