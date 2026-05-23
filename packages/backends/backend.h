#ifndef IDRISML_BACKEND_H
#define IDRISML_BACKEND_H

#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

/* Opaque tensor handle — wraps backend-specific tensor (e.g. at::Tensor*) */
typedef void* TensorHandle;

/* Pair of tensor handles (for functions returning two results) */
typedef struct { TensorHandle first; TensorHandle second; } TensorPair;

/* ---------- Lifecycle ---------- */

/* Per-dtype creation primitives. Each backend defines the dtype variants
 * it actually supports — tape only implements _f64 (no fp32 arena), torch
 * and mlx implement both. The _f32 stubs on tape exist for link-time
 * symbol completeness only; calling them aborts. Idris-side dispatch goes
 * through RuntimeDType typeclass instances (Phase 4): the `RuntimeDType F32`
 * instance is intentionally missing for tape, so F32-typed code can't
 * compile against a tape-primary build. */
TensorHandle tensor_create_scalar_f32(double value, int requires_grad);
TensorHandle tensor_create_scalar_f64(double value, int requires_grad);
TensorHandle tensor_create_f32(double* data, int* shape, int rank, int requires_grad);
TensorHandle tensor_create_f64(double* data, int* shape, int rank, int requires_grad);

/* Per-dtype cast primitives. The destination dtype is in the symbol
 * name (matching the per-dtype create primitives); the source dtype
 * is read from the handle on the C side. Returns a fresh handle whose
 * cast op (where applicable) participates in autograd. Tape's _f32
 * variant aborts (no fp32 arena); tape's _f64 variant handles F64 -> F64
 * as an identity alias. mlx and torch implement both directions. */
TensorHandle tensor_cast_dtype_f32(TensorHandle src);
TensorHandle tensor_cast_dtype_f64(TensorHandle src);

/* Legacy unsuffixed creation primitives. Currently route to whichever
 * dtype the backend has historically used (mlx → _f32, tape/torch → _f64).
 * Kept while Idris-side call sites migrate to RuntimeDType dispatch
 * (Phase 4-5). Removed once migration completes. */
TensorHandle tensor_create_scalar(double value, int requires_grad);
TensorHandle tensor_create(double* data, int* shape, int rank, int requires_grad);
TensorHandle tensor_clone(TensorHandle t);
void         tensor_free(TensorHandle t);

/* Refcount-driven lifecycle (see
 * docs/develop/tensor-lifecycle.md). retain_handle bumps a Tensor's
 * refcount; release_handle decrements and frees when refcount reaches
 * zero. Called by tape capture, param_registry, and Idris-side managed
 * handle finalizers. */
void         tensor_retain_handle(TensorHandle t);
void         tensor_release_handle(TensorHandle t);
/* ---------- Accessors ---------- */

double tensor_item(TensorHandle t);          /* scalar tensor -> double */
int    tensor_numel(TensorHandle t);
int    tensor_dim(TensorHandle t);
int    tensor_size(TensorHandle t, int dim);
void   tensor_to_doubles(TensorHandle t, double* out); /* flatten to buffer */
void   tensor_to_floats(TensorHandle t, float* out);   /* flatten to f32 buffer */
const char* tensor_dtype_name(TensorHandle t); /* "F32" | "F64" — SafeTensors-compatible string */

/* ---------- Arithmetic (element-wise, return new tensor) ---------- */

TensorHandle tensor_add(TensorHandle a, TensorHandle b);
TensorHandle tensor_sub(TensorHandle a, TensorHandle b);
TensorHandle tensor_mul(TensorHandle a, TensorHandle b);
TensorHandle tensor_div(TensorHandle a, TensorHandle b);
TensorHandle tensor_neg(TensorHandle t);
TensorHandle tensor_abs(TensorHandle t);
TensorHandle tensor_exp(TensorHandle t);
TensorHandle tensor_log(TensorHandle t);
TensorHandle tensor_sqrt(TensorHandle t);
TensorHandle tensor_pow(TensorHandle base, TensorHandle exp);
TensorHandle tensor_sigmoid(TensorHandle t);
TensorHandle tensor_tanh(TensorHandle t);
TensorHandle tensor_gelu(TensorHandle t);   /* GELU activation (tanh approx) */
TensorHandle tensor_leaky_relu(TensorHandle t, double alpha);  /* max(alpha*x, x) */
TensorHandle tensor_silu(TensorHandle t);   /* x * sigmoid(x) (Swish) */
TensorHandle tensor_softplus(TensorHandle t);  /* log(1 + exp(x)), backward = sigmoid(x) */

/* Scalar broadcast variants */
TensorHandle tensor_add_scalar(TensorHandle t, double s);
TensorHandle tensor_mul_scalar(TensorHandle t, double s);

/* Clamp: element-wise max(t, min_val). Returns new tensor. */
TensorHandle tensor_clamp_min(TensorHandle t, double min_val);

/* ---------- Reduction ---------- */

TensorHandle tensor_sum(TensorHandle t);
TensorHandle tensor_sum_dim(TensorHandle t, int dim, int keepdim);
TensorHandle tensor_mean(TensorHandle t);
TensorHandle tensor_min(TensorHandle t);     /* scalar reduction: min of all elements */
TensorHandle tensor_max(TensorHandle t);     /* scalar reduction: max of all elements */

/* ---------- Linear algebra ---------- */

TensorHandle tensor_matmul(TensorHandle a, TensorHandle b);     /* general matmul */
TensorHandle tensor_mv(TensorHandle mat, TensorHandle vec);     /* matrix-vector */
TensorHandle tensor_linear(TensorHandle W, TensorHandle x, TensorHandle bias); /* y = Wx + b (fused) */
TensorHandle tensor_linear_2d(TensorHandle W, TensorHandle X, TensorHandle bias);
/* W: [o,i], X: [B,i], bias: [o] -> Y: [B,o] = X @ W^T + broadcast(bias) */
TensorHandle tensor_concat_2d_axis1(TensorHandle A, TensorHandle B);
/* A: [m, n], B: [m, k] -> [m, n+k]. Concat along the last axis. */
TensorHandle tensor_dot(TensorHandle a, TensorHandle b);        /* vector dot */
TensorHandle tensor_outer(TensorHandle a, TensorHandle b);      /* outer product */

/* ---------- Activation / normalization ---------- */

TensorHandle tensor_softmax(TensorHandle t, int dim);
TensorHandle tensor_log_softmax(TensorHandle t, int dim);

/* ---------- Loss ---------- */

TensorHandle tensor_bce_with_logits(TensorHandle input, TensorHandle target);
TensorHandle tensor_cross_entropy(TensorHandle input, TensorHandle target);
TensorHandle tensor_mse_loss(TensorHandle input, TensorHandle target);

/* ---------- Batch Normalization ---------- */

/* Per-channel normalization on [C, *] tensors (instance norm when batch=1).
   input: [C, D] or flat [C*D]. gamma/beta: [C] learnable.
   running_mean/running_var: [C] state (updated in-place when training=1).
   Returns normalized, scaled, shifted tensor (same shape as input). */
TensorHandle tensor_batch_norm(TensorHandle input, TensorHandle gamma, TensorHandle beta,
                               TensorHandle running_mean, TensorHandle running_var,
                               int channels, int spatial, int training,
                               double momentum, double eps);

/* Group normalization: input [C * spatial], gamma/beta [C].
   Normalizes within each group of channels. */
TensorHandle tensor_group_norm(TensorHandle input, TensorHandle gamma, TensorHandle beta,
                               int numGroups, int channels, int spatial, double eps);

/* ---------- Dropout ---------- */

/* Inverted dropout: zero with probability p, scale survivors by 1/(1-p).
   When training=0, returns input unchanged.
   seed controls the random mask (deterministic per call). */
TensorHandle tensor_dropout(TensorHandle input, double p, int training, unsigned int seed);

/* ---------- Convolution & Pooling ---------- */

/* Conv1D: input [inC, L], kernel [outC, inC/groups, kL], bias [outC] or NULL.
   Returns [outC, oL] where oL = (L + 2*pad - kL) / stride + 1. */
TensorHandle tensor_conv1d(TensorHandle input, TensorHandle kernel,
                           TensorHandle bias, int pad, int stride);
/* Grouped Conv1D: same as conv1d but with groups parameter. */
TensorHandle tensor_conv1d_grouped(TensorHandle input, TensorHandle kernel,
                                   TensorHandle bias, int pad, int stride, int groups);

/* MaxPool1D: input [C, L]. Returns [C, oL] where oL = (L - kL) / stride + 1. */
TensorHandle tensor_max_pool1d(TensorHandle input, int kL, int stride);

/* Per-dtype variants; F64 is the legacy default. */
TensorHandle tensor_create_param_3d_f32(int d0, int d1, int d2, double* data);
TensorHandle tensor_create_param_3d_f64(int d0, int d1, int d2, double* data);

/* Conv2D: input [inC, H, W], kernel [outC, inC, kH, kW], bias [outC] or NULL.
   Returns [outC, oH, oW] where oH = (H + 2*padH - kH) / strideH + 1. */
TensorHandle tensor_conv2d(TensorHandle input, TensorHandle kernel,
                           TensorHandle bias, int padH, int padW,
                           int strideH, int strideW);
/* Batched Conv2D: input [B, inC, H, W], kernel [outC, inC, kH, kW], bias [outC] or NULL.
   Returns [B, outC, oH, oW]. The training-loop fast path — one tape entry per
   batched op vs B per-sample ones, single libtorch / mlx native batched call. */
TensorHandle tensor_conv2d_batched(TensorHandle input, TensorHandle kernel,
                                    TensorHandle bias, int padH, int padW,
                                    int strideH, int strideW);
/* Grouped Conv2D: same as conv2d but with groups parameter.
   kernel shape: [outC, inC/groups, kH, kW]. groups=inC for depthwise. */
TensorHandle tensor_conv2d_grouped(TensorHandle input, TensorHandle kernel,
                                   TensorHandle bias, int padH, int padW,
                                   int strideH, int strideW, int groups);

/* ConvTranspose1D: input [inC, L], kernel [inC, outC, kL], bias [outC] or NULL.
   Returns [outC, oL] where oL = (L-1)*stride - 2*pad + kL. */
TensorHandle tensor_conv_transpose1d(TensorHandle input, TensorHandle kernel,
                                     TensorHandle bias, int pad, int stride);

/* ConvTranspose2D: input [inC, H, W], kernel [inC, outC, kH, kW], bias [outC] or NULL.
   Returns [outC, oH, oW]. */
TensorHandle tensor_conv_transpose2d(TensorHandle input, TensorHandle kernel,
                                     TensorHandle bias, int padH, int padW,
                                     int strideH, int strideW);

/* AvgPool1D: input [C, L]. Returns [C, oL] where oL = (L - kL) / stride + 1. */
TensorHandle tensor_avg_pool1d(TensorHandle input, int kL, int stride);

/* AvgPool2D: input [C, H, W]. Returns [C, oH, oW]. */
TensorHandle tensor_avg_pool2d(TensorHandle input, int kH, int kW, int strideH, int strideW);

/* MaxPool2D: input [C, H, W].
   Returns [C, oH, oW] where oH = (H - kH) / strideH + 1. */
TensorHandle tensor_max_pool2d(TensorHandle input, int kH, int kW,
                               int strideH, int strideW);
/* Batched MaxPool2D: input [B, C, H, W]. Returns [B, C, oH, oW]. */
TensorHandle tensor_max_pool2d_batched(TensorHandle input, int kH, int kW,
                                        int strideH, int strideW);

/* ---------- NTM-specific compositions ---------- */

TensorHandle tensor_cosine_similarity(TensorHandle a, TensorHandle b, int dim);
TensorHandle tensor_conv1d_circular(TensorHandle input, TensorHandle kernel);

/* ---------- Shape manipulation ---------- */

TensorHandle tensor_reshape(TensorHandle t, int* shape, int rank);
TensorHandle tensor_reshape_1d(TensorHandle t, int n);
TensorHandle tensor_unsqueeze(TensorHandle t, int dim);
TensorHandle tensor_squeeze(TensorHandle t, int dim);
TensorHandle tensor_select(TensorHandle t, int dim, int index);
TensorHandle tensor_stack(TensorHandle* tensors, int count, int dim);
TensorHandle tensor_cat(TensorHandle* tensors, int count, int dim);
TensorHandle tensor_cat2(TensorHandle a, TensorHandle b);
TensorHandle tensor_narrow(TensorHandle t, int dim, int start, int len);
TensorHandle tensor_mm(TensorHandle a, TensorHandle b);
TensorHandle tensor_bmm(TensorHandle a, TensorHandle b);   /* [B,m,n] x [n,k] -> [B,m,k] */
TensorHandle tensor_batch(TensorHandle* handles, int count); /* B × [m,n] -> [B,m,n] */
TensorHandle* tensor_unbatch(TensorHandle h, int* out_count); /* [B,...] -> B × [...] */
TensorHandle tensor_transpose_2d(TensorHandle t);
TensorHandle tensor_softmax_2d(TensorHandle t);
TensorHandle tensor_masked_fill(TensorHandle t, TensorHandle mask, double value);
TensorHandle tensor_log_softmax_2d(TensorHandle t);
TensorHandle tensor_layer_norm_2d(TensorHandle input, TensorHandle gamma,
                                   TensorHandle bias, double eps);
TensorHandle tensor_reshape_2d(TensorHandle t, int rows, int cols);

/* 3D batched attention ops */
TensorHandle tensor_bmm_3x3(TensorHandle a, TensorHandle b);  /* [B,m,n] x [B,n,k] -> [B,m,k] */
TensorHandle tensor_softmax_3d(TensorHandle t);                /* softmax along last dim */
TensorHandle tensor_transpose_last2(TensorHandle t);           /* [B,m,n] -> [B,n,m] */
TensorHandle tensor_reshape_3d(TensorHandle t, int d0, int d1, int d2);
TensorHandle tensor_reshape_4d(TensorHandle t, int d0, int d1, int d2, int d3);
TensorHandle tensor_expand_mask(TensorHandle mask, int B);     /* [m,n] -> [B,m,n] */
TensorHandle tensor_tile_2d(TensorHandle t, int rep0, int rep1); /* [m,n] -> [m*rep0, n*rep1] */

/* ---------- Autograd ---------- */

void         tensor_backward(TensorHandle loss);
TensorHandle tensor_grad(TensorHandle t);  /* returns .grad, may be NULL */
void         tensor_zero_grad(TensorHandle t);
int          tensor_requires_grad(TensorHandle t);
TensorHandle tensor_detach(TensorHandle t);
TensorHandle tensor_with_grad(TensorHandle t);  /* returns copy with requires_grad=true */
void         tensor_set_requires_grad(TensorHandle t, int requires_grad);

/* No-grad scope (for optimizer steps, inference) */
void         tensor_no_grad_begin(void);
void         tensor_no_grad_end(void);
/* Per-epoch generation-scoped free for grad-mode training: epoch_begin
 * marks the generation, epoch_end deletes wrap-only handles created since.
 * mlx implements the free; tape/torch are no-ops (no buffer ceiling). */
void         tensor_epoch_begin(void);
void         tensor_epoch_end(void);

/* ---------- Device ---------- */

TensorHandle tensor_to_device(TensorHandle t, const char* device);  /* "cpu", "mps", "cuda" */
const char*  tensor_device(TensorHandle t);

/* ---------- GRU ---------- */

/* GRU cell — nn.GRU equation. Takes the two [3*o] half-sums
     ih = W_ih @ x + b_ih, hh = W_hh @ h + b_hh
   and the previous hidden state. Computes:
     z = sigmoid(ih_z + hh_z), r = sigmoid(ih_r + hh_r)
     n = tanh(ih_n + r * hh_n)
     h' = (1 - z) * n + z * prev
   Returns new hidden state [o]. */
TensorHandle tensor_gru_cell(TensorHandle ih, TensorHandle hh,
                              TensorHandle prev_hidden, int o);

/* ---------- LSTM ---------- */

/* Returns (h', c') as two tensors via out pointers */
void tensor_lstm_cell(
    TensorHandle input, TensorHandle hx, TensorHandle cx,
    TensorHandle w_ih, TensorHandle w_hh,
    TensorHandle b_ih, TensorHandle b_hh,
    TensorHandle* out_h, TensorHandle* out_c);

/* Fused LSTM cell from pre-computed gate values.
   combined = mulIW + mulRW + bias  ([4*o] tensor)
   Returns (new_hidden, new_cell) via out pointers. */
void tensor_lstm_gates(
    TensorHandle combined, TensorHandle prev_cell, int o,
    TensorHandle* out_h, TensorHandle* out_c);

TensorPair* tensor_lstm_gates_pair(TensorHandle combined, TensorHandle prev_cell, int o);
TensorHandle tensor_pair_first(TensorPair* p);
TensorHandle tensor_pair_second(TensorPair* p);
void tensor_pair_free(TensorPair* p);

/* ---------- Quantization (BitNet b1.58 — #411) ---------- */

/* Build a Ternary tensor from a packed-2-bit byte buffer.
 *
 * `packed_bytes` carries values in {-1, 0, +1} encoded as 2-bit codes
 * via shared_utils.h `ternary_pack` (00=0, 01=+1, 11=-1). Layout is
 * row-major with each row packed independently: row j occupies
 * `((i + 3) / 4)` bytes starting at offset `j * ((i + 3) / 4)`.
 * Trailing bits in a row's final byte are padded to 0 and ignored on
 * unpack.
 *
 * Per-backend storage (see design-decisions.md "Per-backend ternary
 * storage"):
 *   - tape:   keeps the packed bytes verbatim (2 bits/value, sub-byte
 *             arena branch). Decoded on the inner loop of
 *             `tensor_bitlinear_fwd`.
 *   - torch / mlx: unpacks into int8 storage at construction time (8
 *             bits/value, framework-native dtype). The 4× memory hit
 *             is the cost of staying inside framework op dispatch +
 *             autograd.
 *
 * `packed_byte_count` is the length of the buffer; the call aborts
 * if it doesn't equal `((i + 3) / 4) * o`. */
TensorHandle tensor_create_ternary_packed_2d(
    const uint8_t* packed_bytes, int packed_byte_count,
    int o, int i, int requires_grad);

/* BitLinear inference forward: y = (W_ternary * scale.unsqueeze(1)) @ x + bias.
 *
 *   W:     [o, i] tagged Ternary (see tensor_create_ternary_packed_2d).
 *   scale: [o]    float (compute dtype = output dtype).
 *   x:     [i]    float (compute dtype).
 *   bias:  [o]    float, or NULL for no bias.
 *
 * Output is shape [o] in scale's dtype. NoGrad path (BitNet b1.58
 * weight is a frozen quantized param); bias gradient flow lands in a
 * follow-up. */
TensorHandle tensor_bitlinear_fwd(
    TensorHandle W, TensorHandle scale, TensorHandle x, TensorHandle bias);

/* ---------- Parameter Registry ---------- */

/* Register a named parameter for gradient collection after backward() */
void         param_register(const char* name, TensorHandle t);
void         param_clear(void);
int          param_count(void);
const char*  param_name(int idx);
double       param_grad_item(int idx);          /* read scalar grad for param i */
double       param_grad_item_at(int param_idx, int elem_idx); /* read grad element */
double       param_grad_item_and_zero(int idx); /* read grad, then zero it */
TensorHandle param_tensor(int idx);
void         param_zero_all_grads(void);
void         param_subtract_delta(int idx, double delta); /* in-place: param -= delta */

/* In-place scalar subtract on a tensor (under no_grad). Returns tensor for threading. */
TensorHandle tensor_subtract_scalar_inplace(TensorHandle t, double val);

/* ---------- Convenience: build tensors from scalar arrays ---------- */

/* Create a 1D tensor from n doubles passed as individual args via a C array */
/* dtype-aware: dtag selects the output dtype (see "Unified dtag-dispatch"
   block below for the kind-major layout; 1=Bool, 4=U8, 8-11=I8/I16/I32/I64,
   13-15=F16/F32/F64, 17=BF16) so the result honestly matches the Idris `dt`.
   tape stores all dtypes via the double lingua franca. */
TensorHandle tensor_one_hot(int* tokens, int n_tokens, int vocab_size, int dtag);
/* Per-dtype variants — see lifecycle block comment for dispatch rules. */
TensorHandle tensor_create_1d_f32(int n, double* data, int requires_grad);
TensorHandle tensor_create_1d_f64(int n, double* data, int requires_grad);
TensorHandle tensor_create_2d_f32(int rows, int cols, double* data, int requires_grad);
TensorHandle tensor_create_2d_f64(int rows, int cols, double* data, int requires_grad);
/* Legacy unsuffixed (alias). */
TensorHandle tensor_create_2d(int rows, int cols, double* data, int requires_grad);

/* Buffer helpers (tensor_alloc_doubles / tensor_read_double /
 * tensor_write_double_return / tensor_alloc_ints / tensor_free_ints /
 * tensor_write_int_return / tensor_ptr_array_alloc /
 * tensor_ptr_array_set_return) are backend-agnostic and live in
 * packages/backends/shared_utils.{c,h} with a single unified
 * definition. */

/* ---------- Tensor pointer array (for stack/cat from Idris) ---------- */

TensorHandle   tensor_stack_from_array(TensorHandle* arr, int count, int dim);
TensorHandle   tensor_cat_from_array(TensorHandle* arr, int count, int dim);

/* ---------- Tensor-level parameter creation ---------- */

/* Per-dtype variants */
TensorHandle tensor_create_param_1d_f32(int n, double* data);
TensorHandle tensor_create_param_1d_f64(int n, double* data);
/* Param creators: requires_grad=true. Per-dtype only; F64 is the
 * legacy default. */
TensorHandle tensor_create_param_2d_f32(int rows, int cols, double* data);
TensorHandle tensor_create_param_2d_f64(int rows, int cols, double* data);
TensorHandle tensor_create_param_1d_f32(int n, double* data);
TensorHandle tensor_create_param_1d_f64(int n, double* data);
/* Create state tensors WITHOUT requires_grad. Covers both init-time
 * permanent state (NTM mask, BatchNorm running stats, transformer PE,
 * DNC mask) and per-sequence transient state (Ntm/Dnc zeroState). mlx:
 * is_state=1 + refcount=0; lifecycle is driven by the Idris-side wrap
 * (alive while the model record / per-sequence binding references it).
 * tape/torch: no refcount surface; the backend's own arena / shared_ptr
 * handles freeing. Per-dtype only. */
TensorHandle tensor_create_state_1d_f32(int n, double* data);
TensorHandle tensor_create_state_1d_f64(int n, double* data);
TensorHandle tensor_create_state_2d_f32(int rows, int cols, double* data);
TensorHandle tensor_create_state_2d_f64(int rows, int cols, double* data);
/* Get a scalar view into element [row, col] of a 2D tensor (shares storage) */
TensorHandle tensor_view_2d(TensorHandle mat, int row, int col);
/* Get a scalar view into element [idx] of a 1D tensor (shares storage) */
TensorHandle tensor_view_1d(TensorHandle vec, int idx);
/* Read item from a 2D tensor at [row, col] without creating a new tensor */
double tensor_item_2d(TensorHandle mat, int row, int col);
/* Read item from a 1D tensor at [idx] */
double tensor_item_1d(TensorHandle vec, int idx);

/* ---------- Native Optimizer ---------- */

typedef void* OptimizerHandle;

OptimizerHandle optimizer_create_sgd(double lr);
OptimizerHandle optimizer_create_rmsprop(double lr, double alpha, double eps,
                                          double weight_decay, double momentum);
OptimizerHandle optimizer_create_adam(double lr, double beta1, double beta2, double eps);
/* Adam whose step() / clip_grad only touches params whose registry name starts
 * with `prefix`. Empty prefix behaves identically to optimizer_create_adam. */
OptimizerHandle optimizer_create_adam_group(double lr, double beta1, double beta2,
                                            double eps, const char* prefix);
OptimizerHandle optimizer_create_adamw(double lr, double beta1, double beta2, double eps,
                                       double weight_decay);
void            optimizer_free(OptimizerHandle opt);
void            optimizer_step(OptimizerHandle opt);
void            optimizer_zero_grad(OptimizerHandle opt);
void            optimizer_set_param_lr(OptimizerHandle opt, const char* name, double lr);
/* Set the optimizer's base (global) learning rate. Per-param overrides set
 * via optimizer_set_param_lr remain in effect; only params not overridden
 * pick up the new base lr. Used to apply LR schedules per epoch. */
void            optimizer_set_lr(OptimizerHandle opt, double lr);

/* Gradient clipping (operates on all registered params) */
void optimizer_clip_grad_value(double max_val);
double optimizer_clip_grad_norm(double max_norm);  /* returns actual norm */

/* Polyak soft update for twin-network setups (SAC target Q-nets).
 * For each registered param P whose name starts with `online_scope`,
 * finds the matching target param under `target_scope` and blends:
 *   target.data ← (1 − tau) · target.data + tau · online.data (in-place).
 * Returns the number of param pairs blended. */
int polyak_blend(double tau, const char* online_scope, const char* target_scope);

/* ---------- System ---------- */

int get_rss_mb(void);           /* peak RSS in MB (getrusage) */
int get_current_rss_mb(void);   /* current RSS in MB (macOS mach_task_info) */
/* Count of live backend tensor handles (mlx: all_tensors; torch:
 * intermediates; tape: tape entries). Per-backend so it tracks the
 * thing that actually grows. Takes an ignored arg to defeat Idris-Chez
 * constant-folding of the zero-arg FFI call. */
int tensor_live_count(int dummy);
/* High-water mark of live handles since process start (the figure that
 * actually determines whether a paravirt-Metal buffer ceiling is hit).
 * Ignored arg defeats Idris-Chez constant-folding. */
int tensor_peak_live_count(int dummy);
void backend_reset_for_eval(void); /* reset tape + arena for clean eval forward */

/* Backend-controlled teardown helper. Inference programs that complete
 * with thousands of live `at::Tensor*` / `mlx::array` allocations hit a
 * libtorch CPUAllocator / OS-cleanup tail of up to tens of minutes at
 * `main` exit (large-model CPU lanes only; GPU lanes async-release).
 * `backend_release_all_persistent` performs the explicit deletes inside
 * `main` where they can be timed, then resets the registry. Cheap on
 * tape (no per-tensor heap allocations beyond the arena); meaningful on
 * torch + mlx where it forces `~at::Tensor` / `~mx::array` cascades to
 * run at a controlled point rather than during process shutdown. Idris-
 * side callers (the HF inference examples) pair this with a final
 * `drainManagedHandles` + `forceMajorGc` for guardian-side bookkeeping. */
void backend_release_all_persistent(void);

/* ---------- Profiling ---------- */

void backend_profile_reset(void);
void backend_profile_report(void);
void backend_epoch_begin(void);  /* mark start of forward pass for timing */

/* TODO #393 op-submission diagnostic — count per-forward graph nodes
 * on torch (counts at::Tensor wraps in from_tensor()); no-op stubs on
 * tape and mlx. Use bracketed reset/read at example sites to extract
 * per-forward op counts without instrumenting every kernel wrapper. */
void tensor_perf_reset(void);
long tensor_perf_op_count(void);

/* TODO #399 Commit B — fused scaled-dot-product attention.
 * Replaces the Idris-side per-head attention math (matmul/scale/
 * mask/softmax/matmul) with a single fused libtorch op (on torch:
 * `at::scaled_dot_product_attention`, MPSGraph-fused on MPS) or
 * the analogous fast path on mlx; tape composes the existing
 * matmul/softmax kernels in one C call to save Idris-side FFI hops.
 *
 *   Q : [seq, numHeads   * headDim]
 *   K : [seq, numKvHeads * headDim]
 *   V : [seq, numKvHeads * headDim]
 *   out [seq, numHeads * headDim]
 *
 * GQA (numHeads != numKvHeads) is handled internally. Causal mask is
 * a flag (no mask tensor passed) since the SDPA kernel can construct
 * it directly. Caller's responsibility: Q and K must already have
 * RoPE applied. */
TensorHandle tensor_sdpa_2d(TensorHandle q, TensorHandle k, TensorHandle v,
                            int numHeads, int numKvHeads, int headDim,
                            int isCausal);

/* ---------- Portable FFI helpers (for RefC compatibility) ---------- */

/* These wrap void-returning functions to return an argument for value threading.
   Needed because RefC doesn't support inline Scheme lambdas. */
TensorHandle tensor_backward_return(TensorHandle t);  /* backward(t); return t */
TensorHandle param_register_return(const char* name, TensorHandle t); /* set_requires_grad + register; return t */
int          param_zero_all_grads_return(int dummy);  /* zero_all_grads(); return 0 */
/* tensor_write_double_return / tensor_ptr_array_set_return /
 * tensor_alloc_ints / tensor_free_ints / tensor_write_int_return
 * are unified across backends — declared in shared_utils.h. */
double*      tensor_to_doubles_return(TensorHandle h, double* buf); /* tensor_to_doubles + return buf */
int          tensor_backward_conditional(TensorHandle t); /* backward if requires_grad; return param_count */
double       tensor_backward_return_loss(TensorHandle loss_ptr, double loss_val); /* backward if rg; return loss_val */
double       native_train_step(OptimizerHandle opt, int clip_mode, double clip_val,
                               TensorHandle loss_ptr, double loss_val); /* zero+bwd+clip+step; return loss_val */
double       native_train_step_scaled(OptimizerHandle opt, int clip_mode, double clip_val,
                                      TensorHandle loss_ptr, double loss_val,
                                      double scale); /* GradScaler variant: scaled bwd, unscale grads, NaN if non-finite seen (skip step) */
int          optimizer_step_with_clip(OptimizerHandle opt, int clip_mode, double clip_val, int dummy); /* clip+step+zero; return 0 */
void*        idrisml_seq(void* a, void* b); /* evaluate a, return b */
int          backend_reset_for_eval_return(int dummy);
int          backend_profile_reset_return(int dummy);
int          backend_profile_report_return(int dummy);
/* dropout_random_seed is backend-agnostic and lives in shared_utils.h. */

/* ---------- Backend Info ---------- */

const char* backend_name(void);

/* ---------- Serialization (SafeTensors) ---------- */

/* Save all registered params to a .safetensors file. Returns 0 on success. */
int param_save(const char* path);

/* Load params from a .safetensors file into the existing param registry.
   Matches by name. Skips tensors not in registry. Returns 0 on success.
   Strict mode: errors out if any tensor's on-disk dtype differs from
   the destination param's dtype. Use param_load_with_policy() to opt
   in to silent precision conversion. */
int param_load(const char* path);

/* Load params with explicit dtype-mismatch policy. allow_cast=0 is
   strict (mismatch -> error). allow_cast=1 reads source bytes,
   widens to doubles, then loads via param_load_data (which narrows
   to the destination param's actual dtype as needed). Returns 0 if
   every tensor loaded cleanly, nonzero if any entry was skipped. */
int param_load_with_policy(const char* path, int allow_cast);

/* Overwrite param tensor data in-place from a double buffer (per-backend). */
void param_load_data(int idx, const double* data, int numel);

/* Byte-exact I64 extractor + loader. The double lingua-franca path
   above rounds any int64 magnitude beyond 2^53 (double has 53 bits of
   mantissa); these two symbols bypass the double pivot so every i64
   bit pattern survives the file round-trip when src and dst dtypes
   are both I64. Honest only on backends with native i64 storage
   (torch today); on tape (double-backed) and mlx (no integer
   storage) they route through `tensor_to_doubles` / `param_load_data`
   and inherit the same 2^53 ceiling, matching the existing rounded
   behaviour with no regression. See `safetensors.c`'s I64 save/load
   branches for the call sites. */
void tensor_to_int64(TensorHandle t, int64_t* out);
void param_load_data_int64(int idx, const int64_t* data, int numel);

/* Save optimizer state to a .safetensors file. Returns 0 on success. */
int optimizer_save(OptimizerHandle opt, const char* path);

/* Load optimizer state from a .safetensors file. Returns 0 on success. */
int optimizer_load(OptimizerHandle opt, const char* path);

/* Optimizer buffer accessors (per-backend, for serialization) */
int  optimizer_buf_count(OptimizerHandle opt);
void optimizer_get_m(OptimizerHandle opt, int idx, double* out);
void optimizer_get_v(OptimizerHandle opt, int idx, double* out);
void optimizer_set_m(OptimizerHandle opt, int idx, const double* data);
void optimizer_set_v(OptimizerHandle opt, int idx, const double* data);
void optimizer_get_meta(OptimizerHandle opt, double* out9);
void optimizer_set_meta(OptimizerHandle opt, const double* in9);

/* Per-dtype variants; F64 is the legacy default. */
TensorHandle tensor_create_param_4d_f32(int d0, int d1, int d2, int d3, double* data);
TensorHandle tensor_create_param_4d_f64(int d0, int d1, int d2, int d3, double* data);

/* ---------- Cross-Attention ---------- */

/* Scaled dot-product attention: Q [B,seqQ,d], K [B,seqK,d], V [B,seqK,d].
   Returns [B,seqQ,d]. mask may be NULL (no masking). scale = 1/sqrt(d). */
TensorHandle tensor_cross_attention(TensorHandle Q, TensorHandle K, TensorHandle V,
                                    TensorHandle mask, double scale);

/* ---------- Embedding ---------- */

/* Embedding lookup: weight [vocabSize, embedDim], indices [n] (double-valued ints).
   Returns [n * embedDim] (flat). Backward: scatter_add grads to weight rows. */
TensorHandle tensor_embedding(TensorHandle weight, TensorHandle indices, int n, int embedDim);

/* ---------- Gather / Scatter ---------- */

/* Gather: out[i] = input[index[i]] along dim 0 (1D index into 1D input) */
TensorHandle tensor_gather(TensorHandle input, TensorHandle index, int n);
/* Scatter: out = zeros; out[index[i]] += src[i] (1D scatter-add) */
TensorHandle tensor_scatter_add(TensorHandle index, TensorHandle src, int out_size);

/* ---------- Sort / Scan ---------- */

/* Argsort: returns integer indices that sort input along dim.
   descending=0 for ascending, 1 for descending.
   Input: 1D tensor [n]. Returns: 1D integer tensor [n]. */
TensorHandle tensor_argsort(TensorHandle t, int dim, int descending);

/* Cumulative product along dim. Input: 1D tensor [n]. Returns: 1D tensor [n].
   out[i] = prod(input[0..i]). */
TensorHandle tensor_cumprod(TensorHandle t, int dim);

/* ---------- MNIST data loading ---------- */

void* mnist_load(const char* images_path, const char* labels_path);
int mnist_count(void* handle);
TensorHandle mnist_get_image(void* handle, int index, int dtag);  /* [1, 28, 28] tensor in dtag's dtype */
int mnist_get_label(void* handle, int index);            /* 0-9 */
void mnist_free(void* handle);

/* ---------- DataLoader ---------- */

int* create_index_array(int n);
int* shuffle_index_array(int* arr, int n);
int  index_array_get(int* arr, int i);

/* ---------- MLX compile (Job 3 Phase B) ---------- */

/* Returns 1 if MLX_COMPILE env var is set to a truthy value ("1", "true",
   "yes"), 0 otherwise. Non-mlx backends always return 0. */
int tensor_mlx_compile_enabled(void);

/* Count of how many times tensor_backward has entered the compile-enabled
   code path. Resets via tensor_mlx_compile_reset_stats(). Non-mlx backends
   always return 0. */
int  tensor_mlx_compile_invocations(void);
void tensor_mlx_compile_reset_stats(void);

/* ---------- Debug ---------- */

void tensor_print(TensorHandle t);


/* ---- Unified dtag-dispatch create/cast entry points ----
   One symbol per shape; the trailing `int dtag` selects the RuntimeDType
   under the kind-major layout (closed 2026-05-23; replaces the original
   0=F32, 1=F64, ... incremental order):
     0  invalid (reserved; zero-init traps at backend default-arm abort)
     1  Bool
     4  U8                              (family 1 — U; lanes 5-7 reserved)
     8  I8     9  I16    10 I32    11 I64    (family 2 — I)
     13 F16   14  F32   15 F64               (family 3 — F; F8 reserved)
     17 BF16                                  (family 4 — BF; BF8/32/64 reserved)
     20-23   reserved                          (family 5 — TF)
     24-31   reserved                          (sub-byte quant — U4/I4/NF4/MX/...)
   For numeric families bit_width = 8 << (tag & 3); sub-byte slots use a
   metadata lookup since their semantics aren't pure (family, bit-width).
   Supersedes the per-dtype *_streamed declarations above. Each backend
   switches on dtag internally: torch handles all 10 wired dtypes, mlx
   f32/f64 (rejects the rest), tape stores all via the double lingua
   franca with real F32 storage + kernels (Phase 3). */
TensorHandle tensor_create_scalar_streamed(double value, int requires_grad, int stream_tag, int dtag);
TensorHandle tensor_create_streamed(double* data, int* shape, int rank, int requires_grad, int stream_tag, int dtag);
TensorHandle tensor_create_1d_streamed(int n, double* data, int requires_grad, int stream_tag, int dtag);
TensorHandle tensor_create_2d_streamed(int rows, int cols, double* data, int requires_grad, int stream_tag, int dtag);
TensorHandle tensor_create_param_1d_streamed(int n, double* data, int stream_tag, int dtag);
TensorHandle tensor_create_param_2d_streamed(int rows, int cols, double* data, int stream_tag, int dtag);
TensorHandle tensor_create_param_3d_streamed(int d0, int d1, int d2, double* data, int stream_tag, int dtag);
TensorHandle tensor_create_param_4d_streamed(int d0, int d1, int d2, int d3, double* data, int stream_tag, int dtag);
TensorHandle tensor_create_state_1d_streamed(int n, double* data, int stream_tag, int dtag);
TensorHandle tensor_create_state_2d_streamed(int rows, int cols, double* data, int stream_tag, int dtag);
TensorHandle tensor_cast_dtype_streamed(TensorHandle src, int stream_tag, int dtag);

/* Fused param create + init — allocate + run in-place init in C
   (libtorch's `torch::nn::init::normal_` / `t.fill_`) instead of
   filling element-by-element on the Idris side via traverse +
   packDoubles. Backends that haven't wired their adapter slots abort
   loudly at the FFI boundary (see shared/training/dtype_streamed.c). */
TensorHandle tensor_create_param_1d_normal_streamed(int n,                                       double mean, double std, int stream_tag, int dtag);
TensorHandle tensor_create_param_2d_normal_streamed(int rows, int cols,                          double mean, double std, int stream_tag, int dtag);
TensorHandle tensor_create_param_3d_normal_streamed(int d0, int d1, int d2,                      double mean, double std, int stream_tag, int dtag);
TensorHandle tensor_create_param_4d_normal_streamed(int d0, int d1, int d2, int d3,              double mean, double std, int stream_tag, int dtag);
TensorHandle tensor_create_param_1d_const_streamed (int n,                                       double value,            int stream_tag, int dtag);
TensorHandle tensor_create_param_2d_const_streamed (int rows, int cols,                          double value,            int stream_tag, int dtag);
TensorHandle tensor_create_param_3d_const_streamed (int d0, int d1, int d2,                      double value,            int stream_tag, int dtag);
TensorHandle tensor_create_param_4d_const_streamed (int d0, int d1, int d2, int d3,              double value,            int stream_tag, int dtag);
void         tensor_set_init_seed_streamed(unsigned long long seed, int stream_tag);

#ifdef __cplusplus
}
#endif

#endif /* IDRISML_BACKEND_H */
