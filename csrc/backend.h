#ifndef IDRISML_BACKEND_H
#define IDRISML_BACKEND_H

#ifdef __cplusplus
extern "C" {
#endif

/* Opaque tensor handle — wraps backend-specific tensor (e.g. at::Tensor*) */
typedef void* TensorHandle;

/* Pair of tensor handles (for functions returning two results) */
typedef struct { TensorHandle first; TensorHandle second; } TensorPair;

/* ---------- Lifecycle ---------- */

TensorHandle tensor_create_scalar(double value, int requires_grad);
TensorHandle tensor_create(double* data, int* shape, int rank, int requires_grad);
TensorHandle tensor_clone(TensorHandle t);
void         tensor_free(TensorHandle t);

/* ---------- Accessors ---------- */

double tensor_item(TensorHandle t);          /* scalar tensor -> double */
int    tensor_numel(TensorHandle t);
int    tensor_dim(TensorHandle t);
int    tensor_size(TensorHandle t, int dim);
void   tensor_to_doubles(TensorHandle t, double* out); /* flatten to buffer */

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

/* Scalar broadcast variants */
TensorHandle tensor_add_scalar(TensorHandle t, double s);
TensorHandle tensor_mul_scalar(TensorHandle t, double s);

/* Clamp: element-wise max(t, min_val). Returns new tensor. */
TensorHandle tensor_clamp_min(TensorHandle t, double min_val);

/* ---------- Reduction ---------- */

TensorHandle tensor_sum(TensorHandle t);
TensorHandle tensor_sum_dim(TensorHandle t, int dim, int keepdim);
TensorHandle tensor_mean(TensorHandle t);

/* ---------- Linear algebra ---------- */

TensorHandle tensor_matmul(TensorHandle a, TensorHandle b);     /* general matmul */
TensorHandle tensor_mv(TensorHandle mat, TensorHandle vec);     /* matrix-vector */
TensorHandle tensor_dot(TensorHandle a, TensorHandle b);        /* vector dot */
TensorHandle tensor_outer(TensorHandle a, TensorHandle b);      /* outer product */

/* ---------- Activation / normalization ---------- */

TensorHandle tensor_softmax(TensorHandle t, int dim);
TensorHandle tensor_log_softmax(TensorHandle t, int dim);

/* ---------- Loss ---------- */

TensorHandle tensor_bce_with_logits(TensorHandle input, TensorHandle target);
TensorHandle tensor_cross_entropy(TensorHandle input, TensorHandle target);
TensorHandle tensor_mse_loss(TensorHandle input, TensorHandle target);

/* ---------- NTM-specific compositions ---------- */

TensorHandle tensor_cosine_similarity(TensorHandle a, TensorHandle b, int dim);
TensorHandle tensor_conv1d_circular(TensorHandle input, TensorHandle kernel);

/* Fused NTM read head addressing pipeline (all tensor-level, no scalar stacking).
   Takes: memory [n,w], prev_weights [n], key [w], beta (scalar), g (scalar),
          gamma (scalar), shift_kernel [3]
   Returns: (new_weights [n], read_output [w]) as TensorPair* */
TensorPair* tensor_ntm_read_head(
    TensorHandle memory, TensorHandle prev_weights,
    TensorHandle key, TensorHandle beta, TensorHandle g,
    TensorHandle gamma, TensorHandle shift_kernel);

/* Fused NTM write: interpolation write (no erase).
   memory' = memory + outer(weights, add_vector)
   Returns: new_memory [n,w] */
TensorHandle tensor_ntm_interp_write(
    TensorHandle memory, TensorHandle weights, TensorHandle add_vector);

/* ---------- Shape manipulation ---------- */

TensorHandle tensor_reshape(TensorHandle t, int* shape, int rank);
TensorHandle tensor_unsqueeze(TensorHandle t, int dim);
TensorHandle tensor_squeeze(TensorHandle t, int dim);
TensorHandle tensor_select(TensorHandle t, int dim, int index);
TensorHandle tensor_stack(TensorHandle* tensors, int count, int dim);
TensorHandle tensor_cat(TensorHandle* tensors, int count, int dim);

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

/* ---------- Device ---------- */

TensorHandle tensor_to_device(TensorHandle t, const char* device);  /* "cpu", "mps", "cuda" */
const char*  tensor_device(TensorHandle t);

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
TensorHandle tensor_create_1d(int n, double* data, int requires_grad);
TensorHandle tensor_create_2d(int rows, int cols, double* data, int requires_grad);

/* Allocate a C double buffer (for Scheme-side packing) */
double* tensor_alloc_doubles(int n);
/* Read a double from a C buffer */
double tensor_read_double(double* buf, int idx);
/* Write a double to a C buffer */
void tensor_write_double(double* buf, int idx, double val);

/* ---------- Tensor pointer array (for stack/cat from Idris) ---------- */

TensorHandle* tensor_ptr_array_alloc(int n);
void           tensor_ptr_array_set(TensorHandle* arr, int idx, TensorHandle t);
TensorHandle   tensor_stack_from_array(TensorHandle* arr, int count, int dim);
TensorHandle   tensor_cat_from_array(TensorHandle* arr, int count, int dim);

/* ---------- Tensor-level parameter creation ---------- */

/* Create a [rows, cols] tensor filled with given data, requires_grad=true */
TensorHandle tensor_create_param_2d(int rows, int cols, double* data);
/* Create a [n] tensor filled with given data, requires_grad=true */
TensorHandle tensor_create_param_1d(int n, double* data);
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
void            optimizer_free(OptimizerHandle opt);
void            optimizer_step(OptimizerHandle opt);
void            optimizer_zero_grad(OptimizerHandle opt);

/* Gradient clipping (operates on all registered params) */
void optimizer_clip_grad_value(double max_val);
double optimizer_clip_grad_norm(double max_norm);  /* returns actual norm */

/* ---------- Backend Capabilities ---------- */

/* Returns 1 if the backend supports tensor-level parameters
   (consolidated weight tensors with views that share storage).
   Returns 0 if the backend uses scalar-per-element parameters. */
int backend_supports_tensor_params(void);

/* ---------- System ---------- */

int get_rss_mb(void);           /* peak RSS in MB (getrusage) */
int get_current_rss_mb(void);   /* current RSS in MB (macOS mach_task_info) */

/* ---------- Debug ---------- */

void tensor_print(TensorHandle t);

#ifdef __cplusplus
}
#endif

#endif /* IDRISML_BACKEND_H */
