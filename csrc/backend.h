#ifndef IDRISML_BACKEND_H
#define IDRISML_BACKEND_H

#ifdef __cplusplus
extern "C" {
#endif

/* Opaque tensor handle — wraps backend-specific tensor (e.g. at::Tensor*) */
typedef void* TensorHandle;

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

/* ---------- Parameter Registry ---------- */

/* Register a named parameter for gradient collection after backward() */
void         param_register(const char* name, TensorHandle t);
void         param_clear(void);
int          param_count(void);
const char*  param_name(int idx);
double       param_grad_item(int idx);          /* read scalar grad for param i */
TensorHandle param_tensor(int idx);
void         param_zero_all_grads(void);
void         param_subtract_delta(int idx, double delta); /* in-place: param -= delta */

/* ---------- Debug ---------- */

void tensor_print(TensorHandle t);

#ifdef __cplusplus
}
#endif

#endif /* IDRISML_BACKEND_H */
