/* Shared training port adapter — mlx.
 *
 * Provides the per-tensor accessors that shared/training/param_registry.c
 * uses to talk to mlx (numel / has_grad / grad read+write / zero / data
 * read+write / bulk doubles+int64 loaders). Other port slots stay nullptr
 * because mlx hasn't joined the matching shared TUs (optimizer_*,
 * dtag-streamed creators, ffi_shims) — mlx's own optimizer + streamed
 * creators provide those surfaces directly via local files (training/
 * optimizer.cpp, training/dtype_dispatch.cpp).
 *
 * Element accessors take a per-call hit because mlx arrays are immutable:
 * data_write / grad_write realize the array host-side, mutate one
 * element, then rebuild via mx_array_from_doubles. Param-registry callers
 * hit this rarely (param_subtract_delta / param_grad_item_and_zero on
 * scalar params), so per-call allocate is acceptable for now.
 */
#include "../tensor.h"
#include "../precision.h"
#include "../../shared/training/port.h"
#include <cstdlib>
#include <cstdint>

static int mlx_port_tensor_numel(void* h) {
    return (int)((Tensor*)h)->data.size();
}

static int mlx_port_tensor_requires_grad(void* h) {
    return ((Tensor*)h)->requires_grad ? 1 : 0;
}

static int mlx_port_tensor_has_grad(void* h) {
    return ((Tensor*)h)->has_grad ? 1 : 0;
}

static double mlx_port_grad_read(void* h, int i) {
    auto* t = (Tensor*)h;
    if (!t->has_grad) return 0.0;
    /* mx::vjp may return non-contiguous arrays; force contiguous read. */
    auto contig = mx::contiguous(t->grad);
    mx::eval(contig);
    return mx_read_double(contig, i);
}

static void mlx_port_grad_write(void* h, int i, double v) {
    auto* t = (Tensor*)h;
    if (!t->has_grad) return;
    /* Realize grad host-side, mutate element, push back. mlx arrays are
       immutable, so writing one element requires rebuilding via
       mx_array_from_doubles. Param-registry callers hit this rarely
       (only param_subtract_delta / param_grad_item_and_zero on scalar
       params), so per-call allocate is acceptable. */
    auto contig = mx::contiguous(t->grad);
    mx::eval(contig);
    int n = (int)contig.size();
    double* buf = (double*)malloc((size_t)n * sizeof(double));
    for (int k = 0; k < n; k++) buf[k] = mx_read_double(contig, k);
    buf[i] = v;
    t->grad = mx_array_from_doubles(buf, t->grad.shape(), t->grad.dtype());
    free(buf);
}

static void mlx_port_zero_grad(void* h) {
    auto* t = (Tensor*)h;
    if (t->has_grad) {
        t->grad = mx::zeros(t->data.shape(), t->data.dtype());
    }
}

static double mlx_port_data_read(void* h, int i) {
    auto* t = (Tensor*)h;
    auto contig = mx::contiguous(t->data);
    mx::eval(contig);
    return mx_read_double(contig, i);
}

static void mlx_port_data_write(void* h, int i, double v) {
    auto* t = (Tensor*)h;
    auto contig = mx::contiguous(t->data);
    mx::eval(contig);
    int n = (int)contig.size();
    double* buf = (double*)malloc((size_t)n * sizeof(double));
    for (int k = 0; k < n; k++) buf[k] = mx_read_double(contig, k);
    buf[i] = v;
    t->data = mx_array_from_doubles(buf, t->data.shape(), t->data.dtype());
    free(buf);
}

static void mlx_port_load_doubles(void* h, const double* src, int n) {
    auto* t = (Tensor*)h;
    (void)n;  /* shape already determines size; caller validates against numel */
    t->data = mx_array_from_doubles(src, t->data.shape(), t->data.dtype());
}

static void mlx_port_load_int64(void* h, const int64_t* src, int n) {
    auto* t = (Tensor*)h;
    /* No native I64 storage on mlx — pivot through double. Matches the
       lossy I64 behavior the previous in-file param_load_data_int64
       documented (values above 2^53 lose precision). */
    double* tmp = (double*)malloc((size_t)n * sizeof(double));
    for (int k = 0; k < n; k++) tmp[k] = (double)src[k];
    t->data = mx_array_from_doubles(tmp, t->data.shape(), t->data.dtype());
    free(tmp);
}

const BackendPort g_active_port = {
    /* Tensor introspection + per-element + bulk grad/data ops. */
    .tensor_numel              = mlx_port_tensor_numel,
    .tensor_requires_grad      = mlx_port_tensor_requires_grad,
    .tensor_has_grad           = mlx_port_tensor_has_grad,
    .data_read                 = mlx_port_data_read,
    .data_write                = mlx_port_data_write,
    .grad_read                 = mlx_port_grad_read,
    .grad_write                = mlx_port_grad_write,
    .zero_grad                 = mlx_port_zero_grad,
    .load_doubles              = mlx_port_load_doubles,
    .load_int64                = mlx_port_load_int64,
    /* Remaining slots wait on mlx joining the corresponding shared TUs.
       Order matches port.h's struct declaration order (C++ ISO-required
       for designated init). */
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
    .create_scalar             = nullptr,
    .create                    = nullptr,
    .create_1d                 = nullptr,
    .create_2d                 = nullptr,
    .create_param_1d           = nullptr,
    .create_param_2d           = nullptr,
    .create_param_3d           = nullptr,
    .create_param_4d           = nullptr,
    .create_state_1d           = nullptr,
    .create_state_2d           = nullptr,
    .cast_dtype                = nullptr,
};
