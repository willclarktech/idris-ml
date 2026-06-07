/* Shared training port adapter — torch.
 *
 * Provides the per-tensor accessors that shared/training/param_registry.c
 * uses to talk to libtorch (numel / has_grad / grad read+write / zero /
 * data read+write / bulk doubles+int64 loaders) AND wires the torch
 * dtag-create dispatchers (declared in dtype_dispatch.h) into the port
 * struct as void*-typed trampolines.
 *
 * Slots whose shared TUs torch hasn't opted into yet stay nullptr — the
 * shared trampolines never call them because torch is excluded from
 * those SHARED_BACKENDS_<tu> lists in the Makefile.
 *
 * Hot-path note: for F64/F32 contiguous CPU tensors, the element
 * accessors hit `data_ptr<>()` directly (one load); other dtype / device
 * combos route through the slow `.flatten().index({i}).cpu().item<>()`
 * path. Param storage is always contiguous + same device throughout a
 * run, so the fast path covers ~all live use.
 */
#include "../tensor.h"
#include "dtype_dispatch.h"
#include "../../shared/training/port.h"
#include <torch/torch.h>
#include <cstdint>

/* Port-typed (void*) trampolines for the dtag creators. The internal
   torch_create_*_dtag helpers return TensorHandle (which is void* at the
   C level — but C++ enforces the cast at the function-pointer-init site). */
static void* torch_port_create_scalar(double v, int rg, int dtag) {
	return torch_create_scalar_dtag(v, rg, dtag);
}
static void* torch_port_create(double* d, int* s, int r, int rg, int dtag) {
	return torch_create_dtag(d, s, r, rg, dtag);
}
static void* torch_port_create_1d(int n, double* d, int rg, int dtag) {
	return torch_create_1d_dtag(n, d, rg, dtag);
}
static void* torch_port_create_2d(int rows, int cols, double* d, int rg, int dtag) {
	return torch_create_2d_dtag(rows, cols, d, rg, dtag);
}
static void* torch_port_create_param_1d(int n, double* d, int dtag) {
	return torch_create_param_1d_dtag(n, d, dtag);
}
static void* torch_port_create_param_2d(int rows, int cols, double* d, int dtag) {
	return torch_create_param_2d_dtag(rows, cols, d, dtag);
}
static void* torch_port_create_param_3d(int d0, int d1, int d2, double* d, int dtag) {
	return torch_create_param_3d_dtag(d0, d1, d2, d, dtag);
}
static void* torch_port_create_param_4d(int d0, int d1, int d2, int d3, double* d, int dtag) {
	return torch_create_param_4d_dtag(d0, d1, d2, d3, d, dtag);
}
static void* torch_port_create_state_1d(int n, double* d, int dtag) {
	return torch_create_state_1d_dtag(n, d, dtag);
}
static void* torch_port_create_state_2d(int rows, int cols, double* d, int dtag) {
	return torch_create_state_2d_dtag(rows, cols, d, dtag);
}
static void* torch_port_cast_dtype(void* src, int dtag) {
	return torch_cast_dtype_dtag((TensorHandle)src, dtag);
}

/* Fused param create + init shims — forward to dtype_init.cpp's
   torch_create_param_*_<init>_dtag. */
static void* torch_port_create_param_1d_normal(int n, double mean, double std, int dtag) {
	return torch_create_param_1d_normal_dtag(n, mean, std, dtag);
}
static void* torch_port_create_param_2d_normal(int rows, int cols, double mean, double std,
                                               int dtag) {
	return torch_create_param_2d_normal_dtag(rows, cols, mean, std, dtag);
}
static void* torch_port_create_param_3d_normal(int d0, int d1, int d2, double mean, double std,
                                               int dtag) {
	return torch_create_param_3d_normal_dtag(d0, d1, d2, mean, std, dtag);
}
static void* torch_port_create_param_4d_normal(int d0, int d1, int d2, int d3, double mean,
                                               double std, int dtag) {
	return torch_create_param_4d_normal_dtag(d0, d1, d2, d3, mean, std, dtag);
}
static void* torch_port_create_param_1d_const(int n, double value, int dtag) {
	return torch_create_param_1d_const_dtag(n, value, dtag);
}
static void* torch_port_create_param_2d_const(int rows, int cols, double value, int dtag) {
	return torch_create_param_2d_const_dtag(rows, cols, value, dtag);
}
static void* torch_port_create_param_3d_const(int d0, int d1, int d2, double value, int dtag) {
	return torch_create_param_3d_const_dtag(d0, d1, d2, value, dtag);
}
static void* torch_port_create_param_4d_const(int d0, int d1, int d2, int d3, double value,
                                              int dtag) {
	return torch_create_param_4d_const_dtag(d0, d1, d2, d3, value, dtag);
}
static void torch_port_set_init_seed(uint64_t seed) {
	torch_set_init_seed(seed);
}

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
		if (g.dtype() == torch::kFloat64) {
			((double*)g.data_ptr())[i] = v;
			return;
		}
		if (g.dtype() == torch::kFloat32) {
			((float*)g.data_ptr())[i] = (float)v;
			return;
		}
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
		if (t->dtype() == torch::kFloat64) {
			((double*)t->data_ptr())[i] = v;
			return;
		}
		if (t->dtype() == torch::kFloat32) {
			((float*)t->data_ptr())[i] = (float)v;
			return;
		}
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
    .tensor_numel = torch_port_tensor_numel,
    .tensor_requires_grad = torch_port_tensor_requires_grad,
    .tensor_has_grad = torch_port_tensor_has_grad,
    .data_read = torch_port_data_read,
    .data_write = torch_port_data_write,
    .grad_read = torch_port_grad_read,
    .grad_write = torch_port_grad_write,
    .zero_grad = torch_port_zero_grad,
    .load_doubles = torch_port_load_doubles,
    .load_int64 = torch_port_load_int64,
    /* Slots whose shared TUs torch hasn't opted into yet — see the
       SHARED_BACKENDS_<tu> lists in the Makefile. These stay nullptr
       until torch's adapter ships their bindings AND the shared TU
       gets compiled for torch. Ordering matches port.h's struct
       declaration order (C++ ISO-required for designated init). */
    .backward = nullptr,
    .optimizer_create_sgd = nullptr,
    .optimizer_create_rmsprop = nullptr,
    .optimizer_create_adam = nullptr,
    .optimizer_create_adam_group = nullptr,
    .optimizer_create_adamw = nullptr,
    .optimizer_free = nullptr,
    .optimizer_set_lr = nullptr,
    .optimizer_set_param_lr = nullptr,
    .optimizer_step = nullptr,
    .optimizer_buf_count = nullptr,
    .optimizer_get_m = nullptr,
    .optimizer_get_v = nullptr,
    .optimizer_set_m = nullptr,
    .optimizer_set_v = nullptr,
    .optimizer_get_meta = nullptr,
    .optimizer_set_meta = nullptr,
    .wall_ms = nullptr,
    /* Dtag-streamed creators: torch supplies its libtorch-backed
       dtag dispatchers via the torch_port_create_* shims above. */
    .create_scalar = torch_port_create_scalar,
    .create = torch_port_create,
    .create_1d = torch_port_create_1d,
    .create_2d = torch_port_create_2d,
    .create_param_1d = torch_port_create_param_1d,
    .create_param_2d = torch_port_create_param_2d,
    .create_param_3d = torch_port_create_param_3d,
    .create_param_4d = torch_port_create_param_4d,
    .create_state_1d = torch_port_create_state_1d,
    .create_state_2d = torch_port_create_state_2d,
    .cast_dtype = torch_port_cast_dtype,
    /* Fused param create + init slots (see port.h struct doc). All
       wired on torch; tape/mlx leave these nullptr until their
       follow-up rows land (Phase 7). The shared trampolines in
       dtype_streamed.c abort loudly if any of these is called via a
       backend that hasn't wired the slot. */
    .create_param_1d_normal = torch_port_create_param_1d_normal,
    .create_param_2d_normal = torch_port_create_param_2d_normal,
    .create_param_3d_normal = torch_port_create_param_3d_normal,
    .create_param_4d_normal = torch_port_create_param_4d_normal,
    .create_param_1d_const = torch_port_create_param_1d_const,
    .create_param_2d_const = torch_port_create_param_2d_const,
    .create_param_3d_const = torch_port_create_param_3d_const,
    .create_param_4d_const = torch_port_create_param_4d_const,
    .set_init_seed = torch_port_set_init_seed,
};
