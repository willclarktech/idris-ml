/*
 * backend_byo.c — "Bring Your Own" example backend.
 *
 * Minimal demonstration of how a user-supplied backend plugs into
 * idris-ml's UserExecutorCore typeclass. Each op logs "[byo] <name>"
 * to stderr so you can SEE the dispatch happen, then returns a stub
 * (the input pointer, or a zero scalar). No real math; the goal is
 * to show the type-level mechanism, not implement an autograd
 * engine in 100 lines.
 *
 * Builds as a standalone libbyo.{so,dylib}. No dependency on
 * libidrisml — that's the point. A real custom backend would
 * implement the FFI surface with whatever runtime it wants (CUDA
 * kernels, an FPGA driver, a remote-RPC, whatever).
 *
 * Pairs with `packages/idris-ml-examples/src/Example/BringYourOwn.idr`
 * and the "Custom devices: user-supplied backends" section of
 * docs/grad-mode-and-device-typing.md.
 *
 * Scope note: this file is deliberately the *minimum* — UserExecutorCore
 * only, no training surface, no shared-port adoption. A backend that
 * wants gradient descent + optimizer state + safetensors load/save
 * follows the path documented in `packages/backends/README.md`
 * ("Adding a new backend"): implement the FFI surface in backend.h,
 * expose a `g_active_port_<name>` BackendPort struct, opt the backend
 * into the relevant SHARED_BACKENDS_<tu> Makefile lists. See
 * backend_tape, backend_torch, backend_mlx for working examples.
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

/* Our "tensor" is just a heap-allocated scalar so we have something
 * to return that survives across calls. Real backends would allocate
 * device buffers, build autograd nodes, etc. */
typedef double* byo_tensor;

static byo_tensor make_scalar(double v) {
	byo_tensor t = (byo_tensor)malloc(sizeof(double));
	if (t) *t = v;
	return t;
}

/* ---------- Lifecycle ---------- */

byo_tensor byo_tensor_create_scalar(double value, int requires_grad) {
	fprintf(stderr, "[byo] tensor_create_scalar(%g, rg=%d)\n", value, requires_grad);
	return make_scalar(value);
}

byo_tensor byo_tensor_create(double* data, int* shape, int rank, int requires_grad) {
	(void)shape;
	(void)rank;
	(void)requires_grad;
	fprintf(stderr, "[byo] tensor_create(rank=%d, rg=%d)\n", rank, requires_grad);
	return make_scalar(data ? data[0] : 0.0);
}

void byo_tensor_free(byo_tensor t) {
	fprintf(stderr, "[byo] tensor_free\n");
	free(t);
}

double byo_tensor_item(byo_tensor t) {
	double v = t ? *t : 0.0;
	fprintf(stderr, "[byo] tensor_item -> %g\n", v);
	return v;
}

/* This toy backend stores only a single scalar (see byo_tensor_create),
   so 1-D element reads collapse to that value regardless of idx. */
double byo_tensor_item1d(byo_tensor t, int idx) {
	(void)idx;
	double v = t ? *t : 0.0;
	fprintf(stderr, "[byo] tensor_item1d -> %g\n", v);
	return v;
}

byo_tensor byo_tensor_clone(byo_tensor t) {
	fprintf(stderr, "[byo] tensor_clone\n");
	return make_scalar(t ? *t : 0.0);
}

/* ---------- Elementwise arithmetic ----------
 *
 * Real backends would broadcast + run kernels here. We log + return
 * a new scalar containing a stand-in value (sum / product / etc. of
 * the inputs' scalar slots) so the user can SEE the dispatch fired
 * and the runtime read-back works end-to-end. */

byo_tensor byo_tensor_add(byo_tensor a, byo_tensor b) {
	double va = a ? *a : 0.0, vb = b ? *b : 0.0;
	fprintf(stderr, "[byo] tensor_add(%g, %g)\n", va, vb);
	return make_scalar(va + vb);
}
byo_tensor byo_tensor_sub(byo_tensor a, byo_tensor b) {
	double va = a ? *a : 0.0, vb = b ? *b : 0.0;
	fprintf(stderr, "[byo] tensor_sub(%g, %g)\n", va, vb);
	return make_scalar(va - vb);
}
byo_tensor byo_tensor_mul(byo_tensor a, byo_tensor b) {
	double va = a ? *a : 0.0, vb = b ? *b : 0.0;
	fprintf(stderr, "[byo] tensor_mul(%g, %g)\n", va, vb);
	return make_scalar(va * vb);
}
byo_tensor byo_tensor_div(byo_tensor a, byo_tensor b) {
	double va = a ? *a : 0.0, vb = b ? *b : 0.0;
	fprintf(stderr, "[byo] tensor_div(%g, %g)\n", va, vb);
	return make_scalar(vb != 0.0 ? va / vb : 0.0);
}
byo_tensor byo_tensor_neg(byo_tensor a) {
	fprintf(stderr, "[byo] tensor_neg\n");
	return make_scalar(a ? -(*a) : 0.0);
}
byo_tensor byo_tensor_abs(byo_tensor a) {
	fprintf(stderr, "[byo] tensor_abs\n");
	return make_scalar(a && *a < 0 ? -(*a) : (a ? *a : 0.0));
}
byo_tensor byo_tensor_exp(byo_tensor a) {
	fprintf(stderr, "[byo] tensor_exp\n");
	return make_scalar(0.0);
}
byo_tensor byo_tensor_log(byo_tensor a) {
	fprintf(stderr, "[byo] tensor_log\n");
	return make_scalar(0.0);
}
byo_tensor byo_tensor_sqrt(byo_tensor a) {
	fprintf(stderr, "[byo] tensor_sqrt\n");
	return make_scalar(0.0);
}
byo_tensor byo_tensor_pow(byo_tensor a, byo_tensor b) {
	(void)a;
	(void)b;
	fprintf(stderr, "[byo] tensor_pow\n");
	return make_scalar(0.0);
}
byo_tensor byo_tensor_sigmoid(byo_tensor a) {
	fprintf(stderr, "[byo] tensor_sigmoid\n");
	return make_scalar(0.5);
}
byo_tensor byo_tensor_tanh(byo_tensor a) {
	fprintf(stderr, "[byo] tensor_tanh\n");
	return make_scalar(0.0);
}

/* ---------- Scalar arithmetic ---------- */

byo_tensor byo_tensor_add_scalar(byo_tensor t, double s) {
	double v = t ? *t : 0.0;
	fprintf(stderr, "[byo] tensor_add_scalar(%g, +%g)\n", v, s);
	return make_scalar(v + s);
}
byo_tensor byo_tensor_mul_scalar(byo_tensor t, double s) {
	double v = t ? *t : 0.0;
	fprintf(stderr, "[byo] tensor_mul_scalar(%g, *%g)\n", v, s);
	return make_scalar(v * s);
}
byo_tensor byo_tensor_clamp_min(byo_tensor t, double m) {
	double v = t ? *t : 0.0;
	fprintf(stderr, "[byo] tensor_clamp_min(%g, %g)\n", v, m);
	return make_scalar(v < m ? m : v);
}
