/* backend_tape/training/dtype_init.c — fused param create + in-place init.
 *
 * Closes the symmetry with torch (training/dtype_init.cpp) and mlx
 * (training/dtype_init.cpp). The shared dtype_streamed.c trampolines
 * dispatch to port slots `g_active_port.create_param_<rank>_<init>` and
 * `g_active_port.set_init_seed`; this TU provides tape's implementation
 * of those slots. Without it `tensor_create_param_*_<init>_streamed`
 * aborts loudly at the FFI boundary on tape (the `abort_unwired_init`
 * arm in dtype_streamed.c).
 *
 * Background: HF model state construction used to fill each parameter
 * tensor element-by-element on the host via `traverse normalSample` +
 * `packDoubles` (per-element `prim__setDouble` FFI), costing 58 min for
 * Llama-3.2-1B (~30 min Box-Muller in Chez + ~28 min per-element FFI).
 * The fused-init port lets each backend run the init kernel on its own
 * side at memory-bandwidth speed.
 *
 * Tape has no native init-kernel surface (no `torch::nn::init`, no
 * `mx::random::normal`), so this file ships its own:
 *  - xoshiro256++ + splitmix64 seed expansion (BigCrush-passing,
 *    public-domain, ~30 LOC; smaller state + faster than mt19937).
 *  - Box-Muller (paired output, one-sample cache) for N(mean, std).
 *  - Routes the filled doubles buffer through the existing
 *    `tape_create_param_<rank>_dtag` for dtag/storage dispatch — no
 *    new storage-variant logic to maintain. F32 dtag still gets real
 *    4-byte storage; lingua-franca dtags still retag-round.
 *
 * Determinism: a single process-global xoshiro state, seeded by
 * tape_set_init_seed. Default seed = 0 splatted through splitmix64
 * produces a well-mixed initial state, matching the "implicit seed 0"
 * convention. Not thread-local; tape's call sites are single-threaded
 * (param construction happens once at model build time, before any
 * forward pass).
 */

#include <stdlib.h>
#include <stdint.h>
#include <math.h>
#include "../../backend.h"
#include "../../shared_utils.h"

/* ----------------------------------------------------------------------
   Param-init seeding.

   The generator itself lives in shared_utils.c (compile-once, unified)
   so all three backends can share one definition — see
   `idrisml_portable_fill_normal`. Tape has always used that algorithm
   (xoshiro256++ seeded via splitmix64, paired Box-Muller normals), so
   relocating it leaves tape's numerics untouched.
   ---------------------------------------------------------------------- */

void tape_set_init_seed(uint64_t seed) {
	idrisml_portable_init_seed((unsigned long long)seed);
}

/* ----------------------------------------------------------------------
   Host buffer fills. Caller takes ownership; the downstream
   tape_create_param_<rank>_dtag frees the buffer after the copy
   (matches the F64 streamed-create contract).
   ---------------------------------------------------------------------- */
static double* fill_normal_buf(int n, double mean, double std) {
	double* buf = malloc((size_t)n * sizeof(double));
	idrisml_portable_fill_normal(buf, n, mean, std);
	return buf;
}

static double* fill_const_buf(int n, double value) {
	double* buf = malloc((size_t)n * sizeof(double));
	for (int i = 0; i < n; i++)
		buf[i] = value;
	return buf;
}

/* ----------------------------------------------------------------------
   Fused param creators. Each builds the host buffer then routes through
   the existing dtag dispatch (defined in training/dtype_dispatch.c),
   inheriting F64-direct / F32-real-storage / lingua-franca-retag-round
   behavior for free.
   ---------------------------------------------------------------------- */
extern TensorHandle tape_create_param_1d_dtag(int n, double* data, int dtag);
extern TensorHandle tape_create_param_2d_dtag(int rows, int cols, double* data, int dtag);
extern TensorHandle tape_create_param_3d_dtag(int d0, int d1, int d2, double* data, int dtag);
extern TensorHandle tape_create_param_4d_dtag(int d0, int d1, int d2, int d3, double* data,
                                              int dtag);

TensorHandle tape_create_param_1d_normal_dtag(int n, double mean, double std, int dtag) {
	return tape_create_param_1d_dtag(n, fill_normal_buf(n, mean, std), dtag);
}

TensorHandle tape_create_param_2d_normal_dtag(int rows, int cols, double mean, double std,
                                              int dtag) {
	return tape_create_param_2d_dtag(rows, cols, fill_normal_buf(rows * cols, mean, std), dtag);
}

TensorHandle tape_create_param_3d_normal_dtag(int d0, int d1, int d2, double mean, double std,
                                              int dtag) {
	return tape_create_param_3d_dtag(d0, d1, d2, fill_normal_buf(d0 * d1 * d2, mean, std), dtag);
}

TensorHandle tape_create_param_4d_normal_dtag(int d0, int d1, int d2, int d3, double mean,
                                              double std, int dtag) {
	return tape_create_param_4d_dtag(d0, d1, d2, d3, fill_normal_buf(d0 * d1 * d2 * d3, mean, std),
	                                 dtag);
}

TensorHandle tape_create_param_1d_const_dtag(int n, double value, int dtag) {
	return tape_create_param_1d_dtag(n, fill_const_buf(n, value), dtag);
}

TensorHandle tape_create_param_2d_const_dtag(int rows, int cols, double value, int dtag) {
	return tape_create_param_2d_dtag(rows, cols, fill_const_buf(rows * cols, value), dtag);
}

TensorHandle tape_create_param_3d_const_dtag(int d0, int d1, int d2, double value, int dtag) {
	return tape_create_param_3d_dtag(d0, d1, d2, fill_const_buf(d0 * d1 * d2, value), dtag);
}

TensorHandle tape_create_param_4d_const_dtag(int d0, int d1, int d2, int d3, double value,
                                             int dtag) {
	return tape_create_param_4d_dtag(d0, d1, d2, d3, fill_const_buf(d0 * d1 * d2 * d3, value),
	                                 dtag);
}
