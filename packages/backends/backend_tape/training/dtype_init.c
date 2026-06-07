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

/* ----------------------------------------------------------------------
   xoshiro256++ — process-global state.
   Reference: https://prng.di.unimi.it/xoshiro256plusplus.c (public domain).
   ---------------------------------------------------------------------- */
static uint64_t xs_state[4];
static int xs_seeded = 0;

/* Box-Muller cache state — defined here so tape_set_init_seed can reset
   it across re-seeds. */
static double bm_cached_z1 = 0.0;
static int bm_cached_has = 0;

static uint64_t splitmix64(uint64_t* x) {
	uint64_t z = (*x += 0x9E3779B97F4A7C15ULL);
	z = (z ^ (z >> 30)) * 0xBF58476D1CE4E5B9ULL;
	z = (z ^ (z >> 27)) * 0x94D049BB133111EBULL;
	return z ^ (z >> 31);
}

static inline uint64_t rotl(uint64_t x, int k) {
	return (x << k) | (x >> (64 - k));
}

static uint64_t xs_next(void) {
	const uint64_t r = rotl(xs_state[0] + xs_state[3], 23) + xs_state[0];
	const uint64_t t = xs_state[1] << 17;
	xs_state[2] ^= xs_state[0];
	xs_state[3] ^= xs_state[1];
	xs_state[1] ^= xs_state[2];
	xs_state[0] ^= xs_state[3];
	xs_state[2] ^= t;
	xs_state[3] = rotl(xs_state[3], 45);
	return r;
}

void tape_set_init_seed(uint64_t seed) {
	uint64_t sm = seed;
	xs_state[0] = splitmix64(&sm);
	xs_state[1] = splitmix64(&sm);
	xs_state[2] = splitmix64(&sm);
	xs_state[3] = splitmix64(&sm);
	xs_seeded = 1;
	/* Drop any cached Box-Muller half across re-seeds so the seed change
	   is visible immediately rather than after one stale sample. */
	bm_cached_has = 0;
}

static void ensure_seeded(void) {
	if (!xs_seeded) tape_set_init_seed(0);
}

/* ----------------------------------------------------------------------
   Uniform / normal samplers.
   ---------------------------------------------------------------------- */

/* Uniform in (0, 1). Top 53 bits of xs_next() → [0, 2^53); divide by
   2^53. Resamples on exact zero so Box-Muller's log() is always safe. */
static double uniform01(void) {
	for (;;) {
		uint64_t r = xs_next() >> 11;
		if (r != 0) return (double)r * (1.0 / 9007199254740992.0);
	}
}

/* Box-Muller with a one-sample cache (paired output halves the cost
   of trig + log per sample). Cache state declared above so
   tape_set_init_seed can reset it. */
static double normal01(void) {
	if (bm_cached_has) {
		bm_cached_has = 0;
		return bm_cached_z1;
	}
	double u1 = uniform01();
	double u2 = uniform01();
	double r = sqrt(-2.0 * log(u1));
	double th = 6.283185307179586 * u2;
	bm_cached_z1 = r * sin(th);
	bm_cached_has = 1;
	return r * cos(th);
}

/* ----------------------------------------------------------------------
   Host buffer fills. Caller takes ownership; the downstream
   tape_create_param_<rank>_dtag frees the buffer after the copy
   (matches the F64 streamed-create contract).
   ---------------------------------------------------------------------- */
static double* fill_normal_buf(int n, double mean, double std) {
	ensure_seeded();
	double* buf = malloc((size_t)n * sizeof(double));
	for (int i = 0; i < n; i++)
		buf[i] = mean + std * normal01();
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
