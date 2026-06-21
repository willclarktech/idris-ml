/* Fused param create + init for the mlx backend.
 *
 * Background: Idris's HF model state construction used to fill each
 * parameter tensor element-by-element on the host side via `traverse
 * normalSample` + `packDoubles` (per-element `prim__setDouble` FFI),
 * costing tens of minutes for Llama-class models. The torch backend's
 * dtype_init.cpp fixed this for libtorch via in-place `torch::nn::init`
 * kernels; this file is the mlx peer.
 *
 * mlx's functional-array model means we don't need in-place init —
 * `mx::random::normal(shape, dtype, loc, scale)` and `mx::full(shape,
 * value, dtype)` return a fresh array initialised in one mlx kernel
 * call. The work happens inside mlx's compute graph (CPU or Metal
 * stream, selected via WITH_STREAM); no host loop, no per-element FFI.
 *
 * Unlike torch — where the shared `dtype_streamed.c` trampolines
 * dispatch through `g_active_port` — mlx isn't in
 * SHARED_BACKENDS_dtype_streamed (its existing
 * `tensor_create_param_*_streamed` symbols are defined directly in
 * `training/dtype_dispatch.cpp`). This file follows the same direct
 * pattern: each `tensor_create_param_*_<init>_streamed` symbol is
 * extern "C", dtag-dispatches on f32/f64, and constructs the Tensor +
 * appends OP_CONST exactly like the existing
 * `tensor_create_param_*_<dt>_mlx_streamed` impls do.
 *
 * Seed: `mx::random::seed(seed)` reseeds mlx's KeySequence used by
 * `random::normal` / `random::uniform`. `tensor_set_init_seed_streamed`
 * forwards directly; the stream_tag is ignored (mlx's RNG is per-thread,
 * not per-stream).
 */

#include "../tensor.h"
#include "../tape.h"
#include "../stream.h"
#include "../../backend.h"
#include <mlx/random.h>
#include <mlx/ops.h>
#include <cstdio>
#include <cstdlib>

namespace {

mx::Dtype dt_for_dtag(const char* sym, int dtag) {
	// Each case returns a distinct mx::Dtype; clang-tidy branch-clone FP.
	// NOLINTNEXTLINE(bugprone-branch-clone)
	switch (dtag) {
	case 13:
		return mx::float16;
	case 14:
		return mx::float32;
	case 15:
		return mx::float64;
	case 17:
		return mx::bfloat16;
	default:
		fprintf(stderr,
		        "[mlx backend] %s called with dtag=%d. This randn/const "
		        "init path supports floating dtags only (f16=13, f32=14, "
		        "f64=15, bf16=17). I32 (dtag=10) storage is wired for "
		        "bulk creation + serialization, but randn-initialised "
		        "I32 params don't have semantics; construct I32 tensors "
		        "via the bulk path instead, or build with BACKEND=torch "
		        "for the wider dtype surface.\n",
		        sym, dtag);
		abort();
	}
}

TensorHandle wrap_param(mx::array arr) {
	auto* t = new Tensor(std::move(arr), /*requires_grad=*/true);
	tape_append(OP_CONST, t, nullptr, nullptr, 0);
	return (TensorHandle)t;
}

} // namespace

/* ---- Normal(mean, std) initialisation ---- */
extern "C" TensorHandle tensor_create_param_1d_normal_streamed(int n, double mean, double std,
                                                               int stream_tag, int dtag) {
	WITH_STREAM(stream_tag);
	auto dt = dt_for_dtag("tensor_create_param_1d_normal_streamed", dtag);
	return wrap_param(mx::random::normal(mx::Shape{n}, dt, (float)mean, (float)std));
}

extern "C" TensorHandle tensor_create_param_2d_normal_streamed(int rows, int cols, double mean,
                                                               double std, int stream_tag,
                                                               int dtag) {
	WITH_STREAM(stream_tag);
	auto dt = dt_for_dtag("tensor_create_param_2d_normal_streamed", dtag);
	return wrap_param(mx::random::normal(mx::Shape{rows, cols}, dt, (float)mean, (float)std));
}

extern "C" TensorHandle tensor_create_param_3d_normal_streamed(int d0, int d1, int d2, double mean,
                                                               double std, int stream_tag,
                                                               int dtag) {
	WITH_STREAM(stream_tag);
	auto dt = dt_for_dtag("tensor_create_param_3d_normal_streamed", dtag);
	return wrap_param(mx::random::normal(mx::Shape{d0, d1, d2}, dt, (float)mean, (float)std));
}

extern "C" TensorHandle tensor_create_param_4d_normal_streamed(int d0, int d1, int d2, int d3,
                                                               double mean, double std,
                                                               int stream_tag, int dtag) {
	WITH_STREAM(stream_tag);
	auto dt = dt_for_dtag("tensor_create_param_4d_normal_streamed", dtag);
	return wrap_param(mx::random::normal(mx::Shape{d0, d1, d2, d3}, dt, (float)mean, (float)std));
}

/* ---- Constant fill ---- */
extern "C" TensorHandle tensor_create_param_1d_const_streamed(int n, double value, int stream_tag,
                                                              int dtag) {
	WITH_STREAM(stream_tag);
	auto dt = dt_for_dtag("tensor_create_param_1d_const_streamed", dtag);
	return wrap_param(mx::full(mx::Shape{n}, value, dt));
}

extern "C" TensorHandle tensor_create_param_2d_const_streamed(int rows, int cols, double value,
                                                              int stream_tag, int dtag) {
	WITH_STREAM(stream_tag);
	auto dt = dt_for_dtag("tensor_create_param_2d_const_streamed", dtag);
	return wrap_param(mx::full(mx::Shape{rows, cols}, value, dt));
}

extern "C" TensorHandle tensor_create_param_3d_const_streamed(int d0, int d1, int d2, double value,
                                                              int stream_tag, int dtag) {
	WITH_STREAM(stream_tag);
	auto dt = dt_for_dtag("tensor_create_param_3d_const_streamed", dtag);
	return wrap_param(mx::full(mx::Shape{d0, d1, d2}, value, dt));
}

extern "C" TensorHandle tensor_create_param_4d_const_streamed(int d0, int d1, int d2, int d3,
                                                              double value, int stream_tag,
                                                              int dtag) {
	WITH_STREAM(stream_tag);
	auto dt = dt_for_dtag("tensor_create_param_4d_const_streamed", dtag);
	return wrap_param(mx::full(mx::Shape{d0, d1, d2, d3}, value, dt));
}

/* ---- Init RNG seed ----
   `mx::random::seed` reseeds the default KeySequence used by
   `random::normal` / `random::uniform`. Stream-agnostic; the RNG state
   is per-thread (see mlx/random.h KeySequence::default_), not per-stream. */
extern "C" void tensor_set_init_seed_streamed(unsigned long long seed, int stream_tag) {
	(void)stream_tag;
	mx::random::seed((uint64_t)seed);
}
