/* Per-call stream selection for the mlx backend's modular tree.
 *
 * `Tensor [..] (MlxExecutor MGpu)` and `Tensor [..] (MlxExecutor MCpu)` are
 * distinct types in the Idris-side type system; the C-side runtime
 * honours that distinction by running each op on the right mlx stream.
 * The `UserExecutorCore (MlxExecutor s)` instance derives an int stream tag
 * from `s` (0 = cpu, 1 = gpu) and threads it through the `_streamed`
 * FFI variants. Each streamed entry opens an `mx::StreamContext` from
 * the cached cpu_stream / gpu_stream, so the array's primitive ties
 * to the chosen stream and mlx's autograd (`mx::vjp`) automatically
 * replays the backward on the same stream.
 *
 * The legacy unstreamed entry points (the existing tensor_*_mlx
 * symbols, aliased to unsuffixed names when mlx is primary) keep
 * working as one-line trampolines: they invoke their _streamed
 * counterpart with default_stream_tag() so behaviour matches the
 * pre-streams world for callers that don't have a typed stream.
 *
 * Singletons are leaked at process exit (the cached mx::Stream
 * pointers are never destroyed); see the explanation in
 * backend_mlx.cpp's hot-path scalar constants section for why
 * mx::array destructors racing against mlx's internal statics at
 * shutdown is the actual hazard.
 */
#ifndef IDRISML_BACKEND_MLX_STREAM_H
#define IDRISML_BACKEND_MLX_STREAM_H

#include <cstdlib>
#include <cstring>
#include "tensor.h" /* for `namespace mx = mlx::core;` */

inline mx::Stream& cpu_stream() {
	static const mx::Stream* s = new mx::Stream(mx::default_stream(mx::Device(mx::Device::cpu)));
	return *const_cast<mx::Stream*>(s);
}

inline mx::Stream& gpu_stream() {
	static const mx::Stream* s = new mx::Stream(mx::default_stream(mx::Device(mx::Device::gpu)));
	return *const_cast<mx::Stream*>(s);
}

inline mx::Stream& stream_for_tag(int tag) {
	return tag == 1 ? gpu_stream() : cpu_stream();
}

inline int default_stream_tag() {
	static const int cached = []() {
		const char* env = std::getenv("MLX_DEVICE");
		return (env && (std::strcmp(env, "gpu") == 0 || std::strcmp(env, "metal") == 0)) ? 1 : 0;
	}();
	return cached;
}

#define WITH_STREAM(stream_tag) const mx::StreamContext _stream_guard(stream_for_tag(stream_tag))

#endif /* IDRISML_BACKEND_MLX_STREAM_H */
