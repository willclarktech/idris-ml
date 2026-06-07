/* tensor_tile_2d for the mlx backend.
 *
 * Non-grad inputs (e.g. cached positional encoding) are eagerly
 * materialized so mx::vjp sees a leaf instead of tracing back through
 * the tile op — saves 10-15% wall on small-model shapes. Grad inputs
 * stay lazy so tape replay can reconstruct the graph. */
#include "../../tensor.h"
#include "../../tape.h"
#include "../../stream.h"
#include "../../training/autograd/op_dispatch.h"
#include "../../precision.h"

extern "C" TensorHandle tensor_tile_2d_mlx_streamed(TensorHandle h, int rep0, int rep1,
                                                    int stream_tag) {
	WITH_STREAM(stream_tag);
	auto t = (Tensor*)h;
	auto tiled = mx::tile(t->data, {rep0, rep1});
	if (!t->requires_grad) {
		mx::eval(tiled);
	}
	auto r = new Tensor(tiled, t->requires_grad);
	if (t->requires_grad) {
		int* meta = (int*)std::malloc(sizeof(int) * 2);
		meta[0] = rep0;
		meta[1] = rep1;
		int idx = tape_append(OP_TILE_2D, r, t, nullptr, 0);
		if (idx >= 0)
			tape[idx].meta = meta;
		else
			std::free(meta);
	}
	return (TensorHandle)r;
}

extern "C" TensorHandle tensor_tile_2d(TensorHandle h, int rep0, int rep1) {
	return tensor_tile_2d_mlx_streamed(h, rep0, rep1, default_stream_tag());
}

static void mlx_replay_tile_2d(std::vector<mx::array>& pool, TapeEntry& e) {
	int out = e.result->pool_idx;
	[[maybe_unused]] auto a = e.arg1 ? pool[e.arg1->pool_idx] : kF32_ZERO();
	[[maybe_unused]] auto b = e.arg2 ? pool[e.arg2->pool_idx] : kF32_ZERO();
	int* reps = (int*)e.meta;
	pool[out] = mx::tile(a, {reps[0], reps[1]});
}
MLX_REGISTER_REPLAY(OP_TILE_2D, mlx_replay_tile_2d)
