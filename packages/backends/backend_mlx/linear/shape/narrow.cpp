/* tensor_narrow for the mlx backend.
 *
 * Forward: build start/stop bound vectors for `mx::slice` so the
 * narrowed axis is `dim`, all other axes are full-span. The historical
 * "flatten then 1D-slice" path was a silent shape lie when called with
 * dim > 0 — pinned by `linear_shape_narrow::axis1_correctness_rank2`
 * in the common-backend test suite.
 *
 * Replay (mlx's deferred autograd graph): TapeEntry carries only one
 * scalar slot (`start`), no `dim`. We recover `dim` at replay time by
 * comparing the input array shape to the result tensor's cached shape
 * — exactly one axis differs, and that's the one that was narrowed.
 * Same trick the tape backend's backward path uses.
 */
#include "../../tensor.h"
#include "../../tape.h"
#include "../../stream.h"
#include "../../training/autograd/op_dispatch.h"
#include "../../precision.h"

static std::pair<mx::Shape, mx::Shape> multi_axis_slice_bounds(const mx::Shape& shape, int dim,
                                                               int start, int len) {
	std::vector<int> start_v((size_t)shape.size(), 0);
	std::vector<int> stop_v(shape.begin(), shape.end());
	start_v[(size_t)dim] = start;
	stop_v[(size_t)dim] = start + len;
	return {mx::Shape(start_v.begin(), start_v.end()), mx::Shape(stop_v.begin(), stop_v.end())};
}

extern "C" TensorHandle tensor_narrow_mlx_streamed(TensorHandle h, int dim, int start, int len,
                                                   int stream_tag) {
	WITH_STREAM(stream_tag);
	auto t = (Tensor*)h;
	auto [s_start, s_stop] = multi_axis_slice_bounds(t->data.shape(), dim, start, len);
	auto sliced = mx::slice(t->data, s_start, s_stop);
	auto r = new Tensor(sliced, t->requires_grad);
	if (t->requires_grad) tape_append(OP_NARROW, r, t, nullptr, (double)start);
	return (TensorHandle)r;
}

extern "C" TensorHandle tensor_narrow(TensorHandle h, int dim, int start, int len) {
	return tensor_narrow_mlx_streamed(h, dim, start, len, default_stream_tag());
}

static void mlx_replay_narrow(std::vector<mx::array>& pool, TapeEntry& e) {
	int out = e.result->pool_idx;
	auto a = e.arg1 ? pool[e.arg1->pool_idx] : kF32_ZERO();
	int start = (int)e.scalar_arg;
	/* Infer the narrowed axis from input vs cached-result shapes. */
	auto in_shape = a.shape();
	auto out_shape = e.result->data.shape();
	int dim = 0;
	for (size_t i = 0; i < in_shape.size(); i++) {
		if (i >= out_shape.size() || in_shape[i] != out_shape[i]) {
			dim = (int)i;
			break;
		}
	}
	int len = (int)out_shape[dim];
	auto [s_start, s_stop] = multi_axis_slice_bounds(in_shape, dim, start, len);
	pool[out] = mx::slice(a, s_start, s_stop);
}
MLX_REGISTER_REPLAY(OP_NARROW, mlx_replay_narrow)
