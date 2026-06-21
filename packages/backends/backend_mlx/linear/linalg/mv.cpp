/* tensor_mv for the mlx backend (matrix-vector product).
 *
 * mlx has no native mv — reshape the vec to [n,1], matmul, then
 * reshape result back to [m]. The reshapes are no-ops on the data
 * layout, and OP_MV's backward closes over the matrix/vector shapes
 * to undo them. */
#include "../../tensor.h"
#include "../../tape.h"
#include "../../stream.h"
#include "../../training/autograd/op_dispatch.h"
#include "../../precision.h"

extern "C" TensorHandle tensor_mv_mlx_streamed(TensorHandle hmat, TensorHandle hvec,
                                               int stream_tag) {
	WITH_STREAM(stream_tag);
	auto* mat = (Tensor*)hmat;
	auto* vec = (Tensor*)hvec;
	int const n = (int)vec->data.size();
	int const m_size = (int)mat->data.shape(0);
	auto vec_col = mx::reshape(vec->data, {n, 1});
	auto result_col = mx::matmul(mat->data, vec_col);
	auto result = mx::reshape(result_col, {m_size});
	bool const rg = mat->requires_grad || vec->requires_grad;
	auto* r = new Tensor(result, rg);
	if (rg) tape_append(OP_MV, r, mat, vec, 0);
	return (TensorHandle)r;
}

extern "C" TensorHandle tensor_mv(TensorHandle hmat, TensorHandle hvec) {
	return tensor_mv_mlx_streamed(hmat, hvec, default_stream_tag());
}

static void mlx_replay_mv(std::vector<mx::array>& pool, TapeEntry& e) {
	int const out = e.result->pool_idx;
	[[maybe_unused]] auto a = (e.arg1 != nullptr) ? pool[e.arg1->pool_idx] : kF32_ZERO();
	[[maybe_unused]] auto b = (e.arg2 != nullptr) ? pool[e.arg2->pool_idx] : kF32_ZERO();
	auto col = mx::reshape(b, {(int)b.size(), 1});
	pool[out] = mx::reshape(mx::matmul(a, col), {(int)a.shape(0)});
}
MLX_REGISTER_REPLAY(OP_MV, mlx_replay_mv)
