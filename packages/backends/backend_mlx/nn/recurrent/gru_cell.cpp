/* tensor_gru_cell for the mlx backend.
 *
 * nn.GRU equation. ih = W_ih @ x + b_ih, hh = W_hh @ h + b_hh (caller
 * precomputes both). The replay closure handles backward via vjp;
 * GruCellReplayMeta carries `o` and prev's pool index so backward can
 * find the third input. */
#include "../../tensor.h"
#include "../../tape.h"
#include "../../stream.h"
#include "../../precision.h"
#include "../../training/autograd/op_dispatch.h"

extern "C" TensorHandle tensor_gru_cell_mlx_streamed(TensorHandle hih, TensorHandle hhh,
                                                     TensorHandle hprev, int o, int stream_tag) {
	WITH_STREAM(stream_tag);
	auto ih = (Tensor*)hih;
	auto hh = (Tensor*)hhh;
	auto prev = (Tensor*)hprev;
	auto ih_z = mx::slice(ih->data, {0}, {o});
	auto ih_r = mx::slice(ih->data, {o}, {2 * o});
	auto ih_n = mx::slice(ih->data, {2 * o}, {3 * o});
	auto hh_z = mx::slice(hh->data, {0}, {o});
	auto hh_r = mx::slice(hh->data, {o}, {2 * o});
	auto hh_n = mx::slice(hh->data, {2 * o}, {3 * o});
	auto z = mx::sigmoid(mx::add(ih_z, hh_z));
	auto r_gate = mx::sigmoid(mx::add(ih_r, hh_r));
	auto n = mx::tanh(mx::add(ih_n, mx::multiply(r_gate, hh_n)));
	auto result =
	    mx::add(mx::multiply(mx::subtract(one_like(z), z), n), mx::multiply(z, prev->data));

	bool rg = ih->requires_grad || hh->requires_grad || prev->requires_grad;
	auto r = new Tensor(result, rg);
	if (rg) {
		int idx = tape_append(OP_GRU_CELL, r, ih, hh, 0);
		auto meta = new GruCellReplayMeta();
		meta->o = o;
		meta->prev_pool_idx = prev->pool_idx;
		tape[idx].meta = meta;
	}
	return (TensorHandle)r;
}

extern "C" TensorHandle tensor_gru_cell(TensorHandle hih, TensorHandle hhh, TensorHandle hprev,
                                        int o) {
	return tensor_gru_cell_mlx_streamed(hih, hhh, hprev, o, default_stream_tag());
}

static void mlx_replay_gru_cell(std::vector<mx::array>& pool, TapeEntry& e) {
	int out = e.result->pool_idx;
	[[maybe_unused]] auto a = e.arg1 ? pool[e.arg1->pool_idx] : kF32_ZERO();
	[[maybe_unused]] auto b = e.arg2 ? pool[e.arg2->pool_idx] : kF32_ZERO();
	/* nn.GRU: a=ih, b=hh, prev via meta->prev_pool_idx.
	                     z = sigmoid(ih_z + hh_z), r = sigmoid(ih_r + hh_r)
	                     n = tanh(ih_n + r * hh_n)
	                     h' = (1-z)*n + z*prev                                 */
	auto meta = (GruCellReplayMeta*)e.meta;
	int oo = meta->o;
	auto prev = pool[meta->prev_pool_idx];
	auto ih_z = mx::slice(a, {0}, {oo});
	auto ih_r = mx::slice(a, {oo}, {2 * oo});
	auto ih_n = mx::slice(a, {2 * oo}, {3 * oo});
	auto hh_z = mx::slice(b, {0}, {oo});
	auto hh_r = mx::slice(b, {oo}, {2 * oo});
	auto hh_n = mx::slice(b, {2 * oo}, {3 * oo});
	auto z = mx::sigmoid(mx::add(ih_z, hh_z));
	auto r_gate = mx::sigmoid(mx::add(ih_r, hh_r));
	auto n = mx::tanh(mx::add(ih_n, mx::multiply(r_gate, hh_n)));
	pool[out] = mx::add(mx::multiply(mx::subtract(one_like(z), z), n), mx::multiply(z, prev));
}
MLX_REGISTER_REPLAY(OP_GRU_CELL, mlx_replay_gru_cell)
