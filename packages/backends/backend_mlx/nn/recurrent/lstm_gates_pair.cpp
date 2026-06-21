/* tensor_lstm_gates_pair for the mlx backend.
 *
 * Decomposes the LSTM gate equations into primitives — each tensor_*
 * sub-op records its own tape entry so backward flows through replay.
 * Returns the (hidden, cell) pair via a TensorPair* (Idris unpacks via
 * tensor_pair_first / tensor_pair_second).
 *
 * The pair struct itself is tracked in `all_pairs` (currently in the
 * monolith — Phase 6e moves the intermediates list out alongside the
 * Tensor-tracking machinery) so tape_reset can free it. */
#include "../../tensor.h"
#include "../../tape.h"
#include "../../stream.h"
#include <cstdlib>

/* all_pairs lives in the monolith and is referenced here while the
   intermediates plumbing waits on Phase 6e. */

extern "C" TensorHandle tensor_narrow_mlx_streamed(TensorHandle h, int dim, int start, int len,
                                                   int stream_tag);
extern "C" TensorHandle tensor_sigmoid_mlx_streamed(TensorHandle h, int stream_tag);
extern "C" TensorHandle tensor_tanh_mlx_streamed(TensorHandle h, int stream_tag);
extern "C" TensorHandle tensor_mul_mlx_streamed(TensorHandle ha, TensorHandle hb, int stream_tag);
extern "C" TensorHandle tensor_add_mlx_streamed(TensorHandle ha, TensorHandle hb, int stream_tag);

extern "C" TensorPair* tensor_lstm_gates_pair_mlx_streamed(TensorHandle hcombined,
                                                           TensorHandle hprev_cell, int o,
                                                           int stream_tag) {
	WITH_STREAM(stream_tag);
	/* Split combined [4*o] into 4 gates */
	TensorHandle ig_raw = tensor_narrow_mlx_streamed(hcombined, 0, 0, o, stream_tag);
	TensorHandle fg_raw = tensor_narrow_mlx_streamed(hcombined, 0, o, o, stream_tag);
	TensorHandle gg_raw = tensor_narrow_mlx_streamed(hcombined, 0, 2 * o, o, stream_tag);
	TensorHandle og_raw = tensor_narrow_mlx_streamed(hcombined, 0, 3 * o, o, stream_tag);
	TensorHandle ig = tensor_sigmoid_mlx_streamed(ig_raw, stream_tag);
	TensorHandle fg = tensor_sigmoid_mlx_streamed(fg_raw, stream_tag);
	TensorHandle gg = tensor_tanh_mlx_streamed(gg_raw, stream_tag);
	TensorHandle og = tensor_sigmoid_mlx_streamed(og_raw, stream_tag);
	/* c_t = f_t ⊙ c_{t-1} + i_t ⊙ g_t */
	TensorHandle fc = tensor_mul_mlx_streamed(fg, hprev_cell, stream_tag);
	TensorHandle ig_gg = tensor_mul_mlx_streamed(ig, gg, stream_tag);
	TensorHandle new_cell = tensor_add_mlx_streamed(fc, ig_gg, stream_tag);
	/* h_t = o_t ⊙ tanh(c_t) */
	TensorHandle tanh_cell = tensor_tanh_mlx_streamed(new_cell, stream_tag);
	TensorHandle new_hidden = tensor_mul_mlx_streamed(og, tanh_cell, stream_tag);

	auto* pair = (TensorPair*)malloc(sizeof(TensorPair));
	pair->first = new_hidden;
	pair->second = new_cell;
	all_pairs.push_back(pair);
	return pair;
}

extern "C" TensorPair* tensor_lstm_gates_pair(TensorHandle hcombined, TensorHandle hprev_cell,
                                              int o) {
	return tensor_lstm_gates_pair_mlx_streamed(hcombined, hprev_cell, o, default_stream_tag());
}
