/* NTM-specific compositions + in-place ops for the torch backend.
 *
 * `tensor_lstm_gates` is the void-output variant used by the tape
 * backend's gradient-test harness: same equations as
 * tensor_lstm_gates_pair, but writes results through out pointers
 * instead of returning a TensorPair. (The _pair variant — which
 * tracks the result in all_pairs — stays in the monolith with the
 * intermediates list it depends on; that moves out alongside
 * intermediates in a follow-up commit.)
 *
 * `tensor_subtract_scalar_inplace` is the polyak-blend in-place
 * subtract used by SAC's target-network update. Wrapped in
 * NoGradGuard so the in-place mutation doesn't taint the autograd
 * graph. */
#include "../tensor.h"

extern "C" void tensor_lstm_gates(TensorHandle combined_h, TensorHandle prev_cell_h, int o,
                                  TensorHandle* out_h, TensorHandle* out_c) {
	auto& combined = *to_tensor(combined_h);
	auto& prev_cell = *to_tensor(prev_cell_h);

	auto chunks = combined.split(o);
	auto i_gate = torch::sigmoid(chunks[0]);
	auto f_gate = torch::sigmoid(chunks[1]);
	auto g_gate = torch::tanh(chunks[2]);
	auto o_gate = torch::sigmoid(chunks[3]);

	auto new_cell = f_gate * prev_cell + i_gate * g_gate;
	auto new_hidden = o_gate * torch::tanh(new_cell);

	*out_h = from_tensor(std::move(new_hidden));
	*out_c = from_tensor(std::move(new_cell));
}

extern "C" TensorHandle tensor_subtract_scalar_inplace(TensorHandle h, double val) {
	torch::NoGradGuard no_grad;
	to_tensor(h)->sub_(val);
	return h;
}
