/* tensor_lstm_gates_pair for the torch backend.
 *
 * Splits a combined [4*o] tensor into four gates, applies the
 * sigmoid/tanh/sigmoid/sigmoid composition, then computes the cell
 * + hidden updates. Returns the pair via a TensorPair* that's tracked
 * in `all_pairs_torch` so free_intermediates can clean it up at the
 * next optimizer_step. */
#include "../../tensor.h"
#include "../../training/intermediates.h" /* all_pairs_torch */

extern "C" TensorPair* tensor_lstm_gates_pair(TensorHandle combined_h, TensorHandle prev_cell_h,
                                              int o) {
	auto& combined = *to_tensor(combined_h);
	auto& prev_cell = *to_tensor(prev_cell_h);
	auto chunks = combined.split(o);
	auto i_gate = torch::sigmoid(chunks[0]);
	auto f_gate = torch::sigmoid(chunks[1]);
	auto g_gate = torch::tanh(chunks[2]);
	auto o_gate = torch::sigmoid(chunks[3]);
	auto new_cell = f_gate * prev_cell + i_gate * g_gate;
	auto new_hidden = o_gate * torch::tanh(new_cell);
	auto* p = new TensorPair;
	p->first = from_tensor(std::move(new_hidden));
	p->second = from_tensor(std::move(new_cell));
	all_pairs_torch.push_back(p);
	return p;
}
