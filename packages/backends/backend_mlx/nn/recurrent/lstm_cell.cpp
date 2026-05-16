/* tensor_lstm_cell + tensor_lstm_gates (1D variants) for the mlx
 * backend. lstm_cell composes mv + add to build `combined`, then
 * dispatches to lstm_gates_pair for the gate split + cell update.
 * Each sub-op records its own tape entry; backward flows automatically. */
#include "../../tensor.h"

extern "C" TensorHandle tensor_mv(TensorHandle hmat, TensorHandle hvec);
extern "C" TensorHandle tensor_add(TensorHandle ha, TensorHandle hb);
extern "C" TensorPair*  tensor_lstm_gates_pair(TensorHandle hcombined, TensorHandle hprev_cell, int o);

extern "C" void tensor_lstm_gates(TensorHandle combined, TensorHandle prev_cell, int o,
                                  TensorHandle* out_h, TensorHandle* out_c) {
    /* Void-output variant: same decomposition as tensor_lstm_gates_pair, but
       returns through out_h/out_c pointers instead of a TensorPair.
       The pair struct itself is tracked in all_pairs and cleaned up at
       tape_reset. The caller doesn't own it; the outputs are the standalone
       Tensor handles inside. */
    TensorPair* p = tensor_lstm_gates_pair(combined, prev_cell, o);
    *out_h = p->first;
    *out_c = p->second;
}

extern "C" void tensor_lstm_cell(TensorHandle input, TensorHandle hx, TensorHandle cx,
                                 TensorHandle w_ih, TensorHandle w_hh,
                                 TensorHandle b_ih, TensorHandle b_hh,
                                 TensorHandle* out_h, TensorHandle* out_c) {
    int hidden = (int)((Tensor*)cx)->data.size();
    TensorHandle gi   = tensor_mv(w_ih, input);
    TensorHandle gi_b = tensor_add(gi, b_ih);
    TensorHandle gh   = tensor_mv(w_hh, hx);
    TensorHandle gh_b = tensor_add(gh, b_hh);
    TensorHandle combined = tensor_add(gi_b, gh_b);
    tensor_lstm_gates(combined, cx, hidden, out_h, out_c);
}
