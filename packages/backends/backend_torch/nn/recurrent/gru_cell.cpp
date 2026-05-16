/* tensor_gru_cell for the torch backend.
 *
 * Reproduces nn.GRU's gate equations from already-computed
 * ih = W_ih @ x + b_ih and hh = W_hh @ h + b_hh — the caller
 * (Idris GRU layer) precomputes those so the backend ABI doesn't
 * need to thread per-gate weights. */
#include "../../tensor.h"

extern "C" TensorHandle tensor_gru_cell(TensorHandle hih, TensorHandle hhh,
                                        TensorHandle hprev, int o) {
    auto& ih = *to_tensor(hih);
    auto& hh = *to_tensor(hhh);
    auto& prev = *to_tensor(hprev);
    auto z = torch::sigmoid(ih.slice(0, 0, o) + hh.slice(0, 0, o));
    auto r = torch::sigmoid(ih.slice(0, o, 2*o) + hh.slice(0, o, 2*o));
    auto n = torch::tanh(ih.slice(0, 2*o, 3*o) + r * hh.slice(0, 2*o, 3*o));
    auto h_new = (1.0 - z) * n + z * prev;
    return from_tensor(h_new);
}
