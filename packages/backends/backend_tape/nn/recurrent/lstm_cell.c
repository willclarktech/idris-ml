/* nn/recurrent/lstm_cell.c — torch-style nn.LSTMCell stub (forward only).
 *
 * Phase 1c.7.a (mechanical). Current Idris code uses
 * `tensor_lstm_gates_pair` for the LSTM forward, so this entry point
 * is a passthrough that returns clones of (hx, cx). Retained for ABI
 * compatibility with the backend.h declaration.
 */

#include "../../../backend.h"

void tensor_lstm_cell(
    TensorHandle input, TensorHandle hx, TensorHandle cx,
    TensorHandle w_ih, TensorHandle w_hh,
    TensorHandle b_ih, TensorHandle b_hh,
    TensorHandle* out_h, TensorHandle* out_c)
{
    (void)input; (void)w_ih; (void)w_hh; (void)b_ih; (void)b_hh;
    *out_h = tensor_clone(hx);
    *out_c = tensor_clone(cx);
}
