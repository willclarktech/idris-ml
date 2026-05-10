/* Criterion suite for tape `tensor_lstm_gates_pair` (Phase 1c.7.d).
 *
 * Tape entries:
 *   OP_LSTM_GATES      — hidden output → backward propagates d_h
 *   OP_LSTM_GATES_CELL — cell output   → backward propagates d_cell
 *
 * Both arms share the same LstmGatesMetaLocal cache and accumulate
 * into the same combined[4*o]/prev_cell gradient slots. The test
 * sums hidden + cell so dh = dcell = 1 — exercises BOTH backward
 * arms, so withholding either TAPE_REGISTER_OP zeroes a portion of
 * the expected grad and the asserts fire.
 *
 * RED before Phase 1c.7.d enable: with TAPE_REGISTER_OPs withheld
 * and the monolith arms stripped, the dispatch table has no entries
 * for either OP_LSTM_GATES or OP_LSTM_GATES_CELL, so backward leaves
 * all combined/prev grads at zero → first cr_assert_float_eq fails.
 */

#include <criterion/criterion.h>
#include <math.h>
#include "../../../../backend.h"

Test(tape_nn_recurrent_lstm_gates_pair, backward_grads_both_arms) {
    param_clear();
    int o = 1;
    /* combined raws = [0, 0, 0, 0] → i = f = g_sigmoid_eq = o = 0.5,
       g = tanh(0) = 0. prev_cell = 1.0. */
    double comb_data[4] = {0.0, 0.0, 0.0, 0.0};
    double prev_data[1] = {1.0};
    int sh4[1] = {4 * o};
    int sh1[1] = {o};
    TensorHandle combined  = tensor_create(comb_data, sh4, 1, 1);
    TensorHandle prev_cell = tensor_create(prev_data, sh1, 1, 1);
    param_register("combined",  combined);
    param_register("prev_cell", prev_cell);

    TensorPair* p = tensor_lstm_gates_pair(combined, prev_cell, o);
    TensorHandle h    = tensor_pair_first(p);
    TensorHandle cell = tensor_pair_second(p);

    /* Forward expectations (i=f=o=0.5, g=0, prev=1):
       cell = f*prev + i*g = 0.5*1 + 0.5*0 = 0.5
       h    = o * tanh(cell) = 0.5 * tanh(0.5) */
    double cell_v = 0.5;
    double tanhC  = tanh(0.5);
    cr_assert_float_eq(tensor_item_1d(cell, 0), 0.5, 1e-12);
    cr_assert_float_eq(tensor_item_1d(h, 0),    0.5 * tanhC, 1e-12);

    /* dh = dcell = 1 (loss = sum(h) + sum(cell)) */
    TensorHandle loss = tensor_add(tensor_sum(h), tensor_sum(cell));
    tensor_backward(loss);

    /* Hidden-arm gradient (with d_h=1):
         d_o_raw_h = 1 * tanh(0.5) * 0.5*(1-0.5) = 0.25 * tanh(0.5)
         d_cell_h  = 1 * 0.5 * (1 - tanh(0.5)^2)
       Cell-arm gradient (with d_cell=1) shares grad slots additively.
       Net d_cell into the gate-derivative computation = d_cell_h + 1. */
    double d_o_raw_h  = tanhC * 0.25;
    double d_cell_net = 0.5 * (1.0 - tanhC * tanhC) + 1.0;
    /* fG and iG/gG entries (with prev=1, g=0, i=0.5, f=0.5):
         d_f_raw = d_cell_net * 1 * 0.5*(1-0.5)
         d_i_raw = d_cell_net * 0   * 0.5*(1-0.5)  (g=0 → zero contribution)
         d_g_raw = d_cell_net * 0.5 * (1-0)
       Output gate raw comes only from hidden arm. */
    double d_i_raw = d_cell_net * 0.0 * 0.25;       /* 0 */
    double d_f_raw = d_cell_net * 1.0 * 0.25;
    double d_g_raw = d_cell_net * 0.5 * 1.0;
    double d_o_raw = d_o_raw_h;
    double d_prev  = d_cell_net * 0.5;              /* d_cell * fG */

    /* combined layout: [i_raw, f_raw, g_raw, o_raw], param 0 */
    cr_assert_float_eq(param_grad_item_at(0, 0), d_i_raw, 1e-12, "i_raw");
    cr_assert_float_eq(param_grad_item_at(0, 1), d_f_raw, 1e-12, "f_raw");
    cr_assert_float_eq(param_grad_item_at(0, 2), d_g_raw, 1e-12, "g_raw");
    cr_assert_float_eq(param_grad_item_at(0, 3), d_o_raw, 1e-12, "o_raw");
    /* prev_cell, param 1 */
    cr_assert_float_eq(param_grad_item_at(1, 0), d_prev, 1e-12, "prev");
}
