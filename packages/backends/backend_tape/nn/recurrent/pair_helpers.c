/* nn/recurrent/pair_helpers.c — TensorPair accessor/release helpers.
 *
 * The `tensor_lstm_gates_pair` forward
 * returns a `TensorPair*` (arena-allocated), and Idris unpacks it via
 * these three thin accessors. They carry no autograd state — the
 * gradient flows through the LSTM_GATES / LSTM_GATES_CELL tape
 * entries created in the gates-pair forward.
 */

#include <stdlib.h>
#include "../../../backend.h"

TensorHandle tensor_pair_first(TensorPair* p) {
	return p->first;
}
TensorHandle tensor_pair_second(TensorPair* p) {
	return p->second;
}
void tensor_pair_free(TensorPair* p) {
	free(p);
}
