/* TensorPair accessor + release helpers for the torch backend.
 *
 * `tensor_lstm_gates_pair` (still in the monolith — needs `all_pairs`
 * which Phase 6e formalizes alongside the intermediates list) returns
 * a TensorPair*; Idris unpacks it via these three thin accessors. The
 * gradient flows through the autograd graph carried by `first` and
 * `second`, not through the pair itself. */
#include "../../tensor.h"

extern "C" TensorHandle tensor_pair_first(TensorPair* p)  { return p->first; }
extern "C" TensorHandle tensor_pair_second(TensorPair* p) { return p->second; }
extern "C" void         tensor_pair_free(TensorPair* p)   { delete p; }
