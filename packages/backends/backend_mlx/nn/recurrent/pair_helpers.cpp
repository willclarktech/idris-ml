/* TensorPair accessor + release helpers for the mlx backend.
 *
 * The pair struct is returned by tensor_lstm_gates_pair and tracked
 * in `all_pairs` (still in the monolith — Phase 6e moves it out
 * alongside the intermediates list). The helpers carry no autograd
 * state — the gradient flows through the tape entries on `first` /
 * `second`, not through the pair itself.
 *
 * The streamed variants are no-ops with respect to the stream (the
 * access is a plain pointer read), but the WITH_STREAM wrapper keeps
 * the per-call ABI shape uniform with every other mlx op. */
#include "../../tensor.h"
#include "../../stream.h"
#include <cstdlib>

extern "C" TensorHandle tensor_pair_first_mlx_streamed(TensorPair* p, int stream_tag) {
	WITH_STREAM(stream_tag);
	return p->first;
}

extern "C" TensorHandle tensor_pair_first(TensorPair* p) {
	return tensor_pair_first_mlx_streamed(p, default_stream_tag());
}

extern "C" TensorHandle tensor_pair_second_mlx_streamed(TensorPair* p, int stream_tag) {
	WITH_STREAM(stream_tag);
	return p->second;
}

extern "C" TensorHandle tensor_pair_second(TensorPair* p) {
	return tensor_pair_second_mlx_streamed(p, default_stream_tag());
}

extern "C" void tensor_pair_free(TensorPair* p) {
	if (p != nullptr) free(p);
}
