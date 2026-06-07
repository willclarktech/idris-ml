/* tensor_expand_mask for the torch backend. Expands a [m, n] mask to
 * [B, m, n] for broadcast against a batched attention matrix.
 *
 * No `.contiguous()` — `expand` returns a stride-0 view, and every
 * downstream consumer of the mask (`masked_fill_`, `+`, `bmm`'s
 * broadcasting path) accepts strided tensors. The previous
 * `.contiguous()` materialized a B× duplication into MPS memory once
 * per attention layer, costing an allocation + Metal command buffer
 * submission with no callers reading raw bytes. Mirrors the
 * `ea90238` transpose fix and the matching narrow.cpp removal. */
#include "../../tensor.h"

extern "C" TensorHandle tensor_expand_mask(TensorHandle hmask, int B) {
	return from_tensor(to_tensor(hmask)->unsqueeze(0).expand({(int64_t)B, -1, -1}));
}
