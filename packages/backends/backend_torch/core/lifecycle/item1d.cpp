/* tensor_item_1d for the torch backend.
 *
 * Flat-buffer semantics matching tape (tape_load_d) and mlx
 * (mx_read_double on the flattened data buffer): `idx` is a flat
 * offset into the data layout, not a first-dim index. Required so
 * Idris's `tvecToVector` (Backprop.idr) and backend-agnostic tests
 * see consistent indexing across backends. */
#include "../../tensor.h"

extern "C" double tensor_item_1d(TensorHandle h, int idx) {
    return to_tensor(h)->flatten()[idx].cpu().item<double>();
}
