/* tensor_narrow for the torch backend.
 *
 * Matches the tape backend: always returns 1D (flattened input narrowed
 * along axis 0). The `dim` arg is accepted but ignored — the type-safe
 * Idris surface only narrows the leading axis. */
#include "../../tensor.h"

extern "C" TensorHandle tensor_narrow(TensorHandle h, int dim, int start, int len) {
    (void)dim;
    auto t = to_tensor(h)->flatten().narrow(0, start, len).contiguous();
    return from_tensor(std::move(t));
}
