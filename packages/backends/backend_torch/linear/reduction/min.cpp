/* tensor_min for the torch backend.
 *
 * libtorch's `min()` participates in autograd if used in a grad context;
 * detach to match the legacy unsuffixed scalar-extract semantics. */
#include "../../tensor.h"

extern "C" TensorHandle tensor_min(TensorHandle h) {
    return from_tensor(to_tensor(h)->min().detach());
}
