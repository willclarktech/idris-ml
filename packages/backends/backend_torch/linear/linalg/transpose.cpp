/* tensor_transpose_2d / tensor_transpose_last2 for the torch backend.
 *
 * Both call .contiguous() — the autograd path is unaffected, but
 * downstream FFI consumers expecting row-major storage will see one. */
#include "../../tensor.h"

extern "C" TensorHandle tensor_transpose_2d(TensorHandle h) {
    return from_tensor(to_tensor(h)->t().contiguous());
}

extern "C" TensorHandle tensor_transpose_last2(TensorHandle h) {
    return from_tensor(to_tensor(h)->transpose(-2, -1).contiguous());
}
