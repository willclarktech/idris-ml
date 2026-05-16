/* tensor_item for the torch backend. .cpu() is a no-op on CPU tensors;
 * only MPS / CUDA tensors pay the round-trip. Readback to host memory
 * via .item<double>() requires the tensor live on CPU. */
#include "../../tensor.h"

extern "C" double tensor_item(TensorHandle h) {
    return to_tensor(h)->cpu().item<double>();
}
