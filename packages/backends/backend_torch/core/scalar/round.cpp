/* tensor_round for the torch backend. Element-wise round-to-nearest-
 * even (`at::round`). Inference-only; gradient never flows through. */
#include "../../tensor.h"

extern "C" TensorHandle tensor_round(TensorHandle h) {
    return from_tensor(at::round(*to_tensor(h)));
}
