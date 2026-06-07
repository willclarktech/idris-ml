/* tensor_add for the torch backend. libtorch's autograd records the
 * op automatically when either operand has requires_grad=true; there's
 * no per-op backward to wire here (contrast with tape's add.c, which
 * registers a backward function via TAPE_REGISTER_OP). */
#include "../../tensor.h"

extern "C" TensorHandle tensor_add(TensorHandle a, TensorHandle b) {
	return from_tensor(torch::add(*to_tensor(a), *to_tensor(b)));
}
