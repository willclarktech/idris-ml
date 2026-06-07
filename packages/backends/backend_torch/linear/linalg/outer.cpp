/* tensor_outer for the torch backend (1D outer product). */
#include "../../tensor.h"

extern "C" TensorHandle tensor_outer(TensorHandle a, TensorHandle b) {
	return from_tensor(torch::outer(*to_tensor(a), *to_tensor(b)));
}
