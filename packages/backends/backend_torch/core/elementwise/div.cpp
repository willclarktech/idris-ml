/* tensor_div for the torch backend. See add.cpp for the libtorch-vs-tape
 * autograd contrast. */
#include "../../tensor.h"

extern "C" TensorHandle tensor_div(TensorHandle a, TensorHandle b) {
	return from_tensor(torch::div(*to_tensor(a), *to_tensor(b)));
}
