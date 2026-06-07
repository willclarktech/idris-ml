/* tensor_abs for the torch backend. */
#include "../../tensor.h"

extern "C" TensorHandle tensor_abs(TensorHandle h) {
	return from_tensor(torch::abs(*to_tensor(h)));
}
