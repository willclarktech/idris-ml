/* tensor_neg for the torch backend. */
#include "../../tensor.h"

extern "C" TensorHandle tensor_neg(TensorHandle h) {
	return from_tensor(torch::neg(*to_tensor(h)));
}
