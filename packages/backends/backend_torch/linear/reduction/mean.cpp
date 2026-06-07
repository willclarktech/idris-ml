/* tensor_mean for the torch backend. */
#include "../../tensor.h"

extern "C" TensorHandle tensor_mean(TensorHandle h) {
	return from_tensor(to_tensor(h)->mean());
}
