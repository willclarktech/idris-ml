/* tensor_max for the torch backend. detach() — see min.cpp. */
#include "../../tensor.h"

extern "C" TensorHandle tensor_max(TensorHandle h) {
	return from_tensor(to_tensor(h)->max().detach());
}
