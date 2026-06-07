/* tensor_reshape variants for the torch backend.
 *
 * Generic rank-N entry plus the fixed-rank variants the Idris-side
 * type-safe `treshape{1,2,3,4}` smart constructors bind to (which sidestep
 * marshalling a `Vect rank Nat` over FFI for the common cases). */
#include "../../tensor.h"

extern "C" TensorHandle tensor_reshape(TensorHandle h, int* shape, int rank) {
	std::vector<int64_t> dims(rank);
	for (int i = 0; i < rank; i++)
		dims[i] = shape[i];
	return from_tensor(to_tensor(h)->reshape(dims));
}

extern "C" TensorHandle tensor_reshape_1d(TensorHandle h, int n) {
	return from_tensor(to_tensor(h)->reshape({(int64_t)n}));
}

extern "C" TensorHandle tensor_reshape_2d(TensorHandle h, int rows, int cols) {
	return from_tensor(to_tensor(h)->reshape({(int64_t)rows, (int64_t)cols}));
}

extern "C" TensorHandle tensor_reshape_3d(TensorHandle h, int d0, int d1, int d2) {
	return from_tensor(to_tensor(h)->reshape({(int64_t)d0, (int64_t)d1, (int64_t)d2}));
}

extern "C" TensorHandle tensor_reshape_4d(TensorHandle h, int d0, int d1, int d2, int d3) {
	return from_tensor(to_tensor(h)->reshape({(int64_t)d0, (int64_t)d1, (int64_t)d2, (int64_t)d3}));
}
