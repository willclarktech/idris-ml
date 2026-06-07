/* tensor_max_rows for the torch backend. torch.max(dim) routes the
 * backward to the argmax cells natively (tie-breaking unspecified
 * across backends; tests avoid ties). */
#include "../../tensor.h"

extern "C" TensorHandle tensor_max_rows(TensorHandle hinput, int b, int n) {
	(void)b;
	(void)n;
	auto& inp = *to_tensor(hinput);
	return from_tensor(std::get<0>(inp.max(1)));
}
