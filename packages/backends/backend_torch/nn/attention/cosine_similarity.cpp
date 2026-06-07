/* tensor_cosine_similarity for the torch backend. */
#include "../../tensor.h"

extern "C" TensorHandle tensor_cosine_similarity(TensorHandle a, TensorHandle b, int dim) {
	return from_tensor(torch::nn::functional::cosine_similarity(
	    *to_tensor(a), *to_tensor(b),
	    torch::nn::functional::CosineSimilarityFuncOptions().dim(dim)));
}
