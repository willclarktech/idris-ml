/* tensor_select for the torch backend. Picks one slice along `dim` at
 * `index`, removing that dim from the output. */
#include "../../tensor.h"

extern "C" TensorHandle tensor_select(TensorHandle h, int dim, int index) {
	return from_tensor(to_tensor(h)->select(dim, index));
}
