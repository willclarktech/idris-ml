/* linear/shape/view_1d.c — non-grad view of element [idx] in a 1D tensor.
 *
 * Read-only handle pointing into
 * parent's storage. No tape_append (the consumer is an FFI readback,
 * not part of the autograd graph).
 */

#include <stdlib.h>
#include "../../tensor.h"
#include "../../arena.h"
#include "../../../backend.h"

TensorHandle tensor_view_1d(TensorHandle h, int idx) {
	Tensor* t = (Tensor*)h;
	Tensor* v = calloc(1, sizeof(Tensor));
	v->data = &((double*)t->data)[idx];
	v->shape = NULL;
	v->rank = 0;
	v->numel = 1;
	v->requires_grad = 0;
	v->tape_idx = -1;
	v->grad = NULL;
	return v;
}
