/* linear/shape/view_2d.c — non-grad view of element [row, col] in a 2D tensor.
 *
 * Phase 1b.1.d (mechanical bundle). Read-only handle pointing into
 * parent's storage. No tape_append.
 */

#include <stdlib.h>
#include "../../tensor.h"
#include "../../arena.h"
#include "../../../backend.h"

TensorHandle tensor_view_2d(TensorHandle h, int row, int col) {
    Tensor* t = (Tensor*)h;
    int cols = t->shape[1];
    Tensor* v = calloc(1, sizeof(Tensor));
    v->data = &((double*)t->data)[row * cols + col];
    v->shape = NULL;
    v->rank = 0;
    v->numel = 1;
    v->requires_grad = 0;
    v->tape_idx = -1;
    v->grad = NULL;
    return v;
}
