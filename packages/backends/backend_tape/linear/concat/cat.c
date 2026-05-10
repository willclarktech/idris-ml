/* linear/concat/cat.c — concatenate (simplified: delegates to stack).
 *
 * Today's semantics: identical to
 * tensor_stack for scalar inputs (covers idris-ml's actual usage —
 * the more general N-dim cat surfaces are tensor_cat2 (1D) and
 * tensor_concat_2d_axis1 (2D) in separate files).
 */

#include "../../../backend.h"

TensorHandle tensor_cat(TensorHandle* tensors, int count, int dim) {
    return tensor_stack(tensors, count, dim);
}
