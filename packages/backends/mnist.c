/* mnist.c — tensor-touching MNIST helpers.
 *
 * Compiled per-backend so `mnist_get_image` resolves to the backend's
 * `tensor_create` via the rename header. The non-tensor MNIST helpers
 * (`mnist_load` / `mnist_count` / `mnist_get_label`) live in
 * `shared_utils.c` as a single unified definition.
 */

#include "backend.h"
#include "shared_utils.h"
#include <stdlib.h>

/* Return a [1, 28, 28] tensor for image at index (non-persistent, non-grad).
   `dtag` selects the output dtype (RuntimeDType tag) so the result honestly
   matches the Idris `dt`. Branches on dtag to use the dtype-aware suffixed
   creator: `_f32` for dtag==14 (F32), `_f64` for dtag==15 (F64); per the
   `RuntimeDType` slot layout in `Tensor.idr`. The earlier "create-as-F64
   then cast" two-step crashes on torch-mps after the 8507e50 streamed-path
   migration — `tensor_create` now eagerly moves to `g_torch_target_device`,
   and MPS rejects F64 at construction. The suffixed creator combines cast +
   move into one `.to(opts)`, avoiding the bad F64-on-MPS intermediate. The
   prior `if (dtag == 0) cast_f32` branch was effectively dead code (F32's
   dtag is 14, not 0) which masked the dtype mismatch on F32 builds until
   the MPS-rejection of F64-at-construction surfaced it as a hard crash. */
TensorHandle mnist_get_image(void* handle, int index, int dtag) {
    MnistDataset* ds = (MnistDataset*)handle;
    int dim = ds->rows * ds->cols;  /* 784 */
    int shape[] = {1, ds->rows, ds->cols};
    double* src = ds->images + index * dim;
    return dtag == 14 ? tensor_create_f32(src, shape, 3, 0)
                      : tensor_create_f64(src, shape, 3, 0);
}

void mnist_free(void* handle) {
    if (!handle) return;
    MnistDataset* ds = (MnistDataset*)handle;
    free(ds->images);
    free(ds->labels);
    free(ds);
}
