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
   matches the Idris `dt`. `tensor_create` builds F64 — its 4th arg is
   requires_grad, NOT a dtype selector (passing dtag there was the bug that
   left F32 builds with an F64 image and aborted the F32 conv on torch-mps).
   dtag==0 (F32) then casts down. dtag==0 only occurs on F32-capable backends
   (torch-mps, mlx-gpu); F64 lanes (tape, torch-cpu, mlx-cpu) use dtag==1 and
   skip the cast, so tape's F32-cast abort-stub is never reached. */
TensorHandle mnist_get_image(void* handle, int index, int dtag) {
    MnistDataset* ds = (MnistDataset*)handle;
    int dim = ds->rows * ds->cols;  /* 784 */
    int shape[] = {1, ds->rows, ds->cols};
    TensorHandle img = tensor_create(ds->images + index * dim, shape, 3, 0);
    if (dtag == 0) img = tensor_cast_dtype_f32(img);  /* F64 -> F32 */
    return img;
}

void mnist_free(void* handle) {
    if (!handle) return;
    MnistDataset* ds = (MnistDataset*)handle;
    free(ds->images);
    free(ds->labels);
    free(ds);
}
