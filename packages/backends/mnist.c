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

/* Return a [1, 28, 28] tensor for image at index (non-persistent, non-grad) */
TensorHandle mnist_get_image(void* handle, int index) {
    MnistDataset* ds = (MnistDataset*)handle;
    int dim = ds->rows * ds->cols;  /* 784 */
    int shape[] = {1, ds->rows, ds->cols};
    return tensor_create(ds->images + index * dim, shape, 3, 0);
}

void mnist_free(void* handle) {
    if (!handle) return;
    MnistDataset* ds = (MnistDataset*)handle;
    free(ds->images);
    free(ds->labels);
    free(ds);
}
