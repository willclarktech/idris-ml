/* idx.h — typedef + signatures for the IDX-format dataset helpers
 * (Yann LeCun's IDX file format, used by MNIST + similar small-image
 * datasets). Backend-agnostic; this surface intentionally does NOT
 * touch the Tensor handle ABI — `idx_image_doubles` returns a
 * borrowed pointer into the already-loaded host buffer, and the
 * Idris side feeds that pointer to `primCreate3dStreamed` for tensor
 * construction. Lives outside `backend.h` because it is NOT part of
 * the per-backend dispatch surface — it's an optional dataset utility
 * compiled once into the dylib.
 */

#ifndef IDRISML_IDX_H
#define IDRISML_IDX_H

#include <stdint.h>

typedef struct {
    double*  images;   /* [count * rows * cols], normalized to [0,1] */
    uint8_t* labels;   /* [count] */
    int count;
    int rows;
    int cols;
} IdxDataset;

#ifdef __cplusplus
extern "C" {
#endif

/* Load IDX image + label files. Returns an opaque IdxDataset* handle,
 * or NULL on error (path / magic / shape mismatch reported to stderr). */
void* idx_load(const char* images_path, const char* labels_path);

/* Number of (image, label) pairs in the dataset. */
int idx_count(void* handle);

/* Label (0..numClasses-1) for the image at index. */
int idx_label_at(void* handle, int index);

/* Borrowed pointer to the image at `index`, length `rows * cols` doubles
 * already in [0, 1]. Lifetime tied to the IdxDataset handle. Idris
 * passes this pointer to `primCreate3dStreamed 1 rows cols ...` to
 * construct the tensor; no per-image copy happens on the C side. */
double* idx_image_doubles(void* handle, int index);

/* Free the dataset handle + its buffers. */
void idx_free(void* handle);

#ifdef __cplusplus
}
#endif

#endif /* IDRISML_IDX_H */
