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
	double* images;  /* [count * rows * cols], normalized to [0,1] */
	uint8_t* labels; /* [count] */
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

/* Return a freshly malloc'd double[rows * cols] copy of the image at
 * `index`, normalized to [0, 1]. Ownership transfers to the caller —
 * the streamed-creator path (tape / torch / mlx all consistently free
 * the input buffer after the tensor copy) consumes and frees it.
 * Allocate-and-copy is the convention; a "borrowed pointer" return
 * would be misused by the streamed creators and free a slice of the
 * dataset buffer. The cost (1µs per image × 60k images per MNIST
 * epoch ≈ 60ms) is negligible next to a real training step. */
double* idx_image_doubles(void* handle, int index);

/* Free the dataset handle + its buffers. */
void idx_free(void* handle);

#ifdef __cplusplus
}
#endif

#endif /* IDRISML_IDX_H */
