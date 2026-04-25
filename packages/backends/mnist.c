/* mnist.c — MNIST .idx file reader.
 *
 * Reads the standard MNIST binary format (big-endian .idx files).
 * Images are normalized to [0,1] doubles. Labels are raw uint8.
 * Provides functions to access individual images as C tensors.
 */

#include "backend.h"
#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <string.h>

/* Big-endian to native uint32 */
static uint32_t read_be32(FILE* f) {
    uint8_t buf[4];
    if (fread(buf, 1, 4, f) != 4) return 0;
    return ((uint32_t)buf[0] << 24) | ((uint32_t)buf[1] << 16) |
           ((uint32_t)buf[2] << 8)  | (uint32_t)buf[3];
}

typedef struct {
    double* images;    /* [count * 784], normalized to [0,1] */
    uint8_t* labels;   /* [count] */
    int count;
    int rows;
    int cols;
} MnistDataset;

void* mnist_load(const char* images_path, const char* labels_path) {
    MnistDataset* ds = calloc(1, sizeof(MnistDataset));

    /* Load images */
    FILE* f = fopen(images_path, "rb");
    if (!f) {
        fprintf(stderr, "mnist_load: cannot open %s\n", images_path);
        free(ds);
        return NULL;
    }
    uint32_t magic = read_be32(f);
    if (magic != 2051) {
        fprintf(stderr, "mnist_load: bad magic %u (expected 2051)\n", magic);
        fclose(f); free(ds);
        return NULL;
    }
    ds->count = (int)read_be32(f);
    ds->rows = (int)read_be32(f);
    ds->cols = (int)read_be32(f);
    int pixels = ds->count * ds->rows * ds->cols;

    uint8_t* raw = malloc(pixels);
    if (fread(raw, 1, pixels, f) != (size_t)pixels) {
        fprintf(stderr, "mnist_load: short read on images\n");
        fclose(f); free(raw); free(ds);
        return NULL;
    }
    fclose(f);

    ds->images = malloc(pixels * sizeof(double));
    for (int i = 0; i < pixels; i++)
        ds->images[i] = raw[i] / 255.0;
    free(raw);

    /* Load labels */
    f = fopen(labels_path, "rb");
    if (!f) {
        fprintf(stderr, "mnist_load: cannot open %s\n", labels_path);
        free(ds->images); free(ds);
        return NULL;
    }
    magic = read_be32(f);
    if (magic != 2049) {
        fprintf(stderr, "mnist_load: bad label magic %u (expected 2049)\n", magic);
        fclose(f); free(ds->images); free(ds);
        return NULL;
    }
    int label_count = (int)read_be32(f);
    if (label_count != ds->count) {
        fprintf(stderr, "mnist_load: image count %d != label count %d\n", ds->count, label_count);
    }
    ds->labels = malloc(ds->count);
    if (fread(ds->labels, 1, ds->count, f) != (size_t)ds->count) {
        fprintf(stderr, "mnist_load: short read on labels\n");
    }
    fclose(f);

    return ds;
}

int mnist_count(void* handle) {
    return handle ? ((MnistDataset*)handle)->count : 0;
}

/* Return a [1, 28, 28] tensor for image at index (non-persistent, non-grad) */
TensorHandle mnist_get_image(void* handle, int index) {
    MnistDataset* ds = (MnistDataset*)handle;
    int dim = ds->rows * ds->cols;  /* 784 */
    int shape[] = {1, ds->rows, ds->cols};
    return tensor_create(ds->images + index * dim, shape, 3, 0);
}

/* Return label (0-9) at index */
int mnist_get_label(void* handle, int index) {
    MnistDataset* ds = (MnistDataset*)handle;
    return (int)ds->labels[index];
}

void mnist_free(void* handle) {
    if (!handle) return;
    MnistDataset* ds = (MnistDataset*)handle;
    free(ds->images);
    free(ds->labels);
    free(ds);
}
