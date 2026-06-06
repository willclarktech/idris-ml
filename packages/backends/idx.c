/* idx.c — IDX-format dataset helpers (Yann LeCun's IDX file format).
 *
 * Compiled ONCE into libidrisml.dylib (not per-backend, no rename
 * header). The Idris side reaches these via plain
 * `%foreign "C:idx_*,libidrisml"` bindings, not via a
 * `UserExecutor*` typeclass method — IDX is a dataset utility, not
 * part of the backend dispatch surface.
 *
 * `idx_image_doubles` deliberately returns a borrowed double* into
 * the pre-loaded buffer (zero copy). The Idris side hands that pointer
 * to `primCreate3dStreamed` for tensor construction, going through the
 * generic dtype-streamed creator path instead of a per-backend
 * `mnist_get_image_<b>` symbol. That moves the dataset out of the
 * load-bearing executor interface.
 */

#include "idx.h"
#include <stdio.h>
#include <stdlib.h>

static uint32_t idx_read_be32(FILE* f) {
    uint8_t buf[4];
    if (fread(buf, 1, 4, f) != 4) return 0;
    return ((uint32_t)buf[0] << 24) | ((uint32_t)buf[1] << 16) |
           ((uint32_t)buf[2] << 8)  | (uint32_t)buf[3];
}

void* idx_load(const char* images_path, const char* labels_path) {
    IdxDataset* ds = calloc(1, sizeof(IdxDataset));

    FILE* f = fopen(images_path, "rb");
    if (!f) {
        fprintf(stderr, "idx_load: cannot open %s\n", images_path);
        free(ds);
        return NULL;
    }
    uint32_t magic = idx_read_be32(f);
    if (magic != 2051) {
        fprintf(stderr, "idx_load: bad magic %u (expected 2051)\n", magic);
        fclose(f); free(ds);
        return NULL;
    }
    ds->count = (int)idx_read_be32(f);
    ds->rows  = (int)idx_read_be32(f);
    ds->cols  = (int)idx_read_be32(f);
    int pixels = ds->count * ds->rows * ds->cols;

    uint8_t* raw = malloc(pixels);
    if (fread(raw, 1, pixels, f) != (size_t)pixels) {
        fprintf(stderr, "idx_load: short read on images\n");
        fclose(f); free(raw); free(ds);
        return NULL;
    }
    fclose(f);

    ds->images = malloc(pixels * sizeof(double));
    for (int i = 0; i < pixels; i++)
        ds->images[i] = raw[i] / 255.0;
    free(raw);

    f = fopen(labels_path, "rb");
    if (!f) {
        fprintf(stderr, "idx_load: cannot open %s\n", labels_path);
        free(ds->images); free(ds);
        return NULL;
    }
    magic = idx_read_be32(f);
    if (magic != 2049) {
        fprintf(stderr, "idx_load: bad label magic %u (expected 2049)\n", magic);
        fclose(f); free(ds->images); free(ds);
        return NULL;
    }
    int label_count = (int)idx_read_be32(f);
    if (label_count != ds->count) {
        fprintf(stderr, "idx_load: image count %d != label count %d\n", ds->count, label_count);
    }
    ds->labels = malloc(ds->count);
    if (fread(ds->labels, 1, ds->count, f) != (size_t)ds->count) {
        fprintf(stderr, "idx_load: short read on labels\n");
    }
    fclose(f);

    return ds;
}

int idx_count(void* handle) {
    return handle ? ((IdxDataset*)handle)->count : 0;
}

int idx_label_at(void* handle, int index) {
    IdxDataset* ds = (IdxDataset*)handle;
    return (int)ds->labels[index];
}

double* idx_image_doubles(void* handle, int index) {
    IdxDataset* ds = (IdxDataset*)handle;
    return ds->images + (size_t)index * (size_t)ds->rows * (size_t)ds->cols;
}

void idx_free(void* handle) {
    if (!handle) return;
    IdxDataset* ds = (IdxDataset*)handle;
    free(ds->images);
    free(ds->labels);
    free(ds);
}
