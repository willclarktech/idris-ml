/* Shared utilities: pure-C helpers that don't touch any backend's
 * tensor primitives. Compiled WITHOUT any rename header so the
 * symbols emerge under their unified names natively — these are
 * intentionally backend-agnostic and don't participate in the
 * per-backend dispatch surface. Live as a single TU in the dylib
 * (one definition each, no suffixed variants). */

#include "shared_utils.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/resource.h>

#ifdef __APPLE__
#include <mach/mach.h>
#endif

/* --- Index-array helpers (DataLoader) --- */

int* create_index_array(int n) {
    int* arr = (int*)malloc(n * sizeof(int));
    for (int i = 0; i < n; i++) arr[i] = i;
    return arr;
}

int* shuffle_index_array(int* arr, int n) {
    for (int i = n - 1; i > 0; i--) {
        int j = rand() % (i + 1);
        int tmp = arr[i];
        arr[i] = arr[j];
        arr[j] = tmp;
    }
    return arr;
}

int index_array_get(int* arr, int i) {
    return arr[i];
}

/* --- RSS reporting --- */

/* Peak RSS in MB. macOS reports ru_maxrss in bytes; Linux in KB. */
int get_rss_mb(void) {
    struct rusage usage;
    getrusage(RUSAGE_SELF, &usage);
#ifdef __APPLE__
    return (int)(usage.ru_maxrss / (1024 * 1024));
#else
    return (int)(usage.ru_maxrss / 1024);
#endif
}

/* Current resident-set size in MB. macOS exposes the live RSS via
 * mach_task_basic_info; on Linux we fall back to the peak (the
 * portable rusage path). */
int get_current_rss_mb(void) {
#ifdef __APPLE__
    mach_task_basic_info_data_t info;
    mach_msg_type_number_t count = MACH_TASK_BASIC_INFO_COUNT;
    if (task_info(mach_task_self(), MACH_TASK_BASIC_INFO,
                  (task_info_t)&info, &count) == KERN_SUCCESS)
        return (int)(info.resident_size / (1024 * 1024));
#endif
    return get_rss_mb();
}

/* --- MNIST file/dataset helpers (no tensor surface) --- */

static uint32_t read_be32(FILE* f) {
    uint8_t buf[4];
    if (fread(buf, 1, 4, f) != 4) return 0;
    return ((uint32_t)buf[0] << 24) | ((uint32_t)buf[1] << 16) |
           ((uint32_t)buf[2] << 8)  | (uint32_t)buf[3];
}

void* mnist_load(const char* images_path, const char* labels_path) {
    MnistDataset* ds = calloc(1, sizeof(MnistDataset));

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
    ds->rows  = (int)read_be32(f);
    ds->cols  = (int)read_be32(f);
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

int mnist_get_label(void* handle, int index) {
    MnistDataset* ds = (MnistDataset*)handle;
    return (int)ds->labels[index];
}
