/* shared_utils.h — typedef + signatures for the backend-agnostic
 * helpers in shared_utils.c. The struct definition lives here so
 * mnist.c (which keeps the tensor-touching `mnist_get_image` /
 * `mnist_free`) and shared_utils.c (which owns the rest) agree
 * on the layout. */

#ifndef IDRISML_SHARED_UTILS_H
#define IDRISML_SHARED_UTILS_H

#include <stdint.h>

typedef struct {
    double*  images;   /* [count * rows * cols], normalized to [0,1] */
    uint8_t* labels;   /* [count] */
    int count;
    int rows;
    int cols;
} MnistDataset;

#ifdef __cplusplus
extern "C" {
#endif

/* Index-array helpers (DataLoader). */
int* create_index_array(int n);
int* shuffle_index_array(int* arr, int n);
int  index_array_get(int* arr, int i);

/* RSS reporting. */
int get_rss_mb(void);
int get_current_rss_mb(void);

/* MNIST file/dataset helpers that don't touch the Tensor surface. */
void* mnist_load(const char* images_path, const char* labels_path);
int   mnist_count(void* handle);
int   mnist_get_label(void* handle, int index);

/* C buffer helpers (host malloc / free / element read/write).
 * Backend-agnostic; one definition for all backends. */
double* tensor_alloc_doubles(int n);
void    tensor_free_doubles(double* buf);
double  tensor_read_double(double* buf, int idx);
void*   tensor_write_double_return(void* buf, int off, double val);

int*    tensor_alloc_ints(int n);
void    tensor_free_ints(int* buf);
int*    tensor_write_int_return(int* buf, int off, int val);

/* Tensor-pointer arrays (the Idris-side staging buffer used by
 * stack/cat). Stores opaque void* handles; the per-backend
 * `tensor_stack_from_array` / `tensor_cat_from_array` consumers
 * reinterpret as TensorHandle* internally. */
void**  tensor_ptr_array_alloc(int n);
void*   tensor_ptr_array_set_return(void* arr, int idx, void* t);

#ifdef __cplusplus
}
#endif

#endif /* IDRISML_SHARED_UTILS_H */
