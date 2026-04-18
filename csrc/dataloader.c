#include <stdlib.h>

/* Create array [0, 1, ..., n-1] */
int* create_index_array(int n) {
    int* arr = (int*)malloc(n * sizeof(int));
    for (int i = 0; i < n; i++) arr[i] = i;
    return arr;
}

/* Fisher-Yates in-place shuffle. Returns arr for FFI side-effect threading. */
int* shuffle_index_array(int* arr, int n) {
    for (int i = n - 1; i > 0; i--) {
        int j = rand() % (i + 1);
        int tmp = arr[i];
        arr[i] = arr[j];
        arr[j] = tmp;
    }
    return arr;
}

/* Read arr[i] */
int index_array_get(int* arr, int i) {
    return arr[i];
}
