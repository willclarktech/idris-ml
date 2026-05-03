/* Shared utilities: pure-C helpers that don't touch any backend's
 * tensor primitives. Compiled WITHOUT any rename header so the
 * symbols emerge under their unified names natively — these are
 * intentionally backend-agnostic and don't participate in the
 * per-backend dispatch surface. Live as a single TU in the dylib
 * (one definition each, no suffixed variants). */

#include <stdlib.h>
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
