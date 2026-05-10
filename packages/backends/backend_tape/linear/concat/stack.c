/* linear/concat/stack.c — stack of scalar tensors into a 1D vector.
 *
 * Phase 1b.2.a (mechanical). Simple stack: no tape entry, no backward —
 * this surface is non-grad. The grad-bearing stack variant lives in
 * tensor_stack_from_array (separate file, still in the monolith;
 * OP_STACK backward stays in the monolith switch until that op
 * extracts later — Phase 1e adjacency).
 */

#include <stdlib.h>
#include "../../tape.h"
#include "../../arena.h"
#include "../../tensor.h"
#include "../../../backend.h"

extern void tape_abort_mixed_dtype(const char* op) __attribute__((noreturn));

TensorHandle tensor_stack(TensorHandle* tensors, int count, int dim) {
    (void)dim;
    int shape[] = {count};
    int out_tag = (count > 0) ? ((Tensor*)tensors[0])->dtype_tag : DT_F64;
    for (int i = 1; i < count; i++)
        if (((Tensor*)tensors[i])->dtype_tag != out_tag) tape_abort_mixed_dtype("tensor_stack");
    if (out_tag == DT_F32) {
        float* data = arena_alloc(count * sizeof(float));
        for (int i = 0; i < count; i++) data[i] = (float)tape_load_d((Tensor*)tensors[i], 0);
        return make_tensor_arena_f32(data, count, shape, 1, 0);
    }
    double* data = malloc(count * sizeof(double));
    for (int i = 0; i < count; i++) data[i] = tape_load_d((Tensor*)tensors[i], 0);
    Tensor* r = make_tensor(data, shape, 1, 0);
    free(data);
    return r;
}
