/* linear/concat/stack.c — stack of scalar tensors into a 1D vector.
 *
 * Phase 1b.2.a (initial) + Phase 1d.2.f (OP_STACK backward arm and
 * tensor_stack_from_array migration). The OP_STACK tape entry's
 * backward distributes the upstream vector gradient elementwise to
 * each constituent scalar input's grad[0].
 *
 * tensor_stack (no-grad fast wrapper) and tensor_stack_from_array
 * (grad-bearing variant) share the same OP_STACK tag — the no-grad
 * path doesn't record a tape entry, so it can ignore the dispatch
 * table entirely.
 */

#include <stdlib.h>
#include <string.h>
#include "../../tape.h"
#include "../../arena.h"
#include "../../tensor.h"
#include "../../training/autograd/op_dispatch.h"
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

TensorHandle tensor_stack_from_array(TensorHandle* arr, int count, int dim) {
    (void)dim;
    /* Fast path: if all inputs are consecutive selects from the same parent
       tensor (data pointers are contiguous), skip the copy and return a
       tensor that shares the parent's data. This eliminates the repack
       cost when tensorToScalars → vecStackTensor round-trips. */
    if (count > 0) {
        Tensor* first = (Tensor*)arr[0];
        double* base = first->data;
        int consecutive = 1;
        int rg_check = first->requires_grad;
        for (int i = 1; i < count; i++) {
            Tensor* t = (Tensor*)arr[i];
            if (t->data != base + i) { consecutive = 0; break; }
            if (t->requires_grad) rg_check = 1;
        }
        if (consecutive) {
            /* Create a tensor that shares the parent's data buffer (no copy) */
            Tensor* r = arena_alloc(sizeof(Tensor));
            memset(r, 0, sizeof(Tensor));
            r->data = base;  /* shared with parent */
            r->shape = arena_alloc(sizeof(int));
            r->shape[0] = count;
            r->rank = 1;
            r->numel = count;
            r->requires_grad = rg_check;
            r->persistent = 0;
            /* Still record OP_STACK with input pointers for backward.
               STACK backward distributes ((double*)r->grad)[i] to inputs[i]->grad[0].
               The inputs are SELECT views, so their grad flows to the parent. */
            if (rg_check) {
                Tensor** inputs = malloc(count * sizeof(Tensor*));
                for (int i = 0; i < count; i++) inputs[i] = (Tensor*)arr[i];
                TapeEntry* e = tape_append(OP_STACK, r, NULL, NULL, 0);
                e->inputs = inputs;
                e->input_count = count;
            }
            free(arr);
            return r;
        }
    }

    /* Slow path: copy values and create new tensor */
    double* data = malloc(count * sizeof(double));
    int rg = 0;
    Tensor** inputs = malloc(count * sizeof(Tensor*));
    for (int i = 0; i < count; i++) {
        Tensor* t = (Tensor*)arr[i];
        data[i] = ((double*)t->data)[0];
        inputs[i] = t;
        if (t->requires_grad) rg = 1;
    }
    free(arr);
    int shape[] = {count};
    Tensor* r = make_tensor(data, shape, 1, rg);
    free(data);
    if (rg) {
        TapeEntry* e = tape_append(OP_STACK, r, NULL, NULL, 0);
        e->inputs = inputs;
        e->input_count = count;
    } else {
        free(inputs);
    }
    return r;
}

TensorHandle tensor_cat_from_array(TensorHandle* arr, int count, int dim) {
    return tensor_stack_from_array(arr, count, dim);
}

static void tape_backward_stack(TapeEntry* e) {
    /* Distribute gradient from stacked tensor back to constituent scalars. */
    Tensor* r = e->result;
    if (e->inputs) {
        for (int j = 0; j < e->input_count; j++) {
            Tensor* inp = e->inputs[j];
            if (inp->requires_grad) {
                ensure_grad(inp);
                ensure_grad(r);
                ((double*)inp->grad)[0] += ((double*)r->grad)[j];
            }
        }
    }
}

TAPE_REGISTER_OP(OP_STACK, tape_backward_stack)
