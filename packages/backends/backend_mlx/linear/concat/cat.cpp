/* tensor_cat / tensor_cat2 / tensor_cat_from_array for the mlx backend. */
#include "../../tensor.h"
#include "../../tape.h"
#include "../../stream.h"

static TensorHandle cat_impl(TensorHandle* tensors, int count, int dim) {
    std::vector<mx::array> arrs;
    bool rg = false;
    for (int i = 0; i < count; i++) {
        auto t = (Tensor*)tensors[i];
        arrs.push_back(t->data);
        if (t->requires_grad) rg = true;
    }
    auto r = new Tensor(mx::concatenate(arrs, dim), rg);
    if (rg) {
        int idx = tape_append(OP_CAT_MULTI, r, nullptr, nullptr, (double)dim);
        auto* indices = new std::vector<int>();
        for (int i = 0; i < count; i++)
            indices->push_back(((Tensor*)tensors[i])->pool_idx);
        tape[idx].meta = (void*)indices;
    }
    return (TensorHandle)r;
}

extern "C" TensorHandle tensor_cat_mlx_streamed(TensorHandle* tensors, int count, int dim, int stream_tag) {
    WITH_STREAM(stream_tag);
    return cat_impl(tensors, count, dim);
}

extern "C" TensorHandle tensor_cat(TensorHandle* tensors, int count, int dim) {
    return tensor_cat_mlx_streamed(tensors, count, dim, default_stream_tag());
}

extern "C" TensorHandle tensor_cat_from_array(TensorHandle* arr, int count, int dim) {
    auto r = cat_impl(arr, count, dim);
    free(arr);
    return r;
}

/* cat2 reuses OP_CAT — backward replay needs the split point, which is
 * a->data.size() (the length of the first input along axis 0 when both
 * inputs are flattened). Stored in scalar_arg for replay. */
extern "C" TensorHandle tensor_cat2_mlx_streamed(TensorHandle ha, TensorHandle hb, int stream_tag) {
    WITH_STREAM(stream_tag);
    auto a = (Tensor*)ha; auto b = (Tensor*)hb;
    bool rg = a->requires_grad || b->requires_grad;
    auto r = new Tensor(mx::concatenate({a->data, b->data}, 0), rg);
    if (rg) tape_append(OP_CAT, r, a, b, (double)a->data.size());
    return (TensorHandle)r;
}

extern "C" TensorHandle tensor_cat2(TensorHandle ha, TensorHandle hb) {
    return tensor_cat2_mlx_streamed(ha, hb, default_stream_tag());
}
