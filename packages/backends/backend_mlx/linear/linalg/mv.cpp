/* tensor_mv for the mlx backend (matrix-vector product).
 *
 * mlx has no native mv — reshape the vec to [n,1], matmul, then
 * reshape result back to [m]. The reshapes are no-ops on the data
 * layout, and OP_MV's backward closes over the matrix/vector shapes
 * to undo them. */
#include "../../tensor.h"
#include "../../tape.h"
#include "../../stream.h"

extern "C" TensorHandle tensor_mv_mlx_streamed(TensorHandle hmat, TensorHandle hvec, int stream_tag) {
    WITH_STREAM(stream_tag);
    auto mat = (Tensor*)hmat; auto vec = (Tensor*)hvec;
    int n = (int)vec->data.size();
    int m_size = (int)mat->data.shape(0);
    auto vec_col = mx::reshape(vec->data, {n, 1});
    auto result_col = mx::matmul(mat->data, vec_col);
    auto result = mx::reshape(result_col, {m_size});
    bool rg = mat->requires_grad || vec->requires_grad;
    auto r = new Tensor(result, rg);
    if (rg) tape_append(OP_MV, r, mat, vec, 0);
    return (TensorHandle)r;
}

extern "C" TensorHandle tensor_mv(TensorHandle hmat, TensorHandle hvec) {
    return tensor_mv_mlx_streamed(hmat, hvec, default_stream_tag());
}
