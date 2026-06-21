/* tensor_conv1d_grouped / tensor_conv2d_grouped for the mlx backend.
 *
 * Fast path: groups == 1 delegates to the plain conv{1,2}d (which is
 * what the Idris-side `tconv*_grouped` callers expect when groups is
 * trivial). Otherwise dispatch to mlx::conv{1,2}d with `groups` set
 * — same NCHW↔NHWC transpose dance as the non-grouped variants.
 *
 * Backward isn't recorded: no current example calls grouped conv in
 * a grad context. */
#include "../tensor.h"
#include "../stream.h"

extern "C" TensorHandle tensor_conv1d(TensorHandle hinput, TensorHandle hkernel, TensorHandle hbias,
                                      int pad, int stride);
extern "C" TensorHandle tensor_conv2d(TensorHandle hinput, TensorHandle hkernel, TensorHandle hbias,
                                      int padH, int padW, int strideH, int strideW);

extern "C" TensorHandle tensor_conv1d_grouped(TensorHandle hinput, TensorHandle hkernel,
                                              TensorHandle hbias, int pad, int stride, int groups) {
	if (groups == 1) return tensor_conv1d(hinput, hkernel, hbias, pad, stride);
	auto* inp = (Tensor*)hinput;
	auto* ker = (Tensor*)hkernel;
	Tensor const* bias = (hbias != nullptr) ? (Tensor*)hbias : nullptr;
	int inC = (int)inp->data.shape(0), L = (int)inp->data.shape(1);
	auto inp_lc = mx::transpose(inp->data, {1, 0});
	auto inp_nlc = mx::reshape(inp_lc, {1, L, inC});
	auto ker_mlx = mx::transpose(ker->data, {0, 2, 1});
	auto out = mx::conv1d(inp_nlc, ker_mlx, stride, pad, /*dilation=*/1, groups);
	auto out_sq = mx::squeeze(out, 0);
	auto result = mx::transpose(out_sq, {1, 0});
	if (bias != nullptr) result = mx::add(result, mx::reshape(bias->data, {-1, 1}));
	return (TensorHandle)(new Tensor(result, inp->requires_grad || ker->requires_grad));
}

extern "C" TensorHandle tensor_conv2d_grouped(TensorHandle hinput, TensorHandle hkernel,
                                              TensorHandle hbias, int padH, int padW, int strideH,
                                              int strideW, int groups) {
	if (groups == 1) return tensor_conv2d(hinput, hkernel, hbias, padH, padW, strideH, strideW);
	auto* inp = (Tensor*)hinput;
	auto* ker = (Tensor*)hkernel;
	Tensor const* bias = (hbias != nullptr) ? (Tensor*)hbias : nullptr;
	int inC = (int)inp->data.shape(0), H = (int)inp->data.shape(1), W = (int)inp->data.shape(2);
	auto inp_hwc = mx::transpose(inp->data, {1, 2, 0});
	auto inp_nhwc = mx::reshape(inp_hwc, {1, H, W, inC});
	auto ker_mlx = mx::transpose(ker->data, {0, 2, 3, 1});
	auto out = mx::conv2d(inp_nhwc, ker_mlx, {strideH, strideW}, {padH, padW}, /*dilation=*/{1, 1},
	                      groups);
	auto out_sq = mx::squeeze(out, 0);
	auto result = mx::transpose(out_sq, {2, 0, 1});
	if (bias != nullptr) result = mx::add(result, mx::reshape(bias->data, {-1, 1, 1}));
	return (TensorHandle)(new Tensor(result, inp->requires_grad || ker->requires_grad));
}
