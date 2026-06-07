/* tensor_conv_transpose1d / tensor_conv_transpose2d for the mlx
 * backend.
 *
 * mlx does not expose conv_transpose directly. Implemented on host
 * (eval inputs, allocate double buffers, scatter-add) so the loop is
 * dtype-agnostic. Backward is not implemented — the result tensor
 * is created without a tape entry (no current example calls these in
 * a grad context). */
#include "../tensor.h"
#include "../stream.h"
#include "../precision.h"
#include <cstdlib>
#include <vector>

extern "C" TensorHandle tensor_conv_transpose1d(TensorHandle hinput, TensorHandle hkernel,
                                                TensorHandle hbias, int pad, int stride) {
	auto inp = (Tensor*)hinput;
	auto ker = (Tensor*)hkernel;
	Tensor* bias = hbias ? (Tensor*)hbias : nullptr;
	int inC = (int)inp->data.shape(0), L = (int)inp->data.shape(1);
	int outC = (int)ker->data.shape(1), kL = (int)ker->data.shape(2);
	int oL = (L - 1) * stride - 2 * pad + kL;

	mx::eval(inp->data);
	mx::eval(ker->data);
	std::vector<double> inpD_buf((size_t)inC * L);
	std::vector<double> kerD_buf((size_t)inC * outC * kL);
	mx_to_doubles(inp->data, inpD_buf.data());
	mx_to_doubles(ker->data, kerD_buf.data());
	const double* inpD = inpD_buf.data();
	const double* kerD = kerD_buf.data();
	double* out = (double*)calloc((size_t)outC * oL, sizeof(double));
	if (bias) {
		mx::eval(bias->data);
		std::vector<double> biasD_buf((size_t)outC);
		mx_to_doubles(bias->data, biasD_buf.data());
		const double* biasD = biasD_buf.data();
		for (int oc = 0; oc < outC; oc++)
			for (int ol = 0; ol < oL; ol++)
				out[(size_t)oc * oL + ol] = biasD[oc];
	}
	for (int ic = 0; ic < inC; ic++)
		for (int il = 0; il < L; il++)
			for (int oc = 0; oc < outC; oc++)
				for (int kl = 0; kl < kL; kl++) {
					int ol = il * stride - pad + kl;
					if (ol >= 0 && ol < oL)
						out[(size_t)oc * oL + ol] += inpD[(size_t)ic * L + il] *
						                             kerD[(size_t)ic * outC * kL + oc * kL + kl];
				}
	auto result = mx_array_from_doubles(out, {outC, oL}, inp->data.dtype());
	free(out);
	return (TensorHandle)(new Tensor(result, inp->requires_grad || ker->requires_grad));
}

extern "C" TensorHandle tensor_conv_transpose2d(TensorHandle hinput, TensorHandle hkernel,
                                                TensorHandle hbias, int padH, int padW, int strideH,
                                                int strideW) {
	auto inp = (Tensor*)hinput;
	auto ker = (Tensor*)hkernel;
	Tensor* bias = hbias ? (Tensor*)hbias : nullptr;
	int inC = (int)inp->data.shape(0), H = (int)inp->data.shape(1), W = (int)inp->data.shape(2);
	int outC = (int)ker->data.shape(1), kH = (int)ker->data.shape(2), kW = (int)ker->data.shape(3);
	int oH = (H - 1) * strideH - 2 * padH + kH;
	int oW = (W - 1) * strideW - 2 * padW + kW;
	mx::eval(inp->data);
	mx::eval(ker->data);
	std::vector<double> inpD_buf((size_t)inC * H * W);
	std::vector<double> kerD_buf((size_t)inC * outC * kH * kW);
	mx_to_doubles(inp->data, inpD_buf.data());
	mx_to_doubles(ker->data, kerD_buf.data());
	const double* inpD = inpD_buf.data();
	const double* kerD = kerD_buf.data();
	double* out = (double*)calloc((size_t)outC * oH * oW, sizeof(double));
	if (bias) {
		mx::eval(bias->data);
		std::vector<double> biasD_buf((size_t)outC);
		mx_to_doubles(bias->data, biasD_buf.data());
		const double* biasD = biasD_buf.data();
		for (int oc = 0; oc < outC; oc++)
			for (int oh = 0; oh < oH; oh++)
				for (int ow = 0; ow < oW; ow++)
					out[(size_t)oc * oH * oW + (size_t)oh * oW + ow] = biasD[oc];
	}
	for (int ic = 0; ic < inC; ic++)
		for (int ih = 0; ih < H; ih++)
			for (int iw = 0; iw < W; iw++)
				for (int oc = 0; oc < outC; oc++)
					for (int kh = 0; kh < kH; kh++)
						for (int kw = 0; kw < kW; kw++) {
							int oh = ih * strideH - padH + kh;
							int ow = iw * strideW - padW + kw;
							if (oh >= 0 && oh < oH && ow >= 0 && ow < oW)
								out[(size_t)oc * oH * oW + (size_t)oh * oW + ow] +=
								    inpD[(size_t)ic * H * W + (size_t)ih * W + iw] *
								    kerD[(size_t)ic * outC * kH * kW + (size_t)oc * kH * kW +
								         (size_t)kh * kW + kw];
						}
	auto result = mx_array_from_doubles(out, {outC, oH, oW}, inp->data.dtype());
	free(out);
	return (TensorHandle)(new Tensor(result, inp->requires_grad || ker->requires_grad));
}
