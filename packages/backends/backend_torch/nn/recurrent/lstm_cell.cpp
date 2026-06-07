/* tensor_lstm_cell for the torch backend.
 *
 * torch::lstm_cell expects 2D input/hx/cx ([batch, *]). Mirror
 * nn.LSTMCell which auto-unsqueezes 1D inputs so unbatched callers
 * (including the C-side gradient harness) work. */
#include "../../tensor.h"

extern "C" void tensor_lstm_cell(TensorHandle input, TensorHandle hx, TensorHandle cx,
                                 TensorHandle w_ih, TensorHandle w_hh, TensorHandle b_ih,
                                 TensorHandle b_hh, TensorHandle* out_h, TensorHandle* out_c) {
	auto in1d = *to_tensor(input);
	auto hx1d = *to_tensor(hx);
	auto cx1d = *to_tensor(cx);
	bool unbatched = (in1d.dim() == 1);
	auto in2d = unbatched ? in1d.unsqueeze(0) : in1d;
	auto hx2d = unbatched ? hx1d.unsqueeze(0) : hx1d;
	auto cx2d = unbatched ? cx1d.unsqueeze(0) : cx1d;
	auto result = torch::lstm_cell(in2d, {hx2d, cx2d}, *to_tensor(w_ih), *to_tensor(w_hh),
	                               *to_tensor(b_ih), *to_tensor(b_hh));
	auto new_h = std::get<0>(result);
	auto new_c = std::get<1>(result);
	if (unbatched) {
		new_h = new_h.squeeze(0);
		new_c = new_c.squeeze(0);
	}
	*out_h = from_tensor(new_h);
	*out_c = from_tensor(new_c);
}
