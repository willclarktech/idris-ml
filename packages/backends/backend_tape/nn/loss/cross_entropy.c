/* nn/loss/cross_entropy.c — forward-only cross-entropy.
 *
 * loss = -mean(target * log_softmax(input)).
 * No backward tape entry — callers typically pair the libtorch/MLX
 * native autograd or hand-roll a log-softmax + nllLoss chain that
 * already records a tape entry for each step.
 */

#include "../../arena.h"
#include "../../tensor.h"
#include "../../../backend.h"

extern void tape_abort_mixed_dtype(const char* op) __attribute__((noreturn));

TensorHandle tensor_cross_entropy(TensorHandle hinput, TensorHandle htarget) {
	Tensor* input = (Tensor*)hinput;
	Tensor* target = (Tensor*)htarget;
	if (input->dtype_tag != target->dtype_tag) tape_abort_mixed_dtype("tensor_cross_entropy");
	TensorHandle ls = tensor_log_softmax(hinput, 0);
	Tensor* lsT = (Tensor*)ls;
	double loss = 0;
	for (int i = 0; i < lsT->numel; i++)
		loss -= tape_load_d(target, i) * tape_load_d(lsT, i);
	loss /= lsT->numel;
	return (input->dtype_tag == DT_F32) ? make_scalar_f32(loss, 0) : make_scalar(loss, 0);
}
