/* tensor_rms_norm_2d for the torch backend.
 *
 * HF LlamaRMSNorm formula: variance = (x^2).mean(-1, keepdim=true);
 *                          out = x * rsqrt(variance + eps) * weight.
 * Composed via at::pow + at::mean + at::rsqrt; libtorch's autograd
 * tracks each step so backward "just works" — no manual VJP needed.
 *
 * Replaces the per-row 7-primitive chain in
 * `HfCommon.applyRmsNorm2dRaw` with one fused FFI call. On torch-mps
 * the kernel set composes into a small MPSGraph subgraph.
 */
#include "../../tensor.h"

extern "C" TensorHandle tensor_rms_norm_2d(TensorHandle input, TensorHandle weight,
                                           double eps) {
    auto& x = *to_tensor(input);
    auto& w = *to_tensor(weight);
    auto variance = at::mean(at::pow(x, 2), {-1}, /*keepdim=*/true);
    auto rstd = at::rsqrt(at::add(variance, eps));
    auto out = at::mul(at::mul(x, rstd), w);
    return from_tensor(std::move(out));
}
