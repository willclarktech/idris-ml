/* tensor_gelu for the torch backend.
 *
 * Uses the **tanh approximation** to match tape + mlx semantics:
 *   gelu(x) = x * 0.5 * (1 + tanh(sqrt(2/pi) * (x + 0.044715 * x^3)))
 *
 * libtorch's default `torch::gelu(t)` is the EXACT GELU via `erf`,
 * which diverges from the tape/mlx tanh-approx by ~1.5e-4 at moderate
 * inputs (e.g. x=1 → exact 0.841344746 vs tanh-approx 0.841191991).
 * The codebase's contract (Hendrycks/Gimpel 2016 approximation, same
 * as PyTorch's `nn.GELU(approximate='tanh')`) is what tape and mlx
 * implement; torch must match.
 *
 * Surfaced by `test/common/nn/activation/test_gelu.c` (W3 gap-fill).
 */
#include "../../tensor.h"

extern "C" TensorHandle tensor_gelu(TensorHandle h) {
    return from_tensor(torch::gelu(*to_tensor(h), "tanh"));
}
