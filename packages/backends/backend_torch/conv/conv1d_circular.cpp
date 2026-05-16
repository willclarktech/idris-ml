/* tensor_conv1d_circular for the torch backend.
 *
 * Circular convolution — the NTM shift operator. Pads the input by
 * wrapping the tail/head, then runs a 1D conv with the kernel flipped
 * (torch::conv1d is cross-correlation, not true convolution; flipping
 * the kernel turns it into the latter, matching the NTM paper). */
#include "../tensor.h"

extern "C" TensorHandle tensor_conv1d_circular(TensorHandle input, TensorHandle kernel) {
    auto& inp = *to_tensor(input);
    auto& ker = *to_tensor(kernel);

    int64_t n = inp.size(0);
    int64_t k = ker.size(0);
    int64_t pad = k / 2;

    auto padded = torch::cat({inp.slice(0, n - pad, n), inp, inp.slice(0, 0, pad)});
    auto inp_3d = padded.reshape({1, 1, -1});
    auto ker_3d = ker.flip(0).reshape({1, 1, -1});
    auto out = torch::conv1d(inp_3d, ker_3d);
    return from_tensor(out.reshape({n}));
}
