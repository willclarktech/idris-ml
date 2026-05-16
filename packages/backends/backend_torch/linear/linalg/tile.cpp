/* tensor_tile_2d for the torch backend. torch's .repeat() is the
 * equivalent of numpy.tile (NOT torch.tile, which is element-wise
 * broadcasting — confusing naming history). */
#include "../../tensor.h"

extern "C" TensorHandle tensor_tile_2d(TensorHandle h, int rep0, int rep1) {
    return from_tensor(to_tensor(h)->repeat({(int64_t)rep0, (int64_t)rep1}));
}
