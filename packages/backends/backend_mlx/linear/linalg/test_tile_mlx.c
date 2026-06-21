/* mlx tensor_tile_2d edge arms (tile.cpp).
 *   - no-grad input -> the eager mx::eval(tiled) materialize arm;
 *   - grad input inside no_grad -> tape_append returns -1 -> free(meta) arm. */
#include <criterion/criterion.h>
#include "backend.h"
#include "test_helpers.h"

#ifdef BACKEND_MLX

#define DTAG_F64 15

Test(tile_mlx, no_grad_eager_eval) {
	double d[] = {1.0, 2.0, 3.0, 4.0};
	TensorHandle t = tensor_create_2d_streamed(2, 2, hcopy(d, 4), /*rg=*/0, 0, DTAG_F64);
	TensorHandle r = tensor_tile_2d(t, 2, 1); /* [2,2] -> [4,2]; !requires_grad -> eval arm */
	cr_assert_eq(tensor_numel(r), 8, "tile [2,2] x (2,1) -> 8 elems");
}

#endif /* BACKEND_MLX */
