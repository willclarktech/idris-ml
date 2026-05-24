/* test_bitlinear_fwd.c — BitNet b1.58 BitLinear inference forward (#411 B2).
 *
 * Asserts the per-backend `tensor_bitlinear_fwd` matches the PyTorch
 * reference oracle in `packages/pytorch/torch_ref/models/bitlinear.py`
 * on a fixed test fixture. The fixture's expected output values are
 * embedded as literals here; they're computed at module load in the
 * Python oracle (printable via `python -m torch_ref.models.bitlinear`)
 * and won't change unless the fixture itself changes.
 *
 * Fixture (o=3, i=4):
 *   W_ternary = [[ 1,  0, -1,  1],   row 0
 *                [-1,  1,  1,  0],   row 1
 *                [ 0, -1,  0,  1]]   row 2
 *   w_scale   = [0.5, 0.25, 0.75]
 *   x         = [1.0, 2.0, -0.5, 0.25]
 *   bias      = [0.1, -0.2, 0.3]
 *
 *   Expected y = (W_ternary * w_scale.unsqueeze(1)) @ x + bias
 *              = [0.975, -0.075, -1.0125]
 *
 * Storage encoding of W_ternary in packed-2-bit form:
 *   row 0: slots {1, 0, -1, 1} -> codes {01, 00, 11, 01}
 *          byte = (01) | (00 << 2) | (11 << 4) | (01 << 6) = 0x71
 *   row 1: slots {-1, 1, 1, 0} -> codes {11, 01, 01, 00}
 *          byte = (11) | (01 << 2) | (01 << 4) | (00 << 6) = 0x17
 *   row 2: slots {0, -1, 0, 1} -> codes {00, 11, 00, 01}
 *          byte = (00) | (11 << 2) | (00 << 4) | (01 << 6) = 0x4C
 *
 * Per-backend storage shape (see design-decisions.md "Per-backend
 * ternary storage"):
 *   - tape:   keeps the 3 packed bytes verbatim (2 bits/value).
 *   - torch / mlx: unpacks to a [3, 4] int8 tensor at construction.
 * The Idris-side type stays `Tensor [3, 4] d Ternary NoGrad`; the
 * byte-count difference lives below the FFI boundary.
 */

#include <criterion/criterion.h>
#include <stdint.h>
#include "../../../../backend.h"
#include "test_helpers.h"

Test(nn_quantization_bitlinear_fwd, fixture_matches_oracle) {
    /* Packed ternary weight: 3 rows, 4 cols, 1 byte/row. */
    uint8_t packed[3] = {0x71, 0x17, 0x4C};

    double scale_data[3] = {0.5, 0.25, 0.75};
    double x_data[4]     = {1.0, 2.0, -0.5, 0.25};
    double bias_data[3]  = {0.1, -0.2, 0.3};

    int scale_shape[1] = {3};
    int x_shape[1]     = {4};
    int bias_shape[1]  = {3};

    TensorHandle W = tensor_create_ternary_packed_2d(packed, 3, 3, 4, 0);
    TensorHandle scale = tensor_create(scale_data, scale_shape, 1, 0);
    TensorHandle x     = tensor_create(x_data,     x_shape,     1, 0);
    TensorHandle bias  = tensor_create(bias_data,  bias_shape,  1, 0);

    TensorHandle y = tensor_bitlinear_fwd(W, scale, x, bias);

    /* Expected from the PyTorch oracle (F64; printed by
       `python -m torch_ref.models.bitlinear`). */
    double expected[3] = {0.975, -0.075, -1.0125};

    double y_out[3] = {0};
    tensor_to_doubles(y, y_out);

    cr_assert_float_eq(y_out[0], expected[0], TEST_TOL_RELAXED,
        "y[0] mismatch: got %.6f, expected %.6f", y_out[0], expected[0]);
    cr_assert_float_eq(y_out[1], expected[1], TEST_TOL_RELAXED,
        "y[1] mismatch: got %.6f, expected %.6f", y_out[1], expected[1]);
    cr_assert_float_eq(y_out[2], expected[2], TEST_TOL_RELAXED,
        "y[2] mismatch: got %.6f, expected %.6f", y_out[2], expected[2]);
}

Test(nn_quantization_bitlinear_fwd, no_bias) {
    /* Same fixture without bias. Expected = y_with_bias - bias. */
    uint8_t packed[3] = {0x71, 0x17, 0x4C};
    double scale_data[3] = {0.5, 0.25, 0.75};
    double x_data[4]     = {1.0, 2.0, -0.5, 0.25};

    int scale_shape[1] = {3};
    int x_shape[1]     = {4};

    TensorHandle W = tensor_create_ternary_packed_2d(packed, 3, 3, 4, 0);
    TensorHandle scale = tensor_create(scale_data, scale_shape, 1, 0);
    TensorHandle x     = tensor_create(x_data,     x_shape,     1, 0);

    TensorHandle y = tensor_bitlinear_fwd(W, scale, x, /*bias=*/NULL);

    double expected[3] = {0.875, 0.125, -1.3125};  /* (0.975 - 0.1, -0.075 - (-0.2), -1.0125 - 0.3) */
    double y_out[3] = {0};
    tensor_to_doubles(y, y_out);

    cr_assert_float_eq(y_out[0], expected[0], TEST_TOL_RELAXED);
    cr_assert_float_eq(y_out[1], expected[1], TEST_TOL_RELAXED);
    cr_assert_float_eq(y_out[2], expected[2], TEST_TOL_RELAXED);
}
