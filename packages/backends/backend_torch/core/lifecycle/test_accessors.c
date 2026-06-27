/* torch-only Criterion suite — host-buffer accessors (accessors.cpp).
 *
 * Covers tensor_to_floats: the F32 host-buffer readout has no in-tree torch
 * test (tensor_to_doubles is the common path). Drive it directly on an F32
 * tensor so the .to(kFloat32) + memcpy<float> arm is covered.
 */
#include <criterion/criterion.h>
#include "backend.h"
#include "test_helpers.h"

#ifdef BACKEND_TORCH

#define DTAG_F32 14

Test(torch_accessors_cov, to_floats_f32) {
	double init[] = {1.5, -2.25, 3.0};
	TensorHandle t =
	    tensor_create_1d_streamed(3, hcopy(init, 3), /*rg=*/0, /*stream_tag=*/0, DTAG_F32);
	float buf[3];
	tensor_to_floats(t, buf);
	cr_assert_float_eq(buf[0], 1.5f, 1e-6f, "to_floats[0] (got %.6f)", (double)buf[0]);
	cr_assert_float_eq(buf[1], -2.25f, 1e-6f, "to_floats[1] (got %.6f)", (double)buf[1]);
	cr_assert_float_eq(buf[2], 3.0f, 1e-6f, "to_floats[2] (got %.6f)", (double)buf[2]);
}

#endif /* BACKEND_TORCH */
