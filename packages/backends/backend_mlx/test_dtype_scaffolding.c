/* Dtype-scaffolding Criterion suite.
 *
 * Split out of the original bulk backend port. Shared assertion shims +
 * tolerances live in port_assert.h.
 */
#include "port_assert.h"

#if defined(BACKEND_MLX)
Test(dtype_scaffolding, mlx_bf16_storage) {
	/* mlx BF16 storage gate. Verifies that:
	   (1) creating a BF16-tagged tensor via the unified streamed
	       dispatch lands in mx::bfloat16 storage (not silently downcast
	       to F32 — which is exactly what the pre-2026-05-31 path did
	       when it hit the false abort message);
	   (2) tensor_dtype_name reports "BF16" (the accessor's old
	       F32-or-F64 branching reported "F64" for any non-F32 dtype,
	       which would have lied about BF16 storage);
	   (3) basic elementwise ops preserve BF16 dtype;
	   (4) F32 -> BF16 -> F32 cast roundtrip stays within bf16
	       precision (~1e-2 — bf16 has only 7 bits of mantissa). */
	param_clear();

	/* BF16 create via unified dispatch + dtype name + readback. */
	double bv[] = {1.5, 2.25, -0.5};
	TensorHandle bf = tensor_create_1d_streamed(3, hcopy(bv, 3), 0, 0, 17);
	ASSERT_TRUE("mlx BF16 dtype is BF16", strcmp(tensor_dtype_name(bf), "BF16") == 0);
	double bout[3];
	tensor_to_doubles(bf, bout);
	ASSERT_NEAR("mlx BF16 readback[0]", bout[0], 1.5, 1e-2);
	ASSERT_NEAR("mlx BF16 readback[1]", bout[1], 2.25, 1e-2);
	ASSERT_NEAR("mlx BF16 readback[2]", bout[2], -0.5, 1e-2);

	/* BF16 + BF16 preserves BF16 dtype. */
	TensorHandle bf2 = tensor_create_1d_streamed(3, hcopy(bv, 3), 0, 0, 17);
	TensorHandle bsum = tensor_add(bf, bf2);
	ASSERT_TRUE("mlx BF16 add preserves BF16", strcmp(tensor_dtype_name(bsum), "BF16") == 0);
	double bsout[3];
	tensor_to_doubles(bsum, bsout);
	ASSERT_NEAR("mlx BF16 add[0]", bsout[0], 3.0, 1e-2);

	/* Cast F32 -> BF16 -> F32 roundtrip. */
	double fv[] = {1.5, 2.25};
	TensorHandle f32t = tensor_create_1d_streamed(2, hcopy(fv, 2), 0, 0, 14);
	TensorHandle to_bf = tensor_cast_dtype_streamed(f32t, 0, 17);
	ASSERT_TRUE("mlx cast F32->BF16", strcmp(tensor_dtype_name(to_bf), "BF16") == 0);
	TensorHandle back = tensor_cast_dtype_streamed(to_bf, 0, 14);
	ASSERT_TRUE("mlx cast BF16->F32", strcmp(tensor_dtype_name(back), "F32") == 0);
	double rt[2];
	tensor_to_doubles(back, rt);
	ASSERT_NEAR("mlx BF16 roundtrip[0]", rt[0], 1.5, 1e-2);
	ASSERT_NEAR("mlx BF16 roundtrip[1]", rt[1], 2.25, 1e-2);

	/* `tensor_item` regression — BF16 storage is 2-byte; the previous
	   `item<float>()` cast read 16 bits of valid BF16 + 16 bits of
	   adjacent buffer garbage as a 32-bit float, returning denormal
	   values like 2.3e-41 for an actual 1.1 BF16 scalar. Caught via a
	   silent Supervised-training failure (loss=2.3e-41 from epoch 1)
	   2026-05-31, fixed in core/lifecycle/item.cpp. */
	TensorHandle bf_scalar = tensor_create_scalar_streamed(1.1, 0, 0, 17);
	ASSERT_NEAR("mlx BF16 tensor_item not denormal", tensor_item(bf_scalar), 1.1, 1e-2);

	param_clear();
}
#endif
