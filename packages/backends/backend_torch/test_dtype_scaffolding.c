/* Dtype-scaffolding Criterion suite.
 *
 * Split out of the original bulk backend port. Shared assertion shims +
 * tolerances live in port_assert.h.
 */
#include "port_assert.h"

#if defined(BACKEND_TORCH)
Test(dtype_scaffolding, inference_dtype_scaffolding_torch) {
	param_clear();

	/* BF16 create + dtype; add preserves dtype. */
	double bdata[] = {1.5, 2.25, -0.5};
	TensorHandle bf = tensor_create_1d_streamed(3, hcopy(bdata, 3), 0, 0, 17);
	ASSERT_TRUE("bf16 dtype is BF16", strcmp(tensor_dtype_name(bf), "BF16") == 0);
	double bout[3];
	tensor_to_doubles(bf, bout);
	ASSERT_NEAR("bf16 value[1]", bout[1], 2.25, 1e-2);
	TensorHandle bf2 = tensor_create_1d_streamed(3, hcopy(bdata, 3), 0, 0, 17);
	TensorHandle bsum = tensor_add(bf, bf2);
	ASSERT_TRUE("bf16 add preserves BF16", strcmp(tensor_dtype_name(bsum), "BF16") == 0);
	double bsout[3];
	tensor_to_doubles(bsum, bsout);
	ASSERT_NEAR("bf16 add value[0]", bsout[0], 3.0, 1e-2);

	/* F16 create + dtype. */
	double hdata[] = {0.5, 1.25};
	TensorHandle hf = tensor_create_1d_streamed(2, hcopy(hdata, 2), 0, 0, 13);
	ASSERT_TRUE("f16 dtype is F16", strcmp(tensor_dtype_name(hf), "F16") == 0);

	/* Cast F32 -> BF16 -> F32 round-trip. */
	double fdata[] = {1.5, 2.25};
	TensorHandle f32t = tensor_create_1d_f32(2, hcopy(fdata, 2), 0);
	TensorHandle to_bf = tensor_cast_dtype_streamed(f32t, 0, 17);
	ASSERT_TRUE("cast to BF16", strcmp(tensor_dtype_name(to_bf), "BF16") == 0);
	TensorHandle back_f32 = tensor_cast_dtype_streamed(to_bf, 0, 14);
	ASSERT_TRUE("cast back to F32", strcmp(tensor_dtype_name(back_f32), "F32") == 0);
	double rt[2];
	tensor_to_doubles(back_f32, rt);
	ASSERT_NEAR("bf16 roundtrip value[1]", rt[1], 2.25, 1e-2);

	/* I32 create + dtype + read. */
	double idata[] = {1.0, 2.0, 3.0};
	TensorHandle i32t = tensor_create_1d_streamed(3, hcopy(idata, 3), 0, 0, 10);
	ASSERT_TRUE("i32 dtype is I32", strcmp(tensor_dtype_name(i32t), "I32") == 0);
	double iout[3];
	tensor_to_doubles(i32t, iout);
	ASSERT_NEAR("i32 value[2]", iout[2], 3.0, 1e-10);

	/* Bool create + dtype + read. */
	double booldata[] = {1.0, 0.0, 1.0};
	TensorHandle bt = tensor_create_1d_streamed(3, hcopy(booldata, 3), 0, 0, 1);
	ASSERT_TRUE("bool dtype is BOOL", strcmp(tensor_dtype_name(bt), "BOOL") == 0);
	double boolout[3];
	tensor_to_doubles(bt, boolout);
	ASSERT_NEAR("bool value[0]", boolout[0], 1.0, 1e-10);
	ASSERT_NEAR("bool value[1]", boolout[1], 0.0, 1e-10);

	/* one-hot is dtype-aware: dtag selects the output dtype. */
	int* ohtok = (int*)malloc(2 * sizeof(int));
	ohtok[0] = 1;
	ohtok[1] = 0;                                      /* [2,3] one-hot, flattened to [6] */
	TensorHandle oh = tensor_one_hot(ohtok, 2, 3, 10); /* dtag 10 = I32 */
	ASSERT_TRUE("one_hot honors dtag (I32)", strcmp(tensor_dtype_name(oh), "I32") == 0);
	double ohout[6];
	tensor_to_doubles(oh, ohout);
	ASSERT_NEAR("one_hot tok0->pos1", ohout[1], 1.0, 1e-10);
	ASSERT_NEAR("one_hot tok1->pos3", ohout[3], 1.0, 1e-10);
	free(ohtok);

	param_clear();
}
#endif
