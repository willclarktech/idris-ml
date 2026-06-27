/* Miscellaneous tensor Criterion suite (safetensors roundtrip).
 *
 * Split out of the original bulk backend port. Shared assertion shims +
 * tolerances live in port_assert.h.
 */
#include "port_assert.h"

/* ================================================================
   T11: SafeTensors serialization round-trip
   ================================================================ */

Test(tensor_misc, safetensors_roundtrip) {
	param_clear();

	/* Register a 2D param and a 1D param with known values */
	double w_data[] = {1.0, 2.0, 3.0, 4.0, 5.0, 6.0};
	/* alloc must cover the 6 elements the create copies — the
	   zero-length buffer here was a heap-buffer-overflow READ the
	   ASAN lane caught (values are overwritten just below). */
	TensorHandle w = tensor_create_param_2d_f64(2, 3, tensor_alloc_doubles(6));
	/* Fill via our own buffer */
	{
		double* buf = tensor_alloc_doubles(6);
		for (int i = 0; i < 6; i++)
			tensor_write_double_return(buf, i, w_data[i]);
		tensor_free(w);
		w = tensor_create_param_2d_f64(2, 3, buf);
	}
	param_register("weights", w);

	double b_data[] = {10.0, 20.0};
	{
		double* buf = tensor_alloc_doubles(2);
		for (int i = 0; i < 2; i++)
			tensor_write_double_return(buf, i, b_data[i]);
		TensorHandle b = tensor_create_param_1d_f64(2, buf);
		param_register("biases", b);
	}

	ASSERT_TRUE("param_count == 2", param_count() == 2);

	/* Save */
	const char* path = "/tmp/idrisml_test.safetensors";
	int rc = param_save(path);
	ASSERT_TRUE("param_save returns 0", rc == 0);

	/* Verify file exists and has reasonable size */
	FILE* f = fopen(path, "rb");
	ASSERT_TRUE("file exists", f != NULL);
	if (f) {
		fseek(f, 0, SEEK_END);
		long sz = ftell(f);
		fclose(f);
		printf("  file size: %ld bytes\n", sz);
		ASSERT_TRUE("file size > 8", sz > 8);
	}

	/* Corrupt param data */
	{
		double* buf = (double*)malloc(6 * sizeof(double));
		tensor_to_doubles(w, buf);
		printf("  before corrupt: w[0]=%.1f w[5]=%.1f\n", buf[0], buf[5]);
		free(buf);
	}
	double zeros6[6] = {0};
	param_load_data(0, zeros6, 6);
	double zeros2[2] = {0};
	param_load_data(1, zeros2, 2);
	{
		double* buf = (double*)malloc(6 * sizeof(double));
		tensor_to_doubles(param_tensor(0), buf);
		ASSERT_NEAR("corrupted w[0]", buf[0], 0.0, 1e-15);
		free(buf);
	}

	/* Load */
	rc = param_load(path);
	ASSERT_TRUE("param_load returns 0", rc == 0);

	/* Verify restored values */
	{
		double* buf = (double*)malloc(6 * sizeof(double));
		tensor_to_doubles(param_tensor(0), buf);
		for (int i = 0; i < 6; i++) {
			char msg[64];
			snprintf(msg, sizeof(msg), "restored w[%d]", i);
			ASSERT_NEAR(msg, buf[i], w_data[i], 1e-15);
		}
		free(buf);
	}
	{
		double* buf = (double*)malloc(2 * sizeof(double));
		tensor_to_doubles(param_tensor(1), buf);
		ASSERT_NEAR("restored b[0]", buf[0], b_data[0], 1e-15);
		ASSERT_NEAR("restored b[1]", buf[1], b_data[1], 1e-15);
		free(buf);
	}

	/* Clean up */
	remove(path);
	param_clear();
}
