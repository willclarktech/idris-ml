/* Criterion suite for the named-subset checkpoint FFI:
 *   param_save_by_name / param_save_by_name_renamed (save)
 *   param_load_with_prefix / param_load_renamed     (load)
 *
 * These back the LoRA/PEFT adapter-IO path (save only the trainable
 * subset; load a backbone by prefix; remap idris-ml registry names to/from
 * peft's on-disk decorations). They were exercised end-to-end by the Idris
 * Checkpoint suite + the HF roundtrip but had no C unit — the coverage-gap
 * probe flagged all four as zero-hit. Each test is a save->clobber->load
 * round-trip that asserts the values come back, via the same g_active_port
 * data_read path test_param_registry uses (cross-backend).
 */

#include <criterion/criterion.h>
#include <stdlib.h>
#include <string.h>
#include "backend.h"
#include "shared/training/port.h"

extern void param_clear(void);

static TensorHandle mk_param(const char* name, int n) {
	/* tensor_create_param_1d_f64 takes ownership of the buffer (memcpy + free). */
	double* buf = calloc(n, sizeof(double));
	TensorHandle t = tensor_create_param_1d_f64(n, buf);
	tensor_set_requires_grad(t, 1);
	param_register(name, t);
	return t;
}

Test(param_checkpoint, save_by_name_then_load_with_prefix) {
	param_clear();
	(void)mk_param("layer.weight", 4);
	(void)mk_param("layer.bias", 2);
	double wv[] = {1.0, 2.0, 3.0, 4.0};
	double bv[] = {5.0, 6.0};
	param_load_data(0, wv, 4);
	param_load_data(1, bv, 2);

	/* Save ONLY layer.weight by name (newline-separated list, count=1). */
	const char* path = "/tmp/idrisml_ckpt_byname.safetensors";
	cr_assert_eq(param_save_by_name(path, "layer.weight", 1), 0, "param_save_by_name rc");

	/* Clobber in-registry, then warm-start back via the "layer." prefix. */
	double zero[] = {0, 0, 0, 0};
	param_load_data(0, zero, 4);
	cr_assert_eq(param_load_with_prefix(path, /*allow_cast=*/0, "layer."), 0,
	             "param_load_with_prefix rc");
	cr_assert_float_eq(g_active_port.data_read(param_tensor(0), 0), 1.0, 1e-12);
	cr_assert_float_eq(g_active_port.data_read(param_tensor(0), 3), 4.0, 1e-12);
	/* layer.bias was never saved; the prefix load must not have touched it. */
	cr_assert_float_eq(g_active_port.data_read(param_tensor(1), 1), 6.0, 1e-12);
}

Test(param_checkpoint, save_renamed_then_load_renamed) {
	param_clear();
	(void)mk_param("adapter.lora_A", 3);
	double av[] = {7.0, 8.0, 9.0};
	param_load_data(0, av, 3);

	/* Save registry "adapter.lora_A" under peft's on-disk decoration. */
	const char* path = "/tmp/idrisml_ckpt_renamed.safetensors";
	const char* ondisk = "base_model.adapter.lora_A.default.weight";
	cr_assert_eq(param_save_by_name_renamed(path, "adapter.lora_A", ondisk, 1), 0,
	             "param_save_by_name_renamed rc");

	/* Clobber, then load the on-disk decorated key back into the registry name. */
	double zero[] = {0, 0, 0};
	param_load_data(0, zero, 3);
	cr_assert_eq(param_load_renamed(path, /*allow_cast=*/0, "adapter.lora_A", ondisk, 1), 0,
	             "param_load_renamed rc");
	cr_assert_float_eq(g_active_port.data_read(param_tensor(0), 0), 7.0, 1e-12);
	cr_assert_float_eq(g_active_port.data_read(param_tensor(0), 2), 9.0, 1e-12);
}
