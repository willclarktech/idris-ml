/* Criterion suite for polyak_blend_pair (exact-name EMA target-sync).
 *
 * polyak_blend_pair(tau, online, target) blends ONE exactly-named pair:
 *   target.data <- (1 - tau) * target.data + tau * online.data
 * Names are matched with strcmp, NOT prefix — the property the old
 * prefix `polyak_blend` lacked. The over-match guard below is the RED:
 * with prefix matching, blending ("pk_on" -> "pk_tg") also dragged
 * "pk_on2" -> "pk_tg2" along (both share the "pk_on"/"pk_tg" prefix),
 * corrupting an unrelated pair.
 *
 * Colocated under backend_tape/ but compiled into all three backend test
 * binaries via the test_*.c glob; it calls the public renamed
 * `polyak_blend_pair`, so the same oracle runs on tape/torch/mlx.
 */

#include <criterion/criterion.h>
#include <stdlib.h>
#include <string.h>
#include "backend.h"
#include "test_helpers.h"

static double* heap1(double v) {
	double* buf = (double*)malloc(sizeof(double));
	buf[0] = v;
	return buf;
}

static void reg(const char* name, double v) {
	param_register(name, tensor_create_param_1d_f64(1, heap1(v)));
}

Test(training_polyak_blend_pair, exact_match_no_overmatch) {
	/* Two online/target pairs where one online name is a proper prefix of
	   the other ("pk_on" is a prefix of "pk_on2"). tau=1 => target := online.
	   Exact match must touch ONLY pk_tg; pk_tg2 stays at its init 9.0. */
	param_clear();
	reg("pk_on", 1.0);
	reg("pk_on2", 5.0);
	reg("pk_tg", 0.0);
	reg("pk_tg2", 9.0);

	int n = polyak_blend_pair(1.0, "pk_on", "pk_tg");
	cr_assert_eq(n, 1, "expected exactly 1 pair blended (got %d)", n);

	TensorHandle tg = (TensorHandle)param_tensor(2);
	TensorHandle tg2 = (TensorHandle)param_tensor(3);
	cr_assert_float_eq(tensor_item_1d(tg, 0), 1.0, TEST_TOL_RELAXED,
	                   "pk_tg should copy pk_on=1.0 (got %.9f)", tensor_item_1d(tg, 0));
	cr_assert_float_eq(tensor_item_1d(tg2, 0), 9.0, TEST_TOL_RELAXED,
	                   "pk_tg2 must be UNTOUCHED at 9.0 — prefix over-match bug (got %.9f)",
	                   tensor_item_1d(tg2, 0));
}

Test(training_polyak_blend_pair, blend_formula) {
	/* tau=0.25: target <- 0.75*0 + 0.25*1 = 0.25. */
	param_clear();
	reg("pk_b_on", 1.0);
	reg("pk_b_tg", 0.0);
	int n = polyak_blend_pair(0.25, "pk_b_on", "pk_b_tg");
	cr_assert_eq(n, 1);
	TensorHandle tg = (TensorHandle)param_tensor(1);
	cr_assert_float_eq(tensor_item_1d(tg, 0), 0.25, TEST_TOL_RELAXED,
	                   "blend should be 0.25 (got %.9f)", tensor_item_1d(tg, 0));
}

Test(training_polyak_blend_pair, absent_name_is_noop) {
	param_clear();
	reg("pk_x_on", 1.0);
	reg("pk_x_tg", 0.0);
	cr_assert_eq(polyak_blend_pair(0.5, "missing", "pk_x_tg"), 0, "absent online => 0");
	cr_assert_eq(polyak_blend_pair(0.5, "pk_x_on", "missing"), 0, "absent target => 0");
	TensorHandle tg = (TensorHandle)param_tensor(1);
	cr_assert_float_eq(tensor_item_1d(tg, 0), 0.0, TEST_TOL_RELAXED,
	                   "target unchanged after no-op (got %.9f)", tensor_item_1d(tg, 0));
}

static double val(const char* name) {
	(void)name;
	return 0.0;
}
