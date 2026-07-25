/* Criterion coverage for dropout_random_seed (shared_utils.c).
 *
 * `Ml.Nn.Dropout` calls this once per forward for a fresh mask seed, so
 * successive calls must differ. A constant seed is not dropout at all: it
 * deletes one fixed subset of the activations for the whole run, which the
 * model then adapts to. Only the tape backend consumes the seed — torch and
 * mlx draw from their own RNG — but tape is the default.
 *
 * Backend-agnostic (shared_utils.c is compiled into every backend), hence
 * this lives outside the per-backend test dirs.
 */

#include "port_assert.h" /* pulls in backend.h, shared_utils.h, test_helpers.h */

/* The Idris call site passes 0; pin the argument it actually uses. */
Test(dropout_seed, successive_calls_differ) {
	enum { CALLS = 32 };
	int seen[CALLS];
	for (int i = 0; i < CALLS; i++)
		seen[i] = dropout_random_seed(0);

	int distinct = 0;
	for (int i = 0; i < CALLS; i++) {
		int dup = 0;
		for (int j = 0; j < i; j++)
			if (seen[j] == seen[i]) dup = 1;
		if (!dup) distinct++;
	}
	/* A sound seed source gives ~32 distinct values out of 32 draws. Assert
	   well above 1 rather than exactly 32, so an unlucky collision cannot
	   fail the suite. The broken version scores exactly 1. */
	cr_assert_gt(distinct, CALLS / 2,
	             "dropout_random_seed(0) returned only %d distinct value(s) over %d calls "
	             "— a constant seed means every dropout forward reuses one mask",
	             distinct, CALLS);
}

/* The seed feeds tensor_dropout, so two forwards over the same input must be
   able to drop different elements. This is the property the layer depends on;
   the test above pins its cause. */
Test(dropout_seed, two_forwards_can_differ) {
	param_clear();
	enum { N = 64 };
	double in_src[N];
	for (int i = 0; i < N; i++)
		in_src[i] = 1.0;

	int shape[1] = {N};
	TensorHandle a = tensor_create(in_src, shape, 1, 0);
	TensorHandle b = tensor_create(in_src, shape, 1, 0);

	TensorHandle da = tensor_dropout(a, 0.5, /*training=*/1, (unsigned)dropout_random_seed(0));
	TensorHandle db = tensor_dropout(b, 0.5, /*training=*/1, (unsigned)dropout_random_seed(0));

	double oa[N], ob[N];
	tensor_to_doubles(da, oa);
	tensor_to_doubles(db, ob);

	int differing = 0;
	for (int i = 0; i < N; i++)
		if ((oa[i] == 0.0) != (ob[i] == 0.0)) differing++;

	/* Two independent p=0.5 masks over 64 elements disagree on ~32 of them.
	   Zero disagreement means the two forwards shared a mask. */
	cr_assert_gt(differing, 0,
	             "two dropout forwards produced identical masks over %d elements — "
	             "the seed is not advancing",
	             N);
}
