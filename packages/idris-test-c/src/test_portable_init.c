/* test_portable_init.c — the shared portable parameter-init RNG.
 *
 * This generator (shared_utils.c) is what makes IDRISML_PORTABLE_INIT
 * give bit-identical weights across tape / torch / mlx. It is compiled
 * once with unified symbols, so these assertions hold in every backend's
 * test binary and pin the ONE definition all three share.
 *
 * The end-to-end proof lives outside criterion: `example-supervised
 * --epochs 1 --seed 42` yields loss=0.8479759204519675 on tape and
 * 0.7119292274904563 on torch by default, and 0.8479759204519675 on both
 * under IDRISML_PORTABLE_INIT=1.
 */

#include <criterion/criterion.h>
#include <math.h>
#include "../../backends/shared_utils.h"

/* Same seed → same stream. This is the property the whole feature rests
   on: each backend seeds this generator from the run's seed and fills
   its own host buffer, so identical streams mean identical weights. */
Test(portable_init, same_seed_same_stream) {
	double a[16], b[16];
	idrisml_portable_init_seed(42);
	idrisml_portable_fill_normal(a, 16, 0.0, 1.0);
	idrisml_portable_init_seed(42);
	idrisml_portable_fill_normal(b, 16, 0.0, 1.0);
	for (int i = 0; i < 16; i++)
		cr_assert_float_eq(a[i], b[i], 0.0, "sample %d differs across identical seeds", i);
}

Test(portable_init, different_seed_different_stream) {
	double a[8], b[8];
	idrisml_portable_init_seed(1);
	idrisml_portable_fill_normal(a, 8, 0.0, 1.0);
	idrisml_portable_init_seed(2);
	idrisml_portable_fill_normal(b, 8, 0.0, 1.0);
	int same = 1;
	for (int i = 0; i < 8; i++)
		if (a[i] != b[i]) same = 0;
	cr_assert(same == 0, "seeds 1 and 2 produced an identical stream");
}

/* Re-seeding must drop the cached Box-Muller half, otherwise the first
   sample after a re-seed is a stale value from the previous stream. */
Test(portable_init, reseed_drops_cached_half) {
	double a[2], b[2];
	idrisml_portable_init_seed(7);
	idrisml_portable_fill_normal(a, 1, 0.0, 1.0); /* leaves one half cached */
	idrisml_portable_init_seed(7);
	idrisml_portable_fill_normal(b, 2, 0.0, 1.0);
	cr_assert_float_eq(a[0], b[0], 0.0, "first sample after re-seed is not the stream's first");
}

/* mean/std are an affine map over the standard normal, so the same seed
   at (0,1) and at (mu,sigma) must agree elementwise after scaling. */
Test(portable_init, mean_std_is_affine) {
	double z[32], s[32];
	idrisml_portable_init_seed(99);
	idrisml_portable_fill_normal(z, 32, 0.0, 1.0);
	idrisml_portable_init_seed(99);
	idrisml_portable_fill_normal(s, 32, 5.0, 3.0);
	for (int i = 0; i < 32; i++)
		cr_assert_float_eq(s[i], 5.0 + 3.0 * z[i], 1e-12, "element %d is not the affine image", i);
}

Test(portable_init, produces_plausible_normal_moments) {
	enum { N = 20000 };
	static double buf[N];
	idrisml_portable_init_seed(2024);
	idrisml_portable_fill_normal(buf, N, 0.0, 1.0);
	double sum = 0.0;
	for (int i = 0; i < N; i++)
		sum += buf[i];
	double mean = sum / N;
	double var = 0.0;
	for (int i = 0; i < N; i++)
		var += (buf[i] - mean) * (buf[i] - mean);
	var /= (N - 1);
	/* Generous bands — this catches a broken generator, not a biased one. */
	cr_assert(fabs(mean) < 0.05, "sample mean %g is implausible for N(0,1)", mean);
	cr_assert(fabs(var - 1.0) < 0.05, "sample variance %g is implausible for N(0,1)", var);
}

/* The env gate must be off unless explicitly enabled: the default path
   is each backend's own fast fused init, and flipping that silently
   would be a performance regression on torch and mlx. */
Test(portable_init, disabled_by_default) {
	cr_assert(idrisml_portable_init_enabled() == 0,
	          "portable init must be off unless IDRISML_PORTABLE_INIT is set "
	          "(the criterion suites run without it)");
}
