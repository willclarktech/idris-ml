/* Seeded per-stream index array (the DataStream shuffle engine) — a
 * backend-agnostic shared_utils.c surface. One TU compiled into all
 * three backend criterion binaries (the find test_*.c sweep). Verifies
 * the per-stream xoshiro256++ shuffle is reproducible from its seed,
 * distinct across seeds, advances on reshuffle (so successive epochs
 * differ but stay reproducible), and remains a permutation throughout.
 *
 * Prototypes declared locally rather than via shared_utils.h to dodge
 * the per-backend include-path setup — the symbols resolve against the
 * linked libidrisml (they are excluded from the rename headers). */

#include <criterion/criterion.h>
#include <stdlib.h>

void* create_seeded_index_array(int n, unsigned long long seed);
void* seeded_index_array_shuffle(void* handle);
int seeded_index_array_get(void* handle, int i);

static int orders_equal(void* a, void* b, int n) {
	for (int i = 0; i < n; i++)
		if (seeded_index_array_get(a, i) != seeded_index_array_get(b, i)) return 0;
	return 1;
}

static int is_permutation(void* h, int n) {
	int* seen = (int*)calloc((size_t)n, sizeof(int));
	for (int i = 0; i < n; i++) {
		int v = seeded_index_array_get(h, i);
		if (v < 0 || v >= n || seen[v]) {
			free(seen);
			return 0;
		}
		seen[v] = 1;
	}
	free(seen);
	return 1;
}

Test(seeded_index, same_seed_same_order) {
	int n = 64;
	void* a = create_seeded_index_array(n, 1234ULL);
	void* b = create_seeded_index_array(n, 1234ULL);
	seeded_index_array_shuffle(a);
	seeded_index_array_shuffle(b);
	cr_assert(orders_equal(a, b, n), "same seed must yield the same permutation");
	cr_assert(is_permutation(a, n), "shuffle must remain a permutation");
}

Test(seeded_index, diff_seed_diff_order) {
	int n = 64;
	void* a = create_seeded_index_array(n, 1ULL);
	void* b = create_seeded_index_array(n, 2ULL);
	seeded_index_array_shuffle(a);
	seeded_index_array_shuffle(b);
	cr_assert_not(orders_equal(a, b, n),
	              "different seeds must differ (a 1/64! coincidence otherwise)");
}

Test(seeded_index, reshuffle_advances_reproducibly) {
	int n = 64;
	int ep1[64];
	int ep2[64];

	/* Stream A: shuffle twice — epoch 1, then epoch 2. */
	void* a = create_seeded_index_array(n, 7ULL);
	seeded_index_array_shuffle(a);
	for (int i = 0; i < n; i++)
		ep1[i] = seeded_index_array_get(a, i);
	seeded_index_array_shuffle(a);
	for (int i = 0; i < n; i++)
		ep2[i] = seeded_index_array_get(a, i);

	int same = 1;
	for (int i = 0; i < n; i++)
		if (ep1[i] != ep2[i]) {
			same = 0;
			break;
		}
	cr_assert_not(same, "reshuffle must advance the RNG (epoch 2 != epoch 1)");

	/* Stream B with the same seed must reproduce BOTH epochs in order. */
	void* b = create_seeded_index_array(n, 7ULL);
	seeded_index_array_shuffle(b);
	for (int i = 0; i < n; i++)
		cr_assert_eq(seeded_index_array_get(b, i), ep1[i], "epoch 1 reproducible");
	seeded_index_array_shuffle(b);
	for (int i = 0; i < n; i++)
		cr_assert_eq(seeded_index_array_get(b, i), ep2[i], "epoch 2 reproducible");
}
