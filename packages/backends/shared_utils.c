/* Shared utilities: pure-C helpers that don't touch any backend's
 * tensor primitives. Compiled WITHOUT any rename header so the
 * symbols emerge under their unified names natively — these are
 * intentionally backend-agnostic and don't participate in the
 * per-backend dispatch surface. Live as a single TU in the dylib
 * (one definition each, no suffixed variants). */

#include "shared_utils.h"
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <stdint.h>
#include <string.h>
#include <sys/resource.h>
#include <sys/time.h>

#ifdef __APPLE__
#include <mach/mach.h>
#endif

/* --- Wall clock --- */

double _wall_ms(void) {
	struct timeval tv;
	gettimeofday(&tv, NULL);
	return (double)tv.tv_sec * 1000.0 + (double)tv.tv_usec / 1000.0;
}

/* --- Seeded per-stream index array (DataStream) ---
 *
 * Carries its own xoshiro256++ RNG state so each stream shuffles
 * reproducibly from its seed, independent of the process-global rand().
 * The state is seeded once at creation
 * (splitmix64-expanded from the user seed) and ADVANCES on each
 * reshuffle, so epoch k's permutation is deterministic but distinct from
 * epoch k-1's — and two streams created with the same seed produce the
 * same sequence of permutations regardless of interleaving. */

typedef struct {
	int* idx;
	int n;
	uint64_t s[4];
} SeededIndexArray;

static uint64_t splitmix64_next(uint64_t* x) {
	uint64_t z = (*x += 0x9e3779b97f4a7c15ULL);
	z = (z ^ (z >> 30)) * 0xbf58476d1ce4e5b9ULL;
	z = (z ^ (z >> 27)) * 0x94d049bb133111ebULL;
	return z ^ (z >> 31);
}

static uint64_t xoshiro_rotl(uint64_t x, int k) {
	return (x << k) | (x >> (64 - k));
}

static uint64_t xoshiro_next(uint64_t s[4]) {
	const uint64_t result = xoshiro_rotl(s[1] * 5, 7) * 9;
	const uint64_t t = s[1] << 17;
	s[2] ^= s[0];
	s[3] ^= s[1];
	s[1] ^= s[2];
	s[0] ^= s[3];
	s[2] ^= t;
	s[3] = xoshiro_rotl(s[3], 45);
	return result;
}

void* create_seeded_index_array(int n, unsigned long long seed) {
	SeededIndexArray* h = (SeededIndexArray*)malloc(sizeof(SeededIndexArray));
	h->idx = (int*)malloc((size_t)n * sizeof(int));
	for (int i = 0; i < n; i++)
		h->idx[i] = i;
	h->n = n;
	uint64_t sm = (uint64_t)seed;
	for (int i = 0; i < 4; i++)
		h->s[i] = splitmix64_next(&sm);
	return h;
}

void* seeded_index_array_shuffle(void* handle) {
	SeededIndexArray* h = (SeededIndexArray*)handle;
	for (int i = h->n - 1; i > 0; i--) {
		int j = (int)(xoshiro_next(h->s) % (uint64_t)(i + 1));
		int tmp = h->idx[i];
		h->idx[i] = h->idx[j];
		h->idx[j] = tmp;
	}
	return handle;
}

int seeded_index_array_get(void* handle, int i) {
	return ((SeededIndexArray*)handle)->idx[i];
}

/* --- RSS reporting --- */

/* Peak RSS in MB. macOS reports ru_maxrss in bytes; Linux in KB. */
int get_rss_mb(void) {
	struct rusage usage;
	getrusage(RUSAGE_SELF, &usage);
#ifdef __APPLE__
	return (int)(usage.ru_maxrss / (1024L * 1024));
#else
	return (int)(usage.ru_maxrss / 1024);
#endif
}

/* Current resident-set size in MB. macOS exposes the live RSS via
 * mach_task_basic_info; on Linux we fall back to the peak (the
 * portable rusage path). */
int get_current_rss_mb(void) {
#ifdef __APPLE__
	mach_task_basic_info_data_t info;
	mach_msg_type_number_t count = MACH_TASK_BASIC_INFO_COUNT;
	/* task_info() fills `info` via the pointer on KERN_SUCCESS; cppcheck can't see the syscall */
	/* cppcheck-suppress uninitvar */
	if (task_info(mach_task_self(), MACH_TASK_BASIC_INFO, (task_info_t)&info, &count) ==
	    KERN_SUCCESS)
		return (int)(info.resident_size / (1024ULL * 1024));
#endif
	return get_rss_mb();
}

/* --- Process-global PRNG ---
 *
 * SplitMix64 (Steele, Lea, Flood 2014) rather than libc `rand()`, whose
 * algorithm the C standard leaves to the implementation: glibc and the BSD
 * libc macOS ships give different streams from the same seed, so a run's
 * parameter init and dropout masks would not be reproducible across the two
 * CI legs. This is the same generator `Gym.Rng` implements in pure Idris.
 *
 * Every Idris-side draw (`Ml.Compat.Random`) and the dropout mask seed come
 * through here, so one `idrisml_srand` pins the whole run. */

static uint64_t idrisml_rng_state = 0x9E3779B97F4A7C15ULL;

void idrisml_srand(uint64_t seed) {
	idrisml_rng_state = seed;
}

uint64_t idrisml_rand64(void) {
	idrisml_rng_state += 0x9E3779B97F4A7C15ULL;
	uint64_t z = idrisml_rng_state;
	z = (z ^ (z >> 30)) * 0xBF58476D1CE4E5B9ULL;
	z = (z ^ (z >> 27)) * 0x94D049BB133111EBULL;
	return z ^ (z >> 31);
}

/* Non-negative draw in [0, 2^31), matching what `rand()` handed callers. */
int idrisml_rand(void) {
	return (int)(idrisml_rand64() >> 33);
}

/* Fresh mask seed per dropout forward — successive calls must differ, or the
 * layer deletes a fixed subset of activations instead of regularizing. `x` is
 * a dummy the Idris side passes to keep the call from being folded; it must
 * not constrain the result. Pinned by test_dropout_seed.c. */
int dropout_random_seed(int x) {
	(void)x;
	return idrisml_rand();
}

/* --- C buffer helpers (host malloc/free + element read/write) ---
 *
 * Backend-agnostic host-memory primitives. Each backend used to ship
 * byte-identical malloc wrappers; consolidated here per the
 * 2026-05-20 alias-machinery teardown. The `_return` variants thread
 * the buffer pointer through let-chains so Idris-Chez codegen can't
 * elide the FFI call. The non-`_return` writers are inlined into the
 * `_return` wrappers, so the bare `tensor_write_double` /
 * `tensor_ptr_array_set` are no longer part of the C surface. */

double* tensor_alloc_doubles(int n) {
	return (double*)calloc(n, sizeof(double));
}

void tensor_free_doubles(double* buf) {
	free(buf);
}

double tensor_read_double(double* buf, int idx) {
	return buf[idx];
}

void* tensor_write_double_return(void* buf, int off, double val) {
	((double*)buf)[off] = val;
	return buf;
}

int* tensor_alloc_ints(int n) {
	return (int*)calloc(n, sizeof(int));
}

void tensor_free_ints(int* buf) {
	free(buf);
}

int* tensor_write_int_return(int* buf, int off, int val) {
	buf[off] = val;
	return buf;
}

/* Byte buffer helpers (#411 B2). Used by Idris-side construction of
 * the packed-ternary byte buffer passed to
 * tensor_create_ternary_packed_2d. `val` is an int from Idris because
 * Idris-2 doesn't have a Bits8 FFI type — we narrow to uint8_t here. */
uint8_t* tensor_alloc_bytes(int n) {
	return (uint8_t*)calloc((size_t)n, 1);
}

void tensor_free_bytes(uint8_t* buf) {
	free(buf);
}

uint8_t* tensor_write_byte_return(uint8_t* buf, int off, int val) {
	buf[off] = (uint8_t)(val & 0xff);
	return buf;
}

void** tensor_ptr_array_alloc(int n) {
	return (void**)calloc(n, sizeof(void*));
}

void* tensor_ptr_array_set_return(void* arr, int idx, void* t) {
	((void**)arr)[idx] = t;
	return arr;
}

void tensor_ptr_array_free(void** arr) {
	free(arr);
}

/* ----------------------------------------------------------------------
   bf16 / f16 <-> double bit conversions

   bf16 is the high 16 bits of an IEEE-754 binary32. f16 is IEEE-754
   binary16. Both go through `float` then widen/narrow to `double`. These
   are the only dtypes that aren't a plain integral cast; everything moves
   through the `double` lingua franca.

   Lifted verbatim from safetensors.c (where they shipped first) so the
   tape backend's Phase 4 inference-dtype rounding can call the same code
   path — keeping safetensors round-trips and tape's `tape_round_to_dtype`
   in lockstep on every half-precision value.
   ---------------------------------------------------------------------- */

double bf16_bits_to_double(uint16_t h) {
	uint32_t bits = (uint32_t)h << 16; /* bf16 occupies the f32 high half */
	float f;
	memcpy(&f, &bits, sizeof(f));
	return (double)f;
}

uint16_t double_to_bf16_bits(double d) {
	float f = (float)d;
	uint32_t bits;
	memcpy(&bits, &f, sizeof(bits));
	/* NaN: keep it quiet and non-zero so it survives the round-trip. */
	if ((bits & 0x7f800000u) == 0x7f800000u && (bits & 0x007fffffu) != 0u)
		return (uint16_t)((bits >> 16) | 0x0040u);
	/* Round to nearest, ties to even on the dropped low 16 bits. */
	uint32_t rounding_bias = 0x00007fffu + ((bits >> 16) & 1u);
	bits += rounding_bias;
	return (uint16_t)(bits >> 16);
}

double f16_bits_to_double(uint16_t h) {
	uint32_t sign = (uint32_t)(h & 0x8000u) << 16;
	uint32_t exp = (h >> 10) & 0x1fu;
	uint32_t mant = h & 0x3ffu;
	uint32_t bits;
	if (exp == 0) {
		if (mant == 0) {
			bits = sign; /* +/- zero */
		} else {
			/* Subnormal: normalize into f32. */
			exp = 1;
			while ((mant & 0x400u) == 0) {
				mant <<= 1;
				exp--;
			}
			mant &= 0x3ffu;
			bits = sign | ((exp + (127 - 15)) << 23) | (mant << 13);
		}
	} else if (exp == 0x1fu) {
		bits = sign | 0x7f800000u | (mant << 13); /* Inf / NaN */
	} else {
		bits = sign | ((exp + (127 - 15)) << 23) | (mant << 13);
	}
	float f;
	memcpy(&f, &bits, sizeof(f));
	return (double)f;
}

uint16_t double_to_f16_bits(double d) {
	float f = (float)d;
	uint32_t bits;
	memcpy(&bits, &f, sizeof(bits));
	uint32_t sign = (bits >> 16) & 0x8000u;
	int32_t exp = (int32_t)((bits >> 23) & 0xffu) - 127 + 15; /* rebias */
	uint32_t mant = bits & 0x7fffffu;

	if (((bits >> 23) & 0xffu) == 0xffu) {           /* Inf / NaN */
		if (mant) return (uint16_t)(sign | 0x7e00u); /* quiet NaN */
		return (uint16_t)(sign | 0x7c00u);           /* Inf */
	}
	if (exp >= 0x1f) return (uint16_t)(sign | 0x7c00u); /* overflow -> Inf */
	if (exp <= 0) {
		if (exp < -10) return (uint16_t)sign; /* underflow -> 0 */
		/* Subnormal: add implicit leading 1, shift, round to nearest even. */
		mant |= 0x800000u;
		uint32_t shift = (uint32_t)(14 - exp);
		uint32_t halfm = mant >> shift;
		uint32_t rem = mant & ((1u << shift) - 1u);
		uint32_t half = 1u << (shift - 1);
		if (rem > half || (rem == half && (halfm & 1u))) halfm++;
		return (uint16_t)(sign | halfm);
	}
	/* Normal: round mantissa to 10 bits, nearest, ties to even. */
	uint32_t halfm = mant >> 13;
	uint32_t rem = mant & 0x1fffu;
	if (rem > 0x1000u || (rem == 0x1000u && (halfm & 1u))) {
		halfm++;
		if (halfm == 0x400u) {
			halfm = 0;
			exp++;
		} /* mantissa carry */
		if (exp >= 0x1f) return (uint16_t)(sign | 0x7c00u);
	}
	return (uint16_t)(sign | ((uint32_t)exp << 10) | halfm);
}

/* ----------------------------------------------------------------------
   Ternary {-1, 0, +1} packing (#411 BitNet b1.58)

   Sub-byte storage: 2 bits per element, 4 elements per byte. Encoding
   is 2-bit two's complement so a sign-extending read of a slot yields
   the original integer:

     00 -> 0    01 -> +1    11 -> -1    10 -> reserved / invalid

   Within a byte, slot 0 lives in bits 0..1 (low) and slot 3 in bits
   6..7 (high). The encoding is also the calloc-zero default: an
   all-zero byte buffer unpacks as all-zero ternary, so freshly-
   allocated arena storage for a Ternary tensor has the right semantic
   default.

   Pack/unpack here are pure-C and backend-agnostic — Bitnet inference
   on every backend converges on the same packed layout regardless of
   what compute fabric ultimately reads it. Per-backend C kernels
   (#411 B3) will share the unpack to broadcast the row into compute
   dtype before matmul.
   ---------------------------------------------------------------------- */

int ternary_pack(const int8_t* values, int n, uint8_t* out) {
	int out_bytes = (n + 3) / 4;
	for (int b = 0; b < out_bytes; b++) {
		uint8_t byte = 0;
		for (int slot = 0; slot < 4; slot++) {
			int i = b * 4 + slot;
			if (i >= n) continue; /* trailing slot -> 0 (decodes to zero) */
			uint8_t code;
			switch (values[i]) {
			case 0:
				code = 0x0;
				break;
			case 1:
				code = 0x1;
				break;
			case -1:
				code = 0x3;
				break;
			default:
				fprintf(stderr,
				        "ternary_pack: invalid value %d at index %d "
				        "(expected -1, 0, or +1)\n",
				        (int)values[i], i);
				abort();
			}
			byte |= (uint8_t)((code & 0x3u) << (slot * 2));
		}
		out[b] = byte;
	}
	return out_bytes;
}

void ternary_unpack(const uint8_t* packed, int n, int8_t* out) {
	for (int i = 0; i < n; i++) {
		int b = i / 4;
		int slot = i % 4;
		uint8_t code = (uint8_t)((packed[b] >> (slot * 2)) & 0x3u);
		switch (code) {
		case 0x0:
			out[i] = 0;
			break;
		case 0x1:
			out[i] = 1;
			break;
		case 0x3:
			out[i] = -1;
			break;
		default: /* 0x2 = reserved */
			fprintf(stderr,
			        "ternary_unpack: invalid 2-bit code 0x%x at index %d "
			        "(byte %d slot %d)\n",
			        code, i, b, slot);
			abort();
		}
	}
}

/* ----------------------------------------------------------------------
   Portable parameter-init RNG (see shared_utils.h for the why).

   xoshiro256++ over a process-global state seeded through splitmix64,
   with paired Box-Muller normals. This is the generator tape has always
   used for param init, relocated here verbatim so all three backends can
   share one definition — tape's numerics are therefore unchanged, and
   torch/mlx can opt into identical weights via IDRISML_PORTABLE_INIT.

   Not thread-local: param construction happens once at model build time,
   before any forward pass.
   ---------------------------------------------------------------------- */

static uint64_t pi_state[4];
static int pi_seeded = 0;
static double pi_bm_cached_z1 = 0.0;
static int pi_bm_cached_has = 0;

static uint64_t pi_splitmix64(uint64_t* x) {
	uint64_t z = (*x += 0x9E3779B97F4A7C15ULL);
	z = (z ^ (z >> 30)) * 0xBF58476D1CE4E5B9ULL;
	z = (z ^ (z >> 27)) * 0x94D049BB133111EBULL;
	return z ^ (z >> 31);
}

static inline uint64_t pi_rotl(uint64_t x, int k) {
	return (x << k) | (x >> (64 - k));
}

static uint64_t pi_next(void) {
	const uint64_t r = pi_rotl(pi_state[0] + pi_state[3], 23) + pi_state[0];
	const uint64_t t = pi_state[1] << 17;
	pi_state[2] ^= pi_state[0];
	pi_state[3] ^= pi_state[1];
	pi_state[1] ^= pi_state[2];
	pi_state[0] ^= pi_state[3];
	pi_state[2] ^= t;
	pi_state[3] = pi_rotl(pi_state[3], 45);
	return r;
}

void idrisml_portable_init_seed(unsigned long long seed) {
	uint64_t sm = (uint64_t)seed;
	pi_state[0] = pi_splitmix64(&sm);
	pi_state[1] = pi_splitmix64(&sm);
	pi_state[2] = pi_splitmix64(&sm);
	pi_state[3] = pi_splitmix64(&sm);
	pi_seeded = 1;
	/* Drop any cached Box-Muller half so a re-seed is visible on the next
	   sample rather than after one stale value. */
	pi_bm_cached_has = 0;
}

static void pi_ensure_seeded(void) {
	if (!pi_seeded) idrisml_portable_init_seed(0);
}

/* Uniform in (0, 1). Top 53 bits → [0, 2^53); resamples on exact zero so
   Box-Muller's log() is always safe. */
static double pi_uniform01(void) {
	for (;;) {
		uint64_t r = pi_next() >> 11;
		if (r != 0) return (double)r * (1.0 / 9007199254740992.0);
	}
}

static double pi_normal01(void) {
	if (pi_bm_cached_has) {
		pi_bm_cached_has = 0;
		return pi_bm_cached_z1;
	}
	double u1 = pi_uniform01();
	double u2 = pi_uniform01();
	double r = sqrt(-2.0 * log(u1));
	double th = 6.283185307179586 * u2;
	pi_bm_cached_z1 = r * sin(th);
	pi_bm_cached_has = 1;
	return r * cos(th);
}

void idrisml_portable_fill_normal(double* out, int n, double mean, double std) {
	pi_ensure_seeded();
	for (int i = 0; i < n; i++)
		out[i] = mean + std * pi_normal01();
}

int idrisml_portable_init_enabled(void) {
	static int cached = -1;
	if (cached >= 0) return cached;
	const char* s = getenv("IDRISML_PORTABLE_INIT");
	if (!s || !*s || strcmp(s, "0") == 0 || strcasecmp(s, "false") == 0 ||
	    strcasecmp(s, "no") == 0 || strcasecmp(s, "off") == 0) {
		cached = 0;
	} else {
		cached = 1;
	}
	return cached;
}
