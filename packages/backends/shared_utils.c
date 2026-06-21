/* Shared utilities: pure-C helpers that don't touch any backend's
 * tensor primitives. Compiled WITHOUT any rename header so the
 * symbols emerge under their unified names natively — these are
 * intentionally backend-agnostic and don't participate in the
 * per-backend dispatch surface. Live as a single TU in the dylib
 * (one definition each, no suffixed variants). */

#include "shared_utils.h"
#include <stdio.h>
#include <stdlib.h>
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
	return tv.tv_sec * 1000.0 + tv.tv_usec / 1000.0;
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
	return (int)(usage.ru_maxrss / (1024 * 1024));
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
	if (task_info(mach_task_self(), MACH_TASK_BASIC_INFO, (task_info_t)&info, &count) ==
	    KERN_SUCCESS)
		return (int)(info.resident_size / (1024 * 1024));
#endif
	return get_rss_mb();
}

/* --- Dropout RNG (process-global rand() driver) --- */

int dropout_random_seed(int x) {
	return rand() % (x + 1);
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
