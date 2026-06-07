/* shared_utils.h — typedef + signatures for the backend-agnostic
 * helpers in shared_utils.c. The struct definition lives here so
 * mnist.c (which keeps the tensor-touching `mnist_get_image` /
 * `mnist_free`) and shared_utils.c (which owns the rest) agree
 * on the layout. */

#ifndef IDRISML_SHARED_UTILS_H
#define IDRISML_SHARED_UTILS_H

#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

/* Index-array helpers (DataLoader). */
int* create_index_array(int n);
int* shuffle_index_array(int* arr, int n);
int index_array_get(int* arr, int i);

/* RSS reporting. */
int get_rss_mb(void);
int get_current_rss_mb(void);

/* IDX-format dataset helpers (MNIST + similar) live in idx.{c,h};
 * they are deliberately outside this header because they have their
 * own struct (IdxDataset) and live outside the per-backend dispatch
 * surface (no rename header, no backend.h declaration). */

/* Dropout RNG — drives the process-global rand(). */
int dropout_random_seed(int x);

/* Wall-clock provider — gettimeofday-based monotonic-ish millisecond
   reading. Single unified definition (compile-once, no per-backend
   rename) so multi-link builds avoid duplicate-symbol collisions
   when several backends route through the shared training port. */
double _wall_ms(void);

/* C buffer helpers (host malloc / free / element read/write).
 * Backend-agnostic; one definition for all backends. */
double* tensor_alloc_doubles(int n);
void tensor_free_doubles(double* buf);
double tensor_read_double(double* buf, int idx);
void* tensor_write_double_return(void* buf, int off, double val);

int* tensor_alloc_ints(int n);
void tensor_free_ints(int* buf);
int* tensor_write_int_return(int* buf, int off, int val);

/* Byte buffer helpers (#411 B2 — packed-ternary input to
 * tensor_create_ternary_packed_2d). Mirrors the int helpers above but
 * for unsigned 8-bit values. */
uint8_t* tensor_alloc_bytes(int n);
void tensor_free_bytes(uint8_t* buf);
uint8_t* tensor_write_byte_return(uint8_t* buf, int off, int val);

/* Tensor-pointer arrays (the Idris-side staging buffer used by
 * stack/cat). Stores opaque void* handles; the per-backend
 * `tensor_stack_from_array` / `tensor_cat_from_array` consumers
 * reinterpret as TensorHandle* internally. */
void** tensor_ptr_array_alloc(int n);
void* tensor_ptr_array_set_return(void* arr, int idx, void* t);

/* bf16 / f16 <-> double bit conversions.
 *
 * bf16 is the high 16 bits of an IEEE-754 binary32. f16 is IEEE-754 binary16.
 * Both go through `float` then widen/narrow to `double`. Used by
 * safetensors.c (on-disk half-precision I/O) and backend_tape.c
 * (Phase 4 `tape_round_to_dtype` for DT_BF16 / DT_F16). Backend-agnostic
 * so they live here, one definition for the dylib. */
double bf16_bits_to_double(uint16_t h);
uint16_t double_to_bf16_bits(double d);
double f16_bits_to_double(uint16_t h);
uint16_t double_to_f16_bits(double d);

/* Ternary {-1, 0, +1} packing (#411 BitNet).
 *
 * Storage: 2 bits per element, 4 elements per byte. Encoding chosen as
 * 2-bit two's complement so a sign-extending fetch of a slot yields
 * the original integer:
 *
 *   00 -> 0    01 -> +1    11 -> -1    10 -> reserved / invalid
 *
 * Within a byte, slot 0 lives in bits 0..1 (low) and slot 3 in bits
 * 6..7 (high). Inputs outside {-1, 0, +1} abort.
 *
 * `ternary_pack(values, n, out)` writes ceil(n/4) bytes to `out` and
 * returns that byte count. Trailing slots in the final byte (when
 * n % 4 != 0) are zero-padded.
 *
 * `ternary_unpack(packed, n, out)` reads ceil(n/4) bytes from `packed`
 * and writes n elements to `out`. */
int ternary_pack(const int8_t* values, int n, uint8_t* out);
void ternary_unpack(const uint8_t* packed, int n, int8_t* out);

#ifdef __cplusplus
}
#endif

#endif /* IDRISML_SHARED_UTILS_H */
