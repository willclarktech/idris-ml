/* test_ternary_pack.c — ternary pack/unpack roundtrip (#411 B1).
 *
 * BitNet b1.58 stores weights as ternary values {-1, 0, +1} in a
 * 2-bit-per-element format. Encoding: 2-bit two's complement so that
 * sign-extending a single slot gives the correct integer value:
 *
 *   00 -> 0   01 -> +1   11 -> -1   10 -> reserved/invalid
 *
 * Within a byte, slot 0 occupies bits 0..1 (low) and slot 3 occupies
 * bits 6..7 (high), so the natural decode of `(byte >> (slot*2)) & 3`
 * pulls out the right two bits.
 *
 * The test asserts both directions:
 *   (a) byte-exact packed encoding for a hand-checked length-12 input
 *       (each byte's bits are precomputed in the test docstring so a
 *        mistake in the encoder shows up as a specific byte mismatch);
 *   (b) value-exact unpack roundtrip for the same input + a non-
 *       multiple-of-4 length to exercise the trailing-slot padding.
 */

#include <criterion/criterion.h>
#include <string.h>
#include <stdint.h>
#include "../../../../shared_utils.h"

Test(ternary_pack, roundtrip_length_12) {
    /* 12 values, fits in exactly 3 bytes (4 slots / byte). */
    int8_t values[12] = { 1, 0, -1, 1, -1, 0, 0, 1, -1, -1, 1, 0 };
    uint8_t packed[3] = { 0xAA, 0xAA, 0xAA };  /* poison; pack must overwrite */

    int produced = ternary_pack(values, 12, packed);
    cr_assert_eq(produced, 3,
        "ternary_pack should produce ceil(12/4) = 3 bytes (got %d)", produced);

    /* Hand-computed encoding. Per-slot codes are or'd in at
     * `slot * 2`, so slot 0 lands in the low 2 bits and slot 3 in
     * the high 2 bits. Reading high->low therefore reverses the
     * value list:
     *  byte 0: slots [0..3] = {+1, 0, -1, +1} -> codes {01,00,11,01}
     *           byte high->low: 01 11 00 01 = 0x71
     *  byte 1: slots [0..3] = {-1, 0, 0, +1}  -> codes {11,00,00,01}
     *           byte high->low: 01 00 00 11 = 0x43
     *  byte 2: slots [0..3] = {-1, -1, +1, 0} -> codes {11,11,01,00}
     *           byte high->low: 00 01 11 11 = 0x1F
     */
    cr_assert_eq(packed[0], 0x71,
        "byte 0 expected 0x71 for slots {+1, 0, -1, +1}; got 0x%02X", packed[0]);
    cr_assert_eq(packed[1], 0x43,
        "byte 1 expected 0x43 for slots {-1, 0, 0, +1}; got 0x%02X", packed[1]);
    cr_assert_eq(packed[2], 0x1F,
        "byte 2 expected 0x1F for slots {-1, -1, +1, 0}; got 0x%02X", packed[2]);

    int8_t unpacked[12];
    memset(unpacked, 0x7F, sizeof(unpacked));  /* poison to catch missed writes */
    ternary_unpack(packed, 12, unpacked);
    for (int i = 0; i < 12; i++) {
        cr_assert_eq(unpacked[i], values[i],
            "round-trip mismatch at index %d: expected %d, got %d",
            i, (int)values[i], (int)unpacked[i]);
    }
}

Test(ternary_pack, roundtrip_length_5_padding) {
    /* Non-multiple of 4: 5 elements -> 2 packed bytes; second byte's
       slots 1..3 are unused (encoded as 0 = ternary zero). Round-trip
       must preserve the input length, not bleed into the padding. */
    int8_t values[5] = { -1, 1, -1, 1, -1 };
    uint8_t packed[2] = { 0xAA, 0xAA };

    int produced = ternary_pack(values, 5, packed);
    cr_assert_eq(produced, 2,
        "ternary_pack length=5 should produce ceil(5/4) = 2 bytes (got %d)", produced);

    int8_t unpacked[5];
    ternary_unpack(packed, 5, unpacked);
    for (int i = 0; i < 5; i++)
        cr_assert_eq(unpacked[i], values[i],
            "non-multiple round-trip mismatch at index %d: expected %d, got %d",
            i, (int)values[i], (int)unpacked[i]);
}

Test(ternary_pack, all_zeros) {
    /* All-zero input -> all-zero packed bytes (the encoding chosen so
       calloc-zero buffers decode as ternary zero). */
    int8_t values[8] = {0};
    uint8_t packed[2] = { 0xFF, 0xFF };

    int produced = ternary_pack(values, 8, packed);
    cr_assert_eq(produced, 2);
    cr_assert_eq(packed[0], 0x00, "all-zero ternary -> zero-byte encoding");
    cr_assert_eq(packed[1], 0x00);
}
