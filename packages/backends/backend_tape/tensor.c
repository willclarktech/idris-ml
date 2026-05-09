/* backend_tape/tensor.c — element-size + ABI<->internal dtype-tag
 * translation + lingua-franca rounding helpers.
 *
 * Phase 1.0.1 (per /Users/admin/.claude/plans/modular-petting-minsky.md).
 * Currently #included from backend_tape.c (single-TU build); will be
 * compiled as its own TU once Phase 1.0.4 splits the Makefile rule.
 */

/* Byte size of one element of the given internal DT_* tag. F32 ships as a
   real training dtype (dedicated 4-byte float storage, `tape_load_d` and
   `tape_store_d` element accessors, X-macro stamped kernels); F64 is the
   default and the lingua-franca path. The 8 inference dtypes (BF16, F16,
   I8, I16, I32, I64, U8, Bool) store packed bytes via the `double` lingua
   franca on first construct, so `tape_round_to_dtype` clamps incoming
   doubles into the dtype's representable precision and the per-element
   byte width here matches the ABI on-disk width. Branches on `dtype_tag`
   in hot read paths route through `tape_load_d`. */
static size_t tape_elem_size(int tag) {
    switch (tag) {
        case DT_F64:                                return sizeof(double);
        case DT_F32:                                return sizeof(float);
        case DT_BF16: case DT_F16:                  return sizeof(uint16_t);
        case DT_I8:   case DT_U8:   case DT_BOOL:   return sizeof(int8_t);
        case DT_I16:                                return sizeof(int16_t);
        case DT_I32:                                return sizeof(int32_t);
        case DT_I64:                                return sizeof(int64_t);
        default:                                    return sizeof(double);
    }
}

/* ABI RuntimeDType tag (kind-major layout, closed 2026-05-23: 1=Bool, 4=U8,
   8-11=I8/I16/I32/I64, 13-15=F16/F32/F64, 17=BF16; 0 reserved; sub-byte
   families 24-31 reserved) -> internal DT_* tag (F64=0). The internal enum
   stays dense (0..9) for hot-path switch density (`tape_load_d`,
   `tape_store_d`, the 67-case backward switch) — only this boundary
   translates. Unknown dtags abort loudly via tape_dtype_unsupported
   rather than silently falling back to F64. */
static int tape_tag_from_dtag(int dtag) {
    switch (dtag) {
        case 1:  return DT_BOOL;
        case 4:  return DT_U8;
        case 8:  return DT_I8;
        case 9:  return DT_I16;
        case 10: return DT_I32;
        case 11: return DT_I64;
        case 13: return DT_F16;
        case 14: return DT_F32;
        case 15: return DT_F64;
        case 17: return DT_BF16;
        default:
            fprintf(stderr, "[tape backend] invalid dtag=%d (expected one of "
                "{1=Bool, 4=U8, 8-11=I8/I16/I32/I64, 13-15=F16/F32/F64, 17=BF16})\n",
                dtag);
            abort();
    }
}

/* Round a value through the internal dtype's representable precision, staying
   in `double` (the lingua-franca storage). F32 + the integer/bool dtypes round
   honestly via plain casts; BF16/F16 round through the shared bit helpers
   lifted from safetensors.c — same round-to-nearest-even semantics on disk
   and in tape inference tensors. DT_F64 is identity. */
static double tape_round_to_dtype(double v, int tag) {
    switch (tag) {
        case DT_F32:  return (double)(float)v;
        case DT_BF16: return bf16_bits_to_double(double_to_bf16_bits(v));
        case DT_F16:  return f16_bits_to_double(double_to_f16_bits(v));
        case DT_I8:   return (double)(signed char)(long long)v;
        case DT_I16:  return (double)(short)(long long)v;
        case DT_I32:  return (double)(int)(long long)v;
        case DT_I64:  return (double)(long long)v;
        case DT_U8:   return (double)(unsigned char)(long long)v;
        case DT_BOOL: return v != 0.0 ? 1.0 : 0.0;
        default:      return v;  /* DT_F64 */
    }
}
