/* backend_tape/tensor.h — Tensor struct + internal dtype tags.
 *
 * Phase 1.0.1 (per /Users/admin/.claude/plans/modular-petting-minsky.md).
 * Extracted from backend_tape.c lines 25-120. Currently included only
 * via backend_tape.c (single-TU build); in Phase 1.0.4 this header
 * will be force-included into every backend_tape TU when the
 * Makefile switches to per-TU compile.
 */

#ifndef IDRISML_BACKEND_TAPE_TENSOR_H
#define IDRISML_BACKEND_TAPE_TENSOR_H

#include <stddef.h>
#include <stdint.h>

/* Internal tape dtype tags. Deliberately dense (0..9) **F64 = 0** so that
   every memset/calloc-zeroed Tensor defaults to F64 without touching the
   ~27 constructors. This is NOT the cross-language RuntimeDType ABI, which
   uses the kind-major layout (closed 2026-05-23: 1=Bool, 4=U8, 8-11=I8..I64,
   13-15=F16/F32/F64, 17=BF16; 0 reserved as invalid). The ABI dtag is
   mapped to/from this internal tag only at the create/cast boundary via
   `tape_tag_from_dtag`. Non-F64 tape tensors are inference/storage-only
   except F32 — F32 has real 4-byte float storage + autograd kernels
   (Phase 3); the rest store doubles rounded through the dtype's precision
   (the `double` lingua franca). */
enum { DT_F64 = 0, DT_F32, DT_BF16, DT_F16, DT_I8, DT_I16, DT_I32, DT_I64, DT_U8, DT_BOOL };

typedef struct {
    void* data;         /* owned, heap-allocated; element type per dtype_tag */
    int* shape;         /* owned, heap-allocated (NULL for scalar) */
    int rank;           /* 0 = scalar, 1 = vector, 2 = matrix */
    int numel;
    int requires_grad;
    int tape_idx;       /* index into tape (-1 if not tracked) */
    void* grad;         /* gradient storage (same shape/dtype as data), NULL if not allocated */
    int persistent;     /* 1 = param tensor (malloc'd), 0 = intermediate (arena) */
    int dtype_tag;      /* internal DT_* tag; 0 = DT_F64 (default for zeroed structs) */
} Tensor;

#endif /* IDRISML_BACKEND_TAPE_TENSOR_H */
