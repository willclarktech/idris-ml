/* core/elementwise/_helpers.h — dispatch wrappers + abort helper used
 * by every elementwise op file (add/sub/mul/...).
 *
 * Phase 1a.2 extraction (per /Users/admin/.claude/plans/modular-petting-minsky.md).
 * The X-macro stamped binop_elementwise_inner_{f32,f64} (in the .inc
 * file included from backend_tape.c) drive the actual kernels;
 * binop_elementwise / binop_elementwise_f32_disp wrap them with
 * per-op timing accounting. tape_abort_mixed_dtype is the F32/F64
 * mixed-dtype guard.
 */

#ifndef IDRISML_BACKEND_TAPE_ELEMENTWISE_HELPERS_H
#define IDRISML_BACKEND_TAPE_ELEMENTWISE_HELPERS_H

#include "../../../backend.h"   /* TensorHandle */

TensorHandle binop_elementwise(TensorHandle ha, TensorHandle hb, int op_tag,
                               double (*scalar_fn)(double, double));

TensorHandle binop_elementwise_f32_disp(TensorHandle ha, TensorHandle hb, int op_tag,
                                        float (*scalar_fn)(float, float));

TensorHandle unop_elementwise(TensorHandle ha, int op_tag,
                              double (*scalar_fn)(double));

TensorHandle unop_elementwise_f32_disp(TensorHandle ha, int op_tag,
                                       float (*scalar_fn)(float));

void tape_abort_mixed_dtype(const char* op) __attribute__((noreturn));

#endif /* IDRISML_BACKEND_TAPE_ELEMENTWISE_HELPERS_H */
