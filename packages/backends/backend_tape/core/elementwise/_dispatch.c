/* core/elementwise/_dispatch.c — elementwise kernel stamping + wrappers.
 *
 * The X-macro `_kernels.inc` is included twice (F64 + F32)
 * to stamp two sets of static `*_inner_{f32,f64}` kernels. The four
 * public wrappers (binop_elementwise, binop_elementwise_f32_disp,
 * unop_elementwise, unop_elementwise_f32_disp) bridge per-op files to
 * those stampings with per-op timing accounting.
 *
 * The static scalar function pointers (fn_add/sub/mul/.../fn_*_f32) used
 * to seed the dispatch live here too — they're only consumed by the
 * generated wrappers in this TU.
 *
 * tape_abort_mixed_dtype is the shared F32/F64 mixed-dtype guard used
 * by every op that touches multiple tensors.
 */

#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <Accelerate/Accelerate.h>
#include "../../arena.h"
#include "../../tape.h"
#include "../../tensor.h"
#include "../../broadcast.h"
#include "_helpers.h"
#include "../../../backend.h"

extern double _wall_ms(void);

/* F64 stamping. */
#define SCALAR    double
#define SFX(name) name##_f64
#define VDSP_VADD vDSP_vaddD
#define VDSP_VSUB vDSP_vsubD
#define VDSP_VMUL vDSP_vmulD
#define VDSP_VDIV vDSP_vdivD
#define VDSP_VNEG vDSP_vnegD
#define VV_EXP    vvexp
#define VV_LOG    vvlog
#define VV_SQRT   vvsqrt
#define VV_TANH   vvtanh
#define VV_FABS   vvfabs
#include "_kernels.inc"
#undef SCALAR
#undef SFX
#undef VDSP_VADD
#undef VDSP_VSUB
#undef VDSP_VMUL
#undef VDSP_VDIV
#undef VDSP_VNEG
#undef VV_EXP
#undef VV_LOG
#undef VV_SQRT
#undef VV_TANH
#undef VV_FABS

/* F32 stamping — uses cblas_s* / vDSP_* (no `D` suffix) / vv*f forms. */
#define SCALAR    float
#define SFX(name) name##_f32
#define VDSP_VADD vDSP_vadd
#define VDSP_VSUB vDSP_vsub
#define VDSP_VMUL vDSP_vmul
#define VDSP_VDIV vDSP_vdiv
#define VDSP_VNEG vDSP_vneg
#define VV_EXP    vvexpf
#define VV_LOG    vvlogf
#define VV_SQRT   vvsqrtf
#define VV_TANH   vvtanhf
#define VV_FABS   vvfabsf
#include "_kernels.inc"
#undef SCALAR
#undef SFX
#undef VDSP_VADD
#undef VDSP_VSUB
#undef VDSP_VMUL
#undef VDSP_VDIV
#undef VDSP_VNEG
#undef VV_EXP
#undef VV_LOG
#undef VV_SQRT
#undef VV_TANH
#undef VV_FABS

/* Public wrappers — each per-op file in core/elementwise/ calls these
   via _helpers.h. Per-op timing is accumulated into the binop_inside_*
   counters (read by backend_profile_report). */
TensorHandle binop_elementwise(TensorHandle ha, TensorHandle hb, int op_tag,
                               double (*scalar_fn)(double, double)) {
    extern double prof_binop_inside_ms[];
    extern int prof_binop_inside_count[];
    extern double _wall_ms(void);
    double _b0 = _wall_ms();
    TensorHandle r = binop_elementwise_inner_f64(ha, hb, op_tag, scalar_fn);
    if (op_tag >= 0 && op_tag < OP_COUNT) {
        prof_binop_inside_ms[op_tag] += _wall_ms() - _b0;
        prof_binop_inside_count[op_tag]++;
    }
    return r;
}

TensorHandle binop_elementwise_f32_disp(TensorHandle ha, TensorHandle hb, int op_tag,
                                        float (*scalar_fn)(float, float)) {
    extern double prof_binop_inside_ms[];
    extern int prof_binop_inside_count[];
    extern double _wall_ms(void);
    double _b0 = _wall_ms();
    TensorHandle r = binop_elementwise_inner_f32(ha, hb, op_tag, scalar_fn);
    if (op_tag >= 0 && op_tag < OP_COUNT) {
        prof_binop_inside_ms[op_tag] += _wall_ms() - _b0;
        prof_binop_inside_count[op_tag]++;
    }
    return r;
}

TensorHandle unop_elementwise(TensorHandle ha, int op, double (*fn)(double)) {
    /* Body lives in _kernels.inc as unop_elementwise_f64. */
    return unop_elementwise_f64(ha, op, fn);
}

/* Symmetric F32 dispatch wrapper. unop_elementwise_f32 itself is static
 * (from the .inc stamping); this non-static wrapper exposes it to per-op
 * files in core/elementwise/. */
TensorHandle unop_elementwise_f32_disp(TensorHandle ha, int op_tag, float (*fn)(float)) {
    return unop_elementwise_f32(ha, op_tag, fn);
}

void tape_abort_mixed_dtype(const char* op) {
    fprintf(stderr,
        "[tape backend] %s: mixed-dtype inputs forbidden — both operands must "
        "share a dtype_tag (cast first via tcast / tensor_cast_dtype_streamed).\n", op);
    abort();
}
