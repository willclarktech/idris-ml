/* backend_tape/training/optimizer.c — tape's optimizer implementation.
 *
 * Owns the flat-buffer `Optimizer` struct (one `double* m` + `double* v`
 * spanning every registered param's elements; offsets computed by
 * walking `param_count()` in order) plus the per-element SGD /
 * RMSprop / Adam / AdamW math the tape adapter exposes via the shared
 * port. All public-facing entry points (the FFI-named
 * `optimizer_create_*`, `optimizer_step`, etc.) live in
 * shared/training/optimizer.c as tiny trampolines that dispatch
 * through `g_active_port.optimizer_*` — the entry-point names below
 * are backend-internal and bound into the port struct by
 * backend_tape/training/adapter.c.
 *
 * F64 byte-identical: the per-step math sequence + ordering matches
 * what was in the monolithic backend_tape.c verbatim (preserved
 * through the modular split + Phase 2.3 shared lift + this re-
 * extraction). Moment buffers stay F64 regardless of param dtype —
 * standard mixed-precision practice that keeps F32 param updates
 * exact to the F32-grad mantissa width without polluting the
 * historically-tested F64 trajectory.
 *
 * Adam β1/β2: standard first/second-moment + bias-correction.
 * RMSprop: lr OUTSIDE the momentum buffer (matches
 *   torch.optim.RMSprop under LR schedules; folding lr into the
 *   buffer coincides with PyTorch only at constant lr).
 * AdamW: decoupled weight decay applied to the post-step weight.
 *
 * Epoch boundary (DEBUG_LSTM_TRAJ dump → tape_reset → re-register
 * params on the fresh tape → prof_optimizer_ms accounting → bump
 * epoch counter) is folded into the step function. Backends with
 * no Wengert list (torch/mlx) skip the reset.
 */

#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#ifdef __APPLE__
#include <Accelerate/Accelerate.h>
#else
#include <cblas.h>
#endif
#include "../arena.h"
#include "../tape.h"
#include "../tensor.h"
#include "../../backend.h"

typedef struct {
    double lr;
    int    type;                  /* 0=SGD, 1=RMSprop, 2=Adam, 3=AdamW */
    double alpha, eps, weight_decay, momentum;
    double beta1, beta2;
    double* v;                    /* second moment (RMSprop/Adam) */
    double* m;                    /* first moment (Adam) / momentum buffer (RMSprop) */
    int    t;                     /* step count */
    int    allocated;
    double* param_lr;             /* per-param LR overrides; NULL = use opt->lr */
    int    param_lr_count;
    char   prefix[128];           /* param-name prefix filter (empty = manages all) */
} TapeOptimizer;

/* From training/profiling.c — tape's prof_* accumulators. */
extern double prof_optimizer_ms;
extern int    prof_forward_ops;
extern int    prof_epochs;
extern double prof_epoch_start;
extern double prof_op_t_prev;

/* From training/diagnostics.c — DEBUG_LSTM_TRAJ pre-reset dump. */
extern void _dbg_dump_lstm_traj_if_enabled(void);

static int opt_owns_param(TapeOptimizer* opt, int i) {
    if (opt->prefix[0] == '\0') return 1;
    return strncmp(param_name(i), opt->prefix, strlen(opt->prefix)) == 0;
}

static int param_total_elements(void) {
    int total = 0;
    for (int i = 0; i < param_count(); i++)
        total += ((Tensor*)param_tensor(i))->numel;
    return total;
}

static int param_element_offset(int param_idx) {
    int off = 0;
    for (int i = 0; i < param_idx; i++)
        off += ((Tensor*)param_tensor(i))->numel;
    return off;
}

static void optimizer_ensure_buffers(TapeOptimizer* opt) {
    if (opt->allocated) return;
    int n = param_total_elements();
    opt->v = calloc(n, sizeof(double));
    opt->m = calloc(n, sizeof(double));
    opt->allocated = 1;
}

/* ----------------------------------------------------------------------
   Constructors.
   ---------------------------------------------------------------------- */

void* tape_optimizer_create_sgd(double lr) {
    TapeOptimizer* opt = calloc(1, sizeof(TapeOptimizer));
    opt->lr = lr;
    opt->type = 0;
    return opt;
}

void* tape_optimizer_create_rmsprop(double lr, double alpha, double eps,
                                     double weight_decay, double momentum) {
    TapeOptimizer* opt = calloc(1, sizeof(TapeOptimizer));
    opt->lr = lr; opt->type = 1;
    opt->alpha = alpha; opt->eps = eps;
    opt->weight_decay = weight_decay; opt->momentum = momentum;
    return opt;
}

void* tape_optimizer_create_adam(double lr, double beta1, double beta2, double eps) {
    TapeOptimizer* opt = calloc(1, sizeof(TapeOptimizer));
    opt->lr = lr; opt->type = 2;
    opt->beta1 = beta1; opt->beta2 = beta2; opt->eps = eps;
    return opt;
}

void* tape_optimizer_create_adam_group(double lr, double beta1, double beta2,
                                        double eps, const char* prefix) {
    TapeOptimizer* opt = (TapeOptimizer*)tape_optimizer_create_adam(lr, beta1, beta2, eps);
    if (prefix) {
        strncpy(opt->prefix, prefix, sizeof(opt->prefix) - 1);
        opt->prefix[sizeof(opt->prefix) - 1] = '\0';
    }
    return opt;
}

void* tape_optimizer_create_adamw(double lr, double beta1, double beta2, double eps,
                                   double weight_decay) {
    TapeOptimizer* opt = calloc(1, sizeof(TapeOptimizer));
    opt->lr = lr; opt->type = 3;
    opt->beta1 = beta1; opt->beta2 = beta2; opt->eps = eps;
    opt->weight_decay = weight_decay;
    return opt;
}

void tape_optimizer_free(void* h) {
    TapeOptimizer* opt = (TapeOptimizer*)h;
    free(opt->v); free(opt->m); free(opt->param_lr); free(opt);
}

void tape_optimizer_set_lr(void* h, double lr) {
    ((TapeOptimizer*)h)->lr = lr;
}

void tape_optimizer_set_param_lr(void* h, const char* name, double lr) {
    TapeOptimizer* opt = (TapeOptimizer*)h;
    if (opt->param_lr == NULL || opt->param_lr_count < param_count()) {
        int new_count = param_count() > 64 ? param_count() : 64;
        double* new_lr = realloc(opt->param_lr, new_count * sizeof(double));
        for (int i = opt->param_lr_count; i < new_count; i++) new_lr[i] = -1.0;
        opt->param_lr = new_lr;
        opt->param_lr_count = new_count;
    }
    for (int i = 0; i < param_count(); i++) {
        if (strcmp(param_name(i), name) == 0) {
            opt->param_lr[i] = lr;
            return;
        }
    }
}

/* ----------------------------------------------------------------------
   AdamW foreach fast path — default for F64-tagged params.

   Replaces the per-element AdamW inner loop with a BLAS-1 moment update
   for F64 params. Math sequence preserved: m ← β1·m + (1-β1)·g and
   v ← β2·v + (1-β2)·g² landed before the bias-correct + weight update.
   The BLAS-1 path may differ from the scalar path by ULP via Accelerate
   FMA inside cblas_daxpy; the paired test in test_optimizers.c asserts
   convergence within 1e-12, not bit-identical equality.

   F32-tagged params fall through to the scalar inner switch's case 3
   (mixed-dtype foreach is out of scope; the per-call BLAS overhead
   isn't worth a widen-narrow staging pass for the F32 case).
   ---------------------------------------------------------------------- */
static void adamw_foreach_param(TapeOptimizer* opt, Tensor* t, int base, double lr) {
    int n = t->numel;
    double* g = (double*)t->grad;
    double* w = (double*)t->data;
    double* m = opt->m + base;
    double* v = opt->v + base;

    /* m ← β1·m + (1-β1)·g — BLAS-1 (dscal + daxpy). For tiny n the BLAS
       call overhead outweighs the vectorized inner; gate by n ≥ 256
       (smaller params keep the scalar autovectorized loop). */
    if (n >= 256) {
        cblas_dscal(n, opt->beta1, m, 1);
        cblas_daxpy(n, 1.0 - opt->beta1, g, 1, m, 1);
    } else {
        for (int j = 0; j < n; j++) m[j] = opt->beta1 * m[j] + (1.0 - opt->beta1) * g[j];
    }

    /* v ← β2·v + (1-β2)·g² — single scalar pass; compiler autovectorizes,
       no scratch buffer required. */
    for (int j = 0; j < n; j++) {
        v[j] = opt->beta2 * v[j] + (1.0 - opt->beta2) * g[j] * g[j];
    }

    /* Per-element weight update — same expression as the scalar path so
       the bias-correct factors round identically. */
    double bc1 = 1.0 - pow(opt->beta1, opt->t);
    double bc2 = 1.0 - pow(opt->beta2, opt->t);
    for (int j = 0; j < n; j++) {
        double mhat = m[j] / bc1;
        double vhat = v[j] / bc2;
        double w1 = w[j] - lr * mhat / (sqrt(vhat) + opt->eps);
        w[j] = w1 - lr * opt->weight_decay * w1;
    }
}

/* ----------------------------------------------------------------------
   Step — per-element flat-buffer math + tape epoch hygiene.
   ---------------------------------------------------------------------- */
void tape_optimizer_step(void* h) {
    extern double _wall_ms(void);
    double t0_opt = _wall_ms();
    TapeOptimizer* opt = (TapeOptimizer*)h;
    optimizer_ensure_buffers(opt);
    opt->t++;

    /* SKIP_LSTM_INIT diagnostic — skips updating params whose names end
       in `_h0` / `_c0` (LSTM learned initial state). */
    int skip_lstm_init = getenv("SKIP_LSTM_INIT") != NULL;

    /* AdamW foreach: default for F64 params; F32 params fall through to
       the scalar inner switch's case 3 (mixed-dtype foreach is out of
       scope). The paired test in test_optimizers.c uses the env-var
       opt-out (`TAPE_OPTIMIZER_FOREACH=0`) to force scalar for one
       phase so it can assert |scalar - foreach| < 1e-12 over 50
       AdamW steps; the env var is otherwise an internal debug knob,
       not a user-facing surface. */
    int use_adamw_foreach = (opt->type == 3);
    const char* env = getenv("TAPE_OPTIMIZER_FOREACH");
    if (env && env[0] == '0') use_adamw_foreach = 0;

    for (int i = 0; i < param_count(); i++) {
        if (!opt_owns_param(opt, i)) continue;
        if (skip_lstm_init) {
            const char* nm = param_name(i);
            size_t L = strlen(nm);
            if (L >= 3 && (strcmp(nm + L - 3, "_h0") == 0 ||
                           strcmp(nm + L - 3, "_c0") == 0)) {
                continue;
            }
        }
        Tensor* t = (Tensor*)param_tensor(i);
        if (!t->grad) continue;
        int base = param_element_offset(i);

        double lr = opt->lr;
        if (opt->param_lr && i < opt->param_lr_count && opt->param_lr[i] >= 0)
            lr = opt->param_lr[i];

        if (use_adamw_foreach && t->dtype_tag != DT_F32) {
            adamw_foreach_param(opt, t, base, lr);
            continue;
        }

        for (int j = 0; j < t->numel; j++) {
            double g = tape_grad_load_d(t, j);
            int idx = base + j;

            double w = tape_load_d(t, j);
            switch (opt->type) {
            case 0:
                tape_store_d(t, j, w - lr * g);
                break;
            case 1: {
                opt->v[idx] = opt->alpha * opt->v[idx] + (1.0 - opt->alpha) * g * g;
                double avg = sqrt(opt->v[idx]) + opt->eps;
                if (opt->momentum > 0) {
                    opt->m[idx] = opt->momentum * opt->m[idx] + g / avg;
                    tape_store_d(t, j, w - lr * opt->m[idx]);
                } else {
                    tape_store_d(t, j, w - lr * g / avg);
                }
                break;
            }
            case 2: {
                opt->m[idx] = opt->beta1 * opt->m[idx] + (1.0 - opt->beta1) * g;
                opt->v[idx] = opt->beta2 * opt->v[idx] + (1.0 - opt->beta2) * g * g;
                double mhat = opt->m[idx] / (1.0 - pow(opt->beta1, opt->t));
                double vhat = opt->v[idx] / (1.0 - pow(opt->beta2, opt->t));
                tape_store_d(t, j, w - lr * mhat / (sqrt(vhat) + opt->eps));
                break;
            }
            case 3: {
                opt->m[idx] = opt->beta1 * opt->m[idx] + (1.0 - opt->beta1) * g;
                opt->v[idx] = opt->beta2 * opt->v[idx] + (1.0 - opt->beta2) * g * g;
                double mhat = opt->m[idx] / (1.0 - pow(opt->beta1, opt->t));
                double vhat = opt->v[idx] / (1.0 - pow(opt->beta2, opt->t));
                double w1 = w - lr * mhat / (sqrt(vhat) + opt->eps);
                tape_store_d(t, j, w1 - lr * opt->weight_decay * w1);
                break;
            }
            }
        }
    }

    /* Tape epoch hygiene (formerly port.epoch_boundary). */
    _dbg_dump_lstm_traj_if_enabled();
    prof_forward_ops = tape_size;
    tape_reset();
    for (int j = 0; j < param_count(); j++) {
        Tensor* t = (Tensor*)param_tensor(j);
        t->tape_idx = -1;
        if (t->grad) memset(t->grad, 0, t->numel * sizeof(double));
        tape_append(OP_CONST, t, NULL, NULL, 0);
    }
    prof_optimizer_ms += _wall_ms() - t0_opt;
    prof_epochs++;
    double t_next = _wall_ms();
    prof_epoch_start = t_next;
    prof_op_t_prev = t_next;
}

/* ----------------------------------------------------------------------
   Serialization — flat-buffer m/v access + 9-double meta vector.
   ---------------------------------------------------------------------- */

int tape_optimizer_buf_count(void* h) {
    (void)h;
    return param_count();
}

void tape_optimizer_get_m(void* h, int idx, double* out) {
    TapeOptimizer* opt = (TapeOptimizer*)h;
    int numel = ((Tensor*)param_tensor(idx))->numel;
    if (!opt->allocated) { memset(out, 0, numel * sizeof(double)); return; }
    int offset = param_element_offset(idx);
    memcpy(out, opt->m + offset, numel * sizeof(double));
}

void tape_optimizer_get_v(void* h, int idx, double* out) {
    TapeOptimizer* opt = (TapeOptimizer*)h;
    int numel = ((Tensor*)param_tensor(idx))->numel;
    if (!opt->allocated) { memset(out, 0, numel * sizeof(double)); return; }
    int offset = param_element_offset(idx);
    memcpy(out, opt->v + offset, numel * sizeof(double));
}

void tape_optimizer_set_m(void* h, int idx, const double* data) {
    TapeOptimizer* opt = (TapeOptimizer*)h;
    optimizer_ensure_buffers(opt);
    int offset = param_element_offset(idx);
    int numel = ((Tensor*)param_tensor(idx))->numel;
    memcpy(opt->m + offset, data, numel * sizeof(double));
}

void tape_optimizer_set_v(void* h, int idx, const double* data) {
    TapeOptimizer* opt = (TapeOptimizer*)h;
    optimizer_ensure_buffers(opt);
    int offset = param_element_offset(idx);
    int numel = ((Tensor*)param_tensor(idx))->numel;
    memcpy(opt->v + offset, data, numel * sizeof(double));
}

void tape_optimizer_get_meta(void* h, double* out9) {
    TapeOptimizer* opt = (TapeOptimizer*)h;
    out9[0] = (double)opt->type;
    out9[1] = opt->lr;
    out9[2] = opt->beta1;
    out9[3] = opt->beta2;
    out9[4] = opt->eps;
    out9[5] = opt->alpha;
    out9[6] = opt->weight_decay;
    out9[7] = opt->momentum;
    out9[8] = (double)opt->t;
}

void tape_optimizer_set_meta(void* h, const double* in9) {
    TapeOptimizer* opt = (TapeOptimizer*)h;
    opt->type = (int)in9[0];
    opt->lr = in9[1];
    opt->beta1 = in9[2];
    opt->beta2 = in9[3];
    opt->eps = in9[4];
    opt->alpha = in9[5];
    opt->weight_decay = in9[6];
    opt->momentum = in9[7];
    opt->t = (int)in9[8];
}

/* ----------------------------------------------------------------------
   Prefix-scoped grad clipping. Walks only the params this optimizer
   owns (opt->prefix == ""  means "every registered param"). Used by
   SAC's multi-optimizer training to keep each opt's clip from
   touching other opts' params.
   ---------------------------------------------------------------------- */

void tape_optimizer_clip_grad_value_filtered(void* h, double max_val) {
    TapeOptimizer* opt = (TapeOptimizer*)h;
    for (int i = 0; i < param_count(); i++) {
        if (!opt_owns_param(opt, i)) continue;
        Tensor* t = (Tensor*)param_tensor(i);
        if (!t->grad) continue;
        for (int j = 0; j < t->numel; j++) {
            double v = tape_grad_load_d(t, j);
            if      (v >  max_val) tape_grad_store_d(t, j, max_val);
            else if (v < -max_val) tape_grad_store_d(t, j, -max_val);
        }
    }
}

double tape_optimizer_clip_grad_norm_filtered(void* h, double max_norm) {
    TapeOptimizer* opt = (TapeOptimizer*)h;
    double total = 0;
    for (int i = 0; i < param_count(); i++) {
        if (!opt_owns_param(opt, i)) continue;
        Tensor* t = (Tensor*)param_tensor(i);
        if (!t->grad) continue;
        for (int j = 0; j < t->numel; j++)
            total += tape_grad_load_d(t, j) * tape_grad_load_d(t, j);
    }
    double norm = sqrt(total);
    if (norm > max_norm) {
        double scale = max_norm / norm;
        for (int i = 0; i < param_count(); i++) {
            if (!opt_owns_param(opt, i)) continue;
            Tensor* t = (Tensor*)param_tensor(i);
            if (!t->grad) continue;
            for (int j = 0; j < t->numel; j++)
                tape_grad_store_d(t, j, tape_grad_load_d(t, j) * scale);
        }
    }
    return norm;
}
