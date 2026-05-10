/* training/optimizer.c — SGD / RMSprop / Adam / AdamW + grad clipping
 *                         + Polyak soft update + serialization accessors.
 *
 * Phase 1e.6. The full optimizer surface lives here. State is a single
 * flat per-element buffer (opt->v, opt->m) sized by
 * sum_i tensor_numel(param_i); offsets are computed by walking
 * param_count() in order.
 *
 * Bias toward keeping `lr` outside the momentum buffer for RMSprop —
 * matches torch.optim.RMSprop under LR schedules. Adam/AdamW use the
 * standard β1/β2 + bias-correction formulas.
 *
 * native_train_step / optimizer_step_with_clip are the high-level
 * wrappers (zero_grad → backward → clip → step) used by the typed
 * Idris surface. They live here so the (clip → step) pair is one
 * file's concern.
 *
 * Phase 2 will lift the bulk of this into shared/training/optimizer.c
 * behind an adapter so torch/mlx share the loop structure.
 */

#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "../arena.h"
#include "../tape.h"
#include "../tensor.h"
#include "../../backend.h"

typedef struct {
    double lr;
    int type; /* 0=SGD, 1=RMSprop, 2=Adam, 3=AdamW */
    double alpha, eps, weight_decay, momentum;
    double beta1, beta2;
    double* v;  /* second moment (RMSprop/Adam) */
    double* m;  /* first moment (Adam) / momentum buffer (RMSprop) */
    int t;      /* step count */
    int allocated;
    double* param_lr;      /* per-param LR overrides (NULL = use opt->lr for all) */
    int param_lr_count;    /* number of entries in param_lr */
    char prefix[128];      /* param-name prefix filter (empty = manages all) */
} Optimizer;

/* Profiling globals (in backend_tape.c until Phase 1e.7) */
extern double prof_optimizer_ms;
extern int    prof_forward_ops;
extern int    prof_epochs;
extern double prof_epoch_start;
extern double prof_op_t_prev;
extern double _wall_ms(void);
extern void   _dbg_dump_lstm_traj_if_enabled(void);

/* Returns 1 if param[i]'s name starts with opt->prefix (or prefix is empty). */
static int opt_owns_param(Optimizer* opt, int i) {
    if (opt->prefix[0] == '\0') return 1;
    return strncmp(param_name(i), opt->prefix, strlen(opt->prefix)) == 0;
}

/* Compute total number of elements across all params (for per-element optimizer buffers) */
static int param_total_elements(void) {
    int total = 0;
    for (int i = 0; i < param_count(); i++)
        total += ((Tensor*)param_tensor(i))->numel;
    return total;
}

/* Offset into the flat per-element buffer for param i */
static int param_element_offset(int param_idx) {
    int off = 0;
    for (int i = 0; i < param_idx; i++)
        off += ((Tensor*)param_tensor(i))->numel;
    return off;
}

static void optimizer_ensure_buffers(Optimizer* opt) {
    if (opt->allocated) return;
    int n = param_total_elements();
    opt->v = calloc(n, sizeof(double));
    opt->m = calloc(n, sizeof(double));
    opt->allocated = 1;
}

OptimizerHandle optimizer_create_sgd(double lr) {
    Optimizer* opt = calloc(1, sizeof(Optimizer));
    opt->lr = lr;
    opt->type = 0;
    return opt;
}

OptimizerHandle optimizer_create_rmsprop(double lr, double alpha, double eps,
                                          double weight_decay, double momentum) {
    Optimizer* opt = calloc(1, sizeof(Optimizer));
    opt->lr = lr; opt->type = 1;
    opt->alpha = alpha; opt->eps = eps;
    opt->weight_decay = weight_decay; opt->momentum = momentum;
    return opt;
}

OptimizerHandle optimizer_create_adam(double lr, double beta1, double beta2, double eps) {
    Optimizer* opt = calloc(1, sizeof(Optimizer));
    opt->lr = lr; opt->type = 2;
    opt->beta1 = beta1; opt->beta2 = beta2; opt->eps = eps;
    return opt;
}

/* Adam that only updates params whose registry name starts with `prefix`. */
OptimizerHandle optimizer_create_adam_group(double lr, double beta1, double beta2,
                                            double eps, const char* prefix) {
    Optimizer* opt = (Optimizer*)optimizer_create_adam(lr, beta1, beta2, eps);
    if (prefix) {
        strncpy(opt->prefix, prefix, sizeof(opt->prefix) - 1);
        opt->prefix[sizeof(opt->prefix) - 1] = '\0';
    }
    return opt;
}

OptimizerHandle optimizer_create_adamw(double lr, double beta1, double beta2, double eps,
                                       double weight_decay) {
    Optimizer* opt = calloc(1, sizeof(Optimizer));
    opt->lr = lr; opt->type = 3;
    opt->beta1 = beta1; opt->beta2 = beta2; opt->eps = eps;
    opt->weight_decay = weight_decay;
    return opt;
}

/* Polyak soft update at the param-registry level. For each param P whose
 * name starts with `online_scope`, find the corresponding param Q whose
 * name is `target_scope ++ suffix(P)` and blend Q's data in-place:
 *   Q.data ← (1−tau)·Q.data + tau·P.data.
 *
 * Used by SAC: actor / q1 / q2 networks register with distinct scope
 * prefixes; target-Q params register with `q1_tgt_` / `q2_tgt_` prefixes.
 * Returns number of param-pairs blended (for sanity checking). */
int polyak_blend(double tau, const char* online_scope, const char* target_scope) {
    if (!online_scope || !target_scope) return 0;
    size_t on_len = strlen(online_scope);
    size_t tg_len = strlen(target_scope);
    int blended = 0;
    double one_minus_tau = 1.0 - tau;
    for (int i = 0; i < param_count(); i++) {
        const char* on_name = param_name(i);
        if (strncmp(on_name, online_scope, on_len) != 0) continue;
        /* Build target name: target_scope ++ (on_name + on_len). */
        char tgt_name[256];
        size_t suffix_len = strlen(on_name + on_len);
        if (tg_len + suffix_len + 1 > sizeof(tgt_name)) continue;
        memcpy(tgt_name, target_scope, tg_len);
        memcpy(tgt_name + tg_len, on_name + on_len, suffix_len + 1);
        /* Find target param. */
        for (int j = 0; j < param_count(); j++) {
            if (strcmp(param_name(j), tgt_name) != 0) continue;
            Tensor* on_t = (Tensor*)param_tensor(i);
            Tensor* tg_t = (Tensor*)param_tensor(j);
            if (on_t->numel != tg_t->numel) break;
            for (int k = 0; k < on_t->numel; k++) {
                ((double*)tg_t->data)[k] = one_minus_tau * ((double*)tg_t->data)[k] + tau * ((double*)on_t->data)[k];
            }
            blended++;
            break;
        }
    }
    return blended;
}

void optimizer_free(OptimizerHandle h) {
    Optimizer* opt = (Optimizer*)h;
    free(opt->v); free(opt->m); free(opt->param_lr); free(opt);
}

void optimizer_set_param_lr(OptimizerHandle h, const char* name, double lr) {
    Optimizer* opt = (Optimizer*)h;
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

void optimizer_set_lr(OptimizerHandle h, double lr) {
    Optimizer* opt = (Optimizer*)h;
    opt->lr = lr;
}

void optimizer_zero_grad(OptimizerHandle h) {
    (void)h;
    param_zero_all_grads();
}

void optimizer_step(OptimizerHandle h) {
    double t0_opt = _wall_ms();
    Optimizer* opt = (Optimizer*)h;
    optimizer_ensure_buffers(opt);
    opt->t++;

    for (int i = 0; i < param_count(); i++) {
        if (!opt_owns_param(opt, i)) continue;
        /* DIAGNOSTIC: SKIP_LSTM_INIT skips updating params whose names end in
           _h0 or _c0 (LSTM learned initial state). Equivalent to keeping
           them as zero state tensors. Used to localize whether convergence
           regression is in gradient values applied to h0/c0 vs elsewhere. */
        if (getenv("SKIP_LSTM_INIT")) {
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

        /* Per-param LR: use override if set, otherwise base LR */
        double lr = opt->lr;
        if (opt->param_lr && i < opt->param_lr_count && opt->param_lr[i] >= 0)
            lr = opt->param_lr[i];

        for (int j = 0; j < t->numel; j++) {
            double g = ((double*)t->grad)[j];
            int idx = base + j;

            /* Dtype-aware reads + writes so F32 params take f32-precision
               updates (asserted by the rung-4 F32-exactness check). Moment
               buffers (opt->m / opt->v) stay F64 — standard mixed-precision
               practice and lets the F64 numerics path stay byte-identical. */
            double w = tape_load_d(t, j);
            switch (opt->type) {
            case 0: /* SGD */
                tape_store_d(t, j, w - lr * g);
                break;

            case 1: { /* RMSprop — keep lr OUTSIDE the momentum buffer to match
                         torch.optim.RMSprop. Folding lr into the buffer
                         (buf = m*buf + lr*g/avg) coincides with PyTorch only at
                         constant lr; under an LR schedule the buffer carries
                         stale rates and diverges. */
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

            case 2: { /* Adam */
                opt->m[idx] = opt->beta1 * opt->m[idx] + (1.0 - opt->beta1) * g;
                opt->v[idx] = opt->beta2 * opt->v[idx] + (1.0 - opt->beta2) * g * g;
                double mhat = opt->m[idx] / (1.0 - pow(opt->beta1, opt->t));
                double vhat = opt->v[idx] / (1.0 - pow(opt->beta2, opt->t));
                tape_store_d(t, j, w - lr * mhat / (sqrt(vhat) + opt->eps));
                break;
            }

            case 3: { /* AdamW (decoupled weight decay) */
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

    _dbg_dump_lstm_traj_if_enabled();

    /* Snapshot tape size before reset */
    prof_forward_ops = tape_size;

    /* Reset tape and re-register ONLY the param tensors (from param_registry).
       Ephemeral tensors (select results, intermediates) are not re-registered.
       They will be recreated in the next forward pass. */
    tape_reset();
    for (int j = 0; j < param_count(); j++) {
        Tensor* t = (Tensor*)param_tensor(j);
        t->tape_idx = -1;
        if (t->grad) memset(t->grad, 0, t->numel * sizeof(double));
        tape_append(OP_CONST, t, NULL, NULL, 0);
    }
    prof_optimizer_ms += _wall_ms() - t0_opt;
    prof_epochs++;
    /* Auto-start timing for next epoch's forward pass + per-op accumulator */
    double t_next = _wall_ms();
    prof_epoch_start = t_next;
    prof_op_t_prev = t_next;
}

/* Internal clip-value helper scoped to params owned by `opt`. */
static void clip_grad_value_opt(Optimizer* opt, double max_val) {
    for (int i = 0; i < param_count(); i++) {
        if (opt && !opt_owns_param(opt, i)) continue;
        Tensor* t = (Tensor*)param_tensor(i);
        if (!t->grad) continue;
        for (int j = 0; j < t->numel; j++) {
            if (((double*)t->grad)[j] > max_val) ((double*)t->grad)[j] = max_val;
            if (((double*)t->grad)[j] < -max_val) ((double*)t->grad)[j] = -max_val;
        }
    }
}

/* Internal clip-norm helper scoped to params owned by `opt`. */
static double clip_grad_norm_opt(Optimizer* opt, double max_norm) {
    double total = 0;
    for (int i = 0; i < param_count(); i++) {
        if (opt && !opt_owns_param(opt, i)) continue;
        Tensor* t = (Tensor*)param_tensor(i);
        if (!t->grad) continue;
        for (int j = 0; j < t->numel; j++) total += ((double*)t->grad)[j] * ((double*)t->grad)[j];
    }
    double norm = sqrt(total);
    if (norm > max_norm) {
        double scale = max_norm / norm;
        for (int i = 0; i < param_count(); i++) {
            if (opt && !opt_owns_param(opt, i)) continue;
            Tensor* t = (Tensor*)param_tensor(i);
            if (!t->grad) continue;
            for (int j = 0; j < t->numel; j++) ((double*)t->grad)[j] *= scale;
        }
    }
    return norm;
}

/* Public global clippers retained for direct-FFI callers (backward compat). */
void optimizer_clip_grad_value(double max_val) {
    clip_grad_value_opt(NULL, max_val);
}

double optimizer_clip_grad_norm(double max_norm) {
    return clip_grad_norm_opt(NULL, max_norm);
}

/* ----------------------------------------------------------------------
   Optimizer buffer accessors (for serialization)
   ---------------------------------------------------------------------- */

int optimizer_buf_count(OptimizerHandle h) {
    (void)h;
    return param_count();
}

void optimizer_get_m(OptimizerHandle h, int idx, double* out) {
    Optimizer* opt = (Optimizer*)h;
    if (!opt->allocated) { memset(out, 0, ((Tensor*)param_tensor(idx))->numel * sizeof(double)); return; }
    int offset = param_element_offset(idx);
    int numel = ((Tensor*)param_tensor(idx))->numel;
    memcpy(out, opt->m + offset, numel * sizeof(double));
}

void optimizer_get_v(OptimizerHandle h, int idx, double* out) {
    Optimizer* opt = (Optimizer*)h;
    if (!opt->allocated) { memset(out, 0, ((Tensor*)param_tensor(idx))->numel * sizeof(double)); return; }
    int offset = param_element_offset(idx);
    int numel = ((Tensor*)param_tensor(idx))->numel;
    memcpy(out, opt->v + offset, numel * sizeof(double));
}

void optimizer_set_m(OptimizerHandle h, int idx, const double* data) {
    Optimizer* opt = (Optimizer*)h;
    optimizer_ensure_buffers(opt);
    int offset = param_element_offset(idx);
    int numel = ((Tensor*)param_tensor(idx))->numel;
    memcpy(opt->m + offset, data, numel * sizeof(double));
}

void optimizer_set_v(OptimizerHandle h, int idx, const double* data) {
    Optimizer* opt = (Optimizer*)h;
    optimizer_ensure_buffers(opt);
    int offset = param_element_offset(idx);
    int numel = ((Tensor*)param_tensor(idx))->numel;
    memcpy(opt->v + offset, data, numel * sizeof(double));
}

void optimizer_get_meta(OptimizerHandle h, double* out9) {
    Optimizer* opt = (Optimizer*)h;
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

void optimizer_set_meta(OptimizerHandle h, const double* in9) {
    Optimizer* opt = (Optimizer*)h;
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
   High-level train-step wrappers (zero_grad → backward → clip → step).
   ---------------------------------------------------------------------- */

extern void tensor_backward(TensorHandle h);

double native_train_step(OptimizerHandle opt, int clip_mode, double clip_val,
                         TensorHandle loss_ptr, double loss_val) {
    Tensor* loss = (Tensor*)loss_ptr;
    optimizer_zero_grad(opt);
    if (loss->requires_grad) tensor_backward(loss_ptr);
    /* Scope grad-clipping to this optimizer's owned params, so multi-
     * optimizer setups (SAC actor/q1/q2) each clip their own norm. */
    if (clip_mode == 1) clip_grad_value_opt((Optimizer*)opt, clip_val);
    else if (clip_mode == 2) clip_grad_norm_opt((Optimizer*)opt, clip_val);
    optimizer_step(opt);
    return loss_val;
}

int optimizer_step_with_clip(OptimizerHandle opt, int clip_mode, double clip_val, int dummy) {
    (void)dummy;
    if (clip_mode == 1) clip_grad_value_opt((Optimizer*)opt, clip_val);
    else if (clip_mode == 2) clip_grad_norm_opt((Optimizer*)opt, clip_val);
    optimizer_step(opt);
    optimizer_zero_grad(opt);
    return 0;
}
