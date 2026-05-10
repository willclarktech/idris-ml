/* shared/training/optimizer.c — backend-agnostic optimizer surface.
 *
 * Owns the optimizer's user-facing surface (the Idris training loop
 * binds against it):
 *   - optimizer_create_sgd / rmsprop / adam / adam_group / adamw
 *   - optimizer_step (per-element SGD/RMSprop/Adam/AdamW math)
 *   - polyak_blend (per-element interpolation across param-name pairs)
 *   - clip_grad_value / clip_grad_norm
 *   - optimizer_step_with_clip / native_train_step (training-loop helpers)
 *   - optimizer_buf_count / get_m / get_v / set_m / set_v / get_meta /
 *     set_meta (checkpoint serialization accessors)
 *
 * All per-tensor reads/writes go through `g_active_port` (defined per
 * backend in <backend>/training/adapter.<c|cpp>). The per-element math
 * sequence is preserved verbatim from the monolithic optimizer so the
 * F64 byte-identical guarantee holds across the lift:
 *   - SGD: w -= lr * g
 *   - RMSprop: lr OUTSIDE the momentum buffer (matches torch.optim
 *              under LR schedules; folding lr into the buffer diverges
 *              from PyTorch the moment lr changes).
 *   - Adam: standard β1/β2 first/second-moment + bias correction.
 *   - AdamW: decoupled weight decay applied to the post-step weight.
 *
 * Compiled once per backend in `TRAINING_ADAPTER_BACKENDS` with that
 * backend's rename header, so multi-link gives each backend its own
 * `optimizer_step_<b>` etc.
 *
 * Epoch boundary (tape_reset + re-register params; tape-specific
 * Wengert-list hygiene) is delegated to `g_active_port.epoch_boundary`
 * — torch and mlx will no-op it when their adapters land.
 */

#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "port.h"
#include "../../backend.h"

typedef struct {
    double lr;
    int    type;            /* 0=SGD, 1=RMSprop, 2=Adam, 3=AdamW */
    double alpha, eps, weight_decay, momentum;
    double beta1, beta2;
    double* v;              /* second moment (RMSprop/Adam) */
    double* m;              /* first moment (Adam) / momentum buffer (RMSprop) */
    int    t;               /* step count */
    int    allocated;
    double* param_lr;       /* per-param LR overrides; NULL = use opt->lr for all */
    int    param_lr_count;
    char   prefix[128];     /* param-name prefix filter (empty = manages all) */
} Optimizer;

/* Tape-specific knob — see optimizer_step. Lives at file scope so each
   per-element loop pays one getenv on entry. */

static int opt_owns_param(Optimizer* opt, int i) {
    if (opt->prefix[0] == '\0') return 1;
    return strncmp(param_name(i), opt->prefix, strlen(opt->prefix)) == 0;
}

/* Sum of numel over all registered params. Used to size per-element
   m/v buffers; offsets are computed via param_element_offset. */
static int param_total_elements(void) {
    int total = 0;
    for (int i = 0; i < param_count(); i++)
        total += g_active_port.tensor_numel(param_tensor(i));
    return total;
}

static int param_element_offset(int param_idx) {
    int off = 0;
    for (int i = 0; i < param_idx; i++)
        off += g_active_port.tensor_numel(param_tensor(i));
    return off;
}

static void optimizer_ensure_buffers(Optimizer* opt) {
    if (opt->allocated) return;
    int n = param_total_elements();
    opt->v = calloc(n, sizeof(double));
    opt->m = calloc(n, sizeof(double));
    opt->allocated = 1;
}

/* ----------------------------------------------------------------------
   Constructors. Plain struct allocators, no tensor access.
   ---------------------------------------------------------------------- */

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

/* ----------------------------------------------------------------------
   Polyak soft update — for every param whose name starts with
   `online_scope`, find the corresponding `target_scope ++ suffix` param
   and blend Q.data ← (1−tau)·Q.data + tau·P.data. Used by SAC's target
   networks.
   ---------------------------------------------------------------------- */
int polyak_blend(double tau, const char* online_scope, const char* target_scope) {
    if (!online_scope || !target_scope) return 0;
    size_t on_len = strlen(online_scope);
    size_t tg_len = strlen(target_scope);
    int blended = 0;
    double one_minus_tau = 1.0 - tau;
    for (int i = 0; i < param_count(); i++) {
        const char* on_name = param_name(i);
        if (strncmp(on_name, online_scope, on_len) != 0) continue;
        char tgt_name[256];
        size_t suffix_len = strlen(on_name + on_len);
        if (tg_len + suffix_len + 1 > sizeof(tgt_name)) continue;
        memcpy(tgt_name, target_scope, tg_len);
        memcpy(tgt_name + tg_len, on_name + on_len, suffix_len + 1);
        for (int j = 0; j < param_count(); j++) {
            if (strcmp(param_name(j), tgt_name) != 0) continue;
            void* on_t = param_tensor(i);
            void* tg_t = param_tensor(j);
            int n_on = g_active_port.tensor_numel(on_t);
            if (n_on != g_active_port.tensor_numel(tg_t)) break;
            for (int k = 0; k < n_on; k++) {
                double tg = g_active_port.data_read(tg_t, k);
                double on = g_active_port.data_read(on_t, k);
                g_active_port.data_write(tg_t, k, one_minus_tau * tg + tau * on);
            }
            blended++;
            break;
        }
    }
    return blended;
}

/* ----------------------------------------------------------------------
   The optimizer step. Per-param, per-element math; F64 byte-identical
   with the legacy monolithic loop (operations + ordering preserved).
   ---------------------------------------------------------------------- */
void optimizer_step(OptimizerHandle h) {
    double t0_opt = g_active_port.wall_ms();
    Optimizer* opt = (Optimizer*)h;
    optimizer_ensure_buffers(opt);
    opt->t++;

    /* SKIP_LSTM_INIT diagnostic — skips updating params whose names end
       in `_h0` / `_c0` (LSTM learned initial state). Localizes whether
       a convergence regression lives in h0/c0 grads vs elsewhere. */
    int skip_lstm_init = getenv("SKIP_LSTM_INIT") != NULL;

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
        void* t = param_tensor(i);
        if (!g_active_port.tensor_has_grad(t)) continue;
        int n_elem = g_active_port.tensor_numel(t);
        int base = param_element_offset(i);

        double lr = opt->lr;
        if (opt->param_lr && i < opt->param_lr_count && opt->param_lr[i] >= 0)
            lr = opt->param_lr[i];

        for (int j = 0; j < n_elem; j++) {
            double g = g_active_port.grad_read(t, j);
            int idx = base + j;

            /* dtype-aware read/write: F32 narrows through the port,
               F64 hits the raw double slot (byte-identical with the
               legacy `tape_load_d`/`tape_store_d` pair). Moment buffers
               stay F64 — mixed-precision practice. */
            double w = g_active_port.data_read(t, j);
            switch (opt->type) {
            case 0: /* SGD */
                g_active_port.data_write(t, j, w - lr * g);
                break;

            case 1: { /* RMSprop — lr OUTSIDE the momentum buffer (see file docstring). */
                opt->v[idx] = opt->alpha * opt->v[idx] + (1.0 - opt->alpha) * g * g;
                double avg = sqrt(opt->v[idx]) + opt->eps;
                if (opt->momentum > 0) {
                    opt->m[idx] = opt->momentum * opt->m[idx] + g / avg;
                    g_active_port.data_write(t, j, w - lr * opt->m[idx]);
                } else {
                    g_active_port.data_write(t, j, w - lr * g / avg);
                }
                break;
            }

            case 2: { /* Adam */
                opt->m[idx] = opt->beta1 * opt->m[idx] + (1.0 - opt->beta1) * g;
                opt->v[idx] = opt->beta2 * opt->v[idx] + (1.0 - opt->beta2) * g * g;
                double mhat = opt->m[idx] / (1.0 - pow(opt->beta1, opt->t));
                double vhat = opt->v[idx] / (1.0 - pow(opt->beta2, opt->t));
                g_active_port.data_write(t, j, w - lr * mhat / (sqrt(vhat) + opt->eps));
                break;
            }

            case 3: { /* AdamW (decoupled weight decay) */
                opt->m[idx] = opt->beta1 * opt->m[idx] + (1.0 - opt->beta1) * g;
                opt->v[idx] = opt->beta2 * opt->v[idx] + (1.0 - opt->beta2) * g * g;
                double mhat = opt->m[idx] / (1.0 - pow(opt->beta1, opt->t));
                double vhat = opt->v[idx] / (1.0 - pow(opt->beta2, opt->t));
                double w1 = w - lr * mhat / (sqrt(vhat) + opt->eps);
                g_active_port.data_write(t, j, w1 - lr * opt->weight_decay * w1);
                break;
            }
            }
        }
    }

    /* Adapter-supplied epoch hygiene: dumps, tape_reset + param re-
       registration on tape, prof_optimizer_ms accounting, epoch++. */
    g_active_port.epoch_boundary(t0_opt);
}

/* ----------------------------------------------------------------------
   Grad clipping — clip_grad_value caps each element to [-max, +max];
   clip_grad_norm globally rescales so the L2 norm ≤ max_norm.
   ---------------------------------------------------------------------- */

static void clip_grad_value_opt(Optimizer* opt, double max_val) {
    for (int i = 0; i < param_count(); i++) {
        if (opt && !opt_owns_param(opt, i)) continue;
        void* t = param_tensor(i);
        if (!g_active_port.tensor_has_grad(t)) continue;
        int n = g_active_port.tensor_numel(t);
        for (int j = 0; j < n; j++) {
            double v = g_active_port.grad_read(t, j);
            if (v >  max_val) g_active_port.grad_write(t, j,  max_val);
            else if (v < -max_val) g_active_port.grad_write(t, j, -max_val);
        }
    }
}

static double clip_grad_norm_opt(Optimizer* opt, double max_norm) {
    double total = 0;
    for (int i = 0; i < param_count(); i++) {
        if (opt && !opt_owns_param(opt, i)) continue;
        void* t = param_tensor(i);
        if (!g_active_port.tensor_has_grad(t)) continue;
        int n = g_active_port.tensor_numel(t);
        for (int j = 0; j < n; j++) {
            double v = g_active_port.grad_read(t, j);
            total += v * v;
        }
    }
    double norm = sqrt(total);
    if (norm > max_norm) {
        double scale = max_norm / norm;
        for (int i = 0; i < param_count(); i++) {
            if (opt && !opt_owns_param(opt, i)) continue;
            void* t = param_tensor(i);
            if (!g_active_port.tensor_has_grad(t)) continue;
            int n = g_active_port.tensor_numel(t);
            for (int j = 0; j < n; j++) {
                double v = g_active_port.grad_read(t, j);
                g_active_port.grad_write(t, j, v * scale);
            }
        }
    }
    return norm;
}

void optimizer_clip_grad_value(double max_val) { clip_grad_value_opt(NULL, max_val); }
double optimizer_clip_grad_norm(double max_norm) { return clip_grad_norm_opt(NULL, max_norm); }

/* ----------------------------------------------------------------------
   Optimizer-buffer serialization accessors. m/v are flat per-element
   buffers; meta is a fixed 9-double vector (type, lr, β1, β2, eps,
   alpha, weight_decay, momentum, t).
   ---------------------------------------------------------------------- */

int optimizer_buf_count(OptimizerHandle h) {
    (void)h;
    return param_count();
}

void optimizer_get_m(OptimizerHandle h, int idx, double* out) {
    Optimizer* opt = (Optimizer*)h;
    int n = g_active_port.tensor_numel(param_tensor(idx));
    if (!opt->allocated) { memset(out, 0, n * sizeof(double)); return; }
    int offset = param_element_offset(idx);
    memcpy(out, opt->m + offset, n * sizeof(double));
}

void optimizer_get_v(OptimizerHandle h, int idx, double* out) {
    Optimizer* opt = (Optimizer*)h;
    int n = g_active_port.tensor_numel(param_tensor(idx));
    if (!opt->allocated) { memset(out, 0, n * sizeof(double)); return; }
    int offset = param_element_offset(idx);
    memcpy(out, opt->v + offset, n * sizeof(double));
}

void optimizer_set_m(OptimizerHandle h, int idx, const double* data) {
    Optimizer* opt = (Optimizer*)h;
    optimizer_ensure_buffers(opt);
    int offset = param_element_offset(idx);
    int n = g_active_port.tensor_numel(param_tensor(idx));
    memcpy(opt->m + offset, data, n * sizeof(double));
}

void optimizer_set_v(OptimizerHandle h, int idx, const double* data) {
    Optimizer* opt = (Optimizer*)h;
    optimizer_ensure_buffers(opt);
    int offset = param_element_offset(idx);
    int n = g_active_port.tensor_numel(param_tensor(idx));
    memcpy(opt->v + offset, data, n * sizeof(double));
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

double native_train_step(OptimizerHandle opt, int clip_mode, double clip_val,
                         TensorHandle loss_ptr, double loss_val) {
    optimizer_zero_grad(opt);
    if (g_active_port.tensor_requires_grad((void*)loss_ptr))
        g_active_port.backward((void*)loss_ptr);
    if      (clip_mode == 1) clip_grad_value_opt((Optimizer*)opt, clip_val);
    else if (clip_mode == 2) clip_grad_norm_opt((Optimizer*)opt, clip_val);
    optimizer_step(opt);
    return loss_val;
}

int optimizer_step_with_clip(OptimizerHandle opt, int clip_mode, double clip_val, int dummy) {
    (void)dummy;
    if      (clip_mode == 1) clip_grad_value_opt((Optimizer*)opt, clip_val);
    else if (clip_mode == 2) clip_grad_norm_opt((Optimizer*)opt, clip_val);
    optimizer_step(opt);
    optimizer_zero_grad(opt);
    return 0;
}
