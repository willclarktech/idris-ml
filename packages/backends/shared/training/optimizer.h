/* shared/training/optimizer.h — Optimizer struct + helpers exposed to
 * backend adapters that override port.optimizer_step.
 *
 * The default shared optimizer_step (in optimizer.c) walks every
 * registered param and applies the SGD / RMSprop / Adam / AdamW
 * per-element update through `g_active_port.data_read/data_write/
 * grad_read`. Backends whose native math doesn't match that loop
 * (libtorch's `at::_foreach_adam` fused multi-tensor primitives;
 * mlx's vectorized in-graph updates) override the step by setting
 * `g_active_port.optimizer_step` to a backend-supplied function;
 * the override sees the full Optimizer struct laid out below and
 * touches its m/v buffers directly.
 *
 * `param_total_elements` and `param_element_offset` compute the flat
 * offset into the per-element m/v buffers — same convention every
 * override impl uses.
 */

#ifndef SHARED_TRAINING_OPTIMIZER_H
#define SHARED_TRAINING_OPTIMIZER_H

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

/* Allocate opt->m / opt->v if first call; no-op otherwise. */
void optimizer_ensure_buffers(Optimizer* opt);

/* Sum of numel over all registered params. */
int  param_total_elements(void);

/* Flat offset into the per-element m/v buffers for param `param_idx`. */
int  param_element_offset(int param_idx);

/* Returns 1 if param[i]'s name starts with opt->prefix (or prefix empty). */
int  opt_owns_param(Optimizer* opt, int i);

#endif /* SHARED_TRAINING_OPTIMIZER_H */
