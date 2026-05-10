/* training/param_registry.c — global parameter registry.
 *
 * The optimizer surface (registered learnable tensors) plus
 * the DEBUG_PARAM_GRADS / DEBUG_LSTM_TRAJ diagnostics that walk it.
 *
 * The shared-port lift will move the bulk of this into
 * shared/training/param_registry.c behind an adapter so torch/mlx
 * share the same registry implementation; the tape-local variant
 * stays here as the reference and so the diagnostic functions can
 * keep direct access to the static array.
 */

#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <string.h>
#include <stdint.h>
#include "../arena.h"
#include "../tensor.h"
#include "../../backend.h"

typedef struct {
    char name[256];
    Tensor* tensor;
} ParamEntry;

#define MAX_PARAMS 65536
static ParamEntry param_registry[MAX_PARAMS];
static int param_count_val = 0;

void param_register(const char* name, TensorHandle h) {
    Tensor* t = (Tensor*)h;
    /* Replace if exists */
    for (int i = 0; i < param_count_val; i++) {
        if (strcmp(param_registry[i].name, name) == 0) {
            param_registry[i].tensor = t;
            return;
        }
    }
    if (param_count_val < MAX_PARAMS) {
        strncpy(param_registry[param_count_val].name, name, 255);
        param_registry[param_count_val].tensor = t;
        param_count_val++;
    }
}

void param_clear(void)                   { param_count_val = 0; }
int param_count(void)                    { return param_count_val; }
const char* param_name(int idx)          { return param_registry[idx].name; }
TensorHandle param_tensor(int idx)       { return param_registry[idx].tensor; }

double param_grad_item(int idx) {
    Tensor* t = param_registry[idx].tensor;
    if (!t->grad) return 0.0;
    return ((double*)t->grad)[0];
}

double param_grad_item_at(int param_idx, int elem_idx) {
    Tensor* t = param_registry[param_idx].tensor;
    if (!t->grad || elem_idx >= t->numel) return 0.0;
    return ((double*)t->grad)[elem_idx];
}

double param_grad_item_and_zero(int idx) {
    Tensor* t = param_registry[idx].tensor;
    if (!t->grad) return 0.0;
    double v = ((double*)t->grad)[0];
    ((double*)t->grad)[0] = 0.0;
    return v;
}

void param_zero_all_grads(void) {
    for (int i = 0; i < param_count_val; i++) {
        Tensor* t = param_registry[i].tensor;
        if (t->grad) memset(t->grad, 0, t->numel * sizeof(double));
    }
}

void param_subtract_delta(int idx, double delta) {
    Tensor* t = param_registry[idx].tensor;
    ((double*)t->data)[0] -= delta;
}

void param_load_data(int idx, const double* data, int numel) {
    Tensor* t = param_registry[idx].tensor;
    if (t->numel != numel) {
        fprintf(stderr, "param_load_data: size mismatch for '%s': expected %d, got %d\n",
                param_registry[idx].name, t->numel, numel);
        return;
    }
    memcpy(t->data, data, numel * sizeof(double));
}

/* Byte-level I64 in-place loader — see backend.h. Tape's lingua-franca
   storage routes every int64 through `tape_store_d` (narrows to float
   on F32 storage, plain double-write otherwise). Values above 2^53
   lose precision at this conversion, matching the existing lingua-
   franca behaviour — no regression. */
void param_load_data_int64(int idx, const int64_t* data, int numel) {
    Tensor* t = param_registry[idx].tensor;
    if (t->numel != numel) {
        fprintf(stderr, "param_load_data_int64: size mismatch for '%s': expected %d, got %d\n",
                param_registry[idx].name, t->numel, numel);
        return;
    }
    for (int i = 0; i < numel; i++) {
        tape_store_d(t, i, (double)data[i]);
    }
}

/* ----------------------------------------------------------------------
   Diagnostic dumps (DEBUG_PARAM_GRADS / DEBUG_LSTM_TRAJ).
   ---------------------------------------------------------------------- */

/* Dump per-param gradient L2 norms after a backward pass. Enabled by
   setting DEBUG_PARAM_GRADS in the environment. Called from
   tensor_backward (training/autograd/backward.c). */
void _dbg_dump_param_grads_if_enabled(void) {
    if (!getenv("DEBUG_PARAM_GRADS")) return;
    fprintf(stderr, "=== param grads after backward ===\n");
    for (int i = 0; i < param_count_val; i++) {
        Tensor* t = param_registry[i].tensor;
        double l2 = 0.0;
        int has_nan = 0;
        if (t->grad) {
            for (int j = 0; j < t->numel; j++) {
                double g = ((double*)t->grad)[j];
                if (isnan(g) || isinf(g)) has_nan = 1;
                l2 += g * g;
            }
            l2 = sqrt(l2);
        }
        fprintf(stderr, "  %-40s numel=%-6d l2=%12.6e%s%s\n",
                param_registry[i].name, t->numel, l2,
                t->grad ? "" : " NO_GRAD",
                has_nan ? " NAN_OR_INF!" : "");
    }
}

/* Dump h0/c0 param value trajectories + first 3 element values. Set
   DEBUG_LSTM_TRAJ to print every N epochs (default 100). */
static int _dbg_traj_step = 0;
void _dbg_dump_lstm_traj_if_enabled(void) {
    if (!getenv("DEBUG_LSTM_TRAJ")) return;
    int every = 100;
    const char* every_s = getenv("DEBUG_LSTM_TRAJ_EVERY");
    if (every_s) every = atoi(every_s);
    _dbg_traj_step++;
    if (_dbg_traj_step % every != 0 && _dbg_traj_step != 1) return;
    for (int i = 0; i < param_count_val; i++) {
        const char* nm = param_registry[i].name;
        /* Match _h0 or _c0 (LSTM learned init) */
        size_t L = strlen(nm);
        if (L >= 3 && (strcmp(nm + L - 3, "_h0") == 0 || strcmp(nm + L - 3, "_c0") == 0)) {
            Tensor* t = param_registry[i].tensor;
            double l2 = 0.0, mn = 1e300, mx = -1e300;
            for (int j = 0; j < t->numel; j++) {
                double v = ((double*)t->data)[j];
                l2 += v*v;
                if (v < mn) mn = v;
                if (v > mx) mx = v;
            }
            l2 = sqrt(l2);
            fprintf(stderr, "[traj epoch %d] %s l2=%.10g min=%.10g max=%.10g | t[0..2]=%.10g, %.10g, %.10g\n",
                    _dbg_traj_step, nm, l2, mn, mx,
                    t->numel >= 1 ? ((double*)t->data)[0] : 0.0,
                    t->numel >= 2 ? ((double*)t->data)[1] : 0.0,
                    t->numel >= 3 ? ((double*)t->data)[2] : 0.0);
        }
    }
}
