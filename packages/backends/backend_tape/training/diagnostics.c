/* backend_tape/training/diagnostics.c — DEBUG_PARAM_GRADS / DEBUG_LSTM_TRAJ.
 *
 * Tape-local environment-driven dumpers that walk the shared param
 * registry via its public API (`param_count` / `param_name` /
 * `param_tensor`) and downcast each tensor to tape's `Tensor*` for
 * F64-direct grad/data inspection. Backend-local because the output
 * formats embed tape-specific assumptions (grads always F64,
 * h0/c0 naming convention from the LSTM example), so the equivalent
 * dumpers for torch/mlx live in their own backend directories with
 * formats keyed to those backends' grad surfaces.
 *
 * Called from training/optimizer.c (LSTM trajectory) and
 * training/autograd/backward.c (per-backward grad norms) when the
 * matching environment variable is set.
 */

#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "../tensor.h"
#include "../../backend.h"

void _dbg_dump_param_grads_if_enabled(void) {
    if (!getenv("DEBUG_PARAM_GRADS")) return;
    fprintf(stderr, "=== param grads after backward ===\n");
    for (int i = 0; i < param_count(); i++) {
        Tensor* t = (Tensor*)param_tensor(i);
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
                param_name(i), t->numel, l2,
                t->grad ? "" : " NO_GRAD",
                has_nan ? " NAN_OR_INF!" : "");
    }
}

/* Dumps h0/c0 param value trajectories. Set DEBUG_LSTM_TRAJ to enable;
   DEBUG_LSTM_TRAJ_EVERY=N controls cadence (default 100). */
static int _dbg_traj_step = 0;

void _dbg_dump_lstm_traj_if_enabled(void) {
    if (!getenv("DEBUG_LSTM_TRAJ")) return;
    int every = 100;
    const char* every_s = getenv("DEBUG_LSTM_TRAJ_EVERY");
    if (every_s) every = atoi(every_s);
    _dbg_traj_step++;
    if (_dbg_traj_step % every != 0 && _dbg_traj_step != 1) return;
    for (int i = 0; i < param_count(); i++) {
        const char* nm = param_name(i);
        size_t L = strlen(nm);
        if (L >= 3 && (strcmp(nm + L - 3, "_h0") == 0 ||
                       strcmp(nm + L - 3, "_c0") == 0)) {
            Tensor* t = (Tensor*)param_tensor(i);
            double l2 = 0.0, mn = 1e300, mx = -1e300;
            for (int j = 0; j < t->numel; j++) {
                double v = ((double*)t->data)[j];
                l2 += v*v;
                if (v < mn) mn = v;
                if (v > mx) mx = v;
            }
            l2 = sqrt(l2);
            fprintf(stderr,
                "[traj epoch %d] %s l2=%.10g min=%.10g max=%.10g | t[0..2]=%.10g, %.10g, %.10g\n",
                _dbg_traj_step, nm, l2, mn, mx,
                t->numel >= 1 ? ((double*)t->data)[0] : 0.0,
                t->numel >= 2 ? ((double*)t->data)[1] : 0.0,
                t->numel >= 3 ? ((double*)t->data)[2] : 0.0);
        }
    }
}
