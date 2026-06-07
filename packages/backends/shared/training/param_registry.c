/* shared/training/param_registry.c — backend-agnostic parameter registry.
 *
 * Owns the global parameter-array data structure (`ParamEntry` table)
 * and the surface the Idris training loop binds against:
 * `param_register`, `param_count`, `param_grad_item*`, `param_zero_all_grads`,
 * `param_subtract_delta`, `param_load_data`, `param_load_data_int64`.
 *
 * Compiled once per backend with that backend's rename header (so multi-
 * link builds get `param_register_tape`, `param_register_torch`, etc.),
 * routing all per-tensor accesses through the backend's
 * `g_active_port` adapter. The registry stays a thin coordinator —
 * backend-specific tensor reads/writes live in the adapter.
 *
 * Diagnostic dumps (`_dbg_dump_param_grads_if_enabled` /
 * `_dbg_dump_lstm_traj_if_enabled`) intentionally stay backend-local
 * (see `backend_tape/training/diagnostics.c`): their output formats
 * carry tape-specific assumptions (grads always F64, h0/c0 naming
 * convention) that don't transfer cleanly to torch/mlx, and the
 * registry's public surface (`param_count` / `param_name` / `param_tensor`)
 * is enough for them to walk the table.
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdint.h>
#include "port.h"
#include "../../backend.h"

typedef struct {
	char name[256];
	void* tensor; /* backend-opaque TensorHandle */
} ParamEntry;

#define MAX_PARAMS 65536
static ParamEntry param_registry_arr[MAX_PARAMS];
static int param_count_val = 0;

/* ----------------------------------------------------------------------
   Registration / introspection. Pure storage — no port calls.
   ---------------------------------------------------------------------- */

void param_register(const char* name, TensorHandle h) {
	/* Replace if exists (idempotent re-registration is intentional —
	   safetensors load + the per-epoch tape reset both re-register
	   under existing names). The retain/release pair lets backends
	   with refcount lifecycle (mlx) keep their params anchored against
	   sweeps that walk all_tensors; tape and torch ship no-op
	   retain/release in backend.h. */
	for (int i = 0; i < param_count_val; i++) {
		if (strcmp(param_registry_arr[i].name, name) == 0) {
			void* old = param_registry_arr[i].tensor;
			param_registry_arr[i].tensor = (void*)h;
			tensor_retain_handle((TensorHandle)h);
			tensor_release_handle((TensorHandle)old);
			return;
		}
	}
	if (param_count_val < MAX_PARAMS) {
		strncpy(param_registry_arr[param_count_val].name, name, 255);
		param_registry_arr[param_count_val].name[255] = '\0';
		param_registry_arr[param_count_val].tensor = (void*)h;
		param_count_val++;
		tensor_retain_handle((TensorHandle)h);
	}
}

void param_clear(void) {
	for (int i = 0; i < param_count_val; i++) {
		tensor_release_handle((TensorHandle)param_registry_arr[i].tensor);
	}
	param_count_val = 0;
}

/* Remove every registry entry whose name starts with `prefix`, releasing
   its retain. Used by the activation-dump path in `forwardVarTraced`
   (TRACE level) so transient `__act/<label>/<i>` entries don't survive
   the forward and pollute the optimizer's `nativeTrainStep` walk. */
void param_erase_by_prefix(const char* prefix) {
	if (!prefix || !*prefix) return;
	size_t pl = strlen(prefix);
	int read = 0, write = 0;
	while (read < param_count_val) {
		if (strncmp(param_registry_arr[read].name, prefix, pl) == 0) {
			tensor_release_handle((TensorHandle)param_registry_arr[read].tensor);
			read++;
			continue;
		}
		if (write != read) {
			param_registry_arr[write] = param_registry_arr[read];
		}
		read++;
		write++;
	}
	param_count_val = write;
}

int param_count(void) {
	return param_count_val;
}
const char* param_name(int i) {
	return param_registry_arr[i].name;
}
TensorHandle param_tensor(int i) {
	return (TensorHandle)param_registry_arr[i].tensor;
}

/* ----------------------------------------------------------------------
   Grad inspection. Per-element reads / per-element write-zero, all via
   the port so F64-grad-only is the adapter's choice not the registry's.
   ---------------------------------------------------------------------- */

double param_grad_item(int i) {
	void* t = param_registry_arr[i].tensor;
	if (!g_active_port.tensor_has_grad(t)) return 0.0;
	return g_active_port.grad_read(t, 0);
}

double param_grad_item_at(int param_idx, int elem_idx) {
	void* t = param_registry_arr[param_idx].tensor;
	if (!g_active_port.tensor_has_grad(t)) return 0.0;
	if (elem_idx >= g_active_port.tensor_numel(t)) return 0.0;
	return g_active_port.grad_read(t, elem_idx);
}

double param_grad_item_and_zero(int i) {
	void* t = param_registry_arr[i].tensor;
	if (!g_active_port.tensor_has_grad(t)) return 0.0;
	double v = g_active_port.grad_read(t, 0);
	g_active_port.grad_write(t, 0, 0.0);
	return v;
}

void param_zero_all_grads(void) {
	for (int i = 0; i < param_count_val; i++) {
		g_active_port.zero_grad(param_registry_arr[i].tensor);
	}
}

/* ----------------------------------------------------------------------
   Scalar param update — the `apply_delta` helper a few non-optimizer
   training paths use to subtract a manually-computed delta from a
   scalar param's data slot 0.
   ---------------------------------------------------------------------- */
void param_subtract_delta(int i, double delta) {
	void* t = param_registry_arr[i].tensor;
	double w = g_active_port.data_read(t, 0);
	g_active_port.data_write(t, 0, w - delta);
}

/* ----------------------------------------------------------------------
   Bulk data loaders (safetensors + checkpoint restore). The dbuf array
   the caller hands us is always `double*` even when the destination
   param is non-F64 — the port's `load_doubles` does the dtype narrowing.
   ---------------------------------------------------------------------- */

void param_load_data(int i, const double* data, int numel) {
	void* t = param_registry_arr[i].tensor;
	int dest_numel = g_active_port.tensor_numel(t);
	if (dest_numel != numel) {
		fprintf(stderr, "param_load_data: size mismatch for '%s': expected %d, got %d\n",
		        param_registry_arr[i].name, dest_numel, numel);
		return;
	}
	g_active_port.load_doubles(t, data, numel);
}

void param_load_data_int64(int i, const int64_t* data, int numel) {
	void* t = param_registry_arr[i].tensor;
	int dest_numel = g_active_port.tensor_numel(t);
	if (dest_numel != numel) {
		fprintf(stderr, "param_load_data_int64: size mismatch for '%s': expected %d, got %d\n",
		        param_registry_arr[i].name, dest_numel, numel);
		return;
	}
	g_active_port.load_int64(t, data, numel);
}
