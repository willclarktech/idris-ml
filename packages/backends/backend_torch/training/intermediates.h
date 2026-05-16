/* Intermediate tensor tracking for the torch backend.
 *
 * `intermediates_torch` is the list of non-persistent at::Tensor* nodes
 * created during forward/backward — bulk-freed at optimizer_step via
 * `free_intermediates()`. `all_pairs_torch` mirrors that for TensorPair
 * boxes returned by tensor_lstm_gates_pair.
 *
 * `tracking_enabled_torch` lets the no-grad block + eval path bypass
 * the push (avoids growing intermediates inside withNoGrad).
 *
 * `g_torch_peak_live_intermediates` is a high-water mark exposed by
 * tensor_peak_live_count for diagnostics.
 *
 * The `_torch` suffix sidesteps a tri-link collision with mlx's
 * same-named `all_pairs` global (which is exposed across mlx's own
 * TU boundary, so it can't be static-scoped). */
#ifndef IDRISML_BACKEND_TORCH_INTERMEDIATES_H
#define IDRISML_BACKEND_TORCH_INTERMEDIATES_H

#include <vector>
#include <ATen/ATen.h>
#include "../../backend.h"

extern std::vector<at::Tensor*> intermediates_torch;
extern std::vector<TensorPair*> all_pairs_torch;
extern bool tracking_enabled_torch;
extern long g_torch_peak_live_intermediates;

void free_intermediates(void);

#endif /* IDRISML_BACKEND_TORCH_INTERMEDIATES_H */
