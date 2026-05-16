/* Intermediate tensor tracking + bulk cleanup for the torch backend.
 *
 * Params go via from_tensor_persistent and are never tracked here, so
 * free_intermediates can bulk-delete without filtering. (A previous
 * version built an unordered_set<at::Tensor*> from param_registry per
 * call as a safety net — that was a hot-path hash build for thousands
 * of intermediates on DNC-class workloads.)
 *
 * `freed_by_cleanup` is the per-call set torch's lifecycle helpers can
 * consult to skip already-deleted pointers; tensor_free is a no-op
 * today (see backend_torch/core/lifecycle/free.cpp) so the set isn't
 * read externally, but keeping it here keeps the contract local to
 * the cleanup path. */
#include <unordered_set>
#include "intermediates.h"

std::vector<at::Tensor*> intermediates_torch;
std::vector<TensorPair*> all_pairs_torch;
bool tracking_enabled_torch = true;
long g_torch_peak_live_intermediates = 0;

namespace {
struct _ReserveIntermediates {
    _ReserveIntermediates() {
        intermediates_torch.reserve(4096);
        all_pairs_torch.reserve(256);
    }
};
_ReserveIntermediates _reserve_intermediates_instance;
} // namespace

static std::unordered_set<void*> freed_by_cleanup;

void free_intermediates(void) {
    freed_by_cleanup.clear();
    freed_by_cleanup.reserve(intermediates_torch.size());
    for (auto* p : intermediates_torch) {
        if (p) {
            freed_by_cleanup.insert(p);
            delete p;
        }
    }
    intermediates_torch.clear();
    for (auto* p : all_pairs_torch) delete p;
    all_pairs_torch.clear();
}

extern "C" int tensor_live_count(int dummy) { (void)dummy; return (int)intermediates_torch.size(); }
extern "C" int tensor_peak_live_count(int dummy) { (void)dummy; return (int)g_torch_peak_live_intermediates; }
