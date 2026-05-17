/* Backend meta surface — mlx.
 *
 *   - backend_name              "mlx"
 *   - backend_reset_for_eval    tape_reset + re-anchor each param as
 *                               an OP_CONST entry so backward sees the
 *                               (now-cleared-grad) params on the next
 *                               forward.
 *   - tensor_print              std::cout the realized mx::array.
 *   - tensor_mlx_compile_enabled   gate from $MLX_COMPILE (1/true/yes).
 *   - tensor_mlx_compile_invocations / _reset_stats
 *                               read/clear g_compile_invocations
 *                               (defined here; backward.cpp + optimizer.cpp
 *                               extern-bump it on each cached trace).
 */
#include "../tensor.h"
#include "../tape.h"
#include <cstdlib>
#include <cstring>
#include <iostream>

extern "C" int param_count(void);
extern "C" void* param_tensor(int i);
extern "C" void param_clear(void);

extern "C" const char* backend_name(void) { return "mlx"; }

/* See backend.h: explicit pre-exit cleanup. mlx-cpu shows a smaller
 * post-main tail than torch-cpu (mlx 14:30, torch 20:05 on HfLlama
 * 1.2B). mlx's per-tensor delete sits behind the static-scoped
 * `mlx_sweep_generation` (`autograd.cpp`), unreachable from this TU;
 * the simplest available hooks are `param_clear` (decrements
 * refcount via tensor_release_handle → tensor_release_internal,
 * dropping params to 0) plus `mx::clear_cache` to drop cached
 * MTLBuffer / CPU allocator pool entries. Best-effort only — if
 * mlx-cpu's tail proves stubborn, expose `mlx_sweep_generation`
 * publicly + walk all_tensors here. */
extern "C" void backend_release_all_persistent(void) {
    param_clear();
    try { mx::clear_cache(); } catch (...) { /* best-effort at shutdown */ }
}

extern "C" void backend_reset_for_eval(void) {
    tape_reset();
    for (int i_ = 0; i_ < param_count(); i_++) {
        auto* p_tensor = (Tensor*)param_tensor(i_);
        p_tensor->tape_idx = -1;
        p_tensor->has_grad = false;
        tape_append(OP_CONST, p_tensor, nullptr, nullptr, 0);
    }
}

extern "C" void tensor_print(TensorHandle h) {
    auto t = (Tensor*)h;
    mx::eval(t->data);
    std::cout << t->data << std::endl;
}

extern "C" int tensor_mlx_compile_enabled(void) {
    const char* v = std::getenv("MLX_COMPILE");
    if (!v) return 0;
    if (v[0] == '1' && v[1] == '\0') return 1;
    if (std::strcmp(v, "true") == 0) return 1;
    if (std::strcmp(v, "yes") == 0) return 1;
    return 0;
}

/* Counter g_compile_invocations — incremented on each cached
   mx::compile trace by training/backward.cpp + training/optimizer.cpp.
   Non-static so those TUs can extern it. */
int g_compile_invocations = 0;
extern "C" int  tensor_mlx_compile_invocations(void) { return g_compile_invocations; }
extern "C" void tensor_mlx_compile_reset_stats(void) { g_compile_invocations = 0; }
