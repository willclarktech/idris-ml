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

extern "C" const char* backend_name(void) { return "mlx"; }

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
