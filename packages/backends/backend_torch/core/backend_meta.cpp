/* Backend meta surface — torch.
 *
 *   - backend_name              "torch"
 *   - backend_reset_for_eval    free_intermediates + zero every param's
 *                               grad. Fired between training phases (eval
 *                               loops, checkpoint reload).
 *   - tensor_print              std::cout the .cpu()-routed tensor.
 *   - tensor_mlx_compile_*      mx::compile is mlx-only; the torch
 *                               backend reports disabled / zero
 *                               invocations regardless of MLX_COMPILE.
 */
#include "../tensor.h"
#include "../training/intermediates.h"
#include <torch/torch.h>
#include <iostream>

extern "C" int param_count(void);
extern "C" void* param_tensor(int i);

extern "C" const char* backend_name(void) { return "torch"; }

extern "C" void backend_reset_for_eval(void) {
    free_intermediates();
    for (int i_ = 0; i_ < param_count(); i_++) {
        auto* tensor = (at::Tensor*)param_tensor(i_);
        if (tensor->grad().defined())
            tensor->grad().zero_();
    }
}

extern "C" void tensor_print(TensorHandle h) {
    // std::cout << at::Tensor requires the tensor to live on CPU.
    std::cout << to_tensor(h)->cpu() << std::endl;
}

/* mx::compile is mlx-only; torch backend always reports disabled
   regardless of MLX_COMPILE env var. */
extern "C" int  tensor_mlx_compile_enabled(void) { return 0; }
extern "C" int  tensor_mlx_compile_invocations(void) { return 0; }
extern "C" void tensor_mlx_compile_reset_stats(void) { }
