#!/bin/bash
# test_cuda_colab.sh — Run on Google Colab (or any Linux with CUDA + PyTorch)
#
# Tests the idris-ml torch backend on CUDA GPU.
# No Idris 2 compiler needed — tests the C backend directly.
#
# Usage on Colab:
#   !git clone https://github.com/<user>/idris-ml.git
#   !cd idris-ml && bash scripts/test_cuda_colab.sh
#
# Or as a single cell:
#   !git clone ... && cd idris-ml && bash scripts/test_cuda_colab.sh

set -e

echo "=== idris-ml CUDA backend test ==="
echo ""

# 1. Check CUDA availability
if ! command -v nvidia-smi &>/dev/null; then
    echo "FAIL: nvidia-smi not found. No CUDA GPU available."
    echo "On Colab: Runtime -> Change runtime type -> GPU"
    exit 1
fi
echo "GPU detected:"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
echo ""

# 2. Find PyTorch/libtorch
TORCH_DIR=$(python3 -c "import torch, os; print(os.path.dirname(torch.__file__))")
if [ -z "$TORCH_DIR" ]; then
    echo "FAIL: PyTorch not found. Install with: pip install torch"
    exit 1
fi
echo "PyTorch: $TORCH_DIR"
echo "CUDA available: $(python3 -c 'import torch; print(torch.cuda.is_available())')"
echo "CUDA version: $(python3 -c 'import torch; print(torch.version.cuda)')"
echo ""

# 3. Build the torch backend
echo "Building torch backend..."
TORCH_INC="$TORCH_DIR/include"
TORCH_INC_API="$TORCH_DIR/include/torch/csrc/api/include"
TORCH_LIB="$TORCH_DIR/lib"

mkdir -p build

# Shared objects
cc -O2 -c -o build/safetensors.o csrc/safetensors.c
cc -O2 -c -o build/cJSON.o csrc/cJSON.c
cc -O2 -c -o build/mnist.o csrc/mnist.c
cc -O2 -c -o build/dataloader.o csrc/dataloader.c

# Torch backend (Linux)
c++ -std=c++17 -O2 -shared -fPIC \
    -I"$TORCH_INC" -I"$TORCH_INC_API" \
    -L"$TORCH_LIB" -ltorch -ltorch_cpu -lc10 -ltorch_cuda \
    -Wl,-rpath,"$TORCH_LIB" \
    -o build/libidrisml.so \
    csrc/backend_torch.cpp build/safetensors.o build/cJSON.o build/mnist.o build/dataloader.o
echo "  build/libidrisml.so OK"
echo ""

# 4. Build and run test_backend
echo "Running C backend tests..."
cc -o build/test_backend csrc/test_backend.c -Lbuild -lidrisml -Wl,-rpath,build -lm
./build/test_backend
echo ""

# 5. Test CUDA device placement
echo "Testing CUDA device placement..."
cat > /tmp/test_cuda_device.c << 'CEOF'
#include "backend.h"
#include <stdio.h>
#include <string.h>

int main(void) {
    printf("=== CUDA Device Tests ===\n");

    /* Create a tensor on CPU */
    double data[] = {1.0, 2.0, 3.0, 4.0};
    int shape[] = {4};
    TensorHandle cpu_t = tensor_create(data, shape, 1, 1);
    printf("Created on: %s\n", tensor_device(cpu_t));

    /* Move to CUDA */
    TensorHandle gpu_t = tensor_to_device(cpu_t, "cuda");
    const char* dev = tensor_device(gpu_t);
    printf("After to_device('cuda'): %s\n", dev);

    if (strstr(dev, "cuda") == NULL) {
        printf("FAIL: tensor not on CUDA\n");
        return 1;
    }

    /* Arithmetic on GPU */
    TensorHandle gpu_sum = tensor_add(gpu_t, gpu_t);
    printf("GPU add device: %s\n", tensor_device(gpu_sum));

    /* Move back to CPU and verify values */
    TensorHandle result = tensor_to_device(gpu_sum, "cpu");
    printf("Back on CPU: %s\n", tensor_device(result));

    double out[4];
    tensor_to_doubles(result, out);
    int ok = 1;
    for (int i = 0; i < 4; i++) {
        printf("  result[%d] = %.1f (expected %.1f)\n", i, out[i], data[i] * 2);
        if (out[i] != data[i] * 2) ok = 0;
    }

    /* Test conv2d on GPU */
    double inp[] = {1,2,3,4, 5,6,7,8, 9,10,11,12, 13,14,15,16};
    int inp_s[] = {1, 4, 4};
    TensorHandle inp_t = tensor_to_device(tensor_create(inp, inp_s, 3, 0), "cuda");
    double ker[] = {1, 0, 0, 1};
    int ker_s[] = {1, 1, 2, 2};
    TensorHandle ker_t = tensor_to_device(tensor_create(ker, ker_s, 4, 0), "cuda");
    TensorHandle conv_out = tensor_conv2d(inp_t, ker_t, NULL, 0, 0, 1, 1);
    printf("Conv2D on GPU: %s, numel=%d\n", tensor_device(conv_out), tensor_numel(conv_out));

    /* Test backward on GPU */
    double w_data[] = {2.0};
    int w_shape[] = {1};
    TensorHandle w = tensor_to_device(tensor_create(w_data, w_shape, 1, 1), "cuda");
    double x_data[] = {3.0};
    TensorHandle x = tensor_to_device(tensor_create(x_data, w_shape, 1, 0), "cuda");
    TensorHandle y = tensor_mul(w, x);
    tensor_backward(y);
    TensorHandle w_grad = tensor_grad(w);
    TensorHandle w_grad_cpu = tensor_to_device(w_grad, "cpu");
    double grad_val = tensor_item(w_grad_cpu);
    printf("GPU backward: d(w*x)/dw = %.1f (expected 3.0)\n", grad_val);
    if (grad_val != 3.0) ok = 0;

    printf("\n%s\n", ok ? "All CUDA tests PASSED" : "Some CUDA tests FAILED");
    return ok ? 0 : 1;
}
CEOF

cc -o build/test_cuda_device /tmp/test_cuda_device.c -Icsrc -Lbuild -lidrisml -Wl,-rpath,build -lm
./build/test_cuda_device

echo ""
echo "=== Done ==="
