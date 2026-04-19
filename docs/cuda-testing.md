# CUDA Testing on Google Colab

## Quick Start

1. Go to [colab.research.google.com](https://colab.research.google.com)
2. Runtime -> Change runtime type -> **GPU** (T4)
3. Run:

```
!git clone https://github.com/<your-user>/idris-ml.git
!cd idris-ml && bash scripts/test_cuda_colab.sh
```

## What the Script Tests

- Full C backend test suite (`test_backend`) compiled against libtorch+CUDA
- GPU device placement (`tensor_to_device("cuda")`)
- GPU arithmetic (add on CUDA tensors)
- GPU conv2d forward pass
- GPU backward pass (autograd on CUDA)

No Idris 2 compiler needed — tests the C backend directly.

## If Something Fails

- **"No CUDA GPU available"**: check Colab runtime is set to GPU
- **libtorch link errors**: Colab's PyTorch version may have changed. Check `python3 -c "import torch; print(torch.__version__)"`
- **`-ltorch_cuda` not found**: some PyTorch builds split CUDA libs differently. Try removing `-ltorch_cuda` from the script

## Next Steps After CUDA Works

1. Add `--device cuda` flag to Idris examples (pass through to `tensor_to_device`)
2. Benchmark GPU vs CPU on MNIST and GPT examples
3. Add Linux CI with CUDA for automated testing
