# CUDA testing on Google Colab

The torch backend targets CUDA through the same libtorch build the CPU
and MPS lanes use; the only thing a CUDA test needs is a box with an
NVIDIA GPU. Colab provides one for free, and its preinstalled PyTorch
wheel bundles a CUDA-enabled libtorch, so no toolchain setup is needed
beyond `apt`.

## Quick start

1. Go to [colab.research.google.com](https://colab.research.google.com)
2. Runtime -> Change runtime type -> **GPU** (T4)
3. Run:

```
!git clone <repo-url> idris-ml
%cd idris-ml
!bash scripts/test_cuda_colab.sh
```

`make test-e2e-cuda` is the same entry point for a local Linux CUDA box.

## What the script does

`scripts/test_cuda_colab.sh` drives the repo's own build system — it
maintains no source lists of its own:

1. Preflight: `nvidia-smi`, `import torch`, and
   `torch.cuda.is_available()` must all succeed; versions are printed.
2. Installs criterion (the C test framework) via
   `apt-get install libcriterion-dev`, best-effort.
3. `make BACKEND=torch TORCH_DEVICE=cuda backend` — builds
   `libidrisml.so` from `packages/backends/backend_torch/` against the
   wheel's libtorch (auto-detected by mk/backends.mk). No Idris 2
   toolchain is involved.
4. `make TORCH_DEVICE=cuda test-unit-c-torch` — the full torch criterion
   suite, including the CUDA-specific tests below.

## The CUDA assertions

The CUDA coverage is a normal colocated criterion suite,
`packages/backends/backend_torch/test_cuda_smoke.c`:

- device placement: `tensor_to_device(h, "cuda")` yields a handle whose
  `tensor_device` string contains `cuda`
- on-GPU arithmetic: add on CUDA tensors, migrate back to `cpu`, exact
  F64 value round-trip
- autograd: backward through `w * x` with both operands CUDA-resident;
  the gradient reads back as `x`

Each test starts from the EAFP probe the backend already provides —
`tensor_to_device(h, "cuda")` returns NULL when no CUDA device exists
(`device.cpp` catches the `c10::Error`) — and SKIPs on NULL. So the
suite is green on macOS/CPU lanes, and the Colab script fails if the
tests skipped where a GPU was expected.

## Validating the flow without a GPU

```bash
CUDA_SMOKE_ALLOW_CPU=1 bash scripts/test_cuda_colab.sh
```

runs the identical build + test lane with `TORCH_DEVICE=cpu`; the
`torch_cuda` suite skips via its probe. This is the local regression
check for the script itself.

## If something fails

- **"No NVIDIA GPU visible"**: the Colab runtime type is not GPU.
- **`torch.cuda.is_available()` is False despite nvidia-smi**: the torch
  wheel is CPU-only; on Colab this means the runtime type changed after
  install — restart the runtime.
- **criterion not found**: `apt-get install libcriterion-dev` (the
  script attempts this itself), or set `CRITERION_PREFIX`.
- **libtorch link errors**: Colab's PyTorch version may have changed;
  check `python3 -c "import torch; print(torch.__version__)"` and see
  the `torch_LDFLAGS_Linux` note in mk/backends.mk.

## After the C lane: the full Idris lane

Device selection is a build-time type (`TorchExecutor (TCuda 0)`), so
the Idris-level CUDA lane is not a runtime flag — it is
`make BACKEND=torch TORCH_DEVICE=cuda install` on a CUDA box with an
Idris 2 + pack toolchain, after which examples run on the GPU cell and
`scripts/perf-sweep.sh --cells torch-cuda` benchmarks it. The remaining
work for that lane is tracked in the TODO CUDA row.
