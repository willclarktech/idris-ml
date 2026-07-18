#!/usr/bin/env bash
# test_cuda_colab.sh — torch-backend CUDA smoke on Google Colab (or any
# Linux host with an NVIDIA GPU and a CUDA-enabled PyTorch wheel).
#
# Drives the repo's own build system — no hand-maintained source lists:
#   1. preflight: GPU visible, torch importable, versions printed
#   2. deps: criterion (the C test framework), best-effort via apt
#   3. make BACKEND=torch TORCH_DEVICE=cuda backend      (C dylib only)
#   4. make BACKEND=torch TORCH_DEVICE=cuda test-unit-c-torch
#
# No Idris 2 toolchain needed — this exercises the C backend directly.
# The CUDA-specific assertions live in the colocated criterion suite
# packages/backends/backend_torch/test_cuda_smoke.c (device placement,
# on-GPU add + CPU round-trip, backward onto a CUDA-resident param).
# Those tests SKIP on a box without CUDA (EAFP probe -> NULL), so this
# script's added value is providing the box — it FAILS if the suite
# skipped where a GPU was expected.
#
# Usage on Colab (Runtime -> Change runtime type -> GPU):
#   !git clone <repo-url> idris-ml
#   %cd idris-ml
#   !bash scripts/test_cuda_colab.sh
#
# Local flow validation without an NVIDIA GPU (macOS/Linux):
#   CUDA_SMOKE_ALLOW_CPU=1 bash scripts/test_cuda_colab.sh
# runs the same build + test lane with TORCH_DEVICE=cpu; the cuda
# suite skips via its probe instead of asserting.

set -euo pipefail
cd "$(dirname "$0")/.."

ALLOW_CPU="${CUDA_SMOKE_ALLOW_CPU:-0}"
LOG="$(mktemp)"
trap 'rm -f "$LOG"' EXIT

echo "=== idris-ml torch-backend CUDA smoke ==="
echo ""

# ---- 1. preflight ----------------------------------------------------

if [ "$ALLOW_CPU" != "1" ]; then
	if ! command -v nvidia-smi >/dev/null 2>&1; then
		echo "FAIL: nvidia-smi not found — no NVIDIA GPU visible."
		echo "On Colab: Runtime -> Change runtime type -> GPU."
		echo "(CUDA_SMOKE_ALLOW_CPU=1 runs the CPU-degrade flow instead.)"
		exit 1
	fi
	echo "GPU detected:"
	nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
	echo ""
fi

# torch interpreter: system python3 (Colab), falling back to the repo's
# uv venv (local dev) — the same order mk/backends.mk uses for
# LIBTORCH_PATH detection.
PY=python3
if ! $PY -c "import torch" >/dev/null 2>&1; then
	PY=packages/pytorch/.venv/bin/python3
fi
if ! $PY -c "import torch" >/dev/null 2>&1; then
	echo "FAIL: PyTorch not importable. Install with: pip install torch"
	echo "(or locally: cd packages/pytorch && uv sync)"
	exit 1
fi
echo "PyTorch:        $($PY -c 'import torch; print(torch.__version__)')"
echo "libtorch:       $($PY -c 'import torch, os; print(os.path.dirname(torch.__file__))')"
echo "CUDA available: $($PY -c 'import torch; print(torch.cuda.is_available())')"
echo "CUDA version:   $($PY -c 'import torch; print(torch.version.cuda)')"
echo ""

if [ "$ALLOW_CPU" != "1" ]; then
	if [ "$($PY -c 'import torch; print(torch.cuda.is_available())')" != "True" ]; then
		echo "FAIL: torch.cuda.is_available() is False despite nvidia-smi."
		echo "The installed torch wheel is likely CPU-only; on Colab the"
		echo "preinstalled wheel is CUDA-enabled — check the runtime type."
		exit 1
	fi
fi

# ---- 2. criterion (C test framework) ---------------------------------

if ! pkg-config --exists criterion 2>/dev/null && [ ! -e /usr/include/criterion/criterion.h ]; then
	echo "criterion not found; attempting apt install (best-effort)..."
	APT="apt-get"
	[ "$(id -u)" != "0" ] && command -v sudo >/dev/null 2>&1 && APT="sudo apt-get"
	$APT update -qq >/dev/null 2>&1 || true
	$APT install -y -qq libcriterion-dev pkg-config >/dev/null 2>&1 || true
fi
if ! pkg-config --exists criterion 2>/dev/null && [ ! -e /usr/include/criterion/criterion.h ]; then
	echo "FAIL: criterion still not found. Install it (Debian/Ubuntu:"
	echo "  apt-get install libcriterion-dev) or set CRITERION_PREFIX."
	exit 1
fi
echo "criterion: $(pkg-config --modversion criterion 2>/dev/null || echo 'header found')"
echo ""

# ---- 3 + 4. build + test via the repo's own Make lanes ----------------

DEV=cuda
[ "$ALLOW_CPU" = "1" ] && DEV=cpu
JOBS="$(nproc 2>/dev/null || sysctl -n hw.ncpu 2>/dev/null || echo 2)"

echo "--- make BACKEND=torch TORCH_DEVICE=$DEV backend (-j$JOBS) ---"
make -j"$JOBS" BACKEND=torch TORCH_DEVICE="$DEV" backend
echo ""

echo "--- make BACKEND=torch TORCH_DEVICE=$DEV test-unit-c-torch ---"
# --verbose so criterion prints [SKIP] lines — the verdict below greps
# for the torch_cuda suite's skip message, which is silent by default.
make TORCH_DEVICE="$DEV" CRITERION_FLAGS=--verbose test-unit-c-torch 2>&1 | tee "$LOG"
echo ""

# ---- verdict ----------------------------------------------------------

if [ "$ALLOW_CPU" != "1" ]; then
	# The cuda suite's skip message is unique; on a GPU box it must not
	# appear — a skip here means libtorch never saw the device.
	if grep -q "no CUDA device available" "$LOG"; then
		echo "FAIL: the torch_cuda suite SKIPPED — libtorch did not see the GPU."
		exit 1
	fi
	echo "=== CUDA smoke PASSED (torch_cuda suite asserted on the GPU) ==="
else
	echo "=== CPU-degrade flow completed (torch_cuda suite skipped by probe) ==="
fi
