// bench_rank3_broadcast_wrapped.cpp — rank-3 broadcast mul microbench
// that goes through our C wrapper layer (tensor_mul_torch et al).
//
// Companion to bench_rank3_broadcast.cpp. That bench measures
// torch::mul directly (no idris-ml glue). This bench measures
// tensor_mul_torch — the exact symbol the Idris Scheme wrapper
// dlsyms into. Same shapes, same iteration counts.
//
// The gap between this number and the direct number isolates the
// C-side wrapper cost (from_tensor's `new at::Tensor` + intermediates
// push + prof_op_count_torch++). Any remaining gap up to HfLlama's
// ~10-26 ms/op observed wall lives above the C boundary — in the
// generated Scheme wrapper, the Idris autograd machinery, or the
// per-op typeclass dispatch.
//
// Build + run via `make bench-rank3-broadcast-wrapped`.

#include <torch/torch.h>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <string>
#include <vector>

#include "backend.h"

// Suffixed exports from libidrisml.dylib. Declared by hand so this TU
// doesn't need to include rename_torch.h (which is meant for backend
// internals).
extern "C" {
  TensorHandle tensor_create_f32_torch(double* data, int* shape, int rank, int requires_grad);
  TensorHandle tensor_create_f64_torch(double* data, int* shape, int rank, int requires_grad);
  TensorHandle tensor_mul_torch(TensorHandle a, TensorHandle b);
  TensorHandle tensor_narrow_torch(TensorHandle t, int dim, int start, int len);
  TensorHandle tensor_reshape_3d_torch(TensorHandle t, int d0, int d1, int d2);
  void tensor_perf_reset_torch(void);
  long tensor_perf_op_count_torch(void);
}

static constexpr int SEQ = 6;
static constexpr int NUM_HEADS = 32;
static constexpr int HALF_DIM = 32;
static constexpr int N_WARMUP = 10;
static constexpr int N_ITER = 100;

static void sync_device(torch::Device d) {
#ifdef __APPLE__
  if (d.type() == torch::kMPS) {
    torch::mps::synchronize();
    return;
  }
#endif
}

// Fills `data` with deterministic pseudorandom doubles (small values
// in [-1, 1)). Mirrors what torch::randn would produce in distribution
// but matches the C ABI (double*) `tensor_create_f32` expects.
static void fill_random(std::vector<double>& data, unsigned seed) {
  uint32_t s = seed;
  for (auto& x : data) {
    s = s * 1664525u + 1013904223u;
    x = (double)((int32_t)s) / (double)INT32_MAX;
  }
}

static double bench_strided(torch::Device d, bool use_f32) {
  int x_shape[3] = {SEQ, NUM_HEADS, HALF_DIM};
  int tbl_shape[2] = {2048, HALF_DIM};

  std::vector<double> x_data(SEQ * NUM_HEADS * HALF_DIM);
  std::vector<double> tbl_data(2048 * HALF_DIM);
  fill_random(x_data, 1);
  fill_random(tbl_data, 2);

  TensorHandle x = use_f32
    ? tensor_create_f32_torch(x_data.data(), x_shape, 3, 0)
    : tensor_create_f64_torch(x_data.data(), x_shape, 3, 0);
  TensorHandle tbl = use_f32
    ? tensor_create_f32_torch(tbl_data.data(), tbl_shape, 2, 0)
    : tensor_create_f64_torch(tbl_data.data(), tbl_shape, 2, 0);

  // Strided cos: narrow [2048, halfDim] to [seq, halfDim], reshape
  // to [seq, 1, halfDim]. Mirrors applyRopeAllHeads.
  TensorHandle cos_2d = tensor_narrow_torch(tbl, 0, 0, SEQ);
  TensorHandle cos = tensor_reshape_3d_torch(cos_2d, SEQ, 1, HALF_DIM);

  for (int i = 0; i < N_WARMUP; ++i) {
    (void)tensor_mul_torch(x, cos);
  }
  sync_device(d);

  tensor_perf_reset_torch();
  auto t0 = std::chrono::steady_clock::now();
  for (int i = 0; i < N_ITER; ++i) {
    (void)tensor_mul_torch(x, cos);
  }
  sync_device(d);
  auto t1 = std::chrono::steady_clock::now();
  long ops = tensor_perf_op_count_torch();

  double us = (double)std::chrono::duration_cast<std::chrono::microseconds>(t1 - t0).count();
  std::printf("[strided] op_count from C wrapper: %ld (expected %d)\n", ops, N_ITER);
  return us / N_ITER;
}

static double bench_contig(torch::Device d, bool use_f32) {
  int x_shape[3] = {SEQ, NUM_HEADS, HALF_DIM};
  int cos_shape[3] = {SEQ, 1, HALF_DIM};

  std::vector<double> x_data(SEQ * NUM_HEADS * HALF_DIM);
  std::vector<double> cos_data(SEQ * 1 * HALF_DIM);
  fill_random(x_data, 3);
  fill_random(cos_data, 4);

  TensorHandle x = use_f32
    ? tensor_create_f32_torch(x_data.data(), x_shape, 3, 0)
    : tensor_create_f64_torch(x_data.data(), x_shape, 3, 0);
  TensorHandle cos = use_f32
    ? tensor_create_f32_torch(cos_data.data(), cos_shape, 3, 0)
    : tensor_create_f64_torch(cos_data.data(), cos_shape, 3, 0);

  for (int i = 0; i < N_WARMUP; ++i) {
    (void)tensor_mul_torch(x, cos);
  }
  sync_device(d);

  tensor_perf_reset_torch();
  auto t0 = std::chrono::steady_clock::now();
  for (int i = 0; i < N_ITER; ++i) {
    (void)tensor_mul_torch(x, cos);
  }
  sync_device(d);
  auto t1 = std::chrono::steady_clock::now();
  long ops = tensor_perf_op_count_torch();

  double us = (double)std::chrono::duration_cast<std::chrono::microseconds>(t1 - t0).count();
  std::printf("[contig ] op_count from C wrapper: %ld (expected %d)\n", ops, N_ITER);
  return us / N_ITER;
}

int main(int argc, char** argv) {
  std::string device_str = "cpu";
  if (argc > 1) device_str = argv[1];

  // libidrisml's torch backend reads TORCH_DEVICE in a static-init
  // constructor that runs before main, so we can't setenv() it from
  // here. The make recipe prefixes `TORCH_DEVICE=mps` for the mps
  // run; this CLI arg only picks which sync path we drive.

  // CPU builds (TORCH_DEVICE unset) default to F64; MPS builds force
  // F32. Pick the dtype matching whichever build the recipe selected.
  bool use_f32 = (device_str == "mps");

  torch::Device d(torch::kCPU);
  if (device_str == "mps") {
#ifdef __APPLE__
    if (torch::mps::is_available()) {
      d = torch::Device(torch::kMPS);
    } else {
      std::fprintf(stderr, "MPS requested but not available; falling back to CPU\n");
      use_f32 = false;
    }
#else
    std::fprintf(stderr, "MPS not built; falling back to CPU\n");
    use_f32 = false;
#endif
  } else if (device_str == "cuda") {
    if (torch::cuda::is_available()) {
      d = torch::Device(torch::kCUDA, 0);
    } else {
      std::fprintf(stderr, "CUDA requested but not available; falling back to CPU\n");
    }
  }

  std::printf("device: %s  dtype: %s\n",
              device_str.c_str(), use_f32 ? "f32" : "f64");
  std::printf("shape: x=[%d,%d,%d] cos=[%d,1,%d]\n",
              SEQ, NUM_HEADS, HALF_DIM, SEQ, HALF_DIM);
  std::printf("iterations: warmup=%d measure=%d\n", N_WARMUP, N_ITER);

  double strided_us = bench_strided(d, use_f32);
  std::printf("[strided] %.2f us/op  (= %.3f ms/op)\n", strided_us, strided_us / 1000.0);

  double contig_us = bench_contig(d, use_f32);
  std::printf("[contig ] %.2f us/op  (= %.3f ms/op)\n", contig_us, contig_us / 1000.0);

  std::printf("strided/contig ratio: %.2fx\n", strided_us / contig_us);
  return 0;
}
