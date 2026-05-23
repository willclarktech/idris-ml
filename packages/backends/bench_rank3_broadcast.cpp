// bench_rank3_broadcast.cpp — minimal libtorch rank-3 broadcast mul
// microbenchmark for #402. Bypasses our FFI wrapper entirely: links
// directly against libtorch and measures torch::mul wall on shapes
// that mirror applyRopeAllHeads's hot inner ops.
//
// The wall here is the BASELINE for the same workload PyTorch Python
// runs (~2 ms/op observed in time_inference_llama.py). Our wrapper
// shows ~10-26 ms/op for the same broadcast (`docs/develop/perf-changes.md`
// 2026-05-30 entry). The gap between this benchmark's number and our
// wrapper's number identifies whether the cost is in the wrapper
// (FFI marshalling, from_tensor allocation, intermediates tracking)
// or in libtorch's MPS path itself (strided-view materialization,
// MPSGraph compile cache).
//
// Variants:
//   strided   — cos via narrow(0, ..) + reshape({seq, 1, halfDim})
//               (matches Layer/RoPE.idr's applyRopeAllHeads)
//   contig    — cos materialized contiguous up front
//
// Build + run via `make bench-rank3-broadcast` (the make target sets
// LIBTORCH_PATH via the existing torch-detection block).

#include <torch/torch.h>
#include <chrono>
#include <cstdio>
#include <string>

// Shapes mirror the Llama-3.2-1B Q/K projection passed into RoPE on
// the first decode step: [seq=6, numHeads=32, halfDim=32] for Q,
// [seq=6, numKvHeads=8, halfDim=32] for K. RoPE multiplies both
// against a [seq, 1, halfDim] cos/sin slice. The Q broadcast is the
// hotter of the two because it's 4x the headcount; use that shape.
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
  // CPU: nothing to do; eager ops are synchronous on construction.
}

static double bench_strided(torch::Device d) {
  auto x = torch::randn({SEQ, NUM_HEADS, HALF_DIM}, d);
  // cos starts as a [maxPos, halfDim] table; narrow(0, offset, SEQ)
  // gives a strided view, reshape adds the broadcast dim.
  auto cos_table = torch::randn({2048, HALF_DIM}, d);
  auto cos = cos_table.narrow(0, 0, SEQ).reshape({SEQ, 1, HALF_DIM});

  for (int i = 0; i < N_WARMUP; ++i) {
    auto y = torch::mul(x, cos);
    (void)y;
  }
  sync_device(d);

  auto t0 = std::chrono::steady_clock::now();
  for (int i = 0; i < N_ITER; ++i) {
    auto y = torch::mul(x, cos);
    (void)y;
  }
  sync_device(d);
  auto t1 = std::chrono::steady_clock::now();

  double us = std::chrono::duration_cast<std::chrono::microseconds>(t1 - t0).count();
  return us / N_ITER;
}

static double bench_contig(torch::Device d) {
  auto x = torch::randn({SEQ, NUM_HEADS, HALF_DIM}, d);
  // cos materialized contiguous up front — what the "pre-materialize"
  // fix (Commit 2C in the plan) would build at table-build time.
  auto cos = torch::randn({SEQ, 1, HALF_DIM}, d).contiguous();

  for (int i = 0; i < N_WARMUP; ++i) {
    auto y = torch::mul(x, cos);
    (void)y;
  }
  sync_device(d);

  auto t0 = std::chrono::steady_clock::now();
  for (int i = 0; i < N_ITER; ++i) {
    auto y = torch::mul(x, cos);
    (void)y;
  }
  sync_device(d);
  auto t1 = std::chrono::steady_clock::now();

  double us = std::chrono::duration_cast<std::chrono::microseconds>(t1 - t0).count();
  return us / N_ITER;
}

int main(int argc, char** argv) {
  std::string device_str = "cpu";
  if (argc > 1) device_str = argv[1];

  torch::Device d(torch::kCPU);
  if (device_str == "mps") {
#ifdef __APPLE__
    if (torch::mps::is_available()) {
      d = torch::Device(torch::kMPS);
    } else {
      std::fprintf(stderr, "MPS requested but not available; falling back to CPU\n");
    }
#else
    std::fprintf(stderr, "MPS not built; falling back to CPU\n");
#endif
  } else if (device_str == "cuda") {
    if (torch::cuda::is_available()) {
      d = torch::Device(torch::kCUDA, 0);
    } else {
      std::fprintf(stderr, "CUDA requested but not available; falling back to CPU\n");
    }
  }

  std::printf("device: %s\n", device_str.c_str());
  std::printf("shape: x=[%d,%d,%d] cos=[%d,1,%d]\n",
              SEQ, NUM_HEADS, HALF_DIM, SEQ, HALF_DIM);
  std::printf("iterations: warmup=%d measure=%d\n", N_WARMUP, N_ITER);

  double strided_us = bench_strided(d);
  std::printf("[strided] %.2f us/op  (= %.3f ms/op)\n", strided_us, strided_us / 1000.0);

  double contig_us = bench_contig(d);
  std::printf("[contig ] %.2f us/op  (= %.3f ms/op)\n", contig_us, contig_us / 1000.0);

  std::printf("strided/contig ratio: %.2fx\n", strided_us / contig_us);
  return 0;
}
