-- | MatmulBench: compute-bound microbenchmark demonstrating mlx GPU > CPU
-- |
-- | A pure forward pass: build two N×N matrices, do K matmuls, time
-- | the wall. No training, no gradient, no optimizer — just the
-- | type-safe `Tensor` API exercising the backend's matmul kernel.
-- |
-- | Why this example exists: idris-ml's training examples
-- | (Transformer, NTM/DNC) operate at scales where mlx GPU's per-op
-- | kernel-launch overhead dominates compute — CPU's Accelerate BLAS
-- | wins. This example deliberately picks tensor sizes (N=2048-4096)
-- | where compute >> launch, exposing mlx GPU's parallel-matmul
-- | advantage. Measured 2026-05-15 on M-series in a Tart VM: CPU and
-- | GPU cross at N≈1024; GPU wins 2.3× at N=2048 and 3.75× at N=4096.
-- |
-- | Run:
-- |   MLX_DEVICE=cpu make example-matmul-bench
-- |   MLX_DEVICE=gpu make example-matmul-bench
-- | Override defaults via `MATMUL_BENCH_ARGS`:
-- |   make example-matmul-bench MATMUL_BENCH_ARGS="--size 4096 --iters 8"

module Example.MatmulBench

import Data.List
import Data.String
import System
import System.Clock

import BuildConfig
import Ml.Executor
import Ml.Tensor

record Config where
  constructor MkConfig
  size  : Nat
  iters : Nat

defaultConfig : Config
defaultConfig = MkConfig 2048 5

parseArgs : Config -> List String -> Config
parseArgs cfg []                      = cfg
parseArgs cfg ("--size" :: x :: rest) =
  parseArgs ({size := the Nat (cast (the Int (cast x)))} cfg) rest
parseArgs cfg ("--iters" :: x :: rest) =
  parseArgs ({iters := the Nat (cast (the Int (cast x)))} cfg) rest
parseArgs cfg (_ :: rest) = parseArgs cfg rest

-- Build an n×n fp32 matrix with one nonzero element via a single
-- allocator (no Idris per-element loop). Returns a non-grad state
-- tensor handle.
buildMatrix : (n : Nat) -> AnyPtr
buildMatrix n =
  let nI = the Int (cast n)
      buf  = prim__allocDoubles (nI * nI)
      buf' = prim__setDouble buf 0 0.5
  in dtCreateState2d {ex=ExampleExecutor} {t=ExampleDType} nI nI buf' (deviceStreamTag {ex=ExampleExecutor})

-- Elapsed milliseconds between two monotonic clock readings.
diffMs : Clock Monotonic -> Clock Monotonic -> Double
diffMs t1 t0 =
  let totalNs = (seconds t1 - seconds t0) * 1000000000 + (nanoseconds t1 - nanoseconds t0)
  in cast totalNs / 1000000.0

-- Two-decimal fixed-point formatting.
fmt2 : Double -> String
fmt2 x =
  let scaled = the Integer (cast (x * 100.0))
      whole = scaled `div` 100
      frac  = abs (scaled `mod` 100)
      fracS = if frac < 10 then "0" ++ show frac else show frac
  in show whole ++ "." ++ fracS

-- Force an mlx-side eval of a tensor handle. `prim__item` on the
-- scalar result of `sum` walks the graph.
forceEval : AnyPtr -> IO ()
forceEval h = do
  let v = primItem {ex=ExampleExecutor} (primSum {ex=ExampleExecutor} h)
  ignore (pure v)

-- Run K matmuls, forcing eval after each so we measure real compute
-- (mlx is lazy by default — without per-iter eval it fuses across the
-- whole loop and we'd time graph build, not the work).
loopMatmul : Nat -> AnyPtr -> AnyPtr -> IO ()
loopMatmul Z _ _     = pure ()
loopMatmul (S k) a b = do
  let c = primMm {ex=ExampleExecutor} a b
  forceEval c
  loopMatmul k a b

main : IO ()
main = do
  requireMachine {m = ChosenMachine}
  args <- getArgs
  let cfg = parseArgs defaultConfig (drop 1 args)

  putStrLn "=== MatmulBench: pure forward matmul ==="
  putStrLn $ "Config: N=" ++ show cfg.size
           ++ " iters=" ++ show cfg.iters
  putStrLn "(set MLX_DEVICE=cpu or gpu to compare streams; mlx only)"

  let a = buildMatrix cfg.size
  let b = buildMatrix cfg.size

  -- Warmup pass — flushes any one-time backend setup (Metal pipeline
  -- cache, BLAS thread spin-up, etc.).
  forceEval (primMm {ex=ExampleExecutor} a b)

  t0 <- clockTime Monotonic
  loopMatmul cfg.iters a b
  t1 <- clockTime Monotonic
  let dtMs = diffMs t1 t0

  let perCall      = dtMs / cast cfg.iters
  let flopsPerCall = 2.0 * cast cfg.size * cast cfg.size * cast cfg.size
  let gflops       = flopsPerCall / (perCall / 1000.0) / 1.0e9

  putStrLn $ "Total wall:  " ++ fmt2 dtMs      ++ " ms"
  putStrLn $ "Per matmul:  " ++ fmt2 perCall ++ " ms"
  putStrLn $ "Throughput:  " ++ fmt2 gflops  ++ " GFLOPS"
  putStrLn ""
  putStrLn $ "RESULT\tN=" ++ show cfg.size
           ++ "\titers=" ++ show cfg.iters
           ++ "\ttotal_ms=" ++ fmt2 dtMs
           ++ "\tper_call_ms=" ++ fmt2 perCall
           ++ "\tgflops=" ++ fmt2 gflops
