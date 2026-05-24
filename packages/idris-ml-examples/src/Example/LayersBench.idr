-- | LayersBench: single-layer forward+backward microbenchmark (Axis B).
-- |
-- | Companion to packages/backends/bench_ops.c (Axis A, op-kernel level).
-- | Axis B sits one rung up — instead of pure C-kernel timing, it
-- | exercises the FFI + tape wrap + autograd graph at the *Idris-side
-- | layer* granularity. Each section runs N fwd+bwd+step cycles on a
-- | layer-shaped network and emits a one-line `<label>:\t<ms> ms (<iters>
-- | iters)` row that scripts/perf-fast.sh parses into a `kind: "op_bench"
-- | axis="B"` JSONL entry.
-- |
-- | Run:
-- |   make bench-layers
-- | The Tier-1 driver `scripts/perf-fast.sh` invokes this alongside the
-- | C-level Axis A benches.
-- |
-- | Selection (see docs/develop/testing-taxonomy.md): one entry per
-- | distinct compute pattern. Linear lands first; subsequent commits add
-- | LstmCell, TransformerBlock, NtmHead, Conv2dBlock.

module Example.LayersBench

import Data.List
import Data.String
import Data.Vect
import System
import System.Clock

import Backprop
import DataPoint
import Layer.Core
import Layer.Linear
import Tensor
import Device
import BuildConfig

%default partial


----------------------------------------------------------------------
-- Timing + formatting helpers (copied from MatmulBench; intentional
-- example-level duplication per feedback_example_duplication.md)
----------------------------------------------------------------------

elapsedMs : Clock Monotonic -> Clock Monotonic -> Double
elapsedMs t0 t1 =
  let s  = cast {to=Double} (seconds t1 - seconds t0)
      ns = cast {to=Double} (nanoseconds t1 - nanoseconds t0)
  in s * 1000.0 + ns / 1000000.0

-- Three-decimal fixed-point so the regex in scripts/perf-fast.sh
-- (`([0-9.]+)\s*ms`) reads back a stable wall-clock figure.
fmt3 : Double -> String
fmt3 x =
  let scaled = the Integer (cast (x * 1000.0))
      whole  = scaled `div` 1000
      frac   = abs (scaled `mod` 1000)
      fracS  = if frac < 10 then "00" ++ show frac
                else if frac < 100 then "0" ++ show frac
                else show frac
  in show whole ++ "." ++ fracS

repeatEpoch : Nat -> (m -> IO (m, Double)) -> m -> Double -> IO (m, Double)
repeatEpoch Z _ m loss = pure (m, loss)
repeatEpoch (S k) step m _ = do
  (m', loss') <- step m
  repeatEpoch k step m' loss'

-- Tiny non-grad 1D vector for use as a TensorDataPoint input/target.
-- Same `dtCreateState1d` path Mnist.idr / Transformer.idr take to build
-- their per-sample tensors.
buildDummyVector : (n : Nat) -> AnyPtr
buildDummyVector n =
  let nI   = the Int (cast n)
      buf  = prim__allocDoubles nI
      buf' = prim__setDouble buf 0 0.1
  in dtCreateState1d {d=ExampleDevice} {t=ExampleDType} nI buf' (deviceStreamTag {d=ExampleDevice})


----------------------------------------------------------------------
-- Linear (batch=32, in=512, out=512)
--
-- Measures dense matmul + bias + autograd graph fwd+bwd+step. The
-- batch dim is 32 so we exercise the matmul kernel's batched path
-- (not the per-sample fallback). Loss is sum-reduced MSE over each
-- row, summed over the batch — exactly what `epochVarTensorBatch`
-- does for any feedforward example.
----------------------------------------------------------------------

LinearI : Nat
LinearI = 512

LinearO : Nat
LinearO = 512

LinearBatch : Nat
LinearBatch = 32

LinearIters : Nat
LinearIters = 100

LinearWarmup : Nat
LinearWarmup = 10

benchLinear : IO ()
benchLinear = do
  ll <- linearLayerAny {i=LinearI} {o=LinearO} "axisb_linear_ll"
  let model : Network LinearI [] LinearO ExampleDevice ExampleDType WithGrad
      model = OutputLayer ll
  let opt = nativeSgd 0.01

  -- Build a shared dummy TensorDataPoint and replicate across the
  -- batch dim. Sharing is fine — `epochVarTensorBatch`'s
  -- `catAllTensors` copies on stack-and-reshape.
  let inT  = buildDummyVector LinearI
      tgtT = buildDummyVector LinearO
      dp   = the (TensorDataPoint LinearI LinearO) (MkTensorDataPoint inT tgtT)
      dps  = the (Vect LinearBatch (TensorDataPoint LinearI LinearO))
               (Data.Vect.replicate LinearBatch dp)

  (warmModel, _) <- repeatEpoch LinearWarmup
    (\m => epochVarTensorBatch opt dps tmseLoss m) model 0.0

  t0 <- clockTime Monotonic
  _ <- repeatEpoch LinearIters
    (\m => epochVarTensorBatch opt dps tmseLoss m) warmModel 0.0
  t1 <- clockTime Monotonic

  let ms = elapsedMs t0 t1
  putStrLn $ "linear bs=32 i=512 o=512:\t" ++ fmt3 ms ++ " ms\t("
          ++ show LinearIters ++ " iters)"


----------------------------------------------------------------------
-- Main
----------------------------------------------------------------------

main : IO ()
main = do
  putStrLn "--- Linear ---"
  benchLinear
  putStrLn ""
  putStrLn "=== Done ==="
