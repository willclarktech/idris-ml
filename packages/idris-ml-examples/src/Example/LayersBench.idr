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
import Generate
import Layer.Conv
import Layer.Core
import Layer.Linear
import Layer.Lstm
import Layer.Ntm
import Layer.Transformer
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
-- LstmCell (hidden=256, unbatched)
--
-- Recurrent cell — 4 internal matmuls + per-timestep h/c state. The
-- typeclass-default `applyVarBatch` isn't overridden for LSTM (only
-- `applyVar`), so we benchmark single-sample forward+backward+step
-- per iter. Each iter exercises one timestep; the persistent h/c
-- state threads through across the 100 iters via the LstmState
-- record's `Maybe (Tensor [o])` slots.
----------------------------------------------------------------------

LstmI : Nat
LstmI = 256

LstmO : Nat
LstmO = 256

LstmIters : Nat
LstmIters = 100

LstmWarmup : Nat
LstmWarmup = 10

lstmStep : NativeOptimizer ExampleDevice ->
           Tensor [LstmI] ExampleDevice ExampleDType WithGrad ->
           Tensor [LstmO] ExampleDevice ExampleDType WithGrad ->
           Network LstmI [] LstmO ExampleDevice ExampleDType WithGrad ->
           IO (Network LstmI [] LstmO ExampleDevice ExampleDType WithGrad, Double)
lstmStep opt inp tgt model = do
  (model', pred) <- forwardVar model inp
  loss <- tmseLoss pred tgt
  ms <- nativeTrainStep opt loss
  pure (model', ms)

benchLstmCell : IO ()
benchLstmCell = do
  ll <- lstmLayerAny {i=LstmI} {o=LstmO} "axisb_lstm"
  let model : Network LstmI [] LstmO ExampleDevice ExampleDType WithGrad
      model = OutputLayer ll
  let opt = nativeSgd 0.01

  let inT  = buildDummyVector LstmI
      tgtT = buildDummyVector LstmO
      inp  = the (Tensor [LstmI] ExampleDevice ExampleDType WithGrad)
               (MkTensor inT Nothing)
      tgt  = the (Tensor [LstmO] ExampleDevice ExampleDType WithGrad)
               (MkTensor tgtT Nothing)

  (warmModel, _) <- repeatEpoch LstmWarmup (\m => lstmStep opt inp tgt m) model 0.0

  t0 <- clockTime Monotonic
  _ <- repeatEpoch LstmIters (\m => lstmStep opt inp tgt m) warmModel 0.0
  t1 <- clockTime Monotonic

  let ms = elapsedMs t0 t1
  putStrLn $ "lstm_cell hidden=" ++ show LstmO ++ ":\t" ++ fmt3 ms ++ " ms\t("
          ++ show LstmIters ++ " iters)"


----------------------------------------------------------------------
-- Conv2dBlock (batch=8, c_in=3, h=w=16, c_out=16, k=3)
--
-- Single Conv2D layer with bias. Conv overrides both `applyVar` and
-- `applyVarBatch`, so the batched fwd+bwd exercises the C-side
-- `tensor_conv2d_batched` kernel directly. Input is flat-encoded
-- (`inC * (h * w)` = 768 doubles per sample); the layer reshapes to
-- 4D internally for the kernel call. Smaller than MNIST's 28×28 so
-- the per-iter wall stays in the ~ms range; chosen specifically to
-- avoid the multiplicative-Nat-shape-literal elaborator trap (see
-- `feedback_idris2_tvar_nat_mult`).
----------------------------------------------------------------------

ConvInC : Nat
ConvInC = 3

ConvOutC : Nat
ConvOutC = 16

ConvH : Nat
ConvH = 16

ConvW : Nat
ConvW = 16

ConvKH : Nat
ConvKH = 3

ConvKW : Nat
ConvKW = 3

ConvInputDim : Nat
ConvInputDim = ConvInC * (ConvH * ConvW)        -- 3 * 16 * 16 = 768

ConvOH : Nat
ConvOH = ConvOutDim ConvH ConvKH 0              -- 14

ConvOW : Nat
ConvOW = ConvOutDim ConvW ConvKW 0              -- 14

ConvOutputDim : Nat
ConvOutputDim = ConvOutC * (ConvOH * ConvOW)    -- 16 * 14 * 14 = 3136

ConvBatch : Nat
ConvBatch = 8

ConvIters : Nat
ConvIters = 100

ConvWarmup : Nat
ConvWarmup = 10

benchConv2dBlock : IO ()
benchConv2dBlock = do
  ll <- conv2dLayerAny {inC=ConvInC, outC=ConvOutC, h=ConvH, w=ConvW,
                         kH=ConvKH, kW=ConvKW, padH=0, padW=0} "axisb_conv"
  let model : Network ConvInputDim [] ConvOutputDim ExampleDevice ExampleDType WithGrad
      model = OutputLayer ll
  let opt = nativeSgd 0.01

  let inT  = buildDummyVector ConvInputDim
      tgtT = buildDummyVector ConvOutputDim
      dp   = the (TensorDataPoint ConvInputDim ConvOutputDim)
               (MkTensorDataPoint inT tgtT)
      dps  = the (Vect ConvBatch (TensorDataPoint ConvInputDim ConvOutputDim))
               (Data.Vect.replicate ConvBatch dp)

  (warmModel, _) <- repeatEpoch ConvWarmup
    (\m => epochVarTensorBatch opt dps tmseLoss m) model 0.0

  t0 <- clockTime Monotonic
  _ <- repeatEpoch ConvIters
    (\m => epochVarTensorBatch opt dps tmseLoss m) warmModel 0.0
  t1 <- clockTime Monotonic

  let ms = elapsedMs t0 t1
  putStrLn $ "conv2d_block bs=" ++ show ConvBatch
          ++ " " ++ show ConvInC ++ "x" ++ show ConvH ++ "x" ++ show ConvW
          ++ "->" ++ show ConvOutC
          ++ " k=" ++ show ConvKH ++ "x" ++ show ConvKW
          ++ ":\t" ++ fmt3 ms ++ " ms\t(" ++ show ConvIters ++ " iters)"


----------------------------------------------------------------------
-- Ntm (head + controller, tiny dims, two-phase copy task)
--
-- The NTM layer doesn't expose a head-only AnyLayer constructor — the
-- head is fused inside `ntmLayerAny`. We benchmark the whole NTM
-- (controller + read/write head + memory) at *small* dims so the
-- per-iter wall stays manageable. Each iter = one `epochTwoPhaseVar`
-- call over a single-sample copy task; that runs the input phase
-- (seqLen timesteps reading) + output phase (seqLen timesteps writing)
-- + backward + step. Exercises the content + location-based addressing
-- code path that none of the other Axis B workloads touch.
----------------------------------------------------------------------

NtmW : Nat
NtmW = 3

NtmInputW : Nat
NtmInputW = S NtmW

NtmOutputW : Nat
NtmOutputW = NtmW

NtmN : Nat
NtmN = 8

NtmM : Nat
NtmM = 4

NtmH : Nat
NtmH = 20

NtmBatch : Nat
NtmBatch = 1

NtmIters : Nat
NtmIters = 30

NtmWarmup : Nat
NtmWarmup = 5

benchNtmHead : IO ()
benchNtmHead = do
  ntmAny <- ntmLayerAny {i=NtmInputW, o=NtmOutputW, n=NtmN, m=NtmM, h=NtmH} "axisb_ntm"
  let model : Network NtmInputW [] NtmOutputW ExampleDevice ExampleDType WithGrad
      model = OutputLayer ntmAny

  batch <- copyTaskBinaryBatchVect {w = NtmW} NtmBatch 2 4
  let opt = nativeRmsprop 0.0001 0.95 1.0e-8 10.0 0.0

  (warmModel, _) <- repeatEpoch NtmWarmup
    (\m => epochTwoPhaseVar opt batch tbceLoss m) model 0.0

  t0 <- clockTime Monotonic
  _ <- repeatEpoch NtmIters
    (\m => epochTwoPhaseVar opt batch tbceLoss m) warmModel 0.0
  t1 <- clockTime Monotonic

  let ms = elapsedMs t0 t1
  putStrLn $ "ntm n=" ++ show NtmN ++ " m=" ++ show NtmM
          ++ " h=" ++ show NtmH ++ " batch=" ++ show NtmBatch
          ++ ":\t" ++ fmt3 ms ++ " ms\t(" ++ show NtmIters ++ " iters)"


----------------------------------------------------------------------
-- TransformerBlock (small: batch=2, seq=16, dModel=64, heads=4, vocab=32)
--
-- Approximated as a 1-block transformer (numBlocks=1) — the standard
-- `transformerLayerAny` doesn't expose a block-only AnyLayer
-- constructor, and the embedding + final-norm + vocabProj that wrap
-- the block are insignificant at these dims. Captures the attention
-- + FFN + 2× LayerNorm + 2× residual compute pattern none of the
-- other Axis B workloads touch. Kept deliberately small to avoid the
-- multiplicative-Nat elaborator trap on `seq * vocab` (here 16*32=512).
----------------------------------------------------------------------

TxSeq : Nat
TxSeq = 16

TxDModel : Nat
TxDModel = 64

TxHeads : Nat
TxHeads = 4

TxHeadDim : Nat
TxHeadDim = 16

TxNumBlocks : Nat
TxNumBlocks = 1

TxVocab : Nat
TxVocab = 32

TxOutputDim : Nat
TxOutputDim = TxSeq * TxVocab

TxBatch : Nat
TxBatch = 2

TxIters : Nat
TxIters = 50

TxWarmup : Nat
TxWarmup = 5

benchTransformerBlock : IO ()
benchTransformerBlock = do
  txAny <- transformerLayerAny {seqLen=TxSeq, dModel=TxDModel,
                                numHeads=TxHeads, headDim=TxHeadDim,
                                numBlocks=TxNumBlocks, vocabSize=TxVocab}
                               "axisb_tx"
  let model : Network TxSeq [] TxOutputDim ExampleDevice ExampleDType WithGrad
      model = OutputLayer txAny
  let opt = nativeSgd 0.01

  let inT  = buildDummyVector TxSeq
      tgtT = buildDummyVector TxOutputDim
      dp   = the (TensorDataPoint TxSeq TxOutputDim) (MkTensorDataPoint inT tgtT)
      dps  = the (Vect TxBatch (TensorDataPoint TxSeq TxOutputDim))
               (Data.Vect.replicate TxBatch dp)

  (warmModel, _) <- repeatEpoch TxWarmup
    (\m => epochVarTensorBatch opt dps tmseLoss m) model 0.0

  t0 <- clockTime Monotonic
  _ <- repeatEpoch TxIters
    (\m => epochVarTensorBatch opt dps tmseLoss m) warmModel 0.0
  t1 <- clockTime Monotonic

  let ms = elapsedMs t0 t1
  putStrLn $ "transformer_block bs=" ++ show TxBatch
          ++ " seq=" ++ show TxSeq
          ++ " d=" ++ show TxDModel
          ++ " heads=" ++ show TxHeads
          ++ ":\t" ++ fmt3 ms ++ " ms\t(" ++ show TxIters ++ " iters)"


----------------------------------------------------------------------
-- Main
----------------------------------------------------------------------

main : IO ()
main = do
  putStrLn "--- Linear ---"
  benchLinear
  putStrLn ""
  putStrLn "--- LstmCell ---"
  benchLstmCell
  putStrLn ""
  putStrLn "--- Conv2dBlock ---"
  benchConv2dBlock
  putStrLn ""
  putStrLn "--- Ntm ---"
  benchNtmHead
  putStrLn ""
  putStrLn "--- TransformerBlock ---"
  benchTransformerBlock
  putStrLn ""
  putStrLn "=== Done ==="
