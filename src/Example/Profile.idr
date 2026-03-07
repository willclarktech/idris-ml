module Example.Profile

import Data.List
import Data.Vect
import System
import System.Clock
import System.Random

import Backprop
import DataPoint
import Endofunctor
import Floating
import Generate
import Layer
import Math
import Optimizer
import Tensor
import Util
import Variable


----------------------------------------------------------------------
-- Timing
----------------------------------------------------------------------

elapsedMs : Clock Monotonic -> Clock Monotonic -> Double
elapsedMs t0 t1 =
  let s = cast {to=Double} (seconds t1 - seconds t0)
      ns = cast {to=Double} (nanoseconds t1 - nanoseconds t0)
  in s * 1000.0 + ns / 1000000.0

padL : Nat -> String -> String
padL n s = pack (replicate (minus n (length s)) ' ') ++ s

-- Truncate a Double to 1 decimal place string
showMs : Double -> String
showMs d =
  let whole = cast {to=Integer} d
      frac = cast {to=Integer} (abs ((d - cast whole) * 10))
  in show whole ++ "." ++ show frac

fmtMs : Double -> String
fmtMs d = padL 10 (showMs d)


----------------------------------------------------------------------
-- NTM Copy Task Setup (matches NtmCopy.idr)
----------------------------------------------------------------------

||| Binary vector width (data channels)
W : Nat
W = 8

||| Input width = data + delimiter channel
InputW : Nat
InputW = S W

||| Output width = data channels only
OutputW : Nat
OutputW = W

||| Number of memory slots
N : Nat
N = 128

||| Memory width
M : Nat
M = 20

||| Controller hidden size
H : Nat
H = 100

||| Batch size (data points per epoch)
BatchSize : Nat
BatchSize = 16


----------------------------------------------------------------------
-- Profiled Epoch (inlined epochTwoPhase with timing)
----------------------------------------------------------------------

profileEpoch :
  DenseOptimizer ->
  Vect BatchSize (TwoPhaseDataPoint InputW OutputW Variable) ->
  LossFunction Variable ->
  Network InputW [] OutputW Variable ->
  DenseOptimizerState ->
  Nat ->
  IO (Network InputW [] OutputW Variable, DenseOptimizerState)
profileEpoch opt dataPoints lossFn model st epochNum = do
  -- Phase 1: Forward pass + loss
  t0 <- clockTime Monotonic
  let loss = calculateLossTwoPhaseVar lossFn model dataPoints
  let lossVal = loss.value
  t1 <- clockTime Monotonic

  -- Read tape size between forward and backward
  let ts = tapeSize loss.tapeIdx

  -- Phase 2: Backward pass (dense: accumulates into C array)
  let denseBuf = collectGradsDense 1.0 loss st.buf
  t2 <- clockTime Monotonic

  -- Phase 3: Optimizer step (in-place on dense array)
  let st' = opt.step denseBuf st
  t3 <- clockTime Monotonic

  -- Phase 4: Apply deltas + sync buffers
  let model' = syncNetworkBuffers (emap (applyDeltasDense denseBuf) model)
  t4 <- clockTime Monotonic

  let line = padL 5 (show epochNum)
          ++ fmtMs (elapsedMs t0 t1)
          ++ fmtMs (elapsedMs t1 t2)
          ++ fmtMs (elapsedMs t2 t3)
          ++ fmtMs (elapsedMs t3 t4)
          ++ padL 10 (show ts)
          ++ padL 8 (show st.n)
          ++ "    " ++ show lossVal
  putStrLn line

  pure (model', st')


----------------------------------------------------------------------
-- Profile Loop
----------------------------------------------------------------------

profileLoop :
  DenseOptimizer ->
  Vect BatchSize (TwoPhaseDataPoint InputW OutputW Variable) ->
  LossFunction Variable ->
  Network InputW [] OutputW Variable ->
  DenseOptimizerState ->
  Nat -> Nat ->
  IO (Network InputW [] OutputW Variable, DenseOptimizerState)
profileLoop opt dataPoints lossFn model st cur count =
  if cur >= count
    then pure (model, st)
    else do
      (model', st') <- profileEpoch opt dataPoints lossFn model st (cur + 1)
      profileLoop opt dataPoints lossFn model' st' (cur + 1) count


----------------------------------------------------------------------
-- Main
----------------------------------------------------------------------

main : IO ()
main = do
  srand 123456

  putStrLn "=== NTM Copy Profile ==="
  putStrLn $ "Architecture: N=" ++ show N ++ " M=" ++ show M ++ " H=" ++ show H
  putStrLn $ "Batch=" ++ show BatchSize ++ " seqLen=1-20"
  putStrLn ""

  -- Build NTM model (same as NtmCopy.idr: no output activation, BCE loss)
  ntm <- ntmLayer {inputSize = InputW, outputSize = OutputW, n = N, m = M, h = H}
  let model = autoName $ OutputLayer ntm

  let numPids = getNumPids 0
  let opt = rmspropValueClipDense 0.0001 0.95 1.0e-8 10.0
  let st0 = initDenseState numPids

  -- Generate a fixed batch for consistent profiling
  batch <- copyTaskBinaryBatchVect {w = W} BatchSize 1 20
  let dataPoints = map (map fromDouble) batch

  -- Warmup: 5 epochs (untimed)
  putStrLn "Warmup (5 epochs)..."
  (warmModel, warmSt) <- go 0 model st0
  putStrLn ""

  let header = padL 5 "Epoch"
            ++ padL 10 "Fwd(ms)"
            ++ padL 10 "Bwd(ms)"
            ++ padL 10 "Opt(ms)"
            ++ padL 10 "Sync(ms)"
            ++ padL 10 "TapeSize"
            ++ padL 8 "Params"
            ++ "    Loss"
  putStrLn header

  -- Profile: 10 epochs with per-phase timing
  (finalModel, _) <- profileLoop opt dataPoints binaryCrossEntropyWithLogits warmModel warmSt 0 10

  putStrLn "\nDone."

  where
    -- Warmup loop using epochTwoPhaseDense
    go : Nat -> Network InputW [] OutputW Variable ->
         DenseOptimizerState ->
         IO (Network InputW [] OutputW Variable, DenseOptimizerState)
    go 5 m s = pure (m, s)
    go k m s = do
      batch <- copyTaskBinaryBatchVect {w = W} BatchSize 1 20
      let dps = map (map fromDouble) batch
          opt = rmspropValueClipDense 0.0001 0.95 1.0e-8 10.0
          (m', s', loss) = epochTwoPhaseDense opt dps binaryCrossEntropyWithLogits m s
      putStrLn $ "  warmup " ++ show (k + 1) ++ ": loss=" ++ show loss
      go (k + 1) m' s'
