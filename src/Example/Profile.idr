module Example.Profile

import Data.List
import Data.SortedMap
import Data.Vect
import System
import System.Clock
import System.Random

import Backprop
import DataPoint
import Endofunctor
import Floating
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
-- NTM Copy Task Setup (same config as Bench.idr)
----------------------------------------------------------------------

W : Nat
W = 3

N : Nat
N = 10

H : Nat
H = 20

E : Nat
E = 5

ntmSequences : Vect E (List (Fin W))
ntmSequences =
  [ [1, 2, 1, 2]
  , [1, 1, 2, 2, 1]
  , [2, 1, 1, 2, 2, 1]
  , [2, 1, 2, 1, 2, 1, 2]
  , [1, 2, 1, 1, 2, 2, 1, 2]
  ]

prepNtm : List (Fin W) -> RecurrentDataPoint W W Double
prepNtm sequence =
  let len = length sequence
      blank : Fin W
      blank = 0
      pad = Data.List.replicate len blank
      inp = sequence ++ pad
      outp = pad ++ sequence
      xs = map (oneHotEncode {n=W}) inp
      ys = map (oneHotEncode {n=W}) outp
      toDouble : Vector W Nat -> Vector W Double
      toDouble = map (fromInteger . natToInteger)
  in MkRecurrentDataPoint (map toDouble xs) (map toDouble ys)

ntmRawData : Vect E (RecurrentDataPoint W W Double)
ntmRawData = map prepNtm ntmSequences


----------------------------------------------------------------------
-- Profiled Epoch (inlined epochRecurrent with timing)
----------------------------------------------------------------------

profileEpoch :
  {i, o, n : Nat} ->
  {hs : List Nat} ->
  Optimizer ->
  Vect n (RecurrentDataPoint i o Variable) ->
  LossFunction Variable ->
  Network i hs o Variable ->
  OptimizerState ->
  Nat ->
  IO (Network i hs o Variable, OptimizerState)
profileEpoch opt dataPoints lossFn model st epochNum = do
  -- Phase 1: Forward pass + loss
  t0 <- clockTime Monotonic
  let loss = calculateLossRecurrentVar lossFn model dataPoints
  let lossVal = loss.value
  t1 <- clockTime Monotonic

  -- Read tape size between forward and backward
  let ts = tapeSize loss.tapeIdx

  -- Phase 2: Backward pass
  let grads = collectGrads 1.0 loss
  let nGrads = cast {to=Int} (length (Data.SortedMap.toList grads))
  t2 <- clockTime Monotonic

  -- Phase 3: Optimizer step
  let (deltas, st') = opt.step grads st
  t3 <- clockTime Monotonic

  -- Phase 4: Apply deltas + sync buffers
  let model' = syncNetworkBuffers (emap (applyDeltas deltas) model)
  t4 <- clockTime Monotonic

  let line = padL 5 (show epochNum)
          ++ fmtMs (elapsedMs t0 t1)
          ++ fmtMs (elapsedMs t1 t2)
          ++ fmtMs (elapsedMs t2 t3)
          ++ fmtMs (elapsedMs t3 t4)
          ++ padL 10 (show ts)
          ++ padL 8 (show nGrads)
          ++ "    " ++ show lossVal
  putStrLn line

  pure (model', st')


----------------------------------------------------------------------
-- Profile Loop
----------------------------------------------------------------------

profileLoop :
  {i, o, n : Nat} ->
  {hs : List Nat} ->
  Optimizer ->
  Vect n (RecurrentDataPoint i o Variable) ->
  LossFunction Variable ->
  Network i hs o Variable ->
  OptimizerState ->
  Nat -> Nat ->
  IO (Network i hs o Variable, OptimizerState)
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

  -- Build NTM model (same as Bench.idr)
  controllerHidden <- linearLayer {i = NtmInputWidth W, o = H}
  controllerOut <- linearLayer {i = H, o = NtmOutputWidth N W}
  let controller = controllerHidden ~> sigmoidLayer ~> OutputLayer controllerOut
  ntm <- ntmLayer {n = N, w = W} controller
  let model = nameNetworkParams "ntm" (ntm ~> OutputLayer logSoftmaxLayer)

  let dataPoints = map (map fromDouble) ntmRawData
  let opt = adamGlobalClip 0.001 0.9 0.999 (pow 10 (-8)) 5.0

  -- Warmup: 5 epochs (untimed)
  let (warmModel, warmSt) = trainRecurrentFrom opt model dataPoints nllLoss 5 initState

  putStrLn "=== NTM Profile (10 epochs) ==="
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
  (finalModel, _) <- profileLoop opt dataPoints nllLoss warmModel warmSt 0 10

  -- Final loss
  let finalLoss = calculateLossRecurrent nllLoss finalModel dataPoints
  putStrLn ("\nFinal loss: " ++ show finalLoss.value)
