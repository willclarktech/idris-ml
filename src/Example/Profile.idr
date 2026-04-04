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
-- Profiled Epoch (native optimizer — single step timing)
----------------------------------------------------------------------

profileEpoch :
  NativeOptimizer ->
  Vect BatchSize (TwoPhaseDataPoint InputW OutputW Variable) ->
  Network InputW [] OutputW Variable ->
  Nat ->
  IO (Network InputW [] OutputW Variable)
profileEpoch opt dataPoints model epochNum = do
  t0 <- clockTime Monotonic
  let (model', lossVal) = epochTwoPhaseBceNative opt dataPoints model
  t1 <- clockTime Monotonic

  let line = padL 5 (show epochNum)
          ++ fmtMs (elapsedMs t0 t1)
          ++ "    " ++ show lossVal
  putStrLn line

  pure model'


----------------------------------------------------------------------
-- Profile Loop
----------------------------------------------------------------------

profileLoop :
  NativeOptimizer ->
  Vect BatchSize (TwoPhaseDataPoint InputW OutputW Variable) ->
  Network InputW [] OutputW Variable ->
  Nat -> Nat ->
  IO (Network InputW [] OutputW Variable)
profileLoop opt dataPoints model cur count =
  if cur >= count
    then pure model
    else do
      model' <- profileEpoch opt dataPoints model (cur + 1)
      profileLoop opt dataPoints model' (cur + 1) count


----------------------------------------------------------------------
-- Main
----------------------------------------------------------------------

main : IO ()
main = do
  srand 123456

  putStrLn "=== NTM Copy Forward-Pass Profile ==="
  putStrLn $ "Architecture: N=" ++ show N ++ " M=" ++ show M ++ " H=" ++ show H
  putStrLn $ "Batch=" ++ show BatchSize ++ " seqLen=1-20"
  putStrLn ""

  -- Build NTM model (same as NtmCopy.idr: no output activation, BCE loss)
  ntm <- ntmLayer {inputSize = InputW, outputSize = OutputW, n = N, m = M, h = H}
  let model = autoName $ OutputLayer ntm

  let opt = nativeRmsprop 0.0001 0.95 1.0e-8 10.0 0.9

  -- Generate a fixed batch for consistent profiling
  tGen0 <- clockTime Monotonic
  batch <- copyTaskBinaryBatchVect {w = W} BatchSize 1 20
  let dataPoints = map (map fromDouble) batch
  tGen1 <- clockTime Monotonic
  putStrLn $ "Data generation: " ++ showMs (elapsedMs tGen0 tGen1) ++ " ms"
  putStrLn ""

  -- Warmup: 5 epochs (untimed)
  putStrLn "Warmup (5 epochs)..."
  warmModel <- go 0 model
  putStrLn ""

  let header = padL 5 "Epoch"
            ++ padL 10 "Total(ms)"
            ++ "    Loss"
  putStrLn header

  -- Profile: 10 epochs with timing
  finalModel <- profileLoop opt dataPoints warmModel 0 10

  putStrLn "\nDone."

  where
    -- Warmup loop using epochTwoPhaseBceNative
    go : Nat -> Network InputW [] OutputW Variable ->
         IO (Network InputW [] OutputW Variable)
    go 5 m = pure m
    go k m = do
      batch <- copyTaskBinaryBatchVect {w = W} BatchSize 1 20
      let dps = map (map fromDouble) batch
          opt = nativeRmsprop 0.0001 0.95 1.0e-8 10.0 0.9
          (m', loss) = epochTwoPhaseBceNative opt dps m
      putStrLn $ "  warmup " ++ show (k + 1) ++ ": loss=" ++ show loss
      go (k + 1) m'
