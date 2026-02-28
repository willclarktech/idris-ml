module Example.Bench

import Data.List
import Data.Vect
import System
import System.Clock
import System.Random

import Backprop
import DataPoint
import Floating
import Layer
import Math
import Optimizer
import Tensor
import Variable


----------------------------------------------------------------------
-- NTM Copy Task (same configuration as Ntm.idr)
----------------------------------------------------------------------

W : Nat
W = 3

N : Nat
N = 10

H : Nat
H = 20

E : Nat
E = 5

sequences : Vect E (List (Fin W))
sequences =
  [ [1, 2, 1, 2]
  , [1, 1, 2, 2, 1]
  , [2, 1, 1, 2, 2, 1]
  , [2, 1, 2, 1, 2, 1, 2]
  , [1, 2, 1, 1, 2, 2, 1, 2]
  ]

prep : List (Fin W) -> RecurrentDataPoint W W Double
prep sequence =
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

rawData : Vect E (RecurrentDataPoint W W Double)
rawData = map prep sequences


----------------------------------------------------------------------
-- Timing
----------------------------------------------------------------------

elapsedMs : Clock Monotonic -> Clock Monotonic -> Double
elapsedMs t0 t1 =
  let s = cast {to=Double} (seconds t1 - seconds t0)
      ns = cast {to=Double} (nanoseconds t1 - nanoseconds t0)
  in s * 1000.0 + ns / 1000000.0


----------------------------------------------------------------------
-- Main
----------------------------------------------------------------------

main : IO ()
main = do
  srand 123456

  -- Build NTM model
  controllerHidden <- linearLayer {i = NtmInputWidth W, o = H}
  controllerOut <- linearLayer {i = H, o = NtmOutputWidth N W}
  let controller = controllerHidden ~> sigmoidLayer ~> OutputLayer controllerOut
  ntm <- ntmLayer {n = N, w = W} controller
  let model = nameNetworkParams "ntm" $ ntm ~> OutputLayer logSoftmaxLayer

  let dataPoints = map (map fromDouble) rawData
  let opt = adamGlobalClip 0.001 0.9 0.999 (pow 10 (-8)) 5.0

  -- Warmup: 10 epochs
  let (warmModel, warmSt) = trainRecurrentFrom opt model dataPoints nllLoss 10 initState

  -- Benchmark: 100 epochs
  t0 <- clockTime Monotonic
  let (benchModel, _) = trainRecurrentFrom opt warmModel dataPoints nllLoss 100 warmSt
  let loss = calculateLossRecurrent nllLoss benchModel dataPoints
  t1 <- clockTime Monotonic

  putStrLn $ "100 NTM epochs: " ++ show (elapsedMs t0 t1) ++ " ms"
  putStrLn $ "Final loss:     " ++ show (value loss)
