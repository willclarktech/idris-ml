module Example.NtmDebug

import Data.List
import Data.Vect
import System.Random

import Backprop
import DataPoint
import Floating
import Layer
import Math
import Tensor
import Variable


----------------------------------------------------------------------
-- Configuration (W=3 version — needs more memory/time than W=2)
----------------------------------------------------------------------

W : Nat
W = 3

N : Nat
N = 2

H : Nat
H = 4

E : Nat
E = 1

sequences : Vect E (List (Fin W))
sequences = [ [2] ]

prep : List (Fin W) -> RecurrentDataPoint W W Double
prep sequence =
  let
    len = length sequence
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

main : IO ()
main = do
  srand 123456

  controllerHidden <- linearLayer {i = NtmInputWidth W, o = H}
  controllerOut <- linearLayer {i = H, o = NtmOutputWidth N W}
  let controller = controllerHidden ~> sigmoidLayer ~> OutputLayer controllerOut
  ntm <- ntmLayer {n = N, w = W} controller
  let model = nameNetworkParams "ntm" $ ntm ~> OutputLayer softmaxLayer

  let dataPoints = map (map fromDouble) rawData
  let loss = calculateLossRecurrent crossEntropy model dataPoints
  putStrLn $ "Initial loss: " ++ show (value loss)

  let t1 = trainRecurrent 0.03 model dataPoints crossEntropy 50
  let l1 = calculateLossRecurrent crossEntropy t1 dataPoints
  putStrLn $ "After 50 epochs: " ++ show (value l1)

  let t2 = trainRecurrent 0.03 t1 dataPoints crossEntropy 50
  let l2 = calculateLossRecurrent crossEntropy t2 dataPoints
  putStrLn $ "After 100 epochs: " ++ show (value l2)

  let predictions = evaluateRecurrent t2 dataPoints
  putStrLn $ "Predictions: " ++ show (map (map (map value)) predictions)

  putStrLn "Done!"
