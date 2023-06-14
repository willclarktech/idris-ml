module Example.Rnn

import Data.List
import Data.Stream
import Data.Vect
import System.Random

import Backprop
import DataPoint
import Floating
import Layer
import Math
import Network
import Tensor
import Util
import Variable


generateData : Nat -> (List Double, List Double)
generateData n =
  let infinitePattern = cycle [0, 1, 0]
  in (take n infinitePattern, take n (drop 1 infinitePattern))

generateDataSet : {n : Nat} -> Vect n (List Double, List Double)
generateDataSet = map (generateData. (+3) . finToNat) Data.Vect.Fin.range

rawData : (n : Nat) -> Vect n (RecurrentDataPoint 1 1 Double)
rawData n = map (\(is, os) => MkRecurrentDataPoint (prep is) (prep os)) $ generateDataSet {n}
  where
    prep : (ns : List Double) -> List (Vector 1 Double)
    prep ns = map (flatten . STensor) ns

decodeYs : Vect n (RecurrentDataPoint 1 1 Variable) -> Vect n (List (Vector 1 Double))
decodeYs = map (map (map value) . ys)

decodeOutput : Vect n (List (Vector o Variable)) -> Vect n (List (Vector o Double))
decodeOutput = map (map (map (cast . (0<))))

main : IO ()
main = do
  srand 123456

  let epochs = 1000
  let lr = 0.03
  let lossFn = binaryCrossEntropyWithLogits

  rnn <- nameParams "rnn" <$> rnnLayer
  let model = OutputLayer rnn
  putStr "Model: "
  printLn model
  let dataPoints = map (map fromDouble) (rawData 8)
  putStr "Targets: "
  printLn $ decodeYs dataPoints
  let predictions = decodeOutput $ evaluateRecurrent model dataPoints
  let loss = calculateLossRecurrent lossFn model dataPoints

  putStr "Pre loss: "
  printLn $ value loss
  putStr "Predictions: "
  printLn $ predictions

  let trained = trainRecurrent lr model dataPoints lossFn epochs
  let predictions' = decodeOutput $ evaluateRecurrent trained dataPoints
  let loss' = calculateLossRecurrent lossFn trained dataPoints

  putStr "Post loss: "
  printLn $ value loss'
  putStr "Predictions: "
  printLn $ predictions'
