module Example.Rnn

import Data.List
import Data.Stream
import Data.Vect
import System.Random

import Backprop
import DataPoint
import Layer
import Math
import Network
import Tensor
import Variable


generateData : Nat -> (List Double, List Double)
generateData n =
  let infinitePattern = cycle [1, 0]
  in (take n infinitePattern, take n (drop 1 infinitePattern))

generateDataSet : {n : Nat} -> Vect n (List Double, List Double)
generateDataSet = map (generateData. (+3) . finToNat) Data.Vect.Fin.range

dataPoints : (n : Nat) -> Vect n (RecurrentDataPoint 1 1 Double)
dataPoints n = map (\(is, os) => MkRecurrentDataPoint (prep is) (prep os)) $ generateDataSet {n}
  where
    prep : (ns : List Double) -> List (Vector 1 Double)
    prep ns = map (flatten . STensor) ns

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
  let prepared = map (map fromDouble) (dataPoints 8)
  putStr "Targets: "
  printLn $ map ((map (map value)) . ys) prepared
  let predictions = evaluateRecurrent model prepared
  let loss = calculateLossRecurrent lossFn model prepared

  putStr "Pre loss: "
  printLn $ value loss
  putStr "Predictions: "
  printLn $ map (map (map value)) predictions

  let trained = trainRecurrent lr model prepared lossFn epochs
  let predictions' = evaluateRecurrent trained prepared
  let loss' = calculateLossRecurrent lossFn trained prepared

  putStr "Post loss: "
  printLn $ value loss'
  putStr "Predictions: "
  printLn $ map (map (map value)) predictions'
