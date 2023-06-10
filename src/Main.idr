module Main

import Data.Vect

import Backprop
import Layer
import Math
import Network
import Tensor
import Variable


p : String -> Double -> Scalar Variable
p name = STensor . (param name)

-- f(x, y) = argmax(x - y - 10, -4x + y + 5)
dataPoints : Vect 5 (DataPoint 2 2 Double)
dataPoints =
    [ MkDataPoint (VTensor [1.5, -2.7]) (VTensor [0, 1]),
      MkDataPoint (VTensor [-3.2, 4.1]) (VTensor [0, 1]),
      MkDataPoint (VTensor [5.7, 0]) (VTensor [1, 0]),
      MkDataPoint (VTensor [-1.3, 8.8]) (VTensor [0, 1]),
      MkDataPoint (VTensor [2.9, -1.4]) (VTensor [1, 0])
    ]

initWeights : Matrix 2 2 Variable
initWeights =
  VTensor
      [ VTensor [p "weight1a" 0.123, p "weight1b" 0.234],
        VTensor [p "weight2a" 0.345, p "weight2b" 0.456]
      ]

initBias : Vector 2 Variable
initBias =
  VTensor
    [ p "bias1" 0.314,
      p "bias2" (-0.314)
    ]

prepareDataPoint : DataPoint i o Double -> DataPoint i o Variable
prepareDataPoint = map fromDouble

main : IO ()
main = do
  let epochs = 1000
  let lr = 0.03
  let lossFn = crossEntropy
  let weights = initWeights
  let bias = initBias
  let model = (LinearLayer weights bias) ~> OutputLayer softmaxLayer
  putStr "Model: "
  printLn model
  let prepared = map prepareDataPoint dataPoints
  let predictions = map (forward model . x) prepared

  let loss = calculateLoss lossFn model prepared

  putStr "Pre loss: "
  printLn $ value loss
  putStr "Predictions: "
  printLn $ map (map value) predictions

  let trained = train lr model prepared lossFn epochs
  let predictions' = map (forward trained . x) prepared
  let loss' = calculateLoss lossFn trained prepared

  putStr "Post loss: "
  printLn $ value loss'
  putStr "Predictions: "
  printLn $ map (map value) predictions'
