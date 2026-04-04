module Example.Supervised

import Data.Vect
import System.Random

import Backprop
import DataPoint
import Endofunctor
import Floating
import Layer
import Math
import Optimizer
import Tensor
import Variable


-- f(x, y) = argmax(x - y - 10, -4x + y + 5, 2x + y - 11)
dataPoints : Vect 5 (DataPoint 2 3 Double)
dataPoints =
    [ MkDataPoint (VTensor [1.5, -2.7]) (VTensor [0, 1, 0]),
      MkDataPoint (VTensor [-3.2, 4.1]) (VTensor [0, 1, 0]),
      MkDataPoint (VTensor [5.7, 0]) (VTensor [0, 0, 1]),
      MkDataPoint (VTensor [-1.3, 8.8]) (VTensor [0, 1, 0]),
      MkDataPoint (VTensor [2.9, -1.4]) (VTensor [1, 0, 0])
    ]

main : IO ()
main = do
  srand 123456

  let epochs = 1000
  let lr = 0.03
  let lossFn = crossEntropy

  ll <- linearLayer
  let model = autoName $ ll ~> OutputLayer softmaxLayer
  putStr "Model: "
  printLn model
  let prepared = map (map fromDouble) dataPoints
  let predictions = evaluate model prepared
  let loss = calculateLoss lossFn model prepared

  putStr "Pre loss: "
  printLn $ value loss
  putStr "Predictions: "
  printLn $ map (map value) predictions

  let opt = nativeSgd lr
  let trained = trainNative opt model prepared lossFn epochs
  let dblModel = toDoubleNetwork (emap refreshValue trained)
  let predictions' = evaluate dblModel (map (map fromDouble) dataPoints)
  let loss' = calculateLoss lossFn dblModel (map (map fromDouble) dataPoints)

  putStr "Post loss: "
  printLn loss'
  putStr "Predictions: "
  printLn predictions'
