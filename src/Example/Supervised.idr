module Example.Supervised

import Data.Vect
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
import Train
import Util
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
  tStart <- clockTime Monotonic
  srand 123456

  let epochs = 1000
  let lr = 0.03
  let lossFn = crossEntropy
  let opt = nativeSgd lr
  let prepared = map (map fromDouble) dataPoints

  putStrLn "=== Supervised Classification ==="
  putStrLn $ "Config: lr=" ++ show lr ++ " epochs=" ++ show epochs ++ " seed=123456"

  ll <- linearLayer
  let model = autoName $ ll ~> OutputLayer softmaxLayer
  putStrLn $ "Architecture: " ++ show model
  putStrLn ""

  (trained, epochsDone, _) <- runTraining
    (\m, d => epochNative opt d lossFn m)
    (pure prepared)
    (simpleConfig epochs)
    model
  t1 <- clockTime Monotonic

  let dblModel = toDoubleNetwork (emap refreshValue trained)
  let predictions = evaluate dblModel (map (map fromDouble) dataPoints)
  let finalLoss = calculateLoss lossFn dblModel (map (map fromDouble) dataPoints)

  putStrLn ""
  putStrLn "Eval:"
  putStrLn $ "  Loss: " ++ show finalLoss
  let showSample : DataPoint 2 3 Double -> Vector 3 Double -> IO ()
      showSample dp pred =
        let argmax : Vector 3 Double -> Nat
            argmax (VTensor [STensor a, STensor b, STensor c]) =
              if a >= b && a >= c then 0
              else if b >= c then 1
              else 2
            argmax _ = 0
            showVec : {k : Nat} -> Vector k Double -> String
            showVec (VTensor xs) = "[" ++ go xs ++ "]"
              where
                go : Vect j (Scalar Double) -> String
                go [] = ""
                go [STensor v] = show v
                go (STensor v :: rest) = show v ++ ", " ++ go rest
            target = argmax (y dp)
            predicted = argmax pred
            mark = if target == predicted then " ok" else " WRONG"
        in putStrLn $ "  " ++ showVec (x dp) ++ " -> class " ++ show predicted ++ mark
  traverse_ (\(dp, pred) => showSample dp pred) (zip dataPoints predictions)
  putStrLn ""
  putStrLn $ formatTimingSummary tStart t1 epochsDone
  putStrLn $ "RESULT\tepochs=" ++ show epochsDone ++ "\tloss=" ++ show finalLoss
           ++ "\ttime=" ++ show (seconds t1 - seconds tStart) ++ "s"
