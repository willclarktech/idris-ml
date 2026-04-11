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

  putStrLn "=== Supervised Classification ==="
  putStrLn $ "Config: lr=" ++ show lr ++ " epochs=" ++ show epochs ++ " seed=123456"

  ll <- linearLayer
  let model = autoName $ ll ~> OutputLayer softmaxLayer
  putStrLn $ "Architecture: " ++ show model
  putStrLn ""

  let prepared = map (map fromDouble) dataPoints
  let opt = nativeSgd lr

  putStrLn "Training..."
  t0 <- clockTime Monotonic
  let go : Nat -> Network 2 [3] 3 Variable -> IO (Network 2 [3] 3 Variable, Double)
      go ep m =
        if ep >= epochs then do
          let loss = calculateLoss lossFn (toDoubleNetwork (emap refreshValue m))
                                          (map (map fromDouble) dataPoints)
          pure (m, loss)
        else do
          let (m', loss) = epochNative opt prepared lossFn m
          when (modNatNZ ep 100 ItIsSucc == 0) $ do
            now <- clockTime Monotonic
            putStrLn $ "  " ++ formatElapsed t0 now ++ " " ++ show ep
                     ++ "\tloss=" ++ show loss
          go (S ep) m'

  (trained, finalLoss) <- go 0 model
  t1 <- clockTime Monotonic

  let dblModel = toDoubleNetwork (emap refreshValue trained)
  let predictions = evaluate dblModel (map (map fromDouble) dataPoints)

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
  putStrLn $ formatTimingSummary tStart t1 epochs
