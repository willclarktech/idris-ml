module Example.Rnn

import Data.List
import Data.Stream
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


generateData : Nat -> (List Double, List Double)
generateData n =
  let infinitePattern = cycle [0, 1, 0]
  in (take n infinitePattern, take n (drop 1 infinitePattern))

generateDataSet : {n : Nat} -> Vect n (List Double, List Double)
generateDataSet = map (generateData . (+3) . finToNat) Data.Vect.Fin.range

rawData : (n : Nat) -> Vect n (RecurrentDataPoint 1 1 Double)
rawData n = map (\(is, os) => MkRecurrentDataPoint (prep is) (prep os)) $ generateDataSet {n}
  where
    prep : (ns : List Double) -> List (Vector 1 Double)
    prep ns = map (flatten . STensor) ns

main : IO ()
main = do
  tStart <- clockTime Monotonic
  srand 123456

  let epochs = 1000
  let lr = 0.03
  let lossFn = binaryCrossEntropyWithLogits
  let opt = nativeSgd lr
  let dataPoints = map (map fromDouble) (rawData 8)

  putStrLn "=== RNN Pattern Prediction ==="
  putStrLn $ "Config: lr=" ++ show lr ++ " epochs=" ++ show epochs ++ " seed=123456"

  rnn <- rnnLayer
  let model = autoName $ OutputLayer rnn
  putStrLn $ "Architecture: " ++ show model
  putStrLn ""

  (trained, epochsDone, _) <- runTraining
    (\m, d => epochRecurrentNative opt d lossFn m)
    (pure dataPoints)
    (simpleConfig epochs)
    model
  t1 <- clockTime Monotonic

  let dblModel = toDoubleNetwork (emap refreshValue trained)
  let dblPreds = evaluateRecurrent dblModel (rawData 8)
  let predictions : Vect 8 (List (Vector 1 Double))
      predictions = map (map (map (\x => cast (0 < x)))) dblPreds
  let finalLoss = calculateLossRecurrent lossFn dblModel (rawData 8)

  putStrLn ""
  putStrLn "Eval:"
  putStrLn $ "  Loss: " ++ show finalLoss
  let showSeq : List (Vector 1 Double) -> String
      showSeq xs = concatMap (\(VTensor [STensor v]) => if v >= 0.5 then "1" else "0") xs
  let tgts = map ys (rawData 8)
  putStrLn "  Seq  Target     Predicted"
  traverse_ (\(i, (t, p)) =>
    let ts = showSeq (toList t)
        ps = showSeq (toList p)
        mark = if ts == ps then " ok" else ""
    in putStrLn $ "  " ++ show (finToNat i + 1) ++ ".   " ++ ts ++ "  ->  " ++ ps ++ mark)
    (zip Fin.range (zip tgts predictions))
  putStrLn ""
  putStrLn $ formatTimingSummary tStart t1 epochsDone
  putStrLn $ "RESULT\tepochs=" ++ show epochsDone ++ "\tloss=" ++ show finalLoss
           ++ "\ttime=" ++ show (seconds t1 - seconds tStart) ++ "s"
