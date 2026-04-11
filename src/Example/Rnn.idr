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

decodeYs : Vect n (RecurrentDataPoint 1 1 Variable) -> Vect n (List (Vector 1 Double))
decodeYs = map (map (map value) . ys)

main : IO ()
main = do
  tStart <- clockTime Monotonic
  srand 123456

  let epochs = 1000
  let lr = 0.03
  let lossFn = binaryCrossEntropyWithLogits

  putStrLn "=== RNN Pattern Prediction ==="
  putStrLn $ "Config: lr=" ++ show lr ++ " epochs=" ++ show epochs ++ " seed=123456"

  rnn <- rnnLayer
  let model = autoName $ OutputLayer rnn
  putStrLn $ "Architecture: " ++ show model
  putStrLn ""

  let dataPoints = map (map fromDouble) (rawData 8)
  let opt = nativeSgd lr

  putStrLn "Training..."
  t0 <- clockTime Monotonic
  let go : Nat -> Network 1 [] 1 Variable -> IO (Network 1 [] 1 Variable, Double)
      go ep m =
        if ep >= epochs then do
          let dblM = toDoubleNetwork (emap refreshValue m)
          let loss = calculateLossRecurrent lossFn dblM (rawData 8)
          pure (m, loss)
        else do
          let (m', loss) = epochRecurrentNative opt dataPoints lossFn m
          when (modNatNZ ep 100 ItIsSucc == 0) $ do
            now <- clockTime Monotonic
            putStrLn $ "  " ++ formatElapsed t0 now ++ " " ++ show ep
                     ++ "\tloss=" ++ show loss
          go (S ep) m'

  (trained, finalLoss) <- go 0 model
  t1 <- clockTime Monotonic

  let dblModel = toDoubleNetwork (emap refreshValue trained)
  let dblPreds = evaluateRecurrent dblModel (rawData 8)
  let predictions : Vect 8 (List (Vector 1 Double))
      predictions = map (map (map (\x => cast (0 < x)))) dblPreds

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
  putStrLn $ formatTimingSummary tStart t1 epochs
  putStrLn $ "RESULT\tepochs=" ++ show epochs ++ "\tloss=" ++ show finalLoss
           ++ "\ttime=" ++ show (seconds t1 - seconds tStart) ++ "s"
