module Example.Lstm

import Data.List
import Data.Stream
import Data.String
import Data.Vect
import System
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


----------------------------------------------------------------------
-- Training Loop
----------------------------------------------------------------------

trainLoop :
  NativeOptimizer ->
  Network 1 [1] 1 Variable ->
  Vect 8 (RecurrentDataPoint 1 1 Variable) ->
  LossFunction Variable ->
  (totalEpochs : Nat) -> (patience : Nat) ->
  Clock Monotonic ->
  IO (Network 1 [1] 1 Variable, Nat, Double)
trainLoop opt model dataPoints lossFn totalEpochs patience t0 =
  go 0 model (1.0/0.0) 0
  where
    go : Nat -> Network 1 [1] 1 Variable ->
         Double -> Nat ->
         IO (Network 1 [1] 1 Variable, Nat, Double)
    go ep m bestLoss staleCount =
      if ep >= totalEpochs then pure (m, ep, bestLoss)
      else do
        let (m', loss) = epochRecurrentNative opt dataPoints lossFn m
        when (modNatNZ ep 100 ItIsSucc == 0) $ do
          now <- clockTime Monotonic
          putStrLn $ "  " ++ formatElapsed t0 now ++ " " ++ show ep ++ "\tloss=" ++ show loss
        if loss /= loss
          then do
            now <- clockTime Monotonic
            putStrLn $ "  " ++ formatElapsed t0 now ++ " Diverged (NaN) at epoch " ++ show ep
            pure (m', ep, loss)
          else do
            let improved = loss < bestLoss - 0.001
                bestLoss' = if improved then loss else bestLoss
                sc : Nat
                sc = if improved then 0 else staleCount + 1
            if patience > 0 && sc >= patience
              then do
                now <- clockTime Monotonic
                putStrLn $ "  " ++ formatElapsed t0 now ++ " Early stop at epoch " ++ show (ep + 1)
                         ++ " (patience=" ++ show patience ++ ")"
                pure (m', ep + 1, loss)
              else go (ep + 1) m' bestLoss' sc


----------------------------------------------------------------------
-- CLI Argument Parsing
----------------------------------------------------------------------

record Config where
  constructor MkConfig
  lr : Double
  epochs : Nat
  patience : Nat
  seed : Bits64

defaultConfig : Config
defaultConfig = MkConfig 0.1 2000 500 123456

parseConfig : List String -> Config
parseConfig args = go args defaultConfig
  where
    go : List String -> Config -> Config
    go [] c = c
    go ("--lr" :: v :: rest) c = go rest ({ lr := cast v } c)
    go ("--epochs" :: v :: rest) c = go rest ({ epochs := cast (cast {to=Integer} v) } c)
    go ("--patience" :: v :: rest) c = go rest ({ patience := cast (cast {to=Integer} v) } c)
    go ("--seed" :: v :: rest) c = go rest ({ seed := cast (cast {to=Integer} v) } c)
    go (_ :: rest) c = go rest c


----------------------------------------------------------------------
-- Main
----------------------------------------------------------------------

main : IO ()
main = do
  tStart <- clockTime Monotonic
  args <- getArgs
  let cfg = parseConfig (drop 1 args)

  srand cfg.seed

  let lossFn = binaryCrossEntropyWithLogits

  putStrLn "=== LSTM Pattern Prediction ==="
  putStrLn $ "Config: lr=" ++ show cfg.lr
           ++ " epochs=" ++ show cfg.epochs
           ++ " patience=" ++ show cfg.patience
           ++ " seed=" ++ show cfg.seed

  lstm <- lstmLayer
  ll <- linearLayer {i=1, o=1}
  let model = autoName $ lstm ~> OutputLayer ll
  putStrLn $ "Architecture: " ++ show model
  putStrLn ""

  let dataPoints = map (map fromDouble) (rawData 8)
  let opt = nativeSgd cfg.lr

  putStrLn "Training..."
  t0 <- clockTime Monotonic
  (trained, epochsDone, finalLoss) <- trainLoop opt model dataPoints lossFn
    cfg.epochs cfg.patience t0
  t1 <- clockTime Monotonic

  let dblModel = toDoubleNetwork (emap refreshValue trained)
  let dblData = rawData 8
  let predictions : Vect 8 (List (Vector 1 Double))
      predictions = map (map (map (\x => cast (0 < x)))) (evaluateRecurrent dblModel dblData)
  let loss = calculateLossRecurrent lossFn dblModel dblData

  putStrLn ""
  putStrLn "Eval:"
  putStrLn $ "  Loss: " ++ show loss
  putStr "  Targets:     "
  printLn $ decodeYs dataPoints
  putStr "  Predictions: "
  printLn predictions
  putStrLn ""
  putStrLn $ formatTimingSummary tStart t1 epochsDone
  putStrLn $ "RESULT\tepochs=" ++ show epochsDone ++ "\tloss=" ++ show loss
           ++ "\ttime=" ++ show (seconds t1 - seconds tStart) ++ "s"
           ++ "\tseed=" ++ show cfg.seed
