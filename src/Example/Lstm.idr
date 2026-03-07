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

decodeOutput : Vect n (List (Vector o Variable)) -> Vect n (List (Vector o Double))
decodeOutput = map (map (map (cast . (0<))))


----------------------------------------------------------------------
-- Training Loop
----------------------------------------------------------------------

||| Training loop with early stopping.
||| Returns (model, epochs completed, final loss).
trainLoop :
  Optimizer ->
  Network 1 [1] 1 Variable ->
  Vect 8 (RecurrentDataPoint 1 1 Variable) ->
  LossFunction Variable ->
  (totalEpochs : Nat) -> (patience : Nat) ->
  OptimizerState ->
  Clock Monotonic ->
  IO (Network 1 [1] 1 Variable, Nat, Double)
trainLoop opt model dataPoints lossFn totalEpochs patience st t0 =
  go 0 model st (1.0/0.0) 0
  where
    go : Nat -> Network 1 [1] 1 Variable -> OptimizerState ->
         Double -> Nat ->
         IO (Network 1 [1] 1 Variable, Nat, Double)
    go ep m s bestLoss staleCount =
      if ep >= totalEpochs then do
        let loss = value $ calculateLossRecurrent lossFn m dataPoints
        pure (m, ep, loss)
      else do
        let (m', s', loss) = epochRecurrent opt dataPoints lossFn m s
        when (modNatNZ ep 100 ItIsSucc == 0) $ do
          now <- clockTime Monotonic
          putStrLn $ "  " ++ formatElapsed t0 now ++ " " ++ show ep ++ ":\tloss=" ++ show loss
                   ++ "\trss=" ++ show (getRssMB ep) ++ "MB"
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
              else go (ep + 1) m' s' bestLoss' sc


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
  args <- getArgs
  let cfg = parseConfig (drop 1 args)

  srand cfg.seed

  let lossFn = binaryCrossEntropyWithLogits

  putStrLn "=== LSTM Pattern Prediction ==="
  putStrLn $ "Config: lr=" ++ show cfg.lr
           ++ " epochs=" ++ show cfg.epochs
           ++ " patience=" ++ show cfg.patience
           ++ " seed=" ++ show cfg.seed
  putStrLn ""

  -- LSTM as drop-in replacement for RNN
  lstm <- lstmLayer
  ll <- linearLayer {i=1, o=1}
  let model = autoName $ lstm ~> OutputLayer ll
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

  putStrLn ""
  putStrLn "Training..."
  let opt = sgd cfg.lr (1.0/0.0)
  t0 <- clockTime Monotonic
  (trained, epochsDone, finalLoss) <- trainLoop opt model dataPoints lossFn
    cfg.epochs cfg.patience initState t0

  let predictions' = decodeOutput $ evaluateRecurrent trained dataPoints
  let loss' = calculateLossRecurrent lossFn trained dataPoints

  putStr "Post loss: "
  printLn $ value loss'
  putStr "Predictions: "
  printLn $ predictions'

  -- Machine-readable result line
  putStrLn $ "RESULT\t"
           ++ show cfg.lr ++ "\t"
           ++ show cfg.epochs ++ "\t"
           ++ show cfg.patience ++ "\t"
           ++ show epochsDone ++ "\t"
           ++ show cfg.seed ++ "\t"
           ++ show (value loss')
