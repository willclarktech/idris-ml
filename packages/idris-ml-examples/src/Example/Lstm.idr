module Example.Lstm

import Data.List
import Data.Vect
import System
import System.Clock
import Compat.Random

import Backprop
import DataPoint
import Device
import Generate
import Layer.Core
import Layer.Linear
import Layer.Lstm
import Array
import Train
import Util
import Tensor


-- LSTM pattern-prediction example. Single LSTM(1 -> 4) -> Linear(4 -> 1)
-- network with BCE-with-logits loss.

record Config where
  constructor MkConfig
  lr : Double
  epochs : Nat
  patience : Nat
  seed : Bits64

defaultConfig : Config
defaultConfig = MkConfig 0.5 2000 500 42

specs : List (ArgSpec Config)
specs = [ Arg "--lr" (\v, c => { lr := cast v } c)
        , Arg "--epochs" (\v, c => { epochs := castNat v } c)
        , Arg "--patience" (\v, c => { patience := castNat v } c)
        , Arg "--seed" (\v, c => { seed := castBits64 v } c)
        ]

%default partial

main : IO ()
main = do
  args <- getArgs
  let cfg = parseArgs defaultConfig specs (drop 1 args)

  srand cfg.seed

  let opt = nativeSgd cfg.lr

  putStrLn "=== LSTM Pattern Prediction ==="
  putStrLn $ "Config: lr=" ++ show cfg.lr ++ " epochs=" ++ show cfg.epochs
           ++ " patience=" ++ show cfg.patience ++ " seed=" ++ show cfg.seed

  lstmAny <- lstmLayerAny {i = 1} {o = 4} "lstm"
  llAny <- linearLayerAny {i = 4} {o = 1} "ll"
  let model : Network 1 [4] 1 CPU WithGrad
      model = lstmAny ~~> OutputLayer llAny
  putStrLn ""

  (trained, epochsDone, finalLoss) <- runTraining
    (\m, d => epochRecurrentVar opt d tbceLoss m)
    (pure (patternData 8))
    (patienceConfig cfg.epochs cfg.patience)
    model

  putStrLn ""
  putStrLn $ formatResult [ ("epochs", show epochsDone)
                          , ("loss", show finalLoss)
                          , ("seed", show cfg.seed)
                          ]
