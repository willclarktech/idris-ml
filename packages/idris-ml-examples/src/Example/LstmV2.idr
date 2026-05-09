module Example.LstmV2

import Data.List
import Data.Vect
import System
import System.Clock
import Compat.Random

import BackpropV2
import DataPoint
import Device
import Generate
import Layer.CoreV2
import Layer.LinearV2
import Layer.LstmV2
import Tensor
import Train
import Util
import Variable


-- Path C P3-2: end-to-end LSTM pattern prediction using only V2 surfaces.
-- Mirrors `Example/Lstm.idr` (V1) at the same hyperparameters; validates
-- that NetworkV2 + LstmV2 + LinearV2 + BackpropV2 train an LSTM end-to-end.

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

showSeq : List (Vector 1 Double) -> String
showSeq xs = concatMap (\(VTensor [STensor v]) => if v >= 0.5 then "1" else "0") xs

%default partial

main : IO ()
main = do
  args <- getArgs
  let cfg = parseArgs defaultConfig specs (drop 1 args)

  srand cfg.seed

  let opt = nativeSgd cfg.lr

  putStrLn "=== LSTM Pattern Prediction (V2) ==="
  putStrLn $ "Config: lr=" ++ show cfg.lr ++ " epochs=" ++ show cfg.epochs
           ++ " patience=" ++ show cfg.patience ++ " seed=" ++ show cfg.seed

  -- LSTM(1 -> 4) -> Linear(4 -> 1)
  lstmAny <- lstmLayerV2Any {i = 1} {o = 4} "v2_lstm"
  llAny <- linearLayerV2Any {i = 4} {o = 1} "v2_ll"
  let model : NetworkV2 1 [4] 1 CPU
      model = lstmAny ~~> OutputLayerV2 llAny
  putStrLn ""

  (trained, epochsDone, _) <- runTraining
    (\m, d => epochRecurrentTVar opt d tbceLoss m)
    (pure (patternData 8))
    (patienceConfig cfg.epochs cfg.patience)
    model

  putStrLn ""
  putStrLn $ formatResult [ ("epochs", show epochsDone)
                          , ("seed", show cfg.seed)
                          ]
