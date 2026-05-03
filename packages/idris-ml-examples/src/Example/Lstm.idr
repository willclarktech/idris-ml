module Example.Lstm

import Data.List
import Data.Vect
import System
import System.Clock
import Compat.Random

import Backprop
import Checkpoint
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
import BuildConfig


-- LSTM pattern-prediction example. Single LSTM(1 -> 4) -> Linear(4 -> 1)
-- network with BCE-with-logits loss.

record Config where
  constructor MkConfig
  lr : Double
  epochs : Nat
  patience : Nat
  seed : Bits64
  checkpointDir : String
  checkpointEvery : Nat

defaultConfig : Config
defaultConfig = MkConfig 0.5 2000 500 42 "" 200

specs : List (ArgSpec Config)
specs = [ Arg "--lr" (\v, c => { lr := cast v } c)
        , Arg "--epochs" (\v, c => { epochs := castNat v } c)
        , Arg "--patience" (\v, c => { patience := castNat v } c)
        , Arg "--seed" (\v, c => { seed := castBits64 v } c)
        -- Checkpointing: save to / auto-resume from DIR. `--resume` is an
        -- alias for `--checkpoint-dir` (resumes if DIR/last.* is present).
        , Arg "--checkpoint-dir" (\v, c => { checkpointDir := v } c)
        , Arg "--resume" (\v, c => { checkpointDir := v } c)
        , Arg "--checkpoint-every" (\v, c => { checkpointEvery := castNat v } c)
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
  let model : Network 1 [4] 1 ExampleDevice ExampleDType WithGrad
      model = lstmAny ~~> OutputLayer llAny
  putStrLn ""

  let trainCfgBase = patienceConfig cfg.epochs cfg.patience
      trainCfg = case cfg.checkpointDir of
                   "" => trainCfgBase
                   dir => withCheckpoint
                            (fileCheckpoint dir cfg.checkpointEvery True opt)
                            trainCfgBase

  (trained, epochsDone, finalLoss) <- runTraining {d=ExampleDevice}
    (\m, d => epochRecurrentVar opt d tbceLoss m)
    (pure (patternData 8))
    trainCfg
    model

  putStrLn ""
  putStrLn $ formatResult [ ("epochs", show epochsDone)
                          , ("loss", show finalLoss)
                          , ("seed", show cfg.seed)
                          ]
