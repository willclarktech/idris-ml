||| Cross-dtype SafeTensors round-trip smoke test for L63.
|||
||| Three modes (driven by `--mode`):
|||
|||   1. `save` — build a tiny `LinearState 2 3` model on the active
|||      `(ExampleExecutor, ExampleDType)` pair, train it for a handful
|||      of epochs so the params have meaningful values, and write
|||      the checkpoint to `--path`. The SafeTensors header records
|||      the actual dtype (`F32` or `F64`) per param.
|||
|||   2. `load-strict` — build a model on the active dtype, then
|||      attempt to load the checkpoint via `loadModel` (strict
|||      semantics). If the on-disk dtype matches the destination
|||      dtype, load succeeds. If they differ, `param_load` errors
|||      out and `loadModel` returns False.
|||
|||   3. `load-cast` — same model, load via `loadModelAllowCast`. On
|||      dtype mismatch the on-disk bytes are widened to doubles
|||      (lossless for F32 -> F64) before being loaded into the
|||      destination param via `param_load_data`, which narrows back
|||      to the destination's actual storage dtype (lossy for F64 ->
|||      F32 but well-defined).
|||
||| The `--expect <pass|fail>` flag asserts the outcome and exits
||| nonzero on mismatch — lets the Makefile orchestrator verify the
||| three-step matrix from a single process.
|||
||| Pairs with `make example-precision-checkpoint` which runs the
||| canonical three-step demo: save F32 (BACKEND=mlx MLX_DEVICE=gpu),
||| load-strict into F64 (expect fail), load-cast into F64 (expect
||| pass with eval-loss reproduction).
module Example.PrecisionCheckpoint

import Data.List
import Data.Vect
import System
import Compat.Random

import Backprop
import Checkpoint
import DataPoint
import Layer.Core
import Layer.Linear
import Array
import Train
import Util
import Executor
import Tensor
import BuildConfig

----------------------------------------------------------------------
-- Data (same 5-point classification task as Transfer)
----------------------------------------------------------------------

dataPoints : Vect 5 (DataPoint 2 3 Double)
dataPoints =
  [ MkDataPoint (VArray [1.5, -2.7]) (VArray [0, 1, 0]),
    MkDataPoint (VArray [-3.2, 4.1]) (VArray [0, 1, 0]),
    MkDataPoint (VArray [5.7, 0]) (VArray [0, 0, 1]),
    MkDataPoint (VArray [-1.3, 8.8]) (VArray [0, 1, 0]),
    MkDataPoint (VArray [2.9, -1.4]) (VArray [1, 0, 0])
  ]

----------------------------------------------------------------------
-- Config
----------------------------------------------------------------------

record Config where
  constructor MkConfig
  mode : String
  path : String
  expect : String        -- "pass" | "fail" | "" (no expectation)
  epochs : Nat
  lr : Double
  seed : Bits64

defaultConfig : Config
defaultConfig = MkConfig "save" "" "" 50 0.03 123456

specs : List (ArgSpec Config)
specs =
  [ Arg "--mode"   (\v, c => { mode := v } c)
  , Arg "--path"   (\v, c => { path := v } c)
  , Arg "--expect" (\v, c => { expect := v } c)
  , Arg "--epochs" (\v, c => { epochs := castNat v } c)
  , Arg "--lr"     (\v, c => { lr := cast v } c)
  , Arg "--seed"   (\v, c => { seed := castBits64 v } c)
  ]

----------------------------------------------------------------------
-- Eval — mean NLL loss over the 5 data points.
----------------------------------------------------------------------

evalModel : Network 2 [] 3 ExampleExecutor ExampleDType WithGrad -> IO Double
evalModel model = do
  losses <- traverse (\dp => do
        let inT  = bulkToTensor {ex=ExampleExecutor} {dt=ExampleDType} (x dp)
            inV  = the (TVec 2 ExampleExecutor ExampleDType WithGrad) (MkTensor inT Nothing)
            tgtT = bulkToTensor {ex=ExampleExecutor} {dt=ExampleDType} (y dp)
            tgtV = the (TVec 3 ExampleExecutor ExampleDType WithGrad) (MkTensor tgtT Nothing)
        (_, predV) <- forwardVar model inV
        lossT <- tnllLoss predV tgtV
        pure (primItem {ex=ExampleExecutor} lossT.tensorPtr)) dataPoints
  pure (foldl (+) 0.0 (toList losses) / 5.0)

----------------------------------------------------------------------
-- Modes
----------------------------------------------------------------------

doSave : Config -> Network 2 [] 3 ExampleExecutor ExampleDType WithGrad -> IO Bool
doSave cfg model = do
  let opt = nativeSgd cfg.lr
  putStrLn $ "Training " ++ show cfg.epochs ++ " epochs"
  (trained, _, _) <- runTraining {ex=ExampleExecutor}
    (\m, d => epochVar opt d tnllLoss m) (pure dataPoints)
    (simpleConfig cfg.epochs) model
  trainedLoss <- withNoGrad {ex=ExampleExecutor} (evalModel trained)
  putStrLn $ "Trained eval loss: " ++ show trainedLoss
  ok <- saveModel {ex=ExampleExecutor} cfg.path
  putStrLn $ (if ok then "Saved to " else "FAILED to save to ") ++ cfg.path
  pure ok

doLoad : (allowCast : Bool) -> Config ->
         Network 2 [] 3 ExampleExecutor ExampleDType WithGrad -> IO Bool
doLoad allowCast cfg model = do
  -- Initial eval — captures the untrained / random-init baseline.
  initLoss <- withNoGrad {ex=ExampleExecutor} (evalModel model)
  putStrLn $ "Pre-load eval loss: " ++ show initLoss
  ok <- if allowCast then loadModelAllowCast {ex=ExampleExecutor} cfg.path
                     else loadModel {ex=ExampleExecutor} cfg.path
  let label : String
      label = if allowCast then "load-cast" else "load-strict"
  putStrLn $ (if ok then "Loaded (" ++ label ++ ") from " else "FAILED to load (" ++ label ++ ") from ") ++ cfg.path
  if ok
    then do
      loadedLoss <- withNoGrad {ex=ExampleExecutor} (evalModel model)
      putStrLn $ "Post-load eval loss: " ++ show loadedLoss
    else pure ()
  pure ok

----------------------------------------------------------------------
-- Main
----------------------------------------------------------------------

main : IO ()
main = do
  requireMachine {m = ChosenMachine}
  args <- getArgs
  let cfg = parseArgs defaultConfig specs (drop 1 args)
  srand cfg.seed
  tsetInitSeed {ex = ExampleExecutor} cfg.seed

  when (cfg.path == "") $ do
    putStrLn "ERROR: --path required"
    exitFailure

  -- Use distinct param-prefix per dtype so two-step demos
  -- (BACKEND=mlx MLX_DEVICE=gpu then BACKEND=mlx) don't collide if
  -- a single test runner reuses one process. Within a single run,
  -- the prefix doesn't matter — save and load use the same prefix.
  llAny <- linearLayerAny {i=2} {o=3} "pck_ll"
  let model : Network 2 [] 3 ExampleExecutor ExampleDType WithGrad
      model = OutputLayer llAny

  putStrLn $ "=== PrecisionCheckpoint [" ++ backendName {ex=ExampleExecutor}
           ++ "] mode=" ++ cfg.mode ++ " ==="

  result <- case cfg.mode of
    "save"        => doSave cfg model
    "load-strict" => doLoad False cfg model
    "load-cast"   => doLoad True cfg model
    _ => do
      putStrLn "Unknown mode. Use --mode save|load-strict|load-cast"
      exitFailure

  let actual = if result then "pass" else "fail"
  putStrLn $ "Outcome: " ++ actual
  case cfg.expect of
    "" => pure ()
    expected => if expected == actual
                  then putStrLn ("PASS: expected " ++ expected ++ ", got " ++ actual)
                  else do
                    putStrLn ("FAIL: expected " ++ expected ++ ", got " ++ actual)
                    exitFailure
