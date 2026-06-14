||| Cross-dtype SafeTensors round-trip smoke test.
|||
||| Three modes (driven by `--mode`):
|||
|||   1. `save` — build a tiny `Linear 2 3` model on the active
|||      `(Ex, F)` pair, train it for a handful of epochs so the params
|||      have meaningful values, and write the checkpoint to `--path`.
|||      The SafeTensors header records the actual dtype (`F32` or `F64`)
|||      per param.
|||
|||   2. `load-strict` — build a model on the active dtype, then attempt
|||      to load the checkpoint via `loadModel` (strict semantics). If the
|||      on-disk dtype matches the destination dtype, load succeeds. If
|||      they differ, `param_load` errors and `loadModel` returns False.
|||
|||   3. `load-cast` — same model, load via `loadModelAllowCast`. On dtype
|||      mismatch the on-disk bytes are widened to doubles (lossless for
|||      F32 -> F64) before being loaded into the destination param via
|||      `param_load_data`, which narrows back to the destination's actual
|||      storage dtype (lossy for F64 -> F32 but well-defined).
|||
||| The `--expect <pass|fail>` flag asserts the outcome and exits nonzero
||| on mismatch — lets the Makefile orchestrator verify the three-step
||| matrix from a single process.
|||
||| Pairs with `make example-precision-checkpoint` which runs the
||| canonical three-step demo: save F32 (BACKEND=mlx MLX_DEVICE=gpu),
||| load-strict into F64 (expect fail), load-cast into F64 (expect pass
||| with eval-loss reproduction).
module Example.PrecisionCheckpoint

import Control.Linear.LIO
import Data.Linear.Notation
import Data.List
import Data.Vect
import System

import BuildConfig
import Checkpoint
import Compat.Random
import FitL
import ML.Simple
import Train

----------------------------------------------------------------------
-- Data (same 5-point classification task as Transfer)
----------------------------------------------------------------------

inputsV : Vect 5 (Vect 2 Double)
inputsV = [ [1.5, -2.7], [-3.2, 4.1], [5.7, 0.0], [-1.3, 8.8], [2.9, -1.4] ]

targetsV : Vect 5 (Vect 3 Double)
targetsV = [ [0,1,0], [0,1,0], [0,0,1], [0,1,0], [1,0,0] ]

flatInputs : Vect 10 Double
flatInputs = [1.5, -2.7, -3.2, 4.1, 5.7, 0.0, -1.3, 8.8, 2.9, -1.4]

----------------------------------------------------------------------
-- Config
----------------------------------------------------------------------

record Config where
  constructor MkConfig
  mode   : String
  path   : String
  expect : String        -- "pass" | "fail" | "" (no expectation)
  epochs : Nat
  lr     : Double
  seed   : Bits64

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
-- Data + loss (full-batch, b=5)
----------------------------------------------------------------------

mkPair : (Vect 2 Double, Vect 3 Double) ->
         IO (Tensor [2] Ex F NoGrad, Tensor [3] Ex F NoGrad)
mkPair (xv, yv) = do
  x <- tensor {dims=[2]} (FromVect xv)
  y <- tensor {dims=[3]} (FromVect yv)
  pure (x, y)

sampleAt : Nat -> IO (Tensor [2] Ex F NoGrad, Tensor [3] Ex F NoGrad)
sampleAt n = case natToFin n 5 of
  Just i  => mkPair (index i inputsV, index i targetsV)
  Nothing => assert_total $ idris_crash "PrecisionCheckpoint.sampleAt: index out of range"

buildStream : IO (DataStream (Tensor [5, 2] Ex F NoGrad, Tensor [5, 3] Ex F NoGrad))
buildStream = do
  s <- stream NoShuffle (fromIndexed 5 sampleAt)
  pure (batched {b=5} {i=2} {o=3} s)

-- Linear-resource loss for fitSupervisedL.
nllLossL : (1 _ : Linear 2 3 Ex F WithGrad) ->
           (Tensor [5, 2] Ex F NoGrad, Tensor [5, 3] Ex F NoGrad) ->
           L IO {use = 1} (LPair (!* (Tensor [] Ex F WithGrad)) (Linear 2 3 Ex F WithGrad))
nllLossL model (x, tgt) = do
  (MkBang out # model') <- forwardL {b=5} model (retypeGrad x)
  loss <- tnllLossMeanL {b=5} {n=3} out (retypeGrad tgt)
  pure1 (MkBang loss # model')

----------------------------------------------------------------------
-- Eval — mean NLL loss over the 5 data points.
----------------------------------------------------------------------

evalInput : IO (Tensor [5, 2] Ex F NoGrad)
evalInput = tensor {dims=[5, 2]} (FromVect flatInputs)

-- Mean NLL loss from already-forwarded [5,3] logits (NoGrad → no tape).
lossFrom : Tensor [5, 3] Ex F NoGrad -> IO Double
lossFrom predB = do
  tgt <- tensor {dims=[5,3]} (FromVect (concat targetsV))
  l   <- tnllLossMean {b=5} {n=3} predB tgt
  pure (primItem {ex=Ex} l.tensorPtr)

----------------------------------------------------------------------
-- Modes (linear surface; IO re-entry via run)
----------------------------------------------------------------------

doSave : Config -> Optimizer Ex -> IO Bool
doSave cfg opt = Control.Linear.LIO.run $ do
  model <- runInitL (linear {i=2} {o=3})
  bs <- liftIO1 buildStream
  liftIO1 $ putStrLn $ "Training " ++ show cfg.epochs ++ " epochs"
  (MkBang _ # trained) <- fitSupervisedL opt nllLossL bs (simpleConfig cfg.epochs) model
  infer <- evalL trained
  ein <- liftIO1 evalInput
  (MkBang predB # infer') <- forwardL {b=5} infer ein
  discardL infer'
  liftIO1 $ do
    trainedLoss <- lossFrom predB
    putStrLn $ "Trained eval loss: " ++ show trainedLoss
    ok <- saveAll {ex=Ex} cfg.path
    putStrLn $ (if ok then "Saved to " else "FAILED to save to ") ++ cfg.path
    pure ok

-- Load mode: forward once pre-load and once post-load, threading the (linear)
-- inference model between the two so both reads are single-owner. loadModel
-- mutates the C param registry by name (the model value is unchanged), so the
-- post-load forward reflects the loaded weights.
doLoad : (allowCast : Bool) -> Config -> IO Bool
doLoad allowCast cfg = Control.Linear.LIO.run $ do
  model <- runInitL (linear {i=2} {o=3})
  infer <- evalL model
  ein <- liftIO1 evalInput
  (MkBang predPre # infer') <- forwardL {b=5} infer ein
  liftIO1 $ do
    initLoss <- lossFrom predPre
    putStrLn $ "Pre-load eval loss: " ++ show initLoss
  ok <- liftIO1 $ if allowCast then loadModelAllowCast {ex=Ex} cfg.path
                               else loadModel {ex=Ex} cfg.path
  let label : String
      label = if allowCast then "load-cast" else "load-strict"
  liftIO1 $ putStrLn $ (if ok then "Loaded (" ++ label ++ ") from " else "FAILED to load (" ++ label ++ ") from ") ++ cfg.path
  (MkBang predPost # infer'') <- forwardL {b=5} infer' ein
  discardL infer''
  liftIO1 $ do
    when ok $ do
      loadedLoss <- lossFrom predPost
      putStrLn $ "Post-load eval loss: " ++ show loadedLoss
    pure ok

----------------------------------------------------------------------
-- Main
----------------------------------------------------------------------

%default partial

main : IO ()
main = do
  requireMachine {m = ChosenMachine}
  args <- getArgs
  let cfg = parseArgs defaultConfig specs (drop 1 args)
  srand cfg.seed
  tsetInitSeed {ex = Ex} cfg.seed

  when (cfg.path == "") $ do
    putStrLn "ERROR: --path required"
    exitFailure

  opt <- sgd cfg.lr defaultOpts

  putStrLn $ "=== PrecisionCheckpoint [" ++ backendName {ex=Ex}
           ++ "] mode=" ++ cfg.mode ++ " ==="

  result <- case cfg.mode of
    "save"        => doSave cfg opt
    "load-strict" => doLoad False cfg
    "load-cast"   => doLoad True cfg
    _             => do
      putStrLn "Unknown mode. Use --mode save|load-strict|load-cast"
      exitFailure

  let actual = if result then "pass" else "fail"
  putStrLn $ "Outcome: " ++ actual
  case cfg.expect of
    ""       => pure ()
    expected => if expected == actual
                  then putStrLn ("PASS: expected " ++ expected ++ ", got " ++ actual)
                  else do
                    putStrLn ("FAIL: expected " ++ expected ++ ", got " ++ actual)
                    exitFailure
