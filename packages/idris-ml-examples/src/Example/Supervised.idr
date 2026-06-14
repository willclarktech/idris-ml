module Example.Supervised

import Data.List
import Data.Vect
import System

import BuildConfig
import Compat.Random
import GradScaler
import ML.Simple
import Train

-- f(x, y) = argmax(x - y - 10, -4x + y + 5, 2x + y - 11): a 3-class
-- (mutually-exclusive) problem, so the loss is multiclass NLL (tnllLossMean),
-- matching torch_ref/models/supervised.py's nll_loss(log_softmax(...)).
inputsV : Vect 5 (Vect 2 Double)
inputsV = [ [1.5, -2.7], [-3.2, 4.1], [5.7, 0.0], [-1.3, 8.8], [2.9, -1.4] ]

targetsV : Vect 5 (Vect 3 Double)
targetsV = [ [0,1,0], [0,1,0], [0,0,1], [0,1,0], [1,0,0] ]

-- argmax of each target row — the eval ground truth.
targetClasses : Vect 5 Nat
targetClasses = [1, 1, 2, 1, 0]

-- Flattened inputs for the [5,2] eval batch tensor (FromVect over Numel).
flatInputs : Vect 10 Double
flatInputs = [1.5, -2.7, -3.2, 4.1, 5.7, 0.0, -1.3, 8.8, 2.9, -1.4]

record Config where
  constructor MkConfig
  lr             : Double
  epochs         : Nat
  seed           : Bits64
  mixedPrecision : Bool
  ||| Mixed-precision parameter-storage mode. Only consulted when
  ||| `mixedPrecision = True`. `"native"` (default): paramDt = computeDt = F.
  ||| `"f32"`: paramDt = F32 master, computeDt = F — the F32-master /
  ||| low-precision-compute decoupling (a real F32→F64 widen on tape; the
  ||| autocast-equivalent on a BF16/F16 build).
  paramDtype : String

defaultConfig : Config
defaultConfig = MkConfig 0.03 1000 42 False "native"

boolFlag : String -> Bool
boolFlag v = v == "1" || v == "true" || v == "True" || v == "yes"

specs : List (ArgSpec Config)
specs = [ Arg "--lr" (\v, c => { lr := cast v } c)
        , Arg "--epochs" (\v, c => { epochs := castNat v } c)
        , Arg "--seed" (\v, c => { seed := castBits64 v } c)
        , Arg "--mixed-precision" (\v, c => { mixedPrecision := boolFlag v } c)
        , Arg "--param-dtype" (\v, c => { paramDtype := v } c) ]

----------------------------------------------------------------------
-- Data + loss
----------------------------------------------------------------------

mkPair : (Vect 2 Double, Vect 3 Double) ->
         IO (Tensor [2] Ex F NoGrad, Tensor [3] Ex F NoGrad)
mkPair (xv, yv) = do
  x <- tensor {dims=[2]} (FromVect xv)
  y <- tensor {dims=[3]} (FromVect yv)
  pure (x, y)

-- Materialise FRESH device tensors per access (the `idxDataset` pattern),
-- NOT `fromVect` of persistent handles: collation links each batch's source
-- tensors into the tape, and the per-epoch backward/teardown frees them, so
-- reusing fixed handles across epochs reads freed memory. Fresh-per-pull
-- sidesteps that — every epoch's batch is built from new tensors.
sampleAt : Nat -> IO (Tensor [2] Ex F NoGrad, Tensor [3] Ex F NoGrad)
sampleAt n = case natToFin n 5 of
  Just i  => mkPair (index i inputsV, index i targetsV)
  Nothing => assert_total $ idris_crash "Supervised.sampleAt: index out of range"

-- Full-batch stream: all 5 points in one batch (b=5), so one optimizer
-- step per epoch — full-batch GD, matching the reference's reduction.
buildStream : IO (DataStream (Tensor [5, 2] Ex F NoGrad, Tensor [5, 3] Ex F NoGrad))
buildStream = do
  s <- stream NoShuffle (fromIndexed 5 sampleAt)
  pure (batched {b=5} {i=2} {o=3} s)

-- Default-precision loss: forward [5,2]→[5,3] then batched multiclass NLL.
nllLossDefault : Linear 2 3 Ex F WithGrad ->
                 (Tensor [5, 2] Ex F NoGrad, Tensor [5, 3] Ex F NoGrad) ->
                 IO (Tensor [] Ex F WithGrad)
nllLossDefault model (x, tgt) = do
  out <- forward {b=5} model (retypeGrad x)
  tnllLossMean {b=5} {n=3} out (retypeGrad tgt)

-- Mixed-precision loss: forward casts paramDt → computeDt (F) internally.
nllLossMixed : {0 pDt : DType} -> Backend Ex pDt => IsDType pDt =>
               LinearMixed 2 3 Ex pDt F WithGrad ->
               (Tensor [5, 2] Ex F NoGrad, Tensor [5, 3] Ex F NoGrad) ->
               IO (Tensor [] Ex F WithGrad)
nllLossMixed model (x, tgt) = do
  out <- forwardMixed {b=5} model (retypeGrad x)
  tnllLossMean {b=5} {n=3} out (retypeGrad tgt)

----------------------------------------------------------------------
-- Eval (shared across modes: takes the [5,3] logits)
----------------------------------------------------------------------

argmax3 : Double -> Double -> Double -> Nat
argmax3 a b c = if a >= b && a >= c then 0 else if b >= c then 1 else 2

showVec2 : Vect 2 Double -> String
showVec2 [a, b] = "[" ++ show a ++ ", " ++ show b ++ "]"

evalPredictions : {0 g : GradMode} -> Tensor [5, 3] Ex F g -> IO Nat
evalPredictions predB = do
  counts <- for (toList Fin.range) $ \i => do
    let r  = cast {to=Int} (finToNat i)
        v0 = primItem2d {ex=Ex} predB.tensorPtr r 0
        v1 = primItem2d {ex=Ex} predB.tensorPtr r 1
        v2 = primItem2d {ex=Ex} predB.tensorPtr r 2
        pred = argmax3 v0 v1 v2
        ok   = pred == index i targetClasses
    putStrLn $ "  " ++ showVec2 (index i inputsV) ++ " -> class " ++ show pred
             ++ (if ok then " ok" else " WRONG")
    pure (if ok then the Nat 1 else 0)
  pure (sum counts)

evalInput : IO (Tensor [5, 2] Ex F NoGrad)
evalInput = tensor {dims=[5, 2]} (FromVect flatInputs)

reportResult : Config -> Nat -> Double -> Nat -> IO ()
reportResult cfg epochsDone finalLoss correct =
  putStrLn $ formatResult [ ("epochs", show epochsDone)
                          , ("loss", show finalLoss)
                          , ("seed", show cfg.seed)
                          , ("correct", show correct ++ "/5") ]

----------------------------------------------------------------------
-- Run modes
----------------------------------------------------------------------

runDefault : Config -> Optimizer Ex -> IO ()
runDefault cfg opt = do
  model <- runInit (linear {i=2} {o=3})
  bs <- buildStream
  putStrLn ""
  (trained, epochsDone, finalLoss) <-
    fitSupervised opt nllLossDefault bs (simpleConfig cfg.epochs) model
  putStrLn ""
  putStrLn "Eval:"
  -- Convert to an inference (NoGrad) model: forward is then genuinely
  -- tape-free, and the type witnesses it.
  infer <- eval trained
  predB <- forward {b=5} infer !evalInput
  correct <- evalPredictions predB
  putStrLn ""
  reportResult cfg epochsDone finalLoss correct

-- Mixed-precision run, polymorphic over the master dtype. The caller pins
-- paramDt by the `mkModel` action's result type (Idris can't dispatch types
-- from a runtime string, so each --param-dtype mode is its own typed call).
runMixedGeneric : {0 pDt : DType} -> Backend Ex pDt => IsDType pDt =>
                  Config -> Optimizer Ex -> IO (LinearMixed 2 3 Ex pDt F WithGrad) ->
                  String -> IO ()
runMixedGeneric cfg opt mkModel modeLabel = do
  model <- mkModel
  gs <- defaultGradScaler {ex=Ex} {dt=F}
  bs <- buildStream
  putStrLn modeLabel
  putStrLn ""
  (trained, epochsDone, finalLoss) <-
    fitSupervisedMixed opt gs nllLossMixed bs (simpleConfig cfg.epochs) model
  putStrLn ""
  putStrLn "Eval:"
  -- No `eval` for the mixed (ParamsMixed) path yet; lift the input to
  -- WithGrad so the trained model forwards (predictions read without backward).
  predB <- forwardMixed {b=5} trained (retypeGrad !evalInput)
  correct <- evalPredictions predB
  putStrLn ""
  reportResult cfg epochsDone finalLoss correct

runMixedNative : Config -> Optimizer Ex -> IO ()
runMixedNative cfg opt =
  runMixedGeneric cfg opt
    (runInit (linearMixed {paramDt=F} {computeDt=F} {i=2} {o=3}))
    "Mixed-precision mode: paramDt = computeDt = F (native)"

runMixedF32Master : Config -> Optimizer Ex -> IO ()
runMixedF32Master cfg opt =
  runMixedGeneric cfg opt
    (runInit (linearMixed {paramDt=F32} {computeDt=F} {i=2} {o=3}))
    "Mixed-precision mode: paramDt = F32, computeDt = F (f32-master)"

main : IO ()
main = do
  requireMachine {m = ChosenMachine}
  args <- getArgs
  let cfg = parseArgs defaultConfig specs (drop 1 args)

  srand cfg.seed
  tsetInitSeed {ex = Ex} cfg.seed

  opt <- sgd cfg.lr defaultOpts

  putStrLn "=== Supervised Classification ==="
  putStrLn $ "Config: lr=" ++ show cfg.lr ++ " epochs=" ++ show cfg.epochs
           ++ " seed=" ++ show cfg.seed
           ++ " mixed-precision=" ++ show cfg.mixedPrecision
           ++ " param-dtype=" ++ cfg.paramDtype

  if cfg.mixedPrecision
    then case cfg.paramDtype of
      "f32" => runMixedF32Master cfg opt
      _     => runMixedNative cfg opt
    else runDefault cfg opt
