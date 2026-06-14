||| SafeTensors checkpoint round-trip demo: train → save → load →
||| continue → save → infer. Each phase runs in a separate
||| `BACKEND=` build so the model state crosses backend boundaries
||| via the on-disk `.safetensors` checkpoint format. The live
||| cross-backend tensor transfer demo lives in `Example/Transfer.idr`.
module Example.Checkpoint

import Data.List
import Data.Vect
import System

import BuildConfig
import Checkpoint
import Compat.Random
import Hpo.LrFinder
import ML.Simple
import Train

----------------------------------------------------------------------
-- Data (same 5-point classification task as Supervised)
----------------------------------------------------------------------

inputsV : Vect 5 (Vect 2 Double)
inputsV = [ [1.5, -2.7], [-3.2, 4.1], [5.7, 0.0], [-1.3, 8.8], [2.9, -1.4] ]

targetsV : Vect 5 (Vect 3 Double)
targetsV = [ [0,1,0], [0,1,0], [0,0,1], [0,1,0], [1,0,0] ]

targetClasses : Vect 5 Nat
targetClasses = [1, 1, 2, 1, 0]

flatInputs : Vect 10 Double
flatInputs = [1.5, -2.7, -3.2, 4.1, 5.7, 0.0, -1.3, 8.8, 2.9, -1.4]

----------------------------------------------------------------------
-- Config
----------------------------------------------------------------------

record Config where
  constructor MkConfig
  mode     : String
  epochs   : Nat
  lr       : Double
  seed     : Bits64
  savePath : String
  loadPath : String
  lrFind   : Bool

defaultConfig : Config
defaultConfig = MkConfig "train" 500 0.03 123456 "" "" False

specs : List (ArgSpec Config)
specs = [ Arg "--mode" (\v, c => { mode := v } c)
        , Arg "--epochs" (\v, c => { epochs := castNat v } c)
        , Arg "--lr" (\v, c => { lr := cast v } c)
        , Arg "--seed" (\v, c => { seed := castBits64 v } c)
        , Arg "--save" (\v, c => { savePath := v } c)
        , Arg "--load" (\v, c => { loadPath := v } c)
        , Arg "--lr-find" (\v, c => { lrFind := (v == "1" || v == "true") } c) ]

-- Derive optimizer state path: "model.safetensors" → "model.optimizer.safetensors"
optPath : String -> String
optPath path =
  let chars = unpack path
      suffix = unpack ".safetensors"
      base = pack (take (length chars `minus` length suffix) chars)
  in base ++ ".optimizer.safetensors"

----------------------------------------------------------------------
-- Data + loss (full-batch, b=5 — one optimizer step per epoch)
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
  Nothing => assert_total $ idris_crash "Checkpoint.sampleAt: index out of range"

buildStream : IO (DataStream (Tensor [5, 2] Ex F NoGrad, Tensor [5, 3] Ex F NoGrad))
buildStream = do
  s <- stream NoShuffle (fromIndexed 5 sampleAt)
  pure (batched {b=5} {i=2} {o=3} s)

nllLoss : Linear 2 3 Ex F WithGrad ->
          (Tensor [5, 2] Ex F NoGrad, Tensor [5, 3] Ex F NoGrad) ->
          IO (Tensor [] Ex F WithGrad)
nllLoss model (x, tgt) = do
  out <- forward {b=5} model (retypeGrad x)
  tnllLossMean {b=5} {n=3} out (retypeGrad tgt)

-- Single train step (for lrFind, which owns its own iteration).
trainStep : Optimizer Ex -> Linear 2 3 Ex F WithGrad ->
            (Tensor [5, 2] Ex F NoGrad, Tensor [5, 3] Ex F NoGrad) ->
            IO (Linear 2 3 Ex F WithGrad, Double)
trainStep opt model batch = do
  l <- nllLoss model batch
  d <- nativeTrainStep opt l
  pure (model, d)

----------------------------------------------------------------------
-- Eval (mean NLL loss + per-point predictions over the [5,3] logits)
----------------------------------------------------------------------

argmax3 : Double -> Double -> Double -> Nat
argmax3 a b c = if a >= b && a >= c then 0 else if b >= c then 1 else 2

showVec2 : Vect 2 Double -> String
showVec2 [a, b] = "[" ++ show a ++ ", " ++ show b ++ "]"

evalInput : IO (Tensor [5, 2] Ex F NoGrad)
evalInput = tensor {dims=[5, 2]} (FromVect flatInputs)

evalLoss : {0 g : GradMode} -> Linear 2 3 Ex F g -> IO Double
evalLoss model = withNoGrad {ex=Ex} $ do
  out <- forward {b=5} model (retypeGrad !evalInput)
  l   <- tnllLossMean {b=5} {n=3} out (retypeGrad !(tensor {dims=[5,3]}
           (FromVect (concat targetsV))))
  pure (primItem {ex=Ex} l.tensorPtr)

printPredictions : {0 g : GradMode} -> Linear 2 3 Ex F g -> IO ()
printPredictions model = do
  predB <- forward {b=5} model (retypeGrad !evalInput)
  for_ (toList Fin.range) $ \i => do
    let r  = cast {to=Int} (finToNat i)
        v0 = primItem2d {ex=Ex} predB.tensorPtr r 0
        v1 = primItem2d {ex=Ex} predB.tensorPtr r 1
        v2 = primItem2d {ex=Ex} predB.tensorPtr r 2
        pred = argmax3 v0 v1 v2
        ok   = pred == index i targetClasses
    putStrLn $ "  " ++ showVec2 (index i inputsV) ++ " -> class " ++ show pred
             ++ (if ok then " ok" else " WRONG")

----------------------------------------------------------------------
-- Modes
----------------------------------------------------------------------

doTrain : Config -> Optimizer Ex -> Linear 2 3 Ex F WithGrad -> IO ()
doTrain cfg opt model = do
  bs <- buildStream
  putStrLn $ "Training " ++ show cfg.epochs ++ " epochs..."
  (trained, epochsDone, _) <-
    fitSupervised opt nllLoss bs (simpleConfig cfg.epochs) model
  if cfg.savePath == ""
    then putStrLn "No --save path given; skipping save"
    else do
      ok <- saveAll {ex=Ex} cfg.savePath
      putStrLn $ (if ok then "Saved model to " else "FAILED to save model to ") ++ cfg.savePath
      ok2 <- saveOptimizer (optPath cfg.savePath) opt
      putStrLn $ (if ok2 then "Saved optimizer to " else "FAILED to save optimizer to ") ++ optPath cfg.savePath
  el <- evalLoss trained
  putStrLn $ "Eval loss: " ++ show el
  printPredictions trained
  putStrLn $ formatResult [("mode", "train"), ("epochs", show epochsDone),
                            ("loss", show el), ("backend", backendName {ex=Ex})]

doContinue : Config -> Optimizer Ex -> Linear 2 3 Ex F WithGrad -> IO ()
doContinue cfg opt model = do
  ok <- loadModel {ex=Ex} cfg.loadPath
  putStrLn $ (if ok then "Loaded model from " else "FAILED to load from ") ++ cfg.loadPath
  ok2 <- loadOptimizer (optPath cfg.loadPath) opt
  putStrLn $ (if ok2 then "Loaded optimizer from " else "FAILED to load optimizer from ")
           ++ optPath cfg.loadPath
  bs <- buildStream
  putStrLn $ "Training " ++ show cfg.epochs ++ " more epochs..."
  (trained, epochsDone, _) <-
    fitSupervised opt nllLoss bs (simpleConfig cfg.epochs) model
  if cfg.savePath == ""
    then putStrLn "No --save path given; skipping save"
    else do
      ok3 <- saveAll {ex=Ex} cfg.savePath
      putStrLn $ (if ok3 then "Saved model to " else "FAILED to save model to ") ++ cfg.savePath
      ok4 <- saveOptimizer (optPath cfg.savePath) opt
      putStrLn $ (if ok4 then "Saved optimizer to " else "FAILED to save optimizer to ") ++ optPath cfg.savePath
  el <- evalLoss trained
  putStrLn $ "Eval loss: " ++ show el
  printPredictions trained
  putStrLn $ formatResult [("mode", "continue"), ("epochs", show epochsDone),
                            ("loss", show el), ("backend", backendName {ex=Ex})]

doInfer : Config -> Linear 2 3 Ex F WithGrad -> IO ()
doInfer cfg model = do
  ok <- loadModel {ex=Ex} cfg.loadPath
  putStrLn $ (if ok then "Loaded model from " else "FAILED to load from ") ++ cfg.loadPath
  el <- evalLoss model
  putStrLn $ "Eval loss: " ++ show el
  printPredictions model
  putStrLn $ formatResult [("mode", "infer"), ("loss", show el),
                            ("backend", backendName {ex=Ex})]

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

  opt <- sgd cfg.lr defaultOpts
  model <- runInit (linear {i=2} {o=3})

  putStrLn $ "=== Cross-Backend Transfer [" ++ backendName {ex=Ex} ++ "] -- "
           ++ cfg.mode ++ " ==="

  when cfg.lrFind $ do
    let lrCfg : LrFindConfig
        lrCfg = { numIters := 100 } defaultLrFindConfig
    bs <- buildStream
    _ <- lrFind lrCfg (trainStep opt) bs.next opt model
    putStrLn ""
    putStrLn "Done — re-run without --lr-find at the recommended LR."
    exitSuccess

  case cfg.mode of
    "train"    => doTrain cfg opt model
    "continue" => doContinue cfg opt model
    "infer"    => doInfer cfg model
    _ => putStrLn "Unknown mode. Use --mode train|continue|infer"
