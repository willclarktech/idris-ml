||| SafeTensors checkpoint round-trip demo: train → save → load →
||| continue → save → infer. Each phase runs in a separate
||| `BACKEND=` build so the model state crosses backend boundaries
||| via the on-disk `.safetensors` checkpoint format. The live
||| cross-backend tensor transfer demo lives in `Example/Transfer.idr`.
module Example.Checkpoint

import Data.List
import Data.Vect
import System
import Compat.Random

import Backprop
import Checkpoint
import DataPoint
import Hpo.LrFinder
import Layer.Core
import Layer.Linear
import Array
import Train
import Util
import Executor
import Tensor
import BuildConfig

----------------------------------------------------------------------
-- Data (same 5-point classification task as Supervised)
----------------------------------------------------------------------

dataPoints : Vect 5 (DataPoint 2 3 Double)
dataPoints =
    [ MkDataPoint (VArray [1.5, -2.7]) (VArray [0, 1, 0]),
      MkDataPoint (VArray [-3.2, 4.1]) (VArray [0, 1, 0]),
      MkDataPoint (VArray [5.7, 0]) (VArray [0, 0, 1]),
      MkDataPoint (VArray [-1.3, 8.8]) (VArray [0, 1, 0]),
      MkDataPoint (VArray [2.9, -1.4]) (VArray [1, 0, 0])
    ]

-- Argmax on a 1D tensor (works on logits — softmax is monotonic)
evalPrediction : AnyPtr -> Nat
evalPrediction outT =
  let v0 = primItem1d {ex=ExampleExecutor} outT 0
      v1 = primItem1d {ex=ExampleExecutor} outT 1
      v2 = primItem1d {ex=ExampleExecutor} outT 2
  in if v0 >= v1 && v0 >= v2 then 0 else if v1 >= v2 then 1 else 2

-- Argmax on a one-hot Vector target.
evalPredictionTarget : Vector 3 Double -> Nat
evalPredictionTarget (VArray [SArray a, SArray b, SArray c]) =
  if a >= b && a >= c then 0 else if b >= c then 1 else 2

----------------------------------------------------------------------
-- Config
----------------------------------------------------------------------

record Config where
  constructor MkConfig
  mode : String
  epochs : Nat
  lr : Double
  seed : Bits64
  savePath : String
  loadPath : String
  lrFind : Bool

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
-- Eval helper
----------------------------------------------------------------------

-- Forward each datapoint, compute NLL loss as a Double, average.
evalModel : Network 2 [] 3 ExampleExecutor ExampleDType WithGrad -> IO Double
evalModel model = do
  losses <- traverse (\dp => do
        let inT = bulkToTensor {ex=ExampleExecutor} {dt=ExampleDType} (x dp)
            inV = the (TVec 2 ExampleExecutor ExampleDType WithGrad) (MkTensor inT Nothing)
        (_, predV) <- forwardVar model inV
        let tgtT = bulkToTensor {ex=ExampleExecutor} {dt=ExampleDType} (y dp)
            tgtV = the (TVec 3 ExampleExecutor ExampleDType WithGrad) (MkTensor tgtT Nothing)
        lossT <- tnllLoss predV tgtV
        pure (primItem {ex=ExampleExecutor} lossT.tensorPtr)) dataPoints
  pure (foldl (+) 0.0 (toList losses) / 5.0)

printPredictions : Network 2 [] 3 ExampleExecutor ExampleDType WithGrad -> IO ()
printPredictions model = do
  traverse_ (\dp => do
    let inT = bulkToTensor {ex=ExampleExecutor} {dt=ExampleDType} (x dp)
        inV = the (TVec 2 ExampleExecutor ExampleDType WithGrad) (MkTensor inT Nothing)
    (_, predV) <- forwardVar model inV
    let predClass = evalPrediction predV.tensorPtr
        targetClass = evalPredictionTarget (y dp)
        showVec : {k : Nat} -> Vector k Double -> String
        showVec (VArray xs) = "[" ++ go xs ++ "]"
          where go : Vect j (Scalar Double) -> String
                go [] = ""
                go [SArray v] = show v
                go (SArray v :: rest) = show v ++ ", " ++ go rest
    putStrLn $ "  " ++ showVec (x dp) ++ " -> class " ++ show predClass
                ++ (if targetClass == predClass then " ok" else " WRONG"))
    (toList dataPoints)

----------------------------------------------------------------------
-- Modes
----------------------------------------------------------------------

doTrain : Config -> Network 2 [] 3 ExampleExecutor ExampleDType WithGrad -> IO ()
doTrain cfg model = do
  let opt = nativeSgd cfg.lr
  putStrLn $ "Training " ++ show cfg.epochs ++ " epochs..."
  (trained, epochsDone, _) <- runTraining {ex=ExampleExecutor}
    (\m, d => epochVar opt d tnllLoss m) (pure dataPoints)
    (simpleConfig cfg.epochs) model
  if cfg.savePath == ""
    then putStrLn "No --save path given; skipping save"
    else do
      ok <- saveModel {ex=ExampleExecutor} cfg.savePath
      putStrLn $ (if ok then "Saved model to " else "FAILED to save model to ") ++ cfg.savePath
      ok2 <- saveOptimizer (optPath cfg.savePath) opt
      putStrLn $ (if ok2 then "Saved optimizer to " else "FAILED to save optimizer to ") ++ optPath cfg.savePath
  evalLoss <- withNoGrad {ex=ExampleExecutor} (evalModel trained)
  putStrLn $ "Eval loss: " ++ show evalLoss
  withNoGrad {ex=ExampleExecutor} (printPredictions trained)
  putStrLn $ formatResult [("mode", "train"), ("epochs", show epochsDone),
                            ("loss", show evalLoss), ("backend", backendName {ex=ExampleExecutor})]

doContinue : Config -> Network 2 [] 3 ExampleExecutor ExampleDType WithGrad -> IO ()
doContinue cfg model = do
  ok <- loadModel {ex=ExampleExecutor} cfg.loadPath
  putStrLn $ (if ok then "Loaded model from " else "FAILED to load from ") ++ cfg.loadPath
  let opt = nativeSgd cfg.lr
  ok2 <- loadOptimizer (optPath cfg.loadPath) opt
  putStrLn $ (if ok2 then "Loaded optimizer from " else "FAILED to load optimizer from ")
           ++ optPath cfg.loadPath
  putStrLn $ "Training " ++ show cfg.epochs ++ " more epochs..."
  (trained, epochsDone, _) <- runTraining {ex=ExampleExecutor}
    (\m, d => epochVar opt d tnllLoss m) (pure dataPoints)
    (simpleConfig cfg.epochs) model
  if cfg.savePath == ""
    then putStrLn "No --save path given; skipping save"
    else do
      ok3 <- saveModel {ex=ExampleExecutor} cfg.savePath
      putStrLn $ (if ok3 then "Saved model to " else "FAILED to save model to ") ++ cfg.savePath
      ok4 <- saveOptimizer (optPath cfg.savePath) opt
      putStrLn $ (if ok4 then "Saved optimizer to " else "FAILED to save optimizer to ") ++ optPath cfg.savePath
  evalLoss <- withNoGrad {ex=ExampleExecutor} (evalModel trained)
  putStrLn $ "Eval loss: " ++ show evalLoss
  withNoGrad {ex=ExampleExecutor} (printPredictions trained)
  putStrLn $ formatResult [("mode", "continue"), ("epochs", show epochsDone),
                            ("loss", show evalLoss), ("backend", backendName {ex=ExampleExecutor})]

doInfer : Config -> Network 2 [] 3 ExampleExecutor ExampleDType WithGrad -> IO ()
doInfer cfg model = do
  ok <- loadModel {ex=ExampleExecutor} cfg.loadPath
  putStrLn $ (if ok then "Loaded model from " else "FAILED to load from ") ++ cfg.loadPath
  evalLoss <- withNoGrad {ex=ExampleExecutor} (evalModel model)
  putStrLn $ "Eval loss: " ++ show evalLoss
  withNoGrad {ex=ExampleExecutor} (printPredictions model)
  putStrLn $ formatResult [("mode", "infer"), ("loss", show evalLoss),
                            ("backend", backendName {ex=ExampleExecutor})]

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

  llAny <- linearLayerAny {i=2} {o=3} "ll"
  let model : Network 2 [] 3 ExampleExecutor ExampleDType WithGrad
      model = OutputLayer llAny

  putStrLn $ "=== Cross-Backend Transfer [" ++ backendName {ex=ExampleExecutor} ++ "] -- "
           ++ cfg.mode ++ " ==="

  when cfg.lrFind $ do
    let opt = nativeSgd cfg.lr
    let lrCfg : LrFindConfig
        lrCfg = { numIters := 100 } defaultLrFindConfig
    _ <- lrFind lrCfg
      (\m, d => epochVar opt d tnllLoss m)
      (pure dataPoints) opt model
    putStrLn ""
    putStrLn "Done — re-run without --lr-find at the recommended LR."
    exitSuccess

  case cfg.mode of
    "train"    => doTrain cfg model
    "continue" => doContinue cfg model
    "infer"    => doInfer cfg model
    _ => putStrLn "Unknown mode. Use --mode train|continue|infer"
