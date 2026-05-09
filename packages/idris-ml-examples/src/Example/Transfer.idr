module Example.Transfer

-- Cross-backend transfer demo: train → save → load → continue → save → infer.
-- Run 3 times with different BACKEND= to prove SafeTensors portability.

import Data.List
import Data.Vect
import System
import Compat.Random

import BackpropV2
import Checkpoint
import DataPoint
import Hpo.LrFinder
import Layer.CoreV2
import Layer.LinearV2
import Tensor
import Train
import Util
import Device
import Variable

----------------------------------------------------------------------
-- Data (same 5-point classification task as Supervised)
----------------------------------------------------------------------

dataPoints : Vect 5 (DataPoint 2 3 Double)
dataPoints =
    [ MkDataPoint (VTensor [1.5, -2.7]) (VTensor [0, 1, 0]),
      MkDataPoint (VTensor [-3.2, 4.1]) (VTensor [0, 1, 0]),
      MkDataPoint (VTensor [5.7, 0]) (VTensor [0, 0, 1]),
      MkDataPoint (VTensor [-1.3, 8.8]) (VTensor [0, 1, 0]),
      MkDataPoint (VTensor [2.9, -1.4]) (VTensor [1, 0, 0])
    ]

-- Argmax on a 1D tensor (works on logits — softmax is monotonic)
evalPrediction : AnyPtr -> Nat
evalPrediction outT =
  let v0 = prim__item1d outT 0
      v1 = prim__item1d outT 1
      v2 = prim__item1d outT 2
  in if v0 >= v1 && v0 >= v2 then 0 else if v1 >= v2 then 1 else 2

-- Argmax on a one-hot Vector target.
evalPredictionTarget : Vector 3 Double -> Nat
evalPredictionTarget (VTensor [STensor a, STensor b, STensor c]) =
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
evalModel : NetworkV2 2 [] 3 CPU -> IO Double
evalModel model = do
  let losses = map (\dp =>
        let inT = bulkToTensor (x dp)
            inV = the (TVec 2 CPU) (MkTVar inT Nothing)
            (_, predV) = forwardTVar model inV
            tgtT = bulkToTensor (y dp)
            tgtV = the (TVec 3 CPU) (MkTVar tgtT Nothing)
            lossT = tnllLoss predV tgtV
        in prim__item lossT.tensorPtr) dataPoints
  pure (foldl (+) 0.0 (toList losses) / 5.0)

printPredictions : NetworkV2 2 [] 3 CPU -> IO ()
printPredictions model = do
  traverse_ (\dp =>
    let inT = bulkToTensor (x dp)
        inV = the (TVec 2 CPU) (MkTVar inT Nothing)
        (_, predV) = forwardTVar model inV
        predClass = evalPrediction predV.tensorPtr
        targetClass = evalPredictionTarget (y dp)
        showVec : {k : Nat} -> Vector k Double -> String
        showVec (VTensor xs) = "[" ++ go xs ++ "]"
          where go : Vect j (Scalar Double) -> String
                go [] = ""
                go [STensor v] = show v
                go (STensor v :: rest) = show v ++ ", " ++ go rest
    in putStrLn $ "  " ++ showVec (x dp) ++ " -> class " ++ show predClass
                ++ (if targetClass == predClass then " ok" else " WRONG"))
    (toList dataPoints)


----------------------------------------------------------------------
-- Modes
----------------------------------------------------------------------

doTrain : Config -> NetworkV2 2 [] 3 CPU -> IO ()
doTrain cfg model = do
  let opt = nativeSgd cfg.lr
  putStrLn $ "Training " ++ show cfg.epochs ++ " epochs..."
  (trained, epochsDone, _) <- runTraining
    (\m, d => epochTVar opt d tnllLoss m) (pure dataPoints)
    (simpleConfig cfg.epochs) model
  if cfg.savePath == ""
    then putStrLn "No --save path given; skipping save"
    else do
      ok <- saveModel cfg.savePath
      putStrLn $ (if ok then "Saved model to " else "FAILED to save model to ") ++ cfg.savePath
      ok2 <- saveOptimizer (optPath cfg.savePath) opt
      putStrLn $ (if ok2 then "Saved optimizer to " else "FAILED to save optimizer to ") ++ optPath cfg.savePath
  evalLoss <- evalModel trained
  putStrLn $ "Eval loss: " ++ show evalLoss
  printPredictions trained
  putStrLn $ formatResult [("mode", "train"), ("epochs", show epochsDone),
                            ("loss", show evalLoss), ("backend", backendName)]

doContinue : Config -> NetworkV2 2 [] 3 CPU -> IO ()
doContinue cfg model = do
  ok <- loadModel cfg.loadPath
  putStrLn $ (if ok then "Loaded model from " else "FAILED to load from ") ++ cfg.loadPath
  let opt = nativeSgd cfg.lr
  ok2 <- loadOptimizer (optPath cfg.loadPath) opt
  putStrLn $ (if ok2 then "Loaded optimizer from " else "FAILED to load optimizer from ")
           ++ optPath cfg.loadPath
  putStrLn $ "Training " ++ show cfg.epochs ++ " more epochs..."
  (trained, epochsDone, _) <- runTraining
    (\m, d => epochTVar opt d tnllLoss m) (pure dataPoints)
    (simpleConfig cfg.epochs) model
  if cfg.savePath == ""
    then putStrLn "No --save path given; skipping save"
    else do
      ok3 <- saveModel cfg.savePath
      putStrLn $ (if ok3 then "Saved model to " else "FAILED to save model to ") ++ cfg.savePath
      ok4 <- saveOptimizer (optPath cfg.savePath) opt
      putStrLn $ (if ok4 then "Saved optimizer to " else "FAILED to save optimizer to ") ++ optPath cfg.savePath
  evalLoss <- evalModel trained
  putStrLn $ "Eval loss: " ++ show evalLoss
  printPredictions trained
  putStrLn $ formatResult [("mode", "continue"), ("epochs", show epochsDone),
                            ("loss", show evalLoss), ("backend", backendName)]

doInfer : Config -> NetworkV2 2 [] 3 CPU -> IO ()
doInfer cfg model = do
  ok <- loadModel cfg.loadPath
  putStrLn $ (if ok then "Loaded model from " else "FAILED to load from ") ++ cfg.loadPath
  evalLoss <- evalModel model
  putStrLn $ "Eval loss: " ++ show evalLoss
  printPredictions model
  putStrLn $ formatResult [("mode", "infer"), ("loss", show evalLoss),
                            ("backend", backendName)]


----------------------------------------------------------------------
-- Main
----------------------------------------------------------------------

main : IO ()
main = do
  args <- getArgs
  let cfg = parseArgs defaultConfig specs (drop 1 args)
  srand cfg.seed

  llAny <- linearLayerV2Any {i=2} {o=3} "ll"
  let model : NetworkV2 2 [] 3 CPU
      model = OutputLayerV2 llAny

  putStrLn $ "=== Cross-Backend Transfer [" ++ backendName ++ "] -- "
           ++ cfg.mode ++ " ==="

  when cfg.lrFind $ do
    let opt = nativeSgd cfg.lr
    let lrCfg : LrFindConfig
        lrCfg = { numIters := 100 } defaultLrFindConfig
    _ <- lrFind lrCfg
      (\m, d => let (m', loss) = epochTVar opt d tnllLoss m
                in pure (m', loss))
      (pure dataPoints) opt model
    putStrLn ""
    putStrLn "Done — re-run without --lr-find at the recommended LR."
    exitSuccess

  case cfg.mode of
    "train"    => doTrain cfg model
    "continue" => doContinue cfg model
    "infer"    => doInfer cfg model
    _ => putStrLn "Unknown mode. Use --mode train|continue|infer"
