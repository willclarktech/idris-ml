module Example.Transfer

-- Cross-backend transfer demo: train → save → load → continue → save → infer.
-- Run 3 times with different BACKEND= to prove SafeTensors portability.

import Data.List
import Data.Vect
import System
import System.Random

import Backprop
import Checkpoint
import DataPoint
import Endofunctor
import Layer
import Math
import Optimizer
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

bulkToTensorPersistent : {n : Nat} -> Vector n Double -> AnyPtr
bulkToTensorPersistent {n} (VTensor elems) =
  let nI = cast {to=Int} n
      buf = prim__allocDoubles nI
      buf' = packDoubleBuf buf 0 elems
  in prim__createState1d nI buf'
  where
    packDoubleBuf : AnyPtr -> Int -> Vect k (Scalar Double) -> AnyPtr
    packDoubleBuf buf _ [] = buf
    packDoubleBuf buf off (STensor v :: rest) =
      let buf' = prim__setDouble buf off v
      in packDoubleBuf buf' (off + 1) rest

toTensorDP : DataPoint 2 3 Double -> TensorDataPoint 2 3
toTensorDP dp = MkTensorDataPoint (bulkToTensorPersistent (x dp)) (bulkToTensorPersistent (y dp))

nllLossTensor : LossFnTensor CPU
nllLossTensor predT targetT =
  let logP = prim__logSoftmax predT 0
      product = prim__mul logP targetT
      totalSum = prim__sum product
      loss = prim__mulScalar (prim__neg totalSum) (1.0 / 3.0)
      val = prim__item loss
  in Var loss Nothing val

evalPrediction : AnyPtr -> Nat
evalPrediction outT =
  let v0 = prim__item1d outT 0
      v1 = prim__item1d outT 1
      v2 = prim__item1d outT 2
  in if v0 >= v1 && v0 >= v2 then 0 else if v1 >= v2 then 1 else 2

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

defaultConfig : Config
defaultConfig = MkConfig "train" 500 0.03 123456 "" ""

specs : List (ArgSpec Config)
specs = [ Arg "--mode" (\v, c => { mode := v } c)
        , Arg "--epochs" (\v, c => { epochs := castNat v } c)
        , Arg "--lr" (\v, c => { lr := cast v } c)
        , Arg "--seed" (\v, c => { seed := castBits64 v } c)
        , Arg "--save" (\v, c => { savePath := v } c)
        , Arg "--load" (\v, c => { loadPath := v } c) ]

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

evalModel : Network 2 [] 3 (Variable CPU) -> IO Double
evalModel model = do
  let freshDPs = map toTensorDP dataPoints
      losses = map (\dp =>
        let (_, outT) = forwardVarTensor model (inputTensor dp)
            loss = nllLossTensor outT (targetTensor dp)
        in prim__item loss.tensorPtr) freshDPs
  pure (foldl (+) 0.0 (toList losses) / 5.0)

printPredictions : Network 2 [] 3 (Variable CPU) -> IO ()
printPredictions model = do
  let freshDPs = map toTensorDP dataPoints
  traverse_ (\(dp, orig) =>
    let (_, outT) = forwardVarTensor model (inputTensor dp)
        predClass = evalPrediction outT
        targetClass = evalPrediction (targetTensor dp)
        showVec : {k : Nat} -> Vector k Double -> String
        showVec (VTensor xs) = "[" ++ go xs ++ "]"
          where go : Vect j (Scalar Double) -> String
                go [] = ""
                go [STensor v] = show v
                go (STensor v :: rest) = show v ++ ", " ++ go rest
    in putStrLn $ "  " ++ showVec (x orig) ++ " -> class " ++ show predClass
                ++ (if targetClass == predClass then " ok" else " WRONG"))
    (zip freshDPs dataPoints)

----------------------------------------------------------------------
-- Modes
----------------------------------------------------------------------

doTrain : Config -> Network 2 [] 3 (Variable CPU) -> IO ()
doTrain cfg model = do
  let opt = nativeSgd cfg.lr
      tensorData = map toTensorDP dataPoints
  putStrLn $ "Training " ++ show cfg.epochs ++ " epochs..."
  (trained, epochsDone, _) <- runTraining
    (\m, d => epochNativeTensorPre opt d nllLossTensor m) (pure tensorData)
    (simpleConfig cfg.epochs) model
  -- Save
  ok <- saveModel cfg.savePath
  putStrLn $ (if ok then "Saved model to " else "FAILED to save model to ") ++ cfg.savePath
  ok2 <- saveOptimizer (optPath cfg.savePath) opt
  putStrLn $ (if ok2 then "Saved optimizer to " else "FAILED to save optimizer to ") ++ optPath cfg.savePath
  -- Eval
  evalLoss <- evalModel trained
  putStrLn $ "Eval loss: " ++ show evalLoss
  printPredictions trained
  putStrLn $ formatResult [("mode", "train"), ("epochs", show epochsDone),
                            ("loss", show evalLoss), ("backend", backendName)]

doContinue : Config -> Network 2 [] 3 (Variable CPU) -> IO ()
doContinue cfg model = do
  -- Load model
  ok <- loadModel cfg.loadPath
  putStrLn $ (if ok then "Loaded model from " else "FAILED to load from ") ++ cfg.loadPath
  let model' = emap refreshValue model
  -- Load optimizer
  let opt = nativeSgd cfg.lr
  ok2 <- loadOptimizer (optPath cfg.loadPath) opt
  putStrLn $ (if ok2 then "Loaded optimizer from " else "FAILED to load optimizer from ")
           ++ optPath cfg.loadPath
  -- Continue training
  let tensorData = map toTensorDP dataPoints
  putStrLn $ "Training " ++ show cfg.epochs ++ " more epochs..."
  (trained, epochsDone, _) <- runTraining
    (\m, d => epochNativeTensorPre opt d nllLossTensor m) (pure tensorData)
    (simpleConfig cfg.epochs) model'
  -- Save
  ok3 <- saveModel cfg.savePath
  putStrLn $ (if ok3 then "Saved model to " else "FAILED to save model to ") ++ cfg.savePath
  ok4 <- saveOptimizer (optPath cfg.savePath) opt
  putStrLn $ (if ok4 then "Saved optimizer to " else "FAILED to save optimizer to ") ++ optPath cfg.savePath
  -- Eval
  evalLoss <- evalModel trained
  putStrLn $ "Eval loss: " ++ show evalLoss
  printPredictions trained
  putStrLn $ formatResult [("mode", "continue"), ("epochs", show epochsDone),
                            ("loss", show evalLoss), ("backend", backendName)]

doInfer : Config -> Network 2 [] 3 (Variable CPU) -> IO ()
doInfer cfg model = do
  ok <- loadModel cfg.loadPath
  putStrLn $ (if ok then "Loaded model from " else "FAILED to load from ") ++ cfg.loadPath
  let model' = emap refreshValue model
  evalLoss <- evalModel model'
  putStrLn $ "Eval loss: " ++ show evalLoss
  printPredictions model'
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

  ll <- linearLayer
  let model = autoName $ OutputLayer ll

  putStrLn $ "=== Cross-Backend Transfer [" ++ backendName ++ "] -- "
           ++ cfg.mode ++ " ==="

  case cfg.mode of
    "train"    => doTrain cfg model
    "continue" => doContinue cfg model
    "infer"    => doInfer cfg model
    _ => putStrLn "Unknown mode. Use --mode train|continue|infer"
