module Example.Supervised

import Data.List
import Data.Vect
import System
import System.Clock
import Compat.Random

import Backprop
import DataPoint
import Device
import Endofunctor
import Floating
import Layer
import Math
import Optimizer
import Tensor
import Train
import Util
import Variable


-- f(x, y) = argmax(x - y - 10, -4x + y + 5, 2x + y - 11)
dataPoints : Vect 5 (DataPoint 2 3 Double)
dataPoints =
    [ MkDataPoint (VTensor [1.5, -2.7]) (VTensor [0, 1, 0]),
      MkDataPoint (VTensor [-3.2, 4.1]) (VTensor [0, 1, 0]),
      MkDataPoint (VTensor [5.7, 0]) (VTensor [0, 0, 1]),
      MkDataPoint (VTensor [-1.3, 8.8]) (VTensor [0, 1, 0]),
      MkDataPoint (VTensor [2.9, -1.4]) (VTensor [1, 0, 0])
    ]

-- Convert static DataPoint Double to persistent TensorDataPoint.
-- Uses prim__createState1d (persistent, survives tape resets) because
-- this data is reused across epochs via `pure tensorData`.
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

-- Simplest possible tensor loss: just sum the predictions (for debugging)
debugLoss : LossFnTensor CPU
debugLoss predT targetT =
  let loss = prim__sum predT
      val = prim__item loss
  in Var loss Nothing val

-- Tensor-level NLL loss: -sum(target * logSoftmax(logits)) / n
nllLossTensor : LossFnTensor CPU
nllLossTensor predT targetT =
  let logP = prim__logSoftmax predT 0
      product = prim__mul logP targetT
      totalSum = prim__sum product
      loss = prim__mulScalar (prim__neg totalSum) (1.0 / 3.0)
      val = prim__item loss
  in Var loss Nothing val

-- Argmax on a 1D tensor (works on logits — softmax is monotonic)
evalPrediction : AnyPtr -> Nat
evalPrediction outT =
  let v0 = prim__item1d outT 0
      v1 = prim__item1d outT 1
      v2 = prim__item1d outT 2
  in if v0 >= v1 && v0 >= v2 then 0 else if v1 >= v2 then 1 else 2

record Config where
  constructor MkConfig
  lr : Double
  epochs : Nat
  seed : Bits64

defaultConfig : Config
defaultConfig = MkConfig 0.03 1000 42

specs : List (ArgSpec Config)
specs = [ Arg "--lr" (\v, c => { lr := cast v } c)
        , Arg "--epochs" (\v, c => { epochs := castNat v } c)
        , Arg "--seed" (\v, c => { seed := castBits64 v } c) ]

main : IO ()
main = do
  args <- getArgs
  let cfg = parseArgs defaultConfig specs (drop 1 args)

  srand cfg.seed

  let opt = nativeSgd cfg.lr
  let tensorData = map toTensorDP dataPoints

  putStrLn "=== Supervised Classification ==="
  putStrLn $ "Config: lr=" ++ show cfg.lr ++ " epochs=" ++ show cfg.epochs
           ++ " seed=" ++ show cfg.seed

  ll <- linearLayer
  let model = autoName $ OutputLayer ll
  putStrLn $ "Architecture: " ++ show model
  putStrLn ""

  -- Quick forward test
  let (_, testOut) = forwardVarTensor model (inputTensor (index FZ tensorData))
  putStrLn $ "Forward test: " ++ show (prim__item (prim__sum testOut))

  (trained, epochsDone, _) <- runTraining
    (\m, d => epochNativeTensorPre opt d nllLossTensor m) (pure tensorData) (simpleConfig cfg.epochs) model

  -- Eval: create fresh tensor data (training reset freed arena)
  putStrLn ""
  putStrLn "Eval:"
  let freshEvalDPs = map toTensorDP dataPoints  -- fresh tensors after tape reset
      evalLosses = map (\dp =>
        let (_, outT) = forwardVarTensor trained (inputTensor dp)
            loss = nllLossTensor outT (targetTensor dp)
        in prim__item loss.tensorPtr) freshEvalDPs
      evalLoss = foldl (+) 0.0 (toList evalLosses) / 5.0
  putStrLn $ "  Loss: " ++ show evalLoss

  traverse_ (\(dp, orig) =>
    let (_, outT) = forwardVarTensor trained (inputTensor dp)
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
    (zip freshEvalDPs dataPoints)

  putStrLn ""
  putStrLn $ formatResult [("epochs", show epochsDone), ("loss", show evalLoss),
                            ("seed", show cfg.seed)]
