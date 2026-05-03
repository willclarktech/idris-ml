module Example.Supervised

import Data.List
import Data.Vect
import System
import System.Clock
import Compat.Random

import Backprop
import DataPoint
import Device
import Layer.Core
import Layer.Linear
import Array
import Train
import Util
import Tensor
import BuildConfig


-- f(x, y) = argmax(x - y - 10, -4x + y + 5, 2x + y - 11)
dataPoints : Vect 5 (DataPoint 2 3 Double)
dataPoints =
    [ MkDataPoint (VArray [1.5, -2.7]) (VArray [0, 1, 0]),
      MkDataPoint (VArray [-3.2, 4.1]) (VArray [0, 1, 0]),
      MkDataPoint (VArray [5.7, 0]) (VArray [0, 0, 1]),
      MkDataPoint (VArray [-1.3, 8.8]) (VArray [0, 1, 0]),
      MkDataPoint (VArray [2.9, -1.4]) (VArray [1, 0, 0])
    ]


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


-- Argmax on a TVec (read three values via prim__item1d).
evalPrediction : TVec 3 ExampleDevice ExampleDType WithGrad -> Nat
evalPrediction outV =
  let v0 = primItem1d {d=ExampleDevice} outV.tensorPtr 0
      v1 = primItem1d {d=ExampleDevice} outV.tensorPtr 1
      v2 = primItem1d {d=ExampleDevice} outV.tensorPtr 2
  in if v0 >= v1 && v0 >= v2 then 0 else if v1 >= v2 then 1 else 2

%default partial

main : IO ()
main = do
  args <- getArgs
  let cfg = parseArgs defaultConfig specs (drop 1 args)

  srand cfg.seed

  let opt = nativeSgd cfg.lr

  putStrLn "=== Supervised Classification ==="
  putStrLn $ "Config: lr=" ++ show cfg.lr ++ " epochs=" ++ show cfg.epochs
           ++ " seed=" ++ show cfg.seed

  llAny <- linearLayerAny {i = 2} {o = 3} "ll"
  let model : Network 2 [] 3 ExampleDevice ExampleDType WithGrad
      model = OutputLayer llAny
  putStrLn ""

  (trained, epochsDone, finalLoss) <- runTraining {d=ExampleDevice}
    (\m, d => epochVar opt d tnllLoss m)
    (pure dataPoints)
    (simpleConfig cfg.epochs)
    model

  putStrLn ""
  putStrLn "Eval:"

  -- Build persistent input tensors and forward through the trained model.
  -- Use the dtype-aware constructor (same path training tensorizes through)
  -- so the input matches ExampleDType — a raw F64 creator would crash on an
  -- F32-only device (MPS rejects an F64 tensor at construction).
  traverse_ (\(idx, dp) => do
    let inV = the (TVec 2 ExampleDevice ExampleDType WithGrad)
                  (MkTensor (vectorToTensorPersistent {d=ExampleDevice} {dt=ExampleDType} (x dp)) Nothing)
    (_, predV) <- forwardVar trained inV
    let predClass = evalPrediction predV
        targetClass = evalPredictionTarget (y dp)
        ok = if targetClass == predClass then " ok" else " WRONG"
    putStrLn $ "  " ++ showVecD (x dp) ++ " -> class " ++ show predClass ++ ok)
    (zip Fin.range dataPoints)

  putStrLn ""
  putStrLn $ formatResult [ ("epochs", show epochsDone)
                          , ("loss", show finalLoss)
                          , ("seed", show cfg.seed) ]
  where
    evalPredictionTarget : Vector 3 Double -> Nat
    evalPredictionTarget (VArray [SArray a, SArray b, SArray c]) =
      if a >= b && a >= c then 0 else if b >= c then 1 else 2

    showVecD : Vector 2 Double -> String
    showVecD (VArray [SArray a, SArray b]) = "[" ++ show a ++ ", " ++ show b ++ "]"
