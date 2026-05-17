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
evalPrediction : TVec 3 CPU F64 WithGrad -> Nat
evalPrediction outV =
  let v0 = prim__item1d outV.tensorPtr 0
      v1 = prim__item1d outV.tensorPtr 1
      v2 = prim__item1d outV.tensorPtr 2
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
  let model : Network 2 [] 3 CPU F64 WithGrad
      model = OutputLayer llAny
  putStrLn ""

  (trained, epochsDone, finalLoss) <- runTraining
    (\m, d => epochVar opt d tnllLoss m)
    (pure dataPoints)
    (simpleConfig cfg.epochs)
    model

  putStrLn ""
  putStrLn "Eval:"

  -- Build persistent input tensors and forward through the trained model.
  let inputs = the (Vect 5 AnyPtr) (map mkInputTensor dataPoints)
  traverse_ (\(idx, dp) => do
    let inV = the (TVec 2 CPU F64 WithGrad) (MkTensor (mkInputTensor dp) Nothing)
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
    mkInputTensor : DataPoint 2 3 Double -> AnyPtr
    mkInputTensor dp =
      let (VArray xs) = x dp
          buf = prim__allocDoubles 2
          buf' = packInto buf 0 xs
      in prim__createState1d 2 buf'
      where
        packInto : AnyPtr -> Int -> Vect k (Scalar Double) -> AnyPtr
        packInto b _ [] = b
        packInto b o (SArray v :: rest) =
          packInto (prim__setDouble b o v) (o + 1) rest

    evalPredictionTarget : Vector 3 Double -> Nat
    evalPredictionTarget (VArray [SArray a, SArray b, SArray c]) =
      if a >= b && a >= c then 0 else if b >= c then 1 else 2

    showVecD : Vector 2 Double -> String
    showVecD (VArray [SArray a, SArray b]) = "[" ++ show a ++ ", " ++ show b ++ "]"
