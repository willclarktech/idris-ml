module Example.Supervised

import Data.List
import Data.Vect
import System
import System.Clock
import Compat.Random

import Backprop
import DataPoint
import Device
import GradScaler
import Layer.Core
import Layer.Linear
import Layer.LinearMixed
import Layer.MixedCore
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
  mixedPrecision : Bool

defaultConfig : Config
defaultConfig = MkConfig 0.03 1000 42 False

-- Treat "1" or any "true"-prefix string as true; anything else (the
-- bare flag with no value, "0", "false") falls through to false. The
-- bare-flag idiom is `--mixed-precision true` because ArgSpec's
-- parser is value-taking.
boolFlag : String -> Bool
boolFlag v = v == "1" || v == "true" || v == "True" || v == "yes"

specs : List (ArgSpec Config)
specs = [ Arg "--lr" (\v, c => { lr := cast v } c)
        , Arg "--epochs" (\v, c => { epochs := castNat v } c)
        , Arg "--seed" (\v, c => { seed := castBits64 v } c)
        , Arg "--mixed-precision" (\v, c => { mixedPrecision := boolFlag v } c) ]


-- Argmax on a TVec (read three values via prim__item1d).
evalPrediction : TVec 3 ExampleDevice ExampleDType WithGrad -> Nat
evalPrediction outV =
  let v0 = primItem1d {d=ExampleDevice} outV.tensorPtr 0
      v1 = primItem1d {d=ExampleDevice} outV.tensorPtr 1
      v2 = primItem1d {d=ExampleDevice} outV.tensorPtr 2
  in if v0 >= v1 && v0 >= v2 then 0 else if v1 >= v2 then 1 else 2

%default partial

evalPredictionTarget : Vector 3 Double -> Nat
evalPredictionTarget (VArray [SArray a, SArray b, SArray c]) =
  if a >= b && a >= c then 0 else if b >= c then 1 else 2

showVecD : Vector 2 Double -> String
showVecD (VArray [SArray a, SArray b]) = "[" ++ show a ++ ", " ++ show b ++ "]"

-- Default-precision path: builds a `Network`, trains via `epochVar`,
-- evals via `forwardVar`.
runDefault : Config -> NativeOptimizer ExampleDevice -> IO ()
runDefault cfg opt = do
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

-- Mixed-precision path (F3 of #410): builds a `NetworkMixed`
-- (paramDt = computeDt = ExampleDType so the cast is a structural
-- no-op on default builds; users wanting an actual lossy cast
-- supply a different `MLX_DTYPE` / `TORCH_DTYPE` for ExampleDType),
-- a `defaultGradScaler`, trains via `epochVarMixed`, evals via
-- `forwardVarMixed`.
runMixed : Config -> NativeOptimizer ExampleDevice -> IO ()
runMixed cfg opt = do
  llAny <- mixedLinearLayerAny {paramDt = ExampleDType} {computeDt = ExampleDType}
                               {i = 2} {o = 3} "ll"
  let model : NetworkMixed 2 [] 3 ExampleDevice ExampleDType ExampleDType WithGrad
      model = OutputLayerMixed llAny
  gs <- defaultGradScaler {d=ExampleDevice} {dt=ExampleDType}
  putStrLn "Mixed-precision mode: paramDt = computeDt = ExampleDType"
  putStrLn ""

  (trained, epochsDone, finalLoss) <- runTraining {d=ExampleDevice}
    (\m, d => epochVarMixed opt gs d tnllLoss m)
    (pure dataPoints)
    (simpleConfig cfg.epochs)
    model

  putStrLn ""
  putStrLn "Eval:"

  traverse_ (\(idx, dp) => do
    let inV = the (TVec 2 ExampleDevice ExampleDType WithGrad)
                  (MkTensor (vectorToTensorPersistent {d=ExampleDevice} {dt=ExampleDType} (x dp)) Nothing)
    (_, predV) <- forwardVarMixed trained inV
    let predClass = evalPrediction predV
        targetClass = evalPredictionTarget (y dp)
        ok = if targetClass == predClass then " ok" else " WRONG"
    putStrLn $ "  " ++ showVecD (x dp) ++ " -> class " ++ show predClass ++ ok)
    (zip Fin.range dataPoints)

  scaleAtEnd <- currentScale gs
  putStrLn ""
  putStrLn $ formatResult [ ("epochs", show epochsDone)
                          , ("loss", show finalLoss)
                          , ("seed", show cfg.seed)
                          , ("final_scale", show scaleAtEnd) ]

main : IO ()
main = do
  args <- getArgs
  let cfg = parseArgs defaultConfig specs (drop 1 args)

  srand cfg.seed

  let opt = nativeSgd cfg.lr

  putStrLn "=== Supervised Classification ==="
  putStrLn $ "Config: lr=" ++ show cfg.lr ++ " epochs=" ++ show cfg.epochs
           ++ " seed=" ++ show cfg.seed
           ++ " mixed-precision=" ++ show cfg.mixedPrecision

  if cfg.mixedPrecision then runMixed cfg opt else runDefault cfg opt
