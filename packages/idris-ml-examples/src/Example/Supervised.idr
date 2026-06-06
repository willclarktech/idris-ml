module Example.Supervised

import Data.List
import Data.Vect
import System
import System.Clock
import Compat.Random

import Backprop
import DataPoint
import Executor
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
  ||| Mixed-precision parameter-storage mode (F5 of #410). Only
  ||| consulted when `mixedPrecision = True`. Accepted values:
  ||| - `"native"` (default): `paramDt = computeDt = ExampleDType`
  ||| - `"f32"`: `paramDt = F32`, `computeDt = ExampleDType`. On the
  ||| torch-mps BF16 build this is the actual F32-master /
  ||| BF16-compute decoupling that the autocast equivalent targets.
  paramDtype : String

defaultConfig : Config
defaultConfig = MkConfig 0.03 1000 42 False "native"

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
        , Arg "--mixed-precision" (\v, c => { mixedPrecision := boolFlag v } c)
        , Arg "--param-dtype" (\v, c => { paramDtype := v } c) ]


-- Argmax on a TVec (read three values via prim__item1d).
evalPrediction : TVec 3 ExampleExecutor ExampleDType WithGrad -> Nat
evalPrediction outV =
  let v0 = primItem1d {d=ExampleExecutor} outV.tensorPtr 0
      v1 = primItem1d {d=ExampleExecutor} outV.tensorPtr 1
      v2 = primItem1d {d=ExampleExecutor} outV.tensorPtr 2
  in if v0 >= v1 && v0 >= v2 then 0 else if v1 >= v2 then 1 else 2

%default partial

evalPredictionTarget : Vector 3 Double -> Nat
evalPredictionTarget (VArray [SArray a, SArray b, SArray c]) =
  if a >= b && a >= c then 0 else if b >= c then 1 else 2

showVecD : Vector 2 Double -> String
showVecD (VArray [SArray a, SArray b]) = "[" ++ show a ++ ", " ++ show b ++ "]"

-- Run the per-sample eval, return (printed lines, correct count).
evalOneDefault :
  Network 2 [] 3 ExampleExecutor ExampleDType WithGrad ->
  (Nat, DataPoint 2 3 Double) -> IO Nat
evalOneDefault trained (_, dp) = do
  let inV = the (TVec 2 ExampleExecutor ExampleDType WithGrad)
                (MkTensor (vectorToTensorPersistent {d=ExampleExecutor} {dt=ExampleDType} (x dp)) Nothing)
  (_, predV) <- forwardVar trained inV
  let predClass = evalPrediction predV
      targetClass = evalPredictionTarget (y dp)
      okFlag = targetClass == predClass
      ok = if okFlag then " ok" else " WRONG"
  putStrLn $ "  " ++ showVecD (x dp) ++ " -> class " ++ show predClass ++ ok
  pure (if okFlag then 1 else 0)

-- Default-precision path: builds a `Network`, trains via `epochVar`,
-- evals via `forwardVar`.
runDefault : Config -> NativeOptimizer ExampleExecutor -> IO ()
runDefault cfg opt = do
  llAny <- linearLayerAny {i = 2} {o = 3} "ll"
  let model : Network 2 [] 3 ExampleExecutor ExampleDType WithGrad
      model = OutputLayer llAny
  putStrLn ""

  (trained, epochsDone, finalLoss) <- runTraining {d=ExampleExecutor}
    (\m, d => epochVar opt d tnllLoss m)
    (pure dataPoints)
    (simpleConfig cfg.epochs)
    model

  putStrLn ""
  putStrLn "Eval:"

  let dpListV : Vect 5 (Fin 5, DataPoint 2 3 Double)
      dpListV = zip Fin.range dataPoints
      dpList : List (Nat, DataPoint 2 3 Double)
      dpList = toList (map (\(f, d) => (finToNat f, d)) dpListV)
  correctCounts <- traverse (evalOneDefault trained) dpList
  let correct = the Nat (sum correctCounts)

  putStrLn ""
  putStrLn $ formatResult [ ("epochs", show epochsDone)
                          , ("loss", show finalLoss)
                          , ("seed", show cfg.seed)
                          , ("correct", show correct ++ "/5") ]

-- Mixed-precision eval helper. Polymorphic over the param dtype
-- slot — the forward only sees computeDt at the value level, but
-- the typeclass dispatch needs the paramDt constraints to resolve.
evalOneMixed :
  {0 pDt : DType} ->
  RuntimeDType pDt => IsDType pDt =>
  Compatible ExampleExecutor pDt =>
  NetworkMixed 2 [] 3 ExampleExecutor pDt ExampleDType WithGrad ->
  (Nat, DataPoint 2 3 Double) -> IO Nat
evalOneMixed trained (_, dp) = do
  let inV = the (TVec 2 ExampleExecutor ExampleDType WithGrad)
                (MkTensor (vectorToTensorPersistent {d=ExampleExecutor} {dt=ExampleDType} (x dp)) Nothing)
  (_, predV) <- forwardVarMixed trained inV
  let predClass = evalPrediction predV
      targetClass = evalPredictionTarget (y dp)
      okFlag = targetClass == predClass
      ok = if okFlag then " ok" else " WRONG"
  putStrLn $ "  " ++ showVecD (x dp) ++ " -> class " ++ show predClass ++ ok
  pure (if okFlag then 1 else 0)

-- Mixed-precision train+eval pipeline, polymorphic over the param
-- dtype slot. The caller supplies the layer-construction action so
-- the paramDt is concretely pinned by the `mixedLinearLayerAny`
-- call site (Idris-2 can't dispatch types from a runtime string,
-- so each --param-dtype mode is its own typed sub-program below).
runMixedGeneric :
  {0 pDt : DType} ->
  RuntimeDType pDt => IsDType pDt =>
  Compatible ExampleExecutor pDt =>
  Config -> NativeOptimizer ExampleExecutor ->
  IO (AnyLayerMixed 2 3 ExampleExecutor pDt ExampleDType WithGrad) ->
  String ->
  IO ()
runMixedGeneric cfg opt mkLayer modeLabel = do
  llAny <- mkLayer
  let model : NetworkMixed 2 [] 3 ExampleExecutor pDt ExampleDType WithGrad
      model = OutputLayerMixed llAny
  gs <- defaultGradScaler {d=ExampleExecutor} {dt=ExampleDType}
  putStrLn modeLabel
  putStrLn ""

  (trained, epochsDone, finalLoss) <- runTraining {d=ExampleExecutor}
    (\m, d => epochVarMixed opt gs d tnllLoss m)
    (pure dataPoints)
    (simpleConfig cfg.epochs)
    model

  putStrLn ""
  putStrLn "Eval:"

  let dpListV : Vect 5 (Fin 5, DataPoint 2 3 Double)
      dpListV = zip Fin.range dataPoints
      dpList : List (Nat, DataPoint 2 3 Double)
      dpList = toList (map (\(f, d) => (finToNat f, d)) dpListV)
  correctCounts <- traverse (evalOneMixed trained) dpList
  let correct = the Nat (sum correctCounts)

  scaleAtEnd <- currentScale gs
  putStrLn ""
  putStrLn $ formatResult [ ("epochs", show epochsDone)
                          , ("loss", show finalLoss)
                          , ("seed", show cfg.seed)
                          , ("correct", show correct ++ "/5")
                          , ("final_scale", show scaleAtEnd) ]

-- F3 mixed mode (paramDt = computeDt = ExampleDType): the cast
-- inside LinearMixed.applyVarMixed is structurally a no-op on
-- builds where ExampleDType is the only dtype in play.
runMixedNative : Config -> NativeOptimizer ExampleExecutor -> IO ()
runMixedNative cfg opt =
  runMixedGeneric cfg opt
    (mixedLinearLayerAny {paramDt = ExampleDType} {computeDt = ExampleDType}
                         {i = 2} {o = 3} "ll")
    "Mixed-precision mode: paramDt = computeDt = ExampleDType (native)"

-- F5 mixed mode (paramDt = F32, computeDt = ExampleDType): the
-- F32-master / ExampleDType-compute decoupling. On the torch-mps
-- BF16 build this is the actual autocast-equivalent recipe — F32
-- weights, BF16 forward/backward via the autograd-aware tcast,
-- F32 grad accumulation, optimizer steps F32 directly.
runMixedF32Master : Config -> NativeOptimizer ExampleExecutor -> IO ()
runMixedF32Master cfg opt =
  runMixedGeneric cfg opt
    (mixedLinearLayerAny {paramDt = F32} {computeDt = ExampleDType}
                         {i = 2} {o = 3} "ll")
    "Mixed-precision mode: paramDt = F32, computeDt = ExampleDType (f32-master)"

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
           ++ " param-dtype=" ++ cfg.paramDtype

  if cfg.mixedPrecision
    then case cfg.paramDtype of
      "f32" => runMixedF32Master cfg opt
      _     => runMixedNative cfg opt
    else runDefault cfg opt
