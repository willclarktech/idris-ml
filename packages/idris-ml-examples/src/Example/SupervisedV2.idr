module Example.SupervisedV2

import Data.List
import Data.Vect
import System
import System.Clock
import Compat.Random

import DataPoint
import Device
import Layer.CoreV2
import Layer.LinearV2
import Tensor
import Train
import Util
import Variable


-- Path C P3-1 spike: TVar end-to-end counterpart of Example.Supervised.
-- Identical data, identical hyperparameters — bench against the scalar
-- baseline. Single LinearV2 (2 -> 3), NLL loss against a one-hot target.

-- f(x, y) = argmax(x - y - 10, -4x + y + 5, 2x + y - 11)
-- Same five hand-built points as Example.Supervised.
dataPoints : Vect 5 (DataPoint 2 3 Double)
dataPoints =
  [ MkDataPoint (VTensor [1.5, -2.7]) (VTensor [0, 1, 0])
  , MkDataPoint (VTensor [-3.2, 4.1]) (VTensor [0, 1, 0])
  , MkDataPoint (VTensor [5.7, 0]) (VTensor [0, 0, 1])
  , MkDataPoint (VTensor [-1.3, 8.8]) (VTensor [0, 1, 0])
  , MkDataPoint (VTensor [2.9, -1.4]) (VTensor [1, 0, 0])
  ]


----------------------------------------------------------------------
-- Persistent tensor wrappers
----------------------------------------------------------------------

-- Pack a Vector n Double into a persistent C tensor (survives tape resets).
-- Mirrors Example.Supervised.bulkToTensorPersistent.
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

-- Pair of (input TVar [2], target TVar [3]) per data point.
record TVarDataPoint where
  constructor MkTVarDP
  inputTV  : TVar [2] CPU
  targetTV : TVar [3] CPU

toTVarDP : DataPoint 2 3 Double -> TVarDataPoint
toTVarDP dp =
  MkTVarDP (tinput1d (bulkToTensorPersistent (x dp)))
           (tinput1d (bulkToTensorPersistent (y dp)))


----------------------------------------------------------------------
-- Per-epoch loop
----------------------------------------------------------------------

-- Run one epoch over all data points and return mean loss.
-- Each point: forward + NLL + train step (zero_grad → backward → step).
-- Mirrors the structure of Example.Supervised but threads TVar end-to-end.
runEpoch : NativeOptimizer -> LinearStateV2 2 3 CPU -> List TVarDataPoint -> IO Double
runEpoch opt model dps = do
  losses <- traverse (\dp => do
    let (_, predTV) = applyTVar model (inputTV dp)
        loss = tnllLoss predTV (targetTV dp)
        l = nativeTrainStepTVar opt loss
    pure l) dps
  pure (sum losses / cast (length losses))


----------------------------------------------------------------------
-- CLI config
----------------------------------------------------------------------

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
        , Arg "--seed" (\v, c => { seed := castBits64 v } c)
        ]


----------------------------------------------------------------------
-- Main
----------------------------------------------------------------------

trainLoop : NativeOptimizer -> LinearStateV2 2 3 CPU -> List TVarDataPoint ->
            (n : Nat) -> Nat -> IO Double
trainLoop _ _ _ Z _ = pure 0.0
trainLoop opt model dps (S k) totalEp = do
  loss <- runEpoch opt model dps
  let epochNum = totalEp `minus` k
  when (mod epochNum 100 == 0) $
    putStrLn $ "  epoch " ++ show epochNum ++ "  loss=" ++ show loss
  if k == 0 then pure loss else trainLoop opt model dps k totalEp

main : IO ()
main = do
  args <- getArgs
  let cfg = parseArgs defaultConfig specs (drop 1 args)

  srand cfg.seed

  putStrLn "=== SupervisedV2 (Path C TVar spike) ==="
  putStrLn $ "Config: lr=" ++ show cfg.lr ++ " epochs=" ++ show cfg.epochs
           ++ " seed=" ++ show cfg.seed

  let opt = nativeSgd cfg.lr
  model <- linearLayerV2 {i = 2} {o = 3} "v2_ll0"
  let tDPs = map toTVarDP dataPoints

  start <- clockTime Process
  finalLoss <- trainLoop opt model (toList tDPs) cfg.epochs cfg.epochs
  end <- clockTime Process
  let elapsedNs = timeDifference end start
      elapsedMs : Double
      elapsedMs = cast (toNano elapsedNs) / 1.0e6
      msPerEpoch : Double
      msPerEpoch = elapsedMs / cast cfg.epochs

  putStrLn ""
  putStrLn $ "  Final loss: " ++ show finalLoss
  putStrLn $ "  Total: " ++ show elapsedMs ++ " ms  (" ++ show msPerEpoch ++ " ms/epoch)"
  putStrLn ""
  putStrLn $ formatResult [ ("epochs", show cfg.epochs)
                          , ("loss", show finalLoss)
                          , ("ms_per_epoch", show msPerEpoch)
                          , ("seed", show cfg.seed)
                          ]
