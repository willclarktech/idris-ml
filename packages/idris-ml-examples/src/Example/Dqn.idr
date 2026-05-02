module Example.Dqn

import Data.List
import Data.Vect
import Data.Fin
import Data.IORef
import System
import Compat.Random

import Endofunctor
import Floating
import Gym.ClassicControl.CartPole
import Gym.Env
import Hpo.LrFinder
import Layer
import Layer.Core
import Math
import RL.ReplayBuffer
import Tensor
import Train
import Util
import Device
import Variable


----------------------------------------------------------------------
-- Architecture: MLP 4 -> 64 -> 64 -> 2
----------------------------------------------------------------------

ObsDim : Nat; ObsDim = 4
Hidden : Nat; Hidden = 64
NumActions : Nat; NumActions = 2
MaxSteps : Nat; MaxSteps = cartPoleMaxSteps

-- Intermediate shape after `linear ~> relu` is [64, 64, 64, 64] after two
-- such blocks followed by an output layer (Linear 64 -> 2).
QNet : Type
QNet = Network ObsDim [Hidden, Hidden, Hidden, Hidden] NumActions (Variable CPU)

QNetDouble : Type
QNetDouble = Network ObsDim [Hidden, Hidden, Hidden, Hidden] NumActions Double

mkQNet : IO QNet
mkQNet = do
  ll1 <- linearLayer {i=ObsDim} {o=Hidden}
  ll2 <- linearLayer {i=Hidden} {o=Hidden}
  ll3 <- linearLayer {i=Hidden} {o=NumActions}
  pure (autoName (ll1 ~> reluLayer ~> ll2 ~> reluLayer ~> OutputLayer ll3))


----------------------------------------------------------------------
-- Observation helper
----------------------------------------------------------------------

observeVec : CPState -> Vect ObsDim Double
observeVec s = cpObserve s

-- Convert a plain Vect to the shape-indexed Vector (Tensor [n]) used by
-- the forward / bulkToTensor APIs.
obsTensor : Vect ObsDim Double -> Vector ObsDim Double
obsTensor v = VTensor (map STensor v)


----------------------------------------------------------------------
-- Target network snapshot: frozen Double-valued copy of the online net.
-- Sync = refresh online weights, then copy to Double.
----------------------------------------------------------------------

snapshotTarget : QNet -> QNetDouble
snapshotTarget online = toDoubleNetwork (emap refreshValue online)

-- Max over a Vector NumActions Double
vectorMax : Vector NumActions Double -> Double
vectorMax qr =
  let STensor v = index (argmax qr) qr
  in v


----------------------------------------------------------------------
-- Epsilon-greedy action selection (uses online net's current weights)
----------------------------------------------------------------------

epsilonAt : Nat -> Double -> Double -> Nat -> Double
epsilonAt step start end decaySteps =
  let frac = min (cast (natToInteger step) / cast (natToInteger decaySteps)) 1.0
  in start + frac * (end - start)

-- Argmax over Q(s, *) from the online net.
greedyAction : QNet -> Vect ObsDim Double -> Nat
greedyAction online obs =
  let stateT = bulkToTensor (obsTensor obs)
      qT = snd (forwardVarTensor online stateT)
      q0 = prim__item1d qT 0
      q1 = prim__item1d qT 1
  in if q0 >= q1 then 0 else 1

epsGreedyIO : QNet -> Vect ObsDim Double -> Double -> IO Nat
epsGreedyIO online obs eps = do
  u <- randomRIO (the Double 0.0, 1.0)
  if u < eps
    then do
      u2 <- randomRIO (the Double 0.0, 1.0)
      pure (if u2 < 0.5 then 0 else 1)
    else pure (greedyAction online obs)


----------------------------------------------------------------------
-- DQN loss (batched). Target Q uses the Double snapshot (pure Idris,
-- no FFI per scalar — kept per-sample). Online Q is batched: one
-- [B, ObsDim] forward replaces B per-sample forwards. For 200-step
-- episodes × batch=64, this is ~64× fewer tape entries per train step
-- (B=64 forward calls collapse to 1).
----------------------------------------------------------------------

computeTargetVal : QNetDouble -> Double -> Transition ObsDim 1 -> Double
computeTargetVal tgt gamma t =
  let qNextD = snd (forward tgt (obsTensor t.nextObs))
      nextMaxVal = vectorMax qNextD
      bootstrap  = if t.done then 0.0 else gamma * nextMaxVal
  in t.reward + bootstrap

actionIdx : Vect 1 Double -> Int
actionIdx [a] = cast {to=Int} (cast {to=Integer} a)

-- Build per-sample (Q(s,a) - target)^2 Variables by indexing into the
-- [B, NumActions] online output. Mirrors SAC's qLossBatch pattern
-- (Example/Sac.idr): select row k, then column aIdx[k], wrap as Var to
-- preserve autograd, subtract Double target, square.
perSampleLosses : (qOutB : AnyPtr) -> Vect k (Transition ObsDim 1) ->
                  Vect k Double -> Int -> List (Variable CPU)
perSampleLosses _ [] [] _ = []
perSampleLosses qOutB (t :: tRest) (tv :: tvRest) k =
  let aIdx    = actionIdx t.action
      qRow    = prim__select qOutB 0 k
      qVal    = prim__item1d qRow aIdx
      qV      : Variable CPU
      qV      = Var (prim__select qRow 0 aIdx) Nothing qVal
      targetC : Variable CPU
      targetC = fromDouble tv
      diff    = qV - targetC
  in (diff * diff) :: perSampleLosses qOutB tRest tvRest (k + 1)

batchLossBatched : (n : Nat) -> QNet -> QNetDouble -> Double ->
                   Vect n (Transition ObsDim 1) -> Variable CPU
batchLossBatched n online target gamma batch =
  let targetVals : Vect n Double
      targetVals = map (computeTargetVal target gamma) batch
      obsTensors : Vect n (Vector ObsDim Double)
      obsTensors = map (\t => obsTensor t.obs) batch
      obsBT      = bulkToTensor2d obsTensors                         -- [B, ObsDim]
      qOutB      = snd (forwardVarTensorBatch online n obsBT)         -- [B, NumActions]
      losses     = perSampleLosses qOutB batch targetVals 0
      n_d        = the Double (cast (natToInteger n))
      sumV       = foldl (+) (the (Variable CPU) (fromDouble 0.0)) losses
      nV         = the (Variable CPU) (fromDouble n_d)
  in sumV / nV


----------------------------------------------------------------------
-- DQN state threaded through training
----------------------------------------------------------------------

record DqnState where
  constructor MkDqnState
  qNet      : QNet
  target    : QNetDouble
  buffer    : ReplayBuffer ObsDim 1
  stepRef   : IORef Nat     -- total env steps (for epsilon decay + target sync)
  cfgEpsStart : Double
  cfgEpsEnd   : Double
  cfgEpsDecay : Nat
  cfgSyncEvery : Nat
  cfgBatch    : Nat
  cfgGamma    : Double


----------------------------------------------------------------------
-- Episode rollout with DQN updates
----------------------------------------------------------------------

-- actionToVec converts a Nat action to a 1-vector for buffer storage.
actionToVec : Nat -> Vect 1 Double
actionToVec a = [cast (natToInteger a)]

trainIfReady : NativeOptimizer -> DqnState -> IO DqnState
trainIfReady opt st = do
  bufSz <- bufferSize st.buffer
  if bufSz < st.cfgBatch
    then pure st
    else do
      mBatch <- sampleN st.cfgBatch st.buffer
      case mBatch of
        Just batchVec => do
          let loss = batchLossBatched st.cfgBatch st.qNet st.target st.cfgGamma batchVec
          _ <- pure (nativeTrainStep opt loss)
          pure st
        Nothing => pure st

runEpisode : NativeOptimizer -> DqnState -> IO (DqnState, Double)
runEpisode opt st0 = go st0 (MkCP 0 0 0 0) MaxSteps 0.0
  where
    go : DqnState -> CPState -> Nat -> Double -> IO (DqnState, Double)
    go st _ Z ret = pure (st, ret)
    go st envState (S steps) ret = do
      stepCount <- readIORef st.stepRef
      let obs = observeVec envState
          eps = epsilonAt stepCount st.cfgEpsStart st.cfgEpsEnd st.cfgEpsDecay
      action <- epsGreedyIO st.qNet obs eps
      case cpStep envState action of
        (reward, envState', outcome, _) => do
          let isDone = done outcome
              nextObs = observeVec envState'
              trans = MkTransition obs (actionToVec action) reward nextObs isDone
          push st.buffer trans
          writeIORef st.stepRef (stepCount + 1)
          let ret' = ret + reward

          -- Train if buffer has enough samples
          st' <- trainIfReady opt st

          -- Sync target net on schedule
          let synced : DqnState
              synced = { target := snapshotTarget st'.qNet } st'
          let st'' = if (stepCount + 1) `mod` st.cfgSyncEvery == 0 then synced else st'

          if isDone
            then pure (st'', ret')
            else go st'' envState' steps ret'


----------------------------------------------------------------------
-- Config & epoch
----------------------------------------------------------------------

record Config where
  constructor MkConfig
  lr          : Double
  epochs      : Nat     -- episodes
  gamma       : Double
  batchSize   : Nat
  bufferCap   : Nat
  targetSync  : Nat
  epsStart    : Double
  epsEnd      : Double
  epsDecay    : Nat
  seed        : Bits64
  lrFind      : Bool

defaultConfig : Config
defaultConfig = MkConfig 5.0e-4 300 0.99 64 10000 100 1.0 0.05 10000 42 False

specs : List (ArgSpec Config)
specs = [ Arg "--lr" (\v, c => { lr := cast v } c)
        , Arg "--epochs" (\v, c => { epochs := castNat v } c)
        , Arg "--gamma" (\v, c => { gamma := cast v } c)
        , Arg "--batch" (\v, c => { batchSize := castNat v } c)
        , Arg "--buffer-cap" (\v, c => { bufferCap := castNat v } c)
        , Arg "--target-sync" (\v, c => { targetSync := castNat v } c)
        , Arg "--eps-start" (\v, c => { epsStart := cast v } c)
        , Arg "--eps-end" (\v, c => { epsEnd := cast v } c)
        , Arg "--eps-decay" (\v, c => { epsDecay := castNat v } c)
        , Arg "--seed" (\v, c => { seed := castBits64 v } c)
        , Arg "--lr-find" (\v, c => { lrFind := (v == "1" || v == "true") } c)
        ]


----------------------------------------------------------------------
-- Greedy evaluation
----------------------------------------------------------------------

evalEp : QNet -> CPState -> Nat -> Double -> Double
evalEp _ _ Z acc = acc
evalEp q st (S k) acc =
  let action = greedyAction q (observeVec st)
  in case cpStep st action of
       (reward, st', outcome, _) =>
         if done outcome then acc + reward
         else evalEp q st' k (acc + reward)

evalN : QNet -> Nat -> Double -> Double
evalN _ Z acc = acc
evalN q (S k) acc = evalN q k (acc + evalEp q (MkCP 0 0 0 0) MaxSteps 0.0)


----------------------------------------------------------------------
-- Main
----------------------------------------------------------------------

main : IO ()
main = do
  args <- getArgs
  let cfg = parseArgs defaultConfig specs (drop 1 args)
  srand cfg.seed

  putStrLn "=== DQN on CartPole ==="
  putStrLn $ "Config: lr=" ++ show cfg.lr
           ++ " epochs=" ++ show cfg.epochs
           ++ " gamma=" ++ show cfg.gamma
           ++ " batch=" ++ show cfg.batchSize
           ++ " buffer=" ++ show cfg.bufferCap
           ++ " target_sync=" ++ show cfg.targetSync
           ++ " eps=" ++ show cfg.epsStart ++ "→" ++ show cfg.epsEnd
           ++ " seed=" ++ show cfg.seed

  qNet0 <- mkQNet
  let target0 = snapshotTarget qNet0
  buffer <- mkBuffer {obsDim = ObsDim, actDim = 1} cfg.bufferCap
  stepRef <- newIORef (the Nat 0)
  let st0 = MkDqnState qNet0 target0 buffer stepRef
                       cfg.epsStart cfg.epsEnd cfg.epsDecay
                       cfg.targetSync cfg.batchSize cfg.gamma
      opt = nativeAdamGlobalClip cfg.lr 0.9 0.999 1.0e-8 10.0

  putStrLn ""

  -- HPO branch: --lr-find runs lr_find using episode-return-as-loss.
  -- See hyperparameter-tuning-2026.md for caveats — the per-episode
  -- signal is noisy and the network keeps training across iters, so
  -- the recommendation should be treated as informational only.
  when cfg.lrFind $ do
    let lrCfg : LrFindConfig
        lrCfg = { numIters := 30 } defaultLrFindConfig
    _ <- lrFind lrCfg
      (\st, _ => do (st', ret) <- runEpisode opt st; pure (st', negate ret))
      (pure ()) opt st0
    putStrLn ""
    putStrLn "Done — re-run without --lr-find at the recommended LR."
    exitSuccess

  let trainCfg : TrainConfig DqnState
      trainCfg = MkTrainConfig cfg.epochs 25 NoEarlyStop (const (pure [])) (\_ => pure ())
  (trained, epochsDone, _) <- runTrainingIO
    (\st, _ => do (st', ret) <- runEpisode opt st; pure (st', negate ret))
    (pure ())
    trainCfg st0

  putStrLn ""
  let nEval = the Nat 30
      totalReturn = evalN trained.qNet nEval 0.0
      avgReturn = totalReturn / cast (natToInteger nEval)
  putStrLn $ "Eval (" ++ show nEval ++ " episodes, greedy): avg_return=" ++ show avgReturn
  putStrLn ""
  putStrLn $ formatResult [("avg_return", show avgReturn),
                            ("epochs", show epochsDone),
                            ("seed", show cfg.seed)]
