module Example.Dqn

import Data.List
import Data.Vect
import Data.Fin
import Data.IORef
import System
import Compat.Random

import Floating
import Gym.ClassicControl.CartPole
import Gym.Env
import Hpo.LrFinder
import Layer.Activation
import Layer.Core
import Layer.Linear
import Math
import RL.ReplayBuffer
import Array
import Train
import Util
import Executor
import Tensor
import BuildConfig


----------------------------------------------------------------------
-- Architecture: MLP 4 -> 64 -> 64 -> 2
----------------------------------------------------------------------

ObsDim : Nat; ObsDim = 4
Hidden : Nat; Hidden = 64
NumActions : Nat; NumActions = 2
MaxSteps : Nat; MaxSteps = cartPoleMaxSteps

-- Two `linear ~~> relu` blocks followed by `OutputLayer Linear` give
-- hidden dims [Hidden, Hidden, Hidden, Hidden].
QNet : Type
QNet = Network ObsDim [Hidden, Hidden, Hidden, Hidden] NumActions ExampleExecutor ExampleDType WithGrad

||| Build a Q-network with all params registered under `<scope>...`.
||| Reuse the same architecture for online and target nets, scoped
||| under e.g. "online_" / "target_" so `polyakUpdate` can match
||| online↔target params by suffix.
mkQNet : (scope : String) -> IO QNet
mkQNet scope = do
  ll1 <- linearLayerAny {i=ObsDim} {o=Hidden}     (scope ++ "ll1")
  ll2 <- linearLayerAny {i=Hidden} {o=Hidden}     (scope ++ "ll2")
  ll3 <- linearLayerAny {i=Hidden} {o=NumActions} (scope ++ "ll3")
  pure (ll1 ~~> reluLayerAny ~~> ll2 ~~> reluLayerAny ~~> OutputLayer ll3)


----------------------------------------------------------------------
-- Observation helper
----------------------------------------------------------------------

observeVec : CPState -> Vect ObsDim Double
observeVec s = cpObserve s

obsTensor : Vect ObsDim Double -> Vector ObsDim Double
obsTensor v = VArray (map SArray v)


----------------------------------------------------------------------
-- Epsilon-greedy action selection (uses online net's current weights)
----------------------------------------------------------------------

epsilonAt : Nat -> Double -> Double -> Nat -> Double
epsilonAt step start end decaySteps =
  let frac = min (cast (natToInteger step) / cast (natToInteger decaySteps)) 1.0
  in start + frac * (end - start)

-- Argmax over Q(s, *) from the online net.
greedyAction : QNet -> Vect ObsDim Double -> IO Nat
greedyAction online obs = do
  let stateT = bulkToTensor {ex=ExampleExecutor} {dt=ExampleDType} (obsTensor obs)
      stateV = the (TVec ObsDim ExampleExecutor ExampleDType WithGrad) (MkTensor stateT Nothing)
  (_, qV) <- forwardVar online stateV
  let q0 = primItem1d {ex=ExampleExecutor} qV.tensorPtr 0
      q1 = primItem1d {ex=ExampleExecutor} qV.tensorPtr 1
  pure (if q0 >= q1 then 0 else 1)

epsGreedyIO : QNet -> Vect ObsDim Double -> Double -> IO Nat
epsGreedyIO online obs eps = do
  u <- randomRIO (the Double 0.0, 1.0)
  if u < eps
    then do
      u2 <- randomRIO (the Double 0.0, 1.0)
      pure (if u2 < 0.5 then 0 else 1)
    else greedyAction online obs


----------------------------------------------------------------------
-- DQN loss (batched). Online Q is batched: one [B, ObsDim] forward
-- replaces B per-sample forwards. Target Q is forwarded per-sample
-- (single forwardVar on the target Network) and read back as a
-- Double — no autograd chain into the target's params, since the
-- optimizer is scoped to "online_" only.
----------------------------------------------------------------------

-- Max over a 1D tensor pointer (read NumActions scalars, take max).
vectorMaxPtr : AnyPtr -> Double
vectorMaxPtr t =
  let v0 = primItem1d {ex=ExampleExecutor} t 0
      v1 = primItem1d {ex=ExampleExecutor} t 1
  in if v0 >= v1 then v0 else v1

computeTargetVal : QNet -> Double -> Transition ObsDim 1 -> IO Double
computeTargetVal target gamma t = do
  let stateT = bulkToTensor {ex=ExampleExecutor} {dt=ExampleDType} (obsTensor t.nextObs)
      stateV = the (TVec ObsDim ExampleExecutor ExampleDType WithGrad) (MkTensor stateT Nothing)
  (_, qV) <- forwardVar target stateV
  let nextMax = vectorMaxPtr qV.tensorPtr
      bootstrap = if t.done then 0.0 else gamma * nextMax
  pure (t.reward + bootstrap)

actionIdx : Vect 1 Double -> Int
actionIdx [a] = cast {to=Int} (cast {to=Integer} a)

perSampleLoss : {n : Nat} -> (qOutB : Tensor [n, NumActions] ExampleExecutor ExampleDType WithGrad) ->
                Transition ObsDim 1 -> Double -> Int -> IO (Tensor [] ExampleExecutor ExampleDType WithGrad)
perSampleLoss qOutB t tv k = do
  let aIdx = actionIdx t.action
  qRow    <- trowSelect qOutB k
  qScalar <- telemSelect qRow aIdx
  targetT <- tconstScalar tv
  diff    <- tsub qScalar targetT
  tmul diff diff

meanScalarLoss : (n : Nat) -> List (Tensor [] ExampleExecutor ExampleDType WithGrad) -> IO (Tensor [] ExampleExecutor ExampleDType WithGrad)
meanScalarLoss n losses = do
  zero <- tconstScalar 0.0
  let summed = foldl (\a, b => MkTensor (primAdd {ex=ExampleExecutor} a.tensorPtr b.tensorPtr) Nothing) zero losses
  tmulScalar summed (1.0 / cast n)

batchLossBatched : (n : Nat) -> QNet -> QNet -> Double ->
                   Vect n (Transition ObsDim 1) -> IO (Tensor [] ExampleExecutor ExampleDType WithGrad)
batchLossBatched n online target gamma batch = do
  targetVals <- traverse (computeTargetVal target gamma) batch
  let obsTensors = map (\t => obsTensor t.obs) batch
      obsBT = bulkToTensor2d {ex=ExampleExecutor} {dt=ExampleDType} obsTensors
      obsBV = the (Tensor [n, ObsDim] ExampleExecutor ExampleDType WithGrad) (MkTensor obsBT Nothing)
  (_, qOutB) <- forwardVarBatch online obsBV
  losses <- go qOutB (toList batch) (toList targetVals) 0
  meanScalarLoss n losses
  where
    go : {n : Nat} -> Tensor [n, NumActions] ExampleExecutor ExampleDType WithGrad ->
         List (Transition ObsDim 1) ->
         List Double -> Int -> IO (List (Tensor [] ExampleExecutor ExampleDType WithGrad))
    go _ [] _ _ = pure []
    go _ _ [] _ = pure []
    go qOutB (t :: tRest) (tv :: tvRest) k = do
      l <- perSampleLoss qOutB t tv k
      ls <- go qOutB tRest tvRest (k + 1)
      pure (l :: ls)


----------------------------------------------------------------------
-- DQN state threaded through training
----------------------------------------------------------------------

record DqnState where
  constructor MkDqnState
  qNet      : QNet
  target    : QNet
  buffer    : ReplayBuffer ObsDim 1
  stepRef   : IORef Nat
  cfgEpsStart : Double
  cfgEpsEnd   : Double
  cfgEpsDecay : Nat
  cfgSyncEvery : Nat
  cfgBatch    : Nat
  cfgGamma    : Double


----------------------------------------------------------------------
-- Episode rollout with DQN updates
----------------------------------------------------------------------

actionToVec : Nat -> Vect 1 Double
actionToVec a = [cast (natToInteger a)]

trainIfReady : NativeOptimizer ExampleExecutor -> DqnState -> IO DqnState
trainIfReady opt st = do
  bufSz <- bufferSize st.buffer
  if bufSz < st.cfgBatch
    then pure st
    else do
      mBatch <- sampleN st.cfgBatch st.buffer
      case mBatch of
        Just batchVec => do
          loss <- batchLossBatched st.cfgBatch st.qNet st.target st.cfgGamma batchVec
          _ <- nativeTrainStep opt loss
          pure st
        Nothing => pure st

runEpisode : NativeOptimizer ExampleExecutor -> DqnState -> IO (DqnState, Double)
runEpisode opt st0 = go st0 (MkCP 0 0 0 0) MaxSteps 0.0
  where
    go : DqnState -> CPState -> Nat -> Double -> IO (DqnState, Double)
    go st _ Z ret = pure (st, ret)
    go st envState (S steps) ret = do
      stepCount <- readIORef st.stepRef
      let obs = observeVec envState
          eps = epsilonAt stepCount st.cfgEpsStart st.cfgEpsEnd st.cfgEpsDecay
      -- Action selection forward: no grad needed (just extracting
      -- Q values as Doubles for argmax). Loss-side forward in
      -- trainIfReady runs separately under normal grad tracking.
      action <- withNoGrad {ex=ExampleExecutor} (epsGreedyIO st.qNet obs eps)
      case cpStep envState action of
        (reward, envState', outcome, _) => do
          let isDone = done outcome
              nextObs = observeVec envState'
              trans = MkTransition obs (actionToVec action) reward nextObs isDone
          push st.buffer trans
          writeIORef st.stepRef (stepCount + 1)
          let ret' = ret + reward

          st' <- trainIfReady opt st

          -- Hard-sync target ← online via polyak-blend with tau=1.0
          when ((stepCount + 1) `mod` st.cfgSyncEvery == 0) $ do
            _ <- polyakUpdate {ex=ExampleExecutor} 1.0 "online_" "target_"
            pure ()

          if isDone
            then pure (st', ret')
            else go st' envState' steps ret'


----------------------------------------------------------------------
-- Config & epoch
----------------------------------------------------------------------

record Config where
  constructor MkConfig
  lr          : Double
  epochs      : Nat
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

evalEp : QNet -> CPState -> Nat -> Double -> IO Double
evalEp _ _ Z acc = pure acc
evalEp q st (S k) acc = do
  action <- greedyAction q (observeVec st)
  case cpStep st action of
    (reward, st', outcome, _) =>
      if done outcome then pure (acc + reward)
      else evalEp q st' k (acc + reward)

evalN : QNet -> Nat -> Double -> IO Double
evalN _ Z acc = pure acc
evalN q (S k) acc = do
  ep <- evalEp q (MkCP 0 0 0 0) MaxSteps 0.0
  evalN q k (acc + ep)


----------------------------------------------------------------------
-- Main
----------------------------------------------------------------------

main : IO ()
main = do
  requireMachine {m = ChosenMachine}
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

  qNet0 <- mkQNet "online_"
  target0 <- mkQNet "target_"
  -- Initial hard sync: target ← online (tau=1.0).
  _ <- polyakUpdate {ex=ExampleExecutor} 1.0 "online_" "target_"

  buffer <- mkBuffer {obsDim = ObsDim, actDim = 1} cfg.bufferCap
  stepRef <- newIORef (the Nat 0)
  let st0 = MkDqnState qNet0 target0 buffer stepRef
                       cfg.epsStart cfg.epsEnd cfg.epsDecay
                       cfg.targetSync cfg.batchSize cfg.gamma
      opt = nativeAdamGroup "online_" cfg.lr 0.9 0.999 1.0e-8 10.0

  putStrLn ""

  when cfg.lrFind $ do
    let lrCfg : LrFindConfig
        lrCfg = { numIters := 30 } defaultLrFindConfig
    _ <- lrFind lrCfg
      (\st, _ => do (st', ret) <- runEpisode opt st; pure (st', negate ret))
      (pure ()) opt st0
    putStrLn ""
    putStrLn "Done — re-run without --lr-find at the recommended LR."
    exitSuccess

  metrics <- newRLMetricsState 50
  let trainCfg : TrainConfig DqnState
      trainCfg = mkTrainConfig cfg.epochs 25 NoEarlyStop
                   (\_ => readRLMetrics "recent_50" metrics) (\_ => pure ())
  (trained, epochsDone, _) <- runTrainingIO {ex=ExampleExecutor}
    (\st, _ => do
       (st', ret) <- runEpisode opt st
       recordReturn metrics ret
       pure (st', negate ret))
    (pure ())
    trainCfg st0

  putStrLn ""
  let nEval = the Nat 30
  totalReturn <- withNoGrad {ex=ExampleExecutor} (evalN trained.qNet nEval 0.0)
  let avgReturn = totalReturn / cast (natToInteger nEval)
  putStrLn $ "Eval (" ++ show nEval ++ " episodes, greedy): avg_return=" ++ show avgReturn
  putStrLn ""
  putStrLn $ formatResult [("avg_return", show avgReturn),
                            ("epochs", show epochsDone),
                            ("seed", show cfg.seed)]
