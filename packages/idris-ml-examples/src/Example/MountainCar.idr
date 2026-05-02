module Example.MountainCar

import Data.List
import Data.Vect
import Data.Fin
import Data.IORef
import System
import Compat.Random

import Endofunctor
import Floating
import Gym.ClassicControl.MountainCar
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
-- DQN on MountainCar-v0 with reward shaping.
--
-- MountainCar has a sparse reward (-1 per step until goal at pos >= 0.5).
-- Random exploration almost never reaches the goal in 200 steps, so DQN
-- can't learn from raw reward alone. We add velocity-magnitude shaping
-- (|v| * shapingScale) as a dense intermediate signal that nudges the
-- agent toward building energy. Not policy-invariant in the strict
-- Ng99 sense (alters Q*) but the optimal trajectory is preserved at the
-- shapingScale chosen.
--
-- Architecture: MLP 2 -> 64 -> 64 -> 3.
----------------------------------------------------------------------

ObsDim : Nat; ObsDim = 2
Hidden : Nat; Hidden = 64
NumActions : Nat; NumActions = 3
MaxSteps : Nat; MaxSteps = 200

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

obsTensor : Vect ObsDim Double -> Vector ObsDim Double
obsTensor v = VTensor (map STensor v)


----------------------------------------------------------------------
-- Target network snapshot
----------------------------------------------------------------------

snapshotTarget : QNet -> QNetDouble
snapshotTarget online = toDoubleNetwork (emap refreshValue online)

vectorMax : Vector NumActions Double -> Double
vectorMax qr =
  let STensor v = index (argmax qr) qr
  in v


----------------------------------------------------------------------
-- Epsilon-greedy
----------------------------------------------------------------------

epsilonAt : Nat -> Double -> Double -> Nat -> Double
epsilonAt step start end decaySteps =
  let frac = min (cast (natToInteger step) / cast (natToInteger decaySteps)) 1.0
  in start + frac * (end - start)

greedyAction : QNet -> Vect ObsDim Double -> Nat
greedyAction online obs =
  let stateT = bulkToTensor (obsTensor obs)
      qT = snd (forwardVarTensor online stateT)
      q0 = prim__item1d qT 0
      q1 = prim__item1d qT 1
      q2 = prim__item1d qT 2
  in if q0 >= q1 && q0 >= q2 then 0
     else if q1 >= q2 then 1
     else 2

epsGreedyIO : QNet -> Vect ObsDim Double -> Double -> IO Nat
epsGreedyIO online obs eps = do
  u <- randomRIO (the Double 0.0, 1.0)
  if u < eps
    then do
      u2 <- randomRIO (the Double 0.0, 1.0)
      pure (if u2 < (1.0 / 3.0) then 0
            else if u2 < (2.0 / 3.0) then 1
            else 2)
    else pure (greedyAction online obs)


----------------------------------------------------------------------
-- Batched DQN loss (mirrors Example.Dqn).
----------------------------------------------------------------------

computeTargetVal : QNetDouble -> Double -> Transition ObsDim 1 -> Double
computeTargetVal tgt gamma t =
  let qNextD = snd (forward tgt (obsTensor t.nextObs))
      nextMaxVal = vectorMax qNextD
      bootstrap  = if t.done then 0.0 else gamma * nextMaxVal
  in t.reward + bootstrap

actionIdx : Vect 1 Double -> Int
actionIdx [a] = cast {to=Int} (cast {to=Integer} a)

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
      obsBT      = bulkToTensor2d obsTensors
      qOutB      = snd (forwardVarTensorBatch online n obsBT)
      losses     = perSampleLosses qOutB batch targetVals 0
      n_d        = the Double (cast (natToInteger n))
      sumV       = foldl (+) (the (Variable CPU) (fromDouble 0.0)) losses
      nV         = the (Variable CPU) (fromDouble n_d)
  in sumV / nV


----------------------------------------------------------------------
-- DQN state
----------------------------------------------------------------------

record DqnState where
  constructor MkDqnState
  qNet      : QNet
  target    : QNetDouble
  buffer    : ReplayBuffer ObsDim 1
  stepRef   : IORef Nat
  cfgEpsStart : Double
  cfgEpsEnd   : Double
  cfgEpsDecay : Nat
  cfgSyncEvery : Nat
  cfgBatch    : Nat
  cfgGamma    : Double
  cfgShaping  : Double  -- multiplier on |vel| reward bonus


----------------------------------------------------------------------
-- Episode rollout
----------------------------------------------------------------------

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

-- Reward shaping: add shapingScale * |vel'| to the env reward. Encourages
-- the agent to build kinetic energy, which is the proven intermediate
-- behavior for MountainCar.
shapedReward : DqnState -> MCState -> MCState -> Double -> Double
shapedReward st _ s' baseReward =
  baseReward + st.cfgShaping * abs s'.mcVel

runEpisode : NativeOptimizer -> DqnState -> IO (DqnState, Double)
runEpisode opt st0 = go st0 (MkMC (-0.5) 0.0) MaxSteps 0.0
  where
    go : DqnState -> MCState -> Nat -> Double -> IO (DqnState, Double)
    go st _ Z ret = pure (st, ret)
    go st envState (S steps) ret = do
      stepCount <- readIORef st.stepRef
      let obs = mcObserve envState
          eps = epsilonAt stepCount st.cfgEpsStart st.cfgEpsEnd st.cfgEpsDecay
      action <- epsGreedyIO st.qNet obs eps
      case mcStep envState action of
        (rawReward, envState', outcome, _) => do
          let isDone = done outcome
              nextObs = mcObserve envState'
              shaped = shapedReward st envState envState' rawReward
              trans = MkTransition obs (actionToVec action) shaped nextObs isDone
          push st.buffer trans
          writeIORef st.stepRef (stepCount + 1)
          -- Track the *raw* (unshaped) return so the eval metric is comparable
          -- to standard MountainCar reporting.
          let ret' = ret + rawReward

          st' <- trainIfReady opt st

          let synced : DqnState
              synced = { target := snapshotTarget st'.qNet } st'
          let st'' = if (stepCount + 1) `mod` st.cfgSyncEvery == 0 then synced else st'

          if isDone
            then pure (st'', ret')
            else go st'' envState' steps ret'


----------------------------------------------------------------------
-- Config & main
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
  shaping     : Double
  seed        : Bits64
  lrFind      : Bool

defaultConfig : Config
defaultConfig =
  MkConfig 1.0e-3 1000 0.99 64 50000 200 1.0 0.05 50000 10.0 42 False

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
        , Arg "--shaping" (\v, c => { shaping := cast v } c)
        , Arg "--seed" (\v, c => { seed := castBits64 v } c)
        , Arg "--lr-find" (\v, c => { lrFind := (v == "1" || v == "true") } c)
        ]


----------------------------------------------------------------------
-- Greedy evaluation (raw reward, no shaping).
----------------------------------------------------------------------

evalEp : QNet -> MCState -> Nat -> Double -> Double
evalEp _ _ Z acc = acc
evalEp q st (S k) acc =
  let action = greedyAction q (mcObserve st)
  in case mcStep st action of
       (reward, st', outcome, _) =>
         if done outcome then acc + reward
         else evalEp q st' k (acc + reward)

evalN : QNet -> Nat -> Double -> Double
evalN _ Z acc = acc
evalN q (S k) acc = evalN q k (acc + evalEp q (MkMC (-0.5) 0.0) MaxSteps 0.0)


----------------------------------------------------------------------
-- Main
----------------------------------------------------------------------

main : IO ()
main = do
  args <- getArgs
  let cfg = parseArgs defaultConfig specs (drop 1 args)
  srand cfg.seed

  putStrLn "=== DQN on MountainCar ==="
  putStrLn $ "Config: lr=" ++ show cfg.lr
           ++ " epochs=" ++ show cfg.epochs
           ++ " gamma=" ++ show cfg.gamma
           ++ " batch=" ++ show cfg.batchSize
           ++ " buffer=" ++ show cfg.bufferCap
           ++ " target_sync=" ++ show cfg.targetSync
           ++ " eps=" ++ show cfg.epsStart ++ "→" ++ show cfg.epsEnd
           ++ " shaping=" ++ show cfg.shaping
           ++ " seed=" ++ show cfg.seed

  qNet0 <- mkQNet
  let target0 = snapshotTarget qNet0
  buffer <- mkBuffer {obsDim = ObsDim, actDim = 1} cfg.bufferCap
  stepRef <- newIORef (the Nat 0)
  let st0 = MkDqnState qNet0 target0 buffer stepRef
                       cfg.epsStart cfg.epsEnd cfg.epsDecay
                       cfg.targetSync cfg.batchSize cfg.gamma cfg.shaping
      opt = nativeAdamGlobalClip cfg.lr 0.9 0.999 1.0e-8 10.0

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

  let trainCfg : TrainConfig DqnState
      trainCfg = MkTrainConfig cfg.epochs 50 NoEarlyStop (const (pure [])) (\_ => pure ())
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
