module Example.MountainCar

import Data.List
import Data.Vect
import Data.Fin
import Data.IORef
import System
import Compat.Random

import Floating
import Gym.ClassicControl.MountainCar
import Gym.Env
import Gym.Vector
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
-- DQN on MountainCar-v0 with reward shaping.
--
-- MountainCar has a sparse reward (-1 per step until goal at pos >= 0.5).
-- Random exploration almost never reaches the goal in 200 steps, so DQN
-- can't learn from raw reward alone. We add velocity-magnitude shaping
-- (|v| * shapingScale) as a dense intermediate signal that nudges the
-- agent toward building energy.
--
-- Architecture: MLP 2 -> 64 -> 64 -> 64 -> 64 -> 3 (two relu blocks).
----------------------------------------------------------------------

ObsDim : Nat; ObsDim = 2
Hidden : Nat; Hidden = 64
NumActions : Nat; NumActions = 3
MaxSteps : Nat; MaxSteps = 200

||| Parallel envs collecting transitions in lockstep. Mirrors
||| `Example.Dqn.NumEnvs`. env-0 is the primary; envs 1..N-1 auto-reset
||| and continue feeding the buffer.
NumEnvs : Nat; NumEnvs = 4

QNet : Type
QNet = Network ObsDim [Hidden, Hidden, Hidden, Hidden] NumActions ExampleExecutor ExampleDType WithGrad

mkQNet : (scope : String) -> IO QNet
mkQNet scope = do
  ll1 <- linearLayerAny {i=ObsDim} {o=Hidden}     (scope ++ "ll1")
  ll2 <- linearLayerAny {i=Hidden} {o=Hidden}     (scope ++ "ll2")
  ll3 <- linearLayerAny {i=Hidden} {o=NumActions} (scope ++ "ll3")
  pure (ll1 ~~> reluLayerAny ~~> ll2 ~~> reluLayerAny ~~> OutputLayer ll3)


----------------------------------------------------------------------
-- Observation helper
----------------------------------------------------------------------

obsTensor : Vect ObsDim Double -> Vector ObsDim Double
obsTensor v = VArray (map SArray v)


----------------------------------------------------------------------
-- Epsilon-greedy
----------------------------------------------------------------------

epsilonAt : Nat -> Double -> Double -> Nat -> Double
epsilonAt step start end decaySteps =
  let frac = min (cast (natToInteger step) / cast (natToInteger decaySteps)) 1.0
  in start + frac * (end - start)

greedyAction : QNet -> Vect ObsDim Double -> IO Nat
greedyAction online obs = do
  let stateT = bulkToTensor {ex=ExampleExecutor} {dt=ExampleDType} (obsTensor obs)
      stateV = the (TVec ObsDim ExampleExecutor ExampleDType WithGrad) (MkTensor stateT Nothing)
  (_, qV) <- forwardVar online stateV
  let q0 = primItem1d {ex=ExampleExecutor} qV.tensorPtr 0
      q1 = primItem1d {ex=ExampleExecutor} qV.tensorPtr 1
      q2 = primItem1d {ex=ExampleExecutor} qV.tensorPtr 2
  pure (if q0 >= q1 && q0 >= q2 then 0
        else if q1 >= q2 then 1
        else 2)

epsGreedyIO : QNet -> Vect ObsDim Double -> Double -> IO Nat
epsGreedyIO online obs eps = do
  u <- randomRIO (the Double 0.0, 1.0)
  if u < eps
    then do
      u2 <- randomRIO (the Double 0.0, 1.0)
      pure (if u2 < (1.0 / 3.0) then 0
            else if u2 < (2.0 / 3.0) then 1
            else 2)
    else greedyAction online obs


-- Batched epsilon-greedy across NumEnvs envs: one batched forward,
-- then per-env eps-vs-greedy with independent random draws.
epsGreedyBatched : {n : Nat} -> Tensor [n, NumActions] ExampleExecutor ExampleDType g ->
                   Vect n MCState -> Double -> IO (Vect n Nat)
epsGreedyBatched qB envs eps = go 0 envs
  where
    go : Int -> Vect k MCState -> IO (Vect k Nat)
    go _ [] = pure []
    go i (_ :: rest) = do
      u <- randomRIO (the Double 0.0, 1.0)
      a <- if u < eps
             then do
               u2 <- randomRIO (the Double 0.0, 1.0)
               pure (if u2 < (1.0 / 3.0) then 0
                     else if u2 < (2.0 / 3.0) then 1
                     else 2)
             else do
               let q0 = primItem2d {ex=ExampleExecutor} qB.tensorPtr i 0
                   q1 = primItem2d {ex=ExampleExecutor} qB.tensorPtr i 1
                   q2 = primItem2d {ex=ExampleExecutor} qB.tensorPtr i 2
               pure (if q0 >= q1 && q0 >= q2 then 0
                     else if q1 >= q2 then 1
                     else 2)
      as <- go (i + 1) rest
      pure (a :: as)


----------------------------------------------------------------------
-- Batched DQN loss (mirrors Example.Dqn).
----------------------------------------------------------------------

vectorMaxPtr : AnyPtr -> Double
vectorMaxPtr t =
  let v0 = primItem1d {ex=ExampleExecutor} t 0
      v1 = primItem1d {ex=ExampleExecutor} t 1
      v2 = primItem1d {ex=ExampleExecutor} t 2
  in if v0 >= v1 && v0 >= v2 then v0
     else if v1 >= v2 then v1
     else v2

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
-- DQN state
----------------------------------------------------------------------

record DqnState where
  constructor MkDqnState
  qNet      : QNet
  target    : QNet
  buffer    : ReplayBuffer ObsDim 1
  envsRef   : IORef (VecEnv NumEnvs MCState)
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

trainIfReady : NativeOptimizer ExampleExecutor -> DqnState -> IO DqnState
trainIfReady opt st = do
  bufSz <- bufferSize st.buffer
  if bufSz < st.cfgBatch
    then pure st
    else do
      mBatch <- sampleN st.cfgBatch st.buffer
      case mBatch of
        Just batchVec => do
          -- Per-step generation bracket: free this replay step's grad
          -- intermediates immediately, so the within-epoch live handle
          -- count stays small (one DQN epoch is ~106k ops; without this it
          -- bursts past the paravirt-Metal buffer ceiling). Params update
          -- in-place via the registry (rc>1, spared), so the () result
          -- needs no KeepAlive rescue.
          withGenFree {ex=ExampleExecutor} $ do
            loss <- batchLossBatched st.cfgBatch st.qNet st.target st.cfgGamma batchVec
            _ <- nativeTrainStep opt loss
            pure ()
          pure st
        Nothing => pure st

shapedReward : DqnState -> MCState -> MCState -> Double -> Double
shapedReward st _ s' baseReward =
  baseReward + st.cfgShaping * abs s'.mcVel

runEpisode : NativeOptimizer ExampleExecutor -> DqnState -> IO (DqnState, Double)
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
          let ret' = ret + rawReward

          st' <- trainIfReady opt st

          when ((stepCount + 1) `mod` st.cfgSyncEvery == 0) $ do
            _ <- polyakUpdate {ex=ExampleExecutor} 1.0 "online_" "target_"
            pure ()

          if isDone
            then pure (st', ret')
            else go st' envState' steps ret'


----------------------------------------------------------------------
-- Batched episode rollout: NumEnvs parallel envs collect transitions
-- in lockstep; one batched action-selection forward per outer step.
-- env-0 is the primary — its episode boundary marks the end of an
-- epoch. Envs 1..N-1 auto-reset and continue filling buffer.
-- Reward shaping is applied per-env using each env's old / new state.
----------------------------------------------------------------------

-- Step every env with its action; auto-reset on done. Returns next
-- states, raw rewards, done flags. Per-env shaping applied by caller
-- (needs the OLD state alongside the new state, like the sequential
-- runEpisode's shapedReward helper).
stepAllAutoResetMC : Vect n MCState -> Vect n Nat ->
                     (Vect n MCState, Vect n Double, Vect n Bool)
stepAllAutoResetMC [] [] = ([], [], [])
stepAllAutoResetMC (s :: ss) (a :: as) =
  case mcStep s a of
    (r, s', outcome, _) =>
      let isDone = done outcome
          nextS  = if isDone then MkMC (-0.5) 0.0 else s'
      in case stepAllAutoResetMC ss as of
           (rest, rs, ds) => (nextS :: rest, r :: rs, isDone :: ds)

-- Push N transitions to the replay buffer in order (env-0 first).
-- Uses each env's own (oldState, action, shapedReward, newState, done)
-- tuple; reward shaping is done here to keep callers simple.
pushAllTransitionsMC : ReplayBuffer ObsDim 1 -> Double ->
                       Vect n MCState -> Vect n Nat -> Vect n Double ->
                       Vect n MCState -> Vect n Bool -> IO ()
pushAllTransitionsMC _ _ [] [] [] [] [] = pure ()
pushAllTransitionsMC buf shaping (s :: ss) (a :: as) (r :: rs) (s' :: ss') (d :: ds) = do
  let shapedR = r + shaping * abs s'.mcVel
  push buf (MkTransition (mcObserve s) (actionToVec a) shapedR (mcObserve s') d)
  pushAllTransitionsMC buf shaping ss as rs ss' ds

runEpisodeBatched : NativeOptimizer ExampleExecutor -> DqnState -> IO (DqnState, Double)
runEpisodeBatched opt st0 = do
  startEnvs <- readIORef st0.envsRef
  go st0 startEnvs.envs MaxSteps 0.0
  where
    go : DqnState -> Vect NumEnvs MCState -> Nat -> Double -> IO (DqnState, Double)
    go st _ Z ret = do
      writeIORef st.envsRef (MkVecEnv (replicate NumEnvs (MkMC (-0.5) 0.0)))
      pure (st, ret)
    go st envs (S steps) ret = do
      stepCount <- readIORef st.stepRef
      let eps = epsilonAt stepCount st.cfgEpsStart st.cfgEpsEnd st.cfgEpsDecay
          obsRows : Vect NumEnvs (Vector ObsDim Double)
          obsRows = map (\s => obsTensor (mcObserve s)) envs
          batchPtr = bulkToTensor2d {ex=ExampleExecutor} {dt=ExampleDType} obsRows
          stateV : Tensor [NumEnvs, ObsDim] ExampleExecutor ExampleDType WithGrad
          stateV = MkTensor batchPtr Nothing
      actions <- withNoGrad {ex=ExampleExecutor} $ do
        (_, qB) <- forwardVarBatch st.qNet stateV
        epsGreedyBatched qB envs eps
      case stepAllAutoResetMC envs actions of
        (envs', rewards, dones) => do
          pushAllTransitionsMC st.buffer st.cfgShaping envs actions rewards envs' dones
          writeIORef st.stepRef (stepCount + 1)
          let ret0 : Double
              ret0 = head rewards
              done0 : Bool
              done0 = head dones
              ret' = ret + ret0

          st' <- trainIfReady opt st

          when ((stepCount + 1) `mod` st.cfgSyncEvery == 0) $ do
            _ <- polyakUpdate {ex=ExampleExecutor} 1.0 "online_" "target_"
            pure ()

          if done0
            then do
              writeIORef st'.envsRef (MkVecEnv envs')
              pure (st', ret')
            else go st' envs' steps ret'


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

evalEp : QNet -> MCState -> Nat -> Double -> IO Double
evalEp _ _ Z acc = pure acc
evalEp q st (S k) acc = do
  action <- greedyAction q (mcObserve st)
  case mcStep st action of
    (reward, st', outcome, _) =>
      if done outcome then pure (acc + reward)
      else evalEp q st' k (acc + reward)

evalN : QNet -> Nat -> Double -> IO Double
evalN _ Z acc = pure acc
evalN q (S k) acc = do
  ep <- evalEp q (MkMC (-0.5) 0.0) MaxSteps 0.0
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
  tsetInitSeed {ex = ExampleExecutor} cfg.seed

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

  qNet0 <- mkQNet "online_"
  target0 <- mkQNet "target_"
  _ <- polyakUpdate {ex=ExampleExecutor} 1.0 "online_" "target_"

  buffer <- mkBuffer {obsDim = ObsDim, actDim = 1} cfg.bufferCap
  resetSeedI <- randomInt32
  let initEnvs : VecEnv NumEnvs MCState
      initEnvs = fst (resetAll {state=MCState} {action=Nat} {obs=Vect 2 Double}
                              (cast resetSeedI))
  envsRef <- newIORef initEnvs
  stepRef <- newIORef (the Nat 0)
  let st0 = MkDqnState qNet0 target0 buffer envsRef stepRef
                       cfg.epsStart cfg.epsEnd cfg.epsDecay
                       cfg.targetSync cfg.batchSize cfg.gamma cfg.shaping
      opt = nativeAdamGroup "online_" cfg.lr 0.9 0.999 1.0e-8 10.0

  putStrLn ""

  when cfg.lrFind $ do
    let lrCfg : LrFindConfig
        lrCfg = { numIters := 30 } defaultLrFindConfig
    _ <- lrFind lrCfg
      (\st, _ => do (st', ret) <- runEpisodeBatched opt st; pure (st', negate ret))
      (pure ()) opt st0
    putStrLn ""
    putStrLn "Done — re-run without --lr-find at the recommended LR."
    exitSuccess

  metrics <- newRLMetricsState 50
  let trainCfg : TrainConfig DqnState
      trainCfg = mkTrainConfig cfg.epochs 50 NoEarlyStop
                   (\_ => readRLMetrics "recent_50" metrics) (\_ => pure ())
  (trained, epochsDone, _) <- runTrainingIO {ex=ExampleExecutor}
    (\st, _ => do
       (st', ret) <- runEpisodeBatched opt st
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
