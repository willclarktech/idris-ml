module Example.Dqn

import Data.List
import Data.Vect
import Data.Fin
import Data.IORef
import System
import Compat.Random

import ML.Simple
import Array            -- Vector / VArray / SArray
import Gym.ClassicControl.CartPole
import Gym.Env
import Gym.Vector
import Hpo.LrFinder
import RL.ReplayBuffer
import Train
import BuildConfig

----------------------------------------------------------------------
-- Architecture: MLP 4 -> 64 -> relu -> 64 -> relu -> 2
----------------------------------------------------------------------

ObsDim     : Nat; ObsDim = 4
Hidden     : Nat; Hidden = 64
NumActions : Nat; NumActions = 2
MaxSteps   : Nat; MaxSteps = cartPoleMaxSteps

||| Parallel envs collecting transitions in lockstep. Mirrors PyTorch's
||| `gym.vector.SyncVectorEnv`. Env-0 is the "primary" (its episode
||| boundary marks the end of an Idris epoch).
NumEnvs : Nat; NumEnvs = 4

QNet : Type
QNet = Seq ObsDim NumActions Ex F WithGrad

||| Build a Q-network with all params registered under `<scope>.*`. Reuse
||| the same architecture for online and target nets, scoped "online" /
||| "target" so `polyakUpdate` can match online↔target params by scope.
mkQNet : (scope : String) -> IO QNet
mkQNet scope = runInit $ scoped scope $ do
  l1 <- linear {i=ObsDim} {o=Hidden}
  l2 <- linear {i=Hidden} {o=Hidden}
  l3 <- linear {i=Hidden} {o=NumActions}
  pure (l1 ~~> reluA ~~> l2 ~~> reluA ~~> l3 ~~> Nil)

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

-- Argmax over Q(s, *) from the online net (single [1, ObsDim] forward).
greedyAction : QNet -> Vect ObsDim Double -> IO Nat
greedyAction online obs = do
  let stateT = bulkToTensor2d {ex=Ex} {dt=F} [obsTensor obs]
      stateV = the (Tensor [1, ObsDim] Ex F WithGrad) (MkTensor stateT Nothing)
  qV <- forwardSeq {b=1} online stateV
  let q0 = primItem2d {ex=Ex} qV.tensorPtr 0 0
      q1 = primItem2d {ex=Ex} qV.tensorPtr 0 1
  pure (if q0 >= q1 then 0 else 1)

epsGreedyIO : QNet -> Vect ObsDim Double -> Double -> IO Nat
epsGreedyIO online obs eps = do
  u <- randomRIO (the Double 0.0, 1.0)
  if u < eps
    then do
      u2 <- randomRIO (the Double 0.0, 1.0)
      pure (if u2 < 0.5 then 0 else 1)
    else greedyAction online obs

-- Batched epsilon-greedy: given a [N, NumActions] Q-tensor and the
-- current N envs, sample one action per env (one randomRIO per env).
epsGreedyBatched : {n : Nat} -> Tensor [n, NumActions] Ex F g ->
                   Vect n CPState -> Double -> IO (Vect n Nat)
epsGreedyBatched qB envs eps = go 0 envs
  where
    go : Int -> Vect k CPState -> IO (Vect k Nat)
    go _ [] = pure []
    go i (_ :: rest) = do
      u <- randomRIO (the Double 0.0, 1.0)
      a <- if u < eps
             then do
               u2 <- randomRIO (the Double 0.0, 1.0)
               pure (if u2 < 0.5 then 0 else 1)
             else do
               let q0 = primItem2d {ex=Ex} qB.tensorPtr i 0
                   q1 = primItem2d {ex=Ex} qB.tensorPtr i 1
               pure (if q0 >= q1 then 0 else 1)
      as <- go (i + 1) rest
      pure (a :: as)

----------------------------------------------------------------------
-- DQN loss (batched online Q; target Q forwarded per-sample, read back
-- as a Double — no autograd chain into the target's params, the
-- optimizer is scoped to "online" only).
----------------------------------------------------------------------

-- Max over row 0 of a [1, NumActions] tensor pointer.
rowMax2d : AnyPtr -> Double
rowMax2d t =
  let v0 = primItem2d {ex=Ex} t 0 0
      v1 = primItem2d {ex=Ex} t 0 1
  in if v0 >= v1 then v0 else v1

computeTargetVal : QNet -> Double -> Transition ObsDim 1 -> IO Double
computeTargetVal target gamma t = do
  let stateT = bulkToTensor2d {ex=Ex} {dt=F} [obsTensor t.nextObs]
      stateV = the (Tensor [1, ObsDim] Ex F WithGrad) (MkTensor stateT Nothing)
  qV <- forwardSeq {b=1} target stateV
  let nextMax = rowMax2d qV.tensorPtr
      bootstrap = if t.done then 0.0 else gamma * nextMax
  pure (t.reward + bootstrap)

actionIdx : Vect 1 Double -> Int
actionIdx [a] = cast {to=Int} (cast {to=Integer} a)

perSampleLoss : {n : Nat} -> (qOutB : Tensor [n, NumActions] Ex F WithGrad) ->
                Transition ObsDim 1 -> Double -> Int -> IO (Tensor [] Ex F WithGrad)
perSampleLoss qOutB t tv k = do
  let aIdx = actionIdx t.action
  qRow    <- trowSelect qOutB k
  qScalar <- telemSelect qRow aIdx
  targetT <- tconstScalar tv
  diff    <- tsub qScalar targetT
  tmul diff diff

meanScalarLoss : (n : Nat) -> List (Tensor [] Ex F WithGrad) -> IO (Tensor [] Ex F WithGrad)
meanScalarLoss n losses = do
  zero <- tconstScalar 0.0
  let summed = foldl (\a, b => MkTensor (primAdd {ex=Ex} a.tensorPtr b.tensorPtr) Nothing) zero losses
  tmulScalar summed (1.0 / cast n)

batchLossBatched : (n : Nat) -> QNet -> QNet -> Double ->
                   Vect n (Transition ObsDim 1) -> IO (Tensor [] Ex F WithGrad)
batchLossBatched n online target gamma batch = do
  targetVals <- traverse (computeTargetVal target gamma) batch
  let obsTensors = map (\t => obsTensor t.obs) batch
      obsBT = bulkToTensor2d {ex=Ex} {dt=F} obsTensors
      obsBV = the (Tensor [n, ObsDim] Ex F WithGrad) (MkTensor obsBT Nothing)
  qOutB <- forwardSeq {b=n} online obsBV
  losses <- go qOutB (toList batch) (toList targetVals) 0
  meanScalarLoss n losses
  where
    go : {n : Nat} -> Tensor [n, NumActions] Ex F WithGrad ->
         List (Transition ObsDim 1) ->
         List Double -> Int -> IO (List (Tensor [] Ex F WithGrad))
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
  qNet         : QNet
  target       : QNet
  buffer       : ReplayBuffer ObsDim 1
  envsRef      : IORef (VecEnv NumEnvs CPState)
  stepRef      : IORef Nat
  cfgEpsStart  : Double
  cfgEpsEnd    : Double
  cfgEpsDecay  : Nat
  cfgSyncEvery : Nat
  cfgBatch     : Nat
  cfgGamma     : Double

----------------------------------------------------------------------
-- Episode rollout with DQN updates
----------------------------------------------------------------------

actionToVec : Nat -> Vect 1 Double
actionToVec a = [cast (natToInteger a)]

trainIfReady : Optimizer Ex -> DqnState -> IO DqnState
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

----------------------------------------------------------------------
-- Batched episode rollout: NumEnvs parallel envs collect transitions
-- in lockstep; one batched action-selection forward per outer step.
-- env-0 is the "primary" env — its episode boundary marks the end of
-- a training epoch.
----------------------------------------------------------------------

stepAllAutoResetDqn : Vect n CPState -> Vect n Nat ->
                     (Vect n CPState, Vect n Double, Vect n Bool)
stepAllAutoResetDqn [] [] = ([], [], [])
stepAllAutoResetDqn (s :: ss) (a :: as) =
  case cpStep s a of
    (r, s', outcome, _) =>
      let isDone = done outcome
          nextS  = if isDone then MkCP 0 0 0 0 else s'
      in case stepAllAutoResetDqn ss as of
           (rest, rs, ds) => (nextS :: rest, r :: rs, isDone :: ds)

pushAllTransitions : ReplayBuffer ObsDim 1 ->
                     Vect n CPState -> Vect n Nat -> Vect n Double ->
                     Vect n CPState -> Vect n Bool -> IO ()
pushAllTransitions _ [] [] [] [] [] = pure ()
pushAllTransitions buf (s :: ss) (a :: as) (r :: rs) (s' :: ss') (d :: ds) = do
  push buf (MkTransition (observeVec s) (actionToVec a) r (observeVec s') d)
  pushAllTransitions buf ss as rs ss' ds

runEpisodeBatched : Optimizer Ex -> DqnState -> IO (DqnState, Double)
runEpisodeBatched opt st0 = do
  startEnvs <- readIORef st0.envsRef
  go st0 startEnvs.envs MaxSteps 0.0
  where
    go : DqnState -> Vect NumEnvs CPState -> Nat -> Double -> IO (DqnState, Double)
    go st _ Z ret = do
      writeIORef st.envsRef (MkVecEnv (replicate NumEnvs (MkCP 0 0 0 0)))
      pure (st, ret)
    go st envs (S steps) ret = do
      stepCount <- readIORef st.stepRef
      let eps = epsilonAt stepCount st.cfgEpsStart st.cfgEpsEnd st.cfgEpsDecay
          obsRows : Vect NumEnvs (Vector ObsDim Double)
          obsRows = map (\s => obsTensor (observeVec s)) envs
          batchPtr = bulkToTensor2d {ex=Ex} {dt=F} obsRows
          stateV : Tensor [NumEnvs, ObsDim] Ex F WithGrad
          stateV = MkTensor batchPtr Nothing
      -- Batched action-selection forward: no grad needed.
      actions <- withNoGrad {ex=Ex} $ do
        qB <- forwardSeq {b=NumEnvs} st.qNet stateV
        epsGreedyBatched qB envs eps
      case stepAllAutoResetDqn envs actions of
        (envs', rewards, dones) => do
          pushAllTransitions st.buffer envs actions rewards envs' dones
          writeIORef st.stepRef (stepCount + 1)
          let ret0 : Double
              ret0 = head rewards
              done0 : Bool
              done0 = head dones
              ret' = ret + ret0

          st' <- trainIfReady opt st

          when ((stepCount + 1) `mod` st.cfgSyncEvery == 0) $ do
            _ <- polyakUpdate {ex=Ex} 1.0 "online" "target"
            pure ()

          if done0
            then do
              writeIORef st'.envsRef (MkVecEnv envs')
              pure (st', ret')
            else go st' envs' steps ret'

----------------------------------------------------------------------
-- Config & epoch
----------------------------------------------------------------------

record Config where
  constructor MkConfig
  lr         : Double
  epochs     : Nat
  gamma      : Double
  batchSize  : Nat
  bufferCap  : Nat
  targetSync : Nat
  epsStart   : Double
  epsEnd     : Double
  epsDecay   : Nat
  seed       : Bits64
  lrFind     : Bool

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

%default partial

main : IO ()
main = do
  requireMachine {m = ChosenMachine}
  args <- getArgs
  let cfg = parseArgs defaultConfig specs (drop 1 args)
  srand cfg.seed
  tsetInitSeed {ex = Ex} cfg.seed

  putStrLn "=== DQN on CartPole ==="
  putStrLn $ "Config: lr=" ++ show cfg.lr
           ++ " epochs=" ++ show cfg.epochs
           ++ " gamma=" ++ show cfg.gamma
           ++ " batch=" ++ show cfg.batchSize
           ++ " buffer=" ++ show cfg.bufferCap
           ++ " target_sync=" ++ show cfg.targetSync
           ++ " eps=" ++ show cfg.epsStart ++ "→" ++ show cfg.epsEnd
           ++ " seed=" ++ show cfg.seed

  qNet0 <- mkQNet "online"
  target0 <- mkQNet "target"
  -- Initial hard sync: target ← online (tau=1.0).
  _ <- polyakUpdate {ex=Ex} 1.0 "online" "target"

  buffer <- mkBuffer {obsDim = ObsDim, actDim = 1} cfg.bufferCap
  resetSeedI <- randomInt32
  let initEnvs : VecEnv NumEnvs CPState
      initEnvs = fst (resetAll {state=CPState} {action=Nat} {obs=Vect 4 Double}
                              (cast resetSeedI))
  envsRef <- newIORef initEnvs
  stepRef <- newIORef (the Nat 0)
  let st0 = MkDqnState qNet0 target0 buffer envsRef stepRef
                       cfg.epsStart cfg.epsEnd cfg.epsDecay
                       cfg.targetSync cfg.batchSize cfg.gamma
  -- Adam scoped to "online" only — the target net syncs via polyakUpdate.
  opt <- adam {scope="online"} cfg.lr ({ clip := NormClip 10.0 } defaultOpts)

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
      trainCfg = mkTrainConfig cfg.epochs 25 NoEarlyStop
                   (\_ => readRLMetrics "recent_50" metrics) (\_ => pure ())
  (trained, epochsDone, _) <- fit {batch = ()}
    (\st, _ => do
       (st', ret) <- runEpisodeBatched opt st
       recordReturn metrics ret
       pure (st', negate ret))
    opt (generate (pure ()))
    trainCfg st0

  putStrLn ""
  let nEval = the Nat 30
  totalReturn <- withNoGrad {ex=Ex} (evalN trained.qNet nEval 0.0)
  let avgReturn = totalReturn / cast (natToInteger nEval)
  putStrLn $ "Eval (" ++ show nEval ++ " episodes, greedy): avg_return=" ++ show avgReturn
  putStrLn ""
  putStrLn $ formatResult [("avg_return", show avgReturn),
                            ("epochs", show epochsDone),
                            ("seed", show cfg.seed)]
