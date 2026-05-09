module Example.MountainCarCont

import Data.List
import Data.Vect
import Data.IORef
import System
import Compat.Random

import Floating
import Gym.ClassicControl.MountainCarCont
import Gym.Env
import Layer.ActivationV2
import Layer.CoreV2
import Layer.LinearV2
import Math
import RL.ReplayBuffer
import Sampler
import Tensor
import Train
import Util
import Device
import Variable


----------------------------------------------------------------------
-- SAC on MountainCarContinuous-v0 with velocity-magnitude reward shaping.
--
-- MountainCarCont's reward is sparse (terminal +100 for reaching goal,
-- per-step -0.1*action²). Random Gaussian exploration almost never finds
-- the goal in 999 steps. Shape the *training* reward with `r_shaped =
-- r_raw + shaping * |v_next|` to densify toward kinetic energy. Eval
-- reports the *raw* return.
--
-- Architecture mirrors Example.Sac (separate actor + Q1 + Q2 + 2 target
-- Q nets, scoped paramIds, three Adam optimizers, polyak τ-soft target
-- update). Aligned with `torch_ref/models/mountain_car_cont.py`.
----------------------------------------------------------------------

ObsDim : Nat; ObsDim = 2
ActDim : Nat; ActDim = 1
QInputDim : Nat; QInputDim = 3          -- ObsDim + ActDim
Hidden : Nat; Hidden = 64
EpisodeLen : Nat; EpisodeLen = 999
MaxAct : Double; MaxAct = 1.0


-- --- Architectures --------------------------------------------------

ActorNet : Type
ActorNet = NetworkV2 ObsDim [Hidden, Hidden, Hidden, Hidden] 1 CPU

QNet : Type
QNet = NetworkV2 QInputDim [Hidden, Hidden, Hidden, Hidden] 1 CPU


mkActor : IO ActorNet
mkActor = do
  ll1 <- linearLayerV2Any {i=ObsDim} {o=Hidden} "actor_ll1"
  ll2 <- linearLayerV2Any {i=Hidden} {o=Hidden} "actor_ll2"
  ll3 <- linearLayerV2Any {i=Hidden} {o=1}      "actor_ll3"
  pure (ll1 ~~> reluLayerV2Any ~~> ll2 ~~> reluLayerV2Any ~~> OutputLayerV2 ll3)

mkQ : (scope : String) -> IO QNet
mkQ scope = do
  ll1 <- linearLayerV2Any {i=QInputDim} {o=Hidden} (scope ++ "ll1")
  ll2 <- linearLayerV2Any {i=Hidden} {o=Hidden}    (scope ++ "ll2")
  ll3 <- linearLayerV2Any {i=Hidden} {o=1}         (scope ++ "ll3")
  pure (ll1 ~~> reluLayerV2Any ~~> ll2 ~~> reluLayerV2Any ~~> OutputLayerV2 ll3)


-- --- Observation helpers --------------------------------------------

observeVec : MCCState -> Vect ObsDim Double
observeVec s = mccObserve s

obsTensor : Vect ObsDim Double -> Vector ObsDim Double
obsTensor v = VTensor (map STensor v)

qInput : Vect ObsDim Double -> Double -> Vect QInputDim Double
qInput obs a = obs ++ [a]

qInputTensor : Vect QInputDim Double -> Vector QInputDim Double
qInputTensor v = VTensor (map STensor v)


-- --- Gaussian / squash helpers --------------------------------------

logTwoPiHalf : Double
logTwoPiHalf = 0.5 * Prelude.log (2.0 * 3.141592653589793)

squashCorrection : Double -> Double
squashCorrection u =
  let tu = Math.tanh u
  in Prelude.log (1.0 - tu * tu + 1.0e-6) + Prelude.log MaxAct


-- --- Forward helpers (single-sample) --------------------------------

actorMean : ActorNet -> Vect ObsDim Double -> Double
actorMean actor obs =
  let stateV = the (TVec ObsDim CPU) (MkTVar (bulkToTensor (obsTensor obs)) Nothing)
      outV = snd (forwardTVar actor stateV)
  in prim__item1d outV.tensorPtr 0

qValue : QNet -> Vect ObsDim Double -> Double -> Double
qValue q obs action =
  let inV = the (TVec QInputDim CPU)
                (MkTVar (bulkToTensor (qInputTensor (qInput obs action))) Nothing)
      outV = snd (forwardTVar q inV)
  in prim__item1d outV.tensorPtr 0


sampleActionIO : ActorNet -> TVar [] CPU -> Vect ObsDim Double ->
                 IO (Double, Double)
sampleActionIO actor logStdV obs = do
  let mean = actorMean actor obs
      logStd = prim__item logStdV.tensorPtr
      std = Prelude.exp logStd
  eps <- normalSample
  let u = mean + std * eps
      action = Math.tanh u * MaxAct
      lp_u = -0.5 * ((u - mean) / std) * ((u - mean) / std) - logStd - logTwoPiHalf
      lp = lp_u - squashCorrection u
  pure (action, lp)


-- --- SAC state -------------------------------------------------------

record SACState where
  constructor MkSAC
  actor   : ActorNet
  q1      : QNet
  q2      : QNet
  q1Tgt   : QNet
  q2Tgt   : QNet
  logStdV : TVar [] CPU
  buffer  : ReplayBuffer ObsDim ActDim
  stepRef : IORef Nat
  envRef  : IORef MCCState
  epLenRef : IORef Nat
  retRef  : IORef Double
  lastEpRef : IORef Double


record Config where
  constructor MkConfig
  lr           : Double
  epochs       : Nat
  gamma        : Double
  alpha        : Double
  bufferCap    : Nat
  batchSize    : Nat
  warmupSteps  : Nat
  tau          : Double
  shaping      : Double
  clipNorm     : Double
  seed         : Bits64
  esThreshold  : Double
  esWindow     : Nat
  esPatience   : Nat
  lrFind       : Bool

defaultConfig : Config
defaultConfig = MkConfig 3.0e-4 30000 0.99 0.2 100000 64 1000 0.005 10.0 1.0 42
                         (-85.0) 500 5 False

specs : List (ArgSpec Config)
specs = [ Arg "--lr" (\v, c => { lr := cast v } c)
        , Arg "--epochs" (\v, c => { epochs := castNat v } c)
        , Arg "--gamma" (\v, c => { gamma := cast v } c)
        , Arg "--alpha" (\v, c => { alpha := cast v } c)
        , Arg "--buffer-cap" (\v, c => { bufferCap := castNat v } c)
        , Arg "--batch" (\v, c => { batchSize := castNat v } c)
        , Arg "--warmup" (\v, c => { warmupSteps := castNat v } c)
        , Arg "--tau" (\v, c => { tau := cast v } c)
        , Arg "--shaping" (\v, c => { shaping := cast v } c)
        , Arg "--clip" (\v, c => { clipNorm := cast v } c)
        , Arg "--seed" (\v, c => { seed := castBits64 v } c)
        , Arg "--es-threshold" (\v, c => { esThreshold := cast v } c)
        , Arg "--es-window" (\v, c => { esWindow := castNat v } c)
        , Arg "--es-patience" (\v, c => { esPatience := castNat v } c)
        , Arg "--lr-find" (\v, c => { lrFind := (v == "1" || v == "true") } c)
        ]


-- --- Q-network loss (batched) ---------------------------------------

computeTargetVal : QNet -> QNet -> ActorNet -> TVar [] CPU ->
                   Double -> Double -> Transition ObsDim ActDim -> IO Double
computeTargetVal q1Tgt q2Tgt actor logStdV gamma alpha t = do
  nextPair <- sampleActionIO actor logStdV t.nextObs
  let nextAction = fst nextPair
      nextLogP = snd nextPair
      q1NextD = qValue q1Tgt t.nextObs nextAction
      q2NextD = qValue q2Tgt t.nextObs nextAction
      minQNextD = if q1NextD <= q2NextD then q1NextD else q2NextD
      doneMask = if t.done then 0.0 else 1.0
  pure (t.reward + gamma * doneMask * (minQNextD - alpha * nextLogP))

perSampleQLoss : {n : Nat} -> (qOutB : TVar [n, 1] CPU) -> Double ->
                 Int -> TVar [] CPU
perSampleQLoss qOutB tv k =
  let qRow = the (TVec 1 CPU) (trowSelect qOutB k)
      qScalar = the (TVar [] CPU) (telemSelect qRow 0)
      targetT = the (TVar [] CPU) (tconstScalar tv)
      diff = the (TVar [] CPU) (tsub qScalar targetT)
  in tmul diff diff

meanScalarLoss : (n : Nat) -> List (TVar [] CPU) -> TVar [] CPU
meanScalarLoss n losses =
  let zero = tconstScalar 0.0
      summed = foldl (\a, b => MkTVar (prim__add a.tensorPtr b.tensorPtr) Nothing) zero losses
  in tmulScalar summed (1.0 / cast n)

qLossBatch : (n : Nat) -> QNet -> QNet -> QNet -> ActorNet -> TVar [] CPU ->
             Double -> Double -> Vect n (Transition ObsDim ActDim) ->
             IO (TVar [] CPU)
qLossBatch n qOnline q1Tgt q2Tgt actor logStdV gamma alpha batch = do
  targetVals <- traverse (computeTargetVal q1Tgt q2Tgt actor logStdV gamma alpha) batch
  let qInputs = the (Vect n (Vector QInputDim Double))
                    (map (\t => qInputTensor (qInput t.obs (oneAct t.action))) batch)
      qInputBT = bulkToTensor2d qInputs
      qInputV = the (TVar [n, QInputDim] CPU) (MkTVar qInputBT Nothing)
      qOutB = snd (forwardTVarBatch qOnline qInputV)
      losses = the (List (TVar [] CPU)) (go qOutB (toList targetVals) 0)
  pure (meanScalarLoss n losses)
  where
    oneAct : Vect ActDim Double -> Double
    oneAct [a] = a
    go : {n : Nat} -> TVar [n, 1] CPU -> List Double -> Int -> List (TVar [] CPU)
    go _ [] _ = []
    go qOutB (tv :: rest) k =
      perSampleQLoss qOutB tv k :: go qOutB rest (k + 1)


-- --- Actor loss with reparameterization -----------------------------

buildScalarColumn : {n : Nat} -> Vect n Double -> TVar [n, 1] CPU
buildScalarColumn {n} xs =
  let rows = the (Vect n (Vector 1 Double)) (map (\x => VTensor [STensor x]) xs)
      ptr = bulkToTensor2d rows
  in MkTVar ptr Nothing

actorPerStepLoss : {n : Nat} ->
                   TVar [n, 1] CPU -> TVar [n, 1] CPU ->
                   TVar [n, 1] CPU -> TVar [n, 1] CPU ->
                   TVar [] CPU -> Double ->
                   Int -> TVar [] CPU
actorPerStepLoss meanB uBT q1B q2B logStdV alpha rowIdx =
  let q1Row = the (TVec 1 CPU) (trowSelect q1B rowIdx)
      q1S = the (TVar [] CPU) (telemSelect q1Row 0)
      q1Val = prim__item1d q1Row.tensorPtr 0
      q2Row = the (TVec 1 CPU) (trowSelect q2B rowIdx)
      q2S = the (TVar [] CPU) (telemSelect q2Row 0)
      q2Val = prim__item1d q2Row.tensorPtr 0
      minQS = if q1Val <= q2Val then q1S else q2S

      meanRow = the (TVec 1 CPU) (trowSelect meanB rowIdx)
      meanS = the (TVar [] CPU) (telemSelect meanRow 0)

      uRow = the (TVec 1 CPU) (trowSelect uBT rowIdx)
      uS = the (TVar [] CPU) (telemSelect uRow 0)
      uVal = prim__item1d uRow.tensorPtr 0

      diffM = tsub uS meanS
      negTwoLs = tmulScalar logStdV (-2.0)
      varInv = texp negTwoLs
      diffSq = tmul diffM diffM
      quad = tmulScalar (tmul diffSq varInv) 0.5
      cC = tconstScalar logTwoPiHalf
      lpU = tsub (tsub (tneg quad) logStdV) cC
      corrC = tconstScalar (squashCorrection uVal)
      lpV = tsub lpU corrC

      alphaLogP = tmulScalar lpV alpha
  in tsub alphaLogP minQS

actorLossBatch : (n : Nat) -> ActorNet -> QNet -> QNet -> TVar [] CPU ->
                 Double -> Vect n (Vect ObsDim Double) -> IO (TVar [] CPU)
actorLossBatch n actor q1 q2 logStdV alpha obsBatch = do
  let logStd = prim__item logStdV.tensorPtr
      stdVal = Prelude.exp logStd
  epses <- traverse (\_ => normalSample) obsBatch
  let obsTensors = the (Vect n (Vector ObsDim Double)) (map obsTensor obsBatch)
      obsBT = bulkToTensor2d obsTensors
      obsBV = the (TVar [n, ObsDim] CPU) (MkTVar obsBT Nothing)
      meanB = snd (forwardTVarBatch actor obsBV)
      epsScales = map (\e => stdVal * e) epses
      epsBV = buildScalarColumn epsScales
      uBT = tadd meanB epsBV
      aSquashedBT = ttanh uBT
      aReparamBT = tmulScalar aSquashedBT MaxAct
      qInputBT = tconcat2dAxis1 obsBV aReparamBT
      q1B = snd (forwardTVarBatch q1 qInputBT)
      q2B = snd (forwardTVarBatch q2 qInputBT)
      losses = the (List (TVar [] CPU)) (go meanB uBT q1B q2B (toList epses) 0)
  pure (meanScalarLoss n losses)
  where
    go : {n : Nat} ->
         TVar [n, 1] CPU -> TVar [n, 1] CPU ->
         TVar [n, 1] CPU -> TVar [n, 1] CPU ->
         List Double -> Int -> List (TVar [] CPU)
    go _ _ _ _ [] _ = []
    go meanB uBT q1B q2B (_ :: rest) k =
      actorPerStepLoss meanB uBT q1B q2B logStdV alpha k
        :: go meanB uBT q1B q2B rest (k + 1)


-- --- Batch update ---------------------------------------------------

runBatchUpdate : NativeOptimizer -> NativeOptimizer -> NativeOptimizer ->
                 SACState -> Config -> {n : Nat} ->
                 Vect n (Transition ObsDim ActDim) -> IO ()
runBatchUpdate q1Opt q2Opt actorOpt st cfg {n} batch = do
  q1LossV <- qLossBatch n st.q1 st.q1Tgt st.q2Tgt st.actor st.logStdV
                        cfg.gamma cfg.alpha batch
  _ <- pure (nativeTrainStepTVar q1Opt q1LossV)
  q2LossV <- qLossBatch n st.q2 st.q1Tgt st.q2Tgt st.actor st.logStdV
                        cfg.gamma cfg.alpha batch
  _ <- pure (nativeTrainStepTVar q2Opt q2LossV)
  let obsVec = the (Vect n (Vect ObsDim Double)) (map (\t => t.obs) batch)
  aLossV <- actorLossBatch n st.actor st.q1 st.q2 st.logStdV cfg.alpha obsVec
  _ <- pure (nativeTrainStepTVar actorOpt aLossV)
  pure ()


-- --- Main loop ------------------------------------------------------

sacStep : NativeOptimizer -> NativeOptimizer -> NativeOptimizer ->
          Config -> SACState -> IO (SACState, Double)
sacStep q1Opt q2Opt actorOpt cfg st = do
  stepCount <- readIORef st.stepRef
  envState <- readIORef st.envRef
  epLen <- readIORef st.epLenRef
  let obs = observeVec envState

  action <- if stepCount < cfg.warmupSteps
              then randomRIO (the Double (negate MaxAct), MaxAct)
              else do
                pair <- sampleActionIO st.actor st.logStdV obs
                pure (fst pair)

  case mccStep envState action of
    (rawR, envState', outcome, _) => do
      let nextObs = observeVec envState'
          terminated = case outcome of
                         Terminated => True
                         _          => False
          truncated = (epLen + 1) >= EpisodeLen
          isDone = terminated || truncated
          bufferDone = terminated  -- bootstrap continues at truncation boundaries
          shapedR = rawR + cfg.shaping * abs envState'.mccVel
          nextSt = if isDone then MkMCC (-0.5) 0.0 else envState'
          trans = MkTransition obs [action] shapedR nextObs bufferDone
      push st.buffer trans
      writeIORef st.envRef nextSt
      writeIORef st.stepRef (stepCount + 1)
      writeIORef st.epLenRef (if isDone then 0 else epLen + 1)

      runRet <- readIORef st.retRef
      let newRet = runRet + rawR
      if isDone
        then do writeIORef st.lastEpRef newRet
                writeIORef st.retRef 0.0
        else writeIORef st.retRef newRet

      bufSz <- bufferSize st.buffer
      _ <- if bufSz >= cfg.batchSize && stepCount >= cfg.warmupSteps
             then do
               mBatch <- sampleN cfg.batchSize st.buffer
               case mBatch of
                 Nothing => pure ()
                 Just batch => do
                   runBatchUpdate q1Opt q2Opt actorOpt st cfg batch
                   _ <- polyakUpdate cfg.tau "q1_" "q1tgt_"
                   _ <- polyakUpdate cfg.tau "q2_" "q2tgt_"
                   pure ()
             else pure ()

      lastEp <- readIORef st.lastEpRef
      pure (st, negate lastEp)


-- --- Greedy evaluation ----------------------------------------------

greedyAct : ActorNet -> Vect ObsDim Double -> Double
greedyAct actor obs =
  let mean = actorMean actor obs
  in Math.tanh mean * MaxAct

evalEp : ActorNet -> MCCState -> Nat -> Double -> Double
evalEp _ _ Z acc = acc
evalEp actor st (S k) acc =
  let a = greedyAct actor (mccObserve st)
  in case mccStep st a of
       (r, st', outcome, _) =>
         case outcome of
           Terminated => acc + r
           _          => evalEp actor st' k (acc + r)

evalN : ActorNet -> Nat -> Double -> Double
evalN _ Z acc = acc
evalN actor (S k) acc =
  evalN actor k (acc + evalEp actor (MkMCC (-0.5) 0.0) EpisodeLen 0.0)


-- --- Main -----------------------------------------------------------

main : IO ()
main = do
  args <- getArgs
  let cfg = parseArgs defaultConfig specs (drop 1 args)
  srand cfg.seed

  putStrLn "=== SAC on MountainCarContinuous ==="
  putStrLn $ "Config: lr=" ++ show cfg.lr
           ++ " steps=" ++ show cfg.epochs
           ++ " gamma=" ++ show cfg.gamma
           ++ " alpha=" ++ show cfg.alpha
           ++ " batch=" ++ show cfg.batchSize
           ++ " warmup=" ++ show cfg.warmupSteps
           ++ " tau=" ++ show cfg.tau
           ++ " shaping=" ++ show cfg.shaping
           ++ " seed=" ++ show cfg.seed

  actor <- mkActor
  q1 <- mkQ "q1_"
  q2 <- mkQ "q2_"
  q1Tgt <- mkQ "q1tgt_"
  q2Tgt <- mkQ "q2tgt_"
  let logStdV = the (TVar [] CPU) (tparamScalar "actor_log_std" 0.0)

  _ <- polyakUpdate 1.0 "q1_" "q1tgt_"
  _ <- polyakUpdate 1.0 "q2_" "q2tgt_"

  buffer <- mkBuffer {obsDim=ObsDim, actDim=ActDim} cfg.bufferCap
  stepRef <- newIORef (the Nat 0)
  envRef <- newIORef (the MCCState (MkMCC (-0.5) 0.0))
  epLenRef <- newIORef (the Nat 0)
  retRef <- newIORef (the Double 0.0)
  lastEpRef <- newIORef (the Double 0.0)

  let st0 = MkSAC actor q1 q2 q1Tgt q2Tgt logStdV buffer stepRef envRef epLenRef retRef lastEpRef
      actorOpt = nativeAdamGroup "actor_" cfg.lr 0.9 0.999 1.0e-8 cfg.clipNorm
      q1Opt    = nativeAdamGroup "q1_"    cfg.lr 0.9 0.999 1.0e-8 cfg.clipNorm
      q2Opt    = nativeAdamGroup "q2_"    cfg.lr 0.9 0.999 1.0e-8 cfg.clipNorm

  putStrLn ""

  when cfg.lrFind $ do
    putStrLn "lr_find skipped for SAC: per-step epochs + warmup don't fit"
    putStrLn "the LR-range-test pattern. See docs/develop/hyperparameter-tuning-2026.md."
    exitSuccess

  let trainCfg : TrainConfig SACState
      trainCfg = MkTrainConfig cfg.epochs 2000
                            (WindowedAvg cfg.esThreshold cfg.esWindow cfg.esPatience)
                            (const (pure [])) (\_ => pure ())
  (trained, epochsDone, _) <- runTrainingIO
    (\s, _ => sacStep q1Opt q2Opt actorOpt cfg s)
    (pure ())
    trainCfg st0

  putStrLn ""
  let nEval = the Nat 20
      avgReturn = evalN trained.actor nEval 0.0 / cast (natToInteger nEval)
  putStrLn $ "Eval (" ++ show nEval ++ " episodes, greedy): avg_return=" ++ show avgReturn
  putStrLn ""
  putStrLn $ formatResult [("avg_return", show avgReturn),
                            ("epochs", show epochsDone),
                            ("seed", show cfg.seed)]
