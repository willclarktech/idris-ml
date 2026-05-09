module Example.Sac

import Data.List
import Data.Vect
import Data.IORef
import System
import Compat.Random

import Floating
import Gym.ClassicControl.Pendulum
import Gym.Env
import Layer.Activation
import Layer.Core
import Layer.Linear
import Math
import RL.ReplayBuffer
import Sampler
import Array
import Train
import Util
import Device
import Tensor


----------------------------------------------------------------------
-- Architecture (aligned with `torch_ref/models/sac.py`):
--   Actor : Linear(3,64) → ReLU → Linear(64,64) → ReLU → Linear(64,1) = mean
--   Q1/Q2 : Linear(4,64) → ReLU → Linear(64,64) → ReLU → Linear(64,1) = value
--           (input is obs ++ action)
--   log_std : standalone learnable Tensor [] CPU under "actor_log_std".
--   Target Q nets: same architecture, scoped "q1tgt_" / "q2tgt_". No
--                  optimizer owns them; they move via polyak soft update.
----------------------------------------------------------------------

ObsDim : Nat; ObsDim = 3
ActDim : Nat; ActDim = 1
QInputDim : Nat; QInputDim = 4          -- ObsDim + ActDim
Hidden : Nat; Hidden = 64
EpisodeLen : Nat; EpisodeLen = 200
MaxAction : Double; MaxAction = 2.0


-- --- Architectures --------------------------------------------------

ActorNet : Type
ActorNet = Network ObsDim [Hidden, Hidden, Hidden, Hidden] 1 CPU

QNet : Type
QNet = Network QInputDim [Hidden, Hidden, Hidden, Hidden] 1 CPU


mkActor : IO ActorNet
mkActor = do
  ll1 <- linearLayerAny {i=ObsDim} {o=Hidden} "actor_ll1"
  ll2 <- linearLayerAny {i=Hidden} {o=Hidden} "actor_ll2"
  ll3 <- linearLayerAny {i=Hidden} {o=1}      "actor_ll3"
  pure (ll1 ~~> reluLayerAny ~~> ll2 ~~> reluLayerAny ~~> OutputLayer ll3)

mkQ : (scope : String) -> IO QNet
mkQ scope = do
  ll1 <- linearLayerAny {i=QInputDim} {o=Hidden} (scope ++ "ll1")
  ll2 <- linearLayerAny {i=Hidden} {o=Hidden}    (scope ++ "ll2")
  ll3 <- linearLayerAny {i=Hidden} {o=1}         (scope ++ "ll3")
  pure (ll1 ~~> reluLayerAny ~~> ll2 ~~> reluLayerAny ~~> OutputLayer ll3)


-- --- Observation helpers --------------------------------------------

observeVec : PState -> Vect ObsDim Double
observeVec s = pObserve s

obsTensor : Vect ObsDim Double -> Vector ObsDim Double
obsTensor v = VArray (map SArray v)

qInput : Vect ObsDim Double -> Double -> Vect QInputDim Double
qInput obs a = obs ++ [a]

qInputTensor : Vect QInputDim Double -> Vector QInputDim Double
qInputTensor v = VArray (map SArray v)


-- --- Gaussian / squash helpers --------------------------------------

logTwoPiHalf : Double
logTwoPiHalf = 0.5 * Prelude.log (2.0 * 3.141592653589793)

squashCorrection : Double -> Double
squashCorrection u =
  let tu = Math.tanh u
  in Prelude.log (1.0 - tu * tu + 1.0e-6) + Prelude.log MaxAction


-- --- Forward helpers (single-sample, pure-Double outputs) ------------

actorMean : ActorNet -> Vect ObsDim Double -> Double
actorMean actor obs =
  let stateV = the (TVec ObsDim CPU) (MkTensor (bulkToTensor (obsTensor obs)) Nothing)
      outV = snd (forwardVar actor stateV)
  in prim__item1d outV.tensorPtr 0

qValue : QNet -> Vect ObsDim Double -> Double -> Double
qValue q obs action =
  let inV = the (TVec QInputDim CPU)
                (MkTensor (bulkToTensor (qInputTensor (qInput obs action))) Nothing)
      outV = snd (forwardVar q inV)
  in prim__item1d outV.tensorPtr 0


-- Sample a squashed Gaussian action — pure-Double, used for rollout.
sampleActionIO : ActorNet -> Tensor [] CPU -> Vect ObsDim Double ->
                 IO (Double, Double)
sampleActionIO actor logStdV obs = do
  let mean = actorMean actor obs
      logStd = prim__item logStdV.tensorPtr
      std = Prelude.exp logStd
  eps <- normalSample
  let u = mean + std * eps
      action = Math.tanh u * MaxAction
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
  logStdV : Tensor [] CPU
  buffer  : ReplayBuffer ObsDim ActDim
  stepRef : IORef Nat
  envRef  : IORef PState
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
  clipNorm     : Double
  seed         : Bits64
  esThreshold  : Double
  esWindow     : Nat
  esPatience   : Nat
  lrFind       : Bool

defaultConfig : Config
defaultConfig = MkConfig 3.0e-4 30000 0.99 0.2 100000 64 1000 0.005 1.0 42
                         500.0 1000 100 False

specs : List (ArgSpec Config)
specs = [ Arg "--lr" (\v, c => { lr := cast v } c)
        , Arg "--epochs" (\v, c => { epochs := castNat v } c)
        , Arg "--gamma" (\v, c => { gamma := cast v } c)
        , Arg "--alpha" (\v, c => { alpha := cast v } c)
        , Arg "--buffer-cap" (\v, c => { bufferCap := castNat v } c)
        , Arg "--batch" (\v, c => { batchSize := castNat v } c)
        , Arg "--warmup" (\v, c => { warmupSteps := castNat v } c)
        , Arg "--tau" (\v, c => { tau := cast v } c)
        , Arg "--clip" (\v, c => { clipNorm := cast v } c)
        , Arg "--seed" (\v, c => { seed := castBits64 v } c)
        , Arg "--es-threshold" (\v, c => { esThreshold := cast v } c)
        , Arg "--es-window" (\v, c => { esWindow := castNat v } c)
        , Arg "--es-patience" (\v, c => { esPatience := castNat v } c)
        , Arg "--lr-find" (\v, c => { lrFind := (v == "1" || v == "true") } c)
        ]


-- --- Q-network loss (batched) ---------------------------------------

computeTargetVal : QNet -> QNet -> ActorNet -> Tensor [] CPU ->
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

-- Per-sample MSE loss for a [B, 1] Q-output indexed by row k against
-- a Double target. Mirrors Dqn's perSampleLoss but with a single
-- Q-column (action dim is fixed at the input).
perSampleQLoss : {n : Nat} -> (qOutB : Tensor [n, 1] CPU) -> Double ->
                 Int -> Tensor [] CPU
perSampleQLoss qOutB tv k =
  let qRow = the (TVec 1 CPU) (trowSelect qOutB k)
      qScalar = the (Tensor [] CPU) (telemSelect qRow 0)
      targetT = the (Tensor [] CPU) (tconstScalar tv)
      diff = the (Tensor [] CPU) (tsub qScalar targetT)
  in tmul diff diff

meanScalarLoss : (n : Nat) -> List (Tensor [] CPU) -> Tensor [] CPU
meanScalarLoss n losses =
  let zero = tconstScalar 0.0
      summed = foldl (\a, b => MkTensor (prim__add a.tensorPtr b.tensorPtr) Nothing) zero losses
  in tmulScalar summed (1.0 / cast n)

qLossBatch : (n : Nat) -> QNet -> QNet -> QNet -> ActorNet -> Tensor [] CPU ->
             Double -> Double -> Vect n (Transition ObsDim ActDim) ->
             IO (Tensor [] CPU)
qLossBatch n qOnline q1Tgt q2Tgt actor logStdV gamma alpha batch = do
  targetVals <- traverse (computeTargetVal q1Tgt q2Tgt actor logStdV gamma alpha) batch
  let qInputs = the (Vect n (Vector QInputDim Double))
                    (map (\t => qInputTensor (qInput t.obs (oneAct t.action))) batch)
      qInputBT = bulkToTensor2d qInputs
      qInputV = the (Tensor [n, QInputDim] CPU) (MkTensor qInputBT Nothing)
      qOutB = snd (forwardVarBatch qOnline qInputV)
      losses = the (List (Tensor [] CPU)) (go qOutB (toList targetVals) 0)
  pure (meanScalarLoss n losses)
  where
    oneAct : Vect ActDim Double -> Double
    oneAct [a] = a
    go : {n : Nat} -> Tensor [n, 1] CPU -> List Double -> Int -> List (Tensor [] CPU)
    go _ [] _ = []
    go qOutB (tv :: rest) k =
      perSampleQLoss qOutB tv k :: go qOutB rest (k + 1)


-- --- Actor loss with reparameterization -----------------------------

-- Build a [n, 1] non-grad Tensor from a Vect of Doubles (one row each).
buildScalarColumn : {n : Nat} -> Vect n Double -> Tensor [n, 1] CPU
buildScalarColumn {n} xs =
  let rows = the (Vect n (Vector 1 Double)) (map (\x => VArray [SArray x]) xs)
      ptr = bulkToTensor2d rows
  in MkTensor ptr Nothing

-- Per-sample reparameterized loss. Indexes into the [n, 1] mean / u /
-- q1 / q2 batched outputs and builds the grad-tracked
-- alpha · log_prob - min(Q1, Q2) expression.
actorPerStepLoss : {n : Nat} ->
                   Tensor [n, 1] CPU -> Tensor [n, 1] CPU ->
                   Tensor [n, 1] CPU -> Tensor [n, 1] CPU ->
                   Tensor [] CPU -> Double ->
                   Int ->
                   Tensor [] CPU
actorPerStepLoss meanB uBT q1B q2B logStdV alpha rowIdx =
  let q1Row = the (TVec 1 CPU) (trowSelect q1B rowIdx)
      q1S = the (Tensor [] CPU) (telemSelect q1Row 0)
      q1Val = prim__item1d q1Row.tensorPtr 0
      q2Row = the (TVec 1 CPU) (trowSelect q2B rowIdx)
      q2S = the (Tensor [] CPU) (telemSelect q2Row 0)
      q2Val = prim__item1d q2Row.tensorPtr 0
      minQS = if q1Val <= q2Val then q1S else q2S

      meanRow = the (TVec 1 CPU) (trowSelect meanB rowIdx)
      meanS = the (Tensor [] CPU) (telemSelect meanRow 0)

      uRow = the (TVec 1 CPU) (trowSelect uBT rowIdx)
      uS = the (Tensor [] CPU) (telemSelect uRow 0)
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

actorLossBatch : (n : Nat) -> ActorNet -> QNet -> QNet -> Tensor [] CPU ->
                 Double -> Vect n (Vect ObsDim Double) -> IO (Tensor [] CPU)
actorLossBatch n actor q1 q2 logStdV alpha obsBatch = do
  let logStd = prim__item logStdV.tensorPtr
      stdVal = Prelude.exp logStd
  epses <- traverse (\_ => normalSample) obsBatch
  let obsTensors = the (Vect n (Vector ObsDim Double)) (map obsTensor obsBatch)
      obsBT = bulkToTensor2d obsTensors
      obsBV = the (Tensor [n, ObsDim] CPU) (MkTensor obsBT Nothing)
      meanB = snd (forwardVarBatch actor obsBV)             -- [n, 1] grad
      epsScales = map (\e => stdVal * e) epses
      epsBV = buildScalarColumn epsScales                     -- [n, 1] non-grad
      uBT = tadd meanB epsBV                                  -- [n, 1] grad
      aSquashedBT = ttanh uBT                                 -- [n, 1] grad
      aReparamBT = tmulScalar aSquashedBT MaxAction           -- [n, 1] grad
      qInputBT = tconcat2dAxis1 obsBV aReparamBT              -- [n, 4] grad
      q1B = snd (forwardVarBatch q1 qInputBT)                -- [n, 1] grad
      q2B = snd (forwardVarBatch q2 qInputBT)                -- [n, 1] grad
      losses = the (List (Tensor [] CPU)) (go meanB uBT q1B q2B (toList epses) 0)
  pure (meanScalarLoss n losses)
  where
    go : {n : Nat} ->
         Tensor [n, 1] CPU -> Tensor [n, 1] CPU ->
         Tensor [n, 1] CPU -> Tensor [n, 1] CPU ->
         List Double -> Int -> List (Tensor [] CPU)
    go _ _ _ _ [] _ = []
    go meanB uBT q1B q2B (_ :: rest) k =
      actorPerStepLoss meanB uBT q1B q2B logStdV alpha k
        :: go meanB uBT q1B q2B rest (k + 1)


-- --- One batch update: three group-scoped optimizer steps ------------

runBatchUpdate : NativeOptimizer -> NativeOptimizer -> NativeOptimizer ->
                 SACState -> Config -> {n : Nat} ->
                 Vect n (Transition ObsDim ActDim) -> IO ()
runBatchUpdate q1Opt q2Opt actorOpt st cfg {n} batch = do
  q1LossV <- qLossBatch n st.q1 st.q1Tgt st.q2Tgt st.actor st.logStdV
                        cfg.gamma cfg.alpha batch
  _ <- pure (nativeTrainStep q1Opt q1LossV)

  q2LossV <- qLossBatch n st.q2 st.q1Tgt st.q2Tgt st.actor st.logStdV
                        cfg.gamma cfg.alpha batch
  _ <- pure (nativeTrainStep q2Opt q2LossV)

  let obsVec = the (Vect n (Vect ObsDim Double)) (map (\t => t.obs) batch)
  aLossV <- actorLossBatch n st.actor st.q1 st.q2 st.logStdV cfg.alpha obsVec
  _ <- pure (nativeTrainStep actorOpt aLossV)
  pure ()


-- --- Main loop -------------------------------------------------------

sacStep : NativeOptimizer -> NativeOptimizer -> NativeOptimizer ->
          Config -> SACState -> IO (SACState, Double)
sacStep q1Opt q2Opt actorOpt cfg st = do
  stepCount <- readIORef st.stepRef
  envState <- readIORef st.envRef
  let obs = observeVec envState

  action <- if stepCount < cfg.warmupSteps
              then randomRIO (the Double (negate MaxAction), MaxAction)
              else do
                pair <- sampleActionIO st.actor st.logStdV obs
                pure (fst pair)

  case pStep envState action of
    (r, envState', _, _) => do
      let nextObs = observeVec envState'
          isDone = (stepCount + 1) `mod` EpisodeLen == 0
          nextSt = if isDone then MkP 3.141592653589793 0.0 else envState'
          trans = MkTransition obs [action] r nextObs isDone
      push st.buffer trans
      writeIORef st.envRef nextSt
      writeIORef st.stepRef (stepCount + 1)

      runRet <- readIORef st.retRef
      let newRet = runRet + r
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
  in Math.tanh mean * MaxAction

evalEp : ActorNet -> PState -> Nat -> Double -> Double
evalEp _ _ Z acc = acc
evalEp actor st (S k) acc =
  let a = greedyAct actor (observeVec st)
  in case pStep st a of
       (r, st', _, _) => evalEp actor st' k (acc + r)

evalN : ActorNet -> Nat -> Double -> Double
evalN _ Z acc = acc
evalN actor (S k) acc =
  evalN actor k (acc + evalEp actor (MkP 3.141592653589793 0.0) EpisodeLen 0.0)


-- --- Main -----------------------------------------------------------

main : IO ()
main = do
  args <- getArgs
  let cfg = parseArgs defaultConfig specs (drop 1 args)
  srand cfg.seed

  putStrLn "=== SAC on Pendulum ==="
  putStrLn $ "Config: lr=" ++ show cfg.lr
           ++ " steps=" ++ show cfg.epochs
           ++ " gamma=" ++ show cfg.gamma
           ++ " alpha=" ++ show cfg.alpha
           ++ " batch=" ++ show cfg.batchSize
           ++ " warmup=" ++ show cfg.warmupSteps
           ++ " tau=" ++ show cfg.tau
           ++ " seed=" ++ show cfg.seed

  actor <- mkActor
  q1 <- mkQ "q1_"
  q2 <- mkQ "q2_"
  q1Tgt <- mkQ "q1tgt_"
  q2Tgt <- mkQ "q2tgt_"
  let logStdV = the (Tensor [] CPU) (tparamScalar "actor_log_std" 0.0)

  -- Hard-copy online → target at init.
  _ <- polyakUpdate 1.0 "q1_" "q1tgt_"
  _ <- polyakUpdate 1.0 "q2_" "q2tgt_"

  buffer <- mkBuffer {obsDim=ObsDim, actDim=ActDim} cfg.bufferCap
  stepRef <- newIORef (the Nat 0)
  envRef <- newIORef (the PState (MkP 3.141592653589793 0.0))
  retRef <- newIORef (the Double 0.0)
  lastEpRef <- newIORef (the Double 0.0)

  let st0 = MkSAC actor q1 q2 q1Tgt q2Tgt logStdV buffer stepRef envRef retRef lastEpRef
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
