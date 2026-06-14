module Example.Sac

import Data.List
import Data.Vect
import Data.IORef
import System
import Compat.Random

import Floating
import Gym.ClassicControl.Pendulum
import Gym.Env
import Gym.Vector
import Layer.Activation
import Layer.Core
import Layer.Linear
import Math
import RL.ReplayBuffer
import Sampler
import Array
import Train
import Util
import Executor
import Tensor
import BuildConfig

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

||| Parallel envs collecting transitions in lockstep. Each outer step
||| advances NumEnvs envs through one batched actor forward + one
||| batched gradient update on a sample drawn from the shared buffer.
NumEnvs : Nat; NumEnvs = 4

-- --- Architectures --------------------------------------------------

ActorNet : Type
ActorNet = Network ObsDim [Hidden, Hidden, Hidden, Hidden] 1 ExampleExecutor ExampleDType WithGrad

QNet : Type
QNet = Network QInputDim [Hidden, Hidden, Hidden, Hidden] 1 ExampleExecutor ExampleDType WithGrad

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

actorMean : ActorNet -> Vect ObsDim Double -> IO Double
actorMean actor obs = do
  let stateV = the (TVec ObsDim ExampleExecutor ExampleDType WithGrad) (MkTensor (bulkToTensor {ex=ExampleExecutor} {dt=ExampleDType} (obsTensor obs)) Nothing)
  (_, outV) <- forwardVar actor stateV
  pure (primItem1d {ex=ExampleExecutor} outV.tensorPtr 0)

qValue : QNet -> Vect ObsDim Double -> Double -> IO Double
qValue q obs action = do
  let inV = the (TVec QInputDim ExampleExecutor ExampleDType WithGrad)
                (MkTensor (bulkToTensor {ex=ExampleExecutor} {dt=ExampleDType} (qInputTensor (qInput obs action))) Nothing)
  (_, outV) <- forwardVar q inV
  pure (primItem1d {ex=ExampleExecutor} outV.tensorPtr 0)

-- Sample a squashed Gaussian action — pure-Double, used for rollout.
sampleActionIO : ActorNet -> Tensor [] ExampleExecutor ExampleDType WithGrad -> Vect ObsDim Double ->
                 IO (Double, Double)
sampleActionIO actor logStdV obs = do
  mean <- actorMean actor obs
  let logStd = primItem {ex=ExampleExecutor} logStdV.tensorPtr
      std = Prelude.exp logStd
  eps <- normalSample
  let u = mean + std * eps
      action = Math.tanh u * MaxAction
      lp_u = -0.5 * ((u - mean) / std) * ((u - mean) / std) - logStd - logTwoPiHalf
      lp = lp_u - squashCorrection u
  pure (action, lp)

-- Batched action sampling: one batched actor forward → N means, then N
-- independent eps draws. Shared logStd scalar. Mirrors MountainCarCont's
-- helper of the same name.
sampleActionsBatched : {n : Nat} -> ActorNet -> Tensor [] ExampleExecutor ExampleDType WithGrad ->
                       Vect n PState -> IO (Vect n Double)
sampleActionsBatched actor logStdV envs = do
  let obsRows : Vect n (Vector ObsDim Double)
      obsRows = map (\s => obsTensor (observeVec s)) envs
      batchPtr = bulkToTensor2d {ex=ExampleExecutor} {dt=ExampleDType} obsRows
      stateV : Tensor [n, ObsDim] ExampleExecutor ExampleDType WithGrad
      stateV = MkTensor batchPtr Nothing
  (_, meanB) <- forwardVarBatch actor stateV
  let logStd = primItem {ex=ExampleExecutor} logStdV.tensorPtr
      std = Prelude.exp logStd
  go meanB std 0 envs
  where
    go : {n : Nat} -> Tensor [n, 1] ExampleExecutor ExampleDType WithGrad ->
         Double -> Int -> Vect k PState -> IO (Vect k Double)
    go _ _ _ [] = pure []
    go meanB std i (_ :: rest) = do
      let mean = primItem2d {ex=ExampleExecutor} meanB.tensorPtr i 0
      eps <- normalSample
      let u = mean + std * eps
          action = Math.tanh u * MaxAction
      as <- go meanB std (i + 1) rest
      pure (action :: as)

-- --- SAC state -------------------------------------------------------

record SACState where
  constructor MkSAC
  actor   : ActorNet
  q1      : QNet
  q2      : QNet
  q1Tgt   : QNet
  q2Tgt   : QNet
  logStdV : Tensor [] ExampleExecutor ExampleDType WithGrad
  buffer  : ReplayBuffer ObsDim ActDim
  stepRef : IORef Nat
  envRef  : IORef (VecEnv NumEnvs PState)
  epLenRef : IORef (Vect NumEnvs Nat)
  retRef  : IORef (Vect NumEnvs Double)
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

computeTargetVal : QNet -> QNet -> ActorNet -> Tensor [] ExampleExecutor ExampleDType WithGrad ->
                   Double -> Double -> Transition ObsDim ActDim -> IO Double
computeTargetVal q1Tgt q2Tgt actor logStdV gamma alpha t = do
  nextPair <- sampleActionIO actor logStdV t.nextObs
  let nextAction = fst nextPair
      nextLogP = snd nextPair
  q1NextD <- qValue q1Tgt t.nextObs nextAction
  q2NextD <- qValue q2Tgt t.nextObs nextAction
  let minQNextD = if q1NextD <= q2NextD then q1NextD else q2NextD
      doneMask = if t.done then 0.0 else 1.0
  pure (t.reward + gamma * doneMask * (minQNextD - alpha * nextLogP))

-- Per-sample MSE loss for a [B, 1] Q-output indexed by row k against
-- a Double target. Mirrors Dqn's perSampleLoss but with a single
-- Q-column (action dim is fixed at the input).
perSampleQLoss : {n : Nat} -> (qOutB : Tensor [n, 1] ExampleExecutor ExampleDType WithGrad) -> Double ->
                 Int -> IO (Tensor [] ExampleExecutor ExampleDType WithGrad)
perSampleQLoss qOutB tv k = do
  qRow    <- trowSelect qOutB k
  qScalar <- telemSelect qRow 0
  targetT <- tconstScalar tv
  diff    <- tsub qScalar targetT
  tmul diff diff

meanScalarLoss : (n : Nat) -> List (Tensor [] ExampleExecutor ExampleDType WithGrad) -> IO (Tensor [] ExampleExecutor ExampleDType WithGrad)
meanScalarLoss n losses = do
  zero <- tconstScalar 0.0
  let summed = foldl (\a, b => MkTensor (primAdd {ex=ExampleExecutor} a.tensorPtr b.tensorPtr) Nothing) zero losses
  tmulScalar summed (1.0 / cast n)

qLossBatch : (n : Nat) -> QNet -> QNet -> QNet -> ActorNet -> Tensor [] ExampleExecutor ExampleDType WithGrad ->
             Double -> Double -> Vect n (Transition ObsDim ActDim) ->
             IO (Tensor [] ExampleExecutor ExampleDType WithGrad)
qLossBatch n qOnline q1Tgt q2Tgt actor logStdV gamma alpha batch = do
  targetVals <- traverse (computeTargetVal q1Tgt q2Tgt actor logStdV gamma alpha) batch
  let qInputs = the (Vect n (Vector QInputDim Double))
                    (map (\t => qInputTensor (qInput t.obs (oneAct t.action))) batch)
      qInputBT = bulkToTensor2d {ex=ExampleExecutor} {dt=ExampleDType} qInputs
      qInputV = the (Tensor [n, QInputDim] ExampleExecutor ExampleDType WithGrad) (MkTensor qInputBT Nothing)
  (_, qOutB) <- forwardVarBatch qOnline qInputV
  losses <- go qOutB (toList targetVals) 0
  meanScalarLoss n losses
  where
    oneAct : Vect ActDim Double -> Double
    oneAct [a] = a
    go : {n : Nat} -> Tensor [n, 1] ExampleExecutor ExampleDType WithGrad -> List Double -> Int -> IO (List (Tensor [] ExampleExecutor ExampleDType WithGrad))
    go _ [] _ = pure []
    go qOutB (tv :: rest) k = do
      l <- perSampleQLoss qOutB tv k
      ls <- go qOutB rest (k + 1)
      pure (l :: ls)

-- --- Actor loss with reparameterization -----------------------------

-- Build a [n, 1] non-grad Tensor from a Vect of Doubles (one row each).
buildScalarColumn : {n : Nat} -> Vect n Double -> Tensor [n, 1] ExampleExecutor ExampleDType WithGrad
buildScalarColumn {n} xs =
  let rows = the (Vect n (Vector 1 Double)) (map (\x => VArray [SArray x]) xs)
      ptr = bulkToTensor2d {ex=ExampleExecutor} {dt=ExampleDType} rows
  in MkTensor ptr Nothing

actorPerStepLoss : {n : Nat} ->
                   Tensor [n, 1] ExampleExecutor ExampleDType WithGrad -> Tensor [n, 1] ExampleExecutor ExampleDType WithGrad ->
                   Tensor [n, 1] ExampleExecutor ExampleDType WithGrad -> Tensor [n, 1] ExampleExecutor ExampleDType WithGrad ->
                   Tensor [] ExampleExecutor ExampleDType WithGrad -> Double ->
                   Int ->
                   IO (Tensor [] ExampleExecutor ExampleDType WithGrad)
actorPerStepLoss meanB uBT q1B q2B logStdV alpha rowIdx = do
  q1Row <- trowSelect q1B rowIdx
  q1S   <- telemSelect q1Row 0
  let q1Val = primItem1d {ex=ExampleExecutor} q1Row.tensorPtr 0
  q2Row <- trowSelect q2B rowIdx
  q2S   <- telemSelect q2Row 0
  let q2Val = primItem1d {ex=ExampleExecutor} q2Row.tensorPtr 0
      minQS = if q1Val <= q2Val then q1S else q2S
  meanRow <- trowSelect meanB rowIdx
  meanS   <- telemSelect meanRow 0
  uRow    <- trowSelect uBT rowIdx
  uS      <- telemSelect uRow 0
  let uVal = primItem1d {ex=ExampleExecutor} uRow.tensorPtr 0
  diffM    <- tsub uS meanS
  negTwoLs <- tmulScalar logStdV (-2.0)
  varInv   <- texp negTwoLs
  diffSq   <- tmul diffM diffM
  diffSqV  <- tmul diffSq varInv
  quad     <- tmulScalar diffSqV 0.5
  cC       <- tconstScalar logTwoPiHalf
  negQ     <- tneg quad
  negQLs   <- tsub negQ logStdV
  lpU      <- tsub negQLs cC
  corrC    <- tconstScalar (squashCorrection uVal)
  lpV      <- tsub lpU corrC
  alphaLogP <- tmulScalar lpV alpha
  tsub alphaLogP minQS

actorLossBatch : (n : Nat) -> ActorNet -> QNet -> QNet -> Tensor [] ExampleExecutor ExampleDType WithGrad ->
                 Double -> Vect n (Vect ObsDim Double) -> IO (Tensor [] ExampleExecutor ExampleDType WithGrad)
actorLossBatch n actor q1 q2 logStdV alpha obsBatch = do
  let logStd = primItem {ex=ExampleExecutor} logStdV.tensorPtr
      stdVal = Prelude.exp logStd
  epses <- traverse (\_ => normalSample) obsBatch
  let obsTensors = the (Vect n (Vector ObsDim Double)) (map obsTensor obsBatch)
      obsBT = bulkToTensor2d {ex=ExampleExecutor} {dt=ExampleDType} obsTensors
      obsBV = the (Tensor [n, ObsDim] ExampleExecutor ExampleDType WithGrad) (MkTensor obsBT Nothing)
  (_, meanB) <- forwardVarBatch actor obsBV
  let epsScales = map (\e => stdVal * e) epses
      epsBV = buildScalarColumn epsScales
  uBT         <- tadd meanB epsBV
  aSquashedBT <- ttanh uBT
  aReparamBT  <- tmulScalar aSquashedBT MaxAction
  qInputBT    <- tconcat2dAxis1 obsBV aReparamBT
  (_, q1B)    <- forwardVarBatch q1 qInputBT
  (_, q2B)    <- forwardVarBatch q2 qInputBT
  losses <- go meanB uBT q1B q2B (toList epses) 0
  meanScalarLoss n losses
  where
    go : {n : Nat} ->
         Tensor [n, 1] ExampleExecutor ExampleDType WithGrad -> Tensor [n, 1] ExampleExecutor ExampleDType WithGrad ->
         Tensor [n, 1] ExampleExecutor ExampleDType WithGrad -> Tensor [n, 1] ExampleExecutor ExampleDType WithGrad ->
         List Double -> Int -> IO (List (Tensor [] ExampleExecutor ExampleDType WithGrad))
    go _ _ _ _ [] _ = pure []
    go meanB uBT q1B q2B (_ :: rest) k = do
      l <- actorPerStepLoss meanB uBT q1B q2B logStdV alpha k
      ls <- go meanB uBT q1B q2B rest (k + 1)
      pure (l :: ls)

-- --- One batch update: three group-scoped optimizer steps ------------

runBatchUpdate : NativeOptimizer ExampleExecutor -> NativeOptimizer ExampleExecutor -> NativeOptimizer ExampleExecutor ->
                 SACState -> Config -> {n : Nat} ->
                 Vect n (Transition ObsDim ActDim) -> IO ()
runBatchUpdate q1Opt q2Opt actorOpt st cfg {n} batch = do
  q1LossV <- qLossBatch n st.q1 st.q1Tgt st.q2Tgt st.actor st.logStdV
                        cfg.gamma cfg.alpha batch
  _ <- nativeTrainStep q1Opt q1LossV

  q2LossV <- qLossBatch n st.q2 st.q1Tgt st.q2Tgt st.actor st.logStdV
                        cfg.gamma cfg.alpha batch
  _ <- nativeTrainStep q2Opt q2LossV

  let obsVec = the (Vect n (Vect ObsDim Double)) (map (\t => t.obs) batch)
  aLossV <- actorLossBatch n st.actor st.q1 st.q2 st.logStdV cfg.alpha obsVec
  _ <- nativeTrainStep actorOpt aLossV
  pure ()

-- --- Main loop -------------------------------------------------------

-- Step every env with its action; auto-reset on per-env EpisodeLen.
-- Pendulum has no termination (continuous task), so we mark isDone
-- purely on per-env step counter hitting EpisodeLen.
stepAllAutoResetP : Vect n PState -> Vect n Double -> Vect n Nat ->
                    (Vect n PState, Vect n Double, Vect n Bool, Vect n Nat)
stepAllAutoResetP [] [] [] = ([], [], [], [])
stepAllAutoResetP (s :: ss) (a :: as) (l :: ls) =
  case pStep s a of
    (r, s', _, _) =>
      let isDone = (l + 1) >= EpisodeLen
          nextS  = if isDone then MkP 3.141592653589793 0.0 else s'
          nextL  = the Nat (if isDone then 0 else l + 1)
      in case stepAllAutoResetP ss as ls of
           (rest, rs, ds, restL) =>
             (nextS :: rest, r :: rs, isDone :: ds, nextL :: restL)

sacStepBatched : NativeOptimizer ExampleExecutor -> NativeOptimizer ExampleExecutor -> NativeOptimizer ExampleExecutor ->
                 Config -> SACState -> IO (SACState, Double)
sacStepBatched q1Opt q2Opt actorOpt cfg st = do
  stepCount <- readIORef st.stepRef
  envs0 <- readIORef st.envRef
  epLens <- readIORef st.epLenRef
  oldRets <- readIORef st.retRef

  actions <- if stepCount < cfg.warmupSteps
               then traverse (\_ => randomRIO (the Double (negate MaxAction), MaxAction)) envs0.envs
               else withNoGrad {ex=ExampleExecutor} (sampleActionsBatched st.actor st.logStdV envs0.envs)

  case stepAllAutoResetP envs0.envs actions epLens of
    (envs', rewards, isDones, newEpLens) => do
      pushAll envs0.envs actions rewards envs' isDones
      writeIORef st.envRef (MkVecEnv envs')
      writeIORef st.stepRef (stepCount + 1)
      writeIORef st.epLenRef newEpLens

      let newRets : Vect NumEnvs Double
          newRets = zipWith3 (\old, r, d => if d then 0.0 else old + r) oldRets rewards isDones
          completed : List Double
          completed = getCompleted (toList oldRets) (toList rewards) (toList isDones)
      writeIORef st.retRef newRets
      case completed of
        []        => pure ()
        (e :: es) => writeIORef st.lastEpRef (last (e :: es))

      bufSz <- bufferSize st.buffer
      _ <- if bufSz >= cfg.batchSize && stepCount >= cfg.warmupSteps
             then do
               mBatch <- sampleN cfg.batchSize st.buffer
               case mBatch of
                 Nothing => pure ()
                 Just batch => do
                   runBatchUpdate q1Opt q2Opt actorOpt st cfg batch
                   _ <- polyakUpdate {ex=ExampleExecutor} cfg.tau "q1_" "q1tgt_"
                   _ <- polyakUpdate {ex=ExampleExecutor} cfg.tau "q2_" "q2tgt_"
                   pure ()
             else pure ()

      lastEp <- readIORef st.lastEpRef
      pure (st, negate lastEp)
  where
    zipWith3 : (a -> b -> c -> d) -> Vect n a -> Vect n b -> Vect n c -> Vect n d
    zipWith3 _ [] [] [] = []
    zipWith3 f (x :: xs) (y :: ys) (z :: zs) = f x y z :: zipWith3 f xs ys zs

    getCompleted : List Double -> List Double -> List Bool -> List Double
    getCompleted [] _ _ = []
    getCompleted _ [] _ = []
    getCompleted _ _ [] = []
    getCompleted (run :: rs) (rw :: rws) (d :: ds) =
      let recur = getCompleted rs rws ds
      in if d then (run + rw) :: recur else recur

    pushAll : Vect n PState -> Vect n Double -> Vect n Double ->
              Vect n PState -> Vect n Bool -> IO ()
    pushAll [] [] [] [] [] = pure ()
    pushAll (s :: ss) (a :: as) (r :: rs) (s' :: ss') (d :: ds) = do
      push st.buffer (MkTransition (observeVec s) [a] r (observeVec s') d)
      pushAll ss as rs ss' ds

-- --- Greedy evaluation ----------------------------------------------

greedyAct : ActorNet -> Vect ObsDim Double -> IO Double
greedyAct actor obs = do
  mean <- actorMean actor obs
  pure (Math.tanh mean * MaxAction)

evalEp : ActorNet -> PState -> Nat -> Double -> IO Double
evalEp _ _ Z acc = pure acc
evalEp actor st (S k) acc = do
  a <- greedyAct actor (observeVec st)
  case pStep st a of
    (r, st', _, _) => evalEp actor st' k (acc + r)

evalN : ActorNet -> Nat -> Double -> IO Double
evalN _ Z acc = pure acc
evalN actor (S k) acc = do
  v <- evalEp actor (MkP 3.141592653589793 0.0) EpisodeLen 0.0
  evalN actor k (acc + v)

-- --- Main -----------------------------------------------------------

main : IO ()
main = do
  requireMachine {m = ChosenMachine}
  args <- getArgs
  let cfg = parseArgs defaultConfig specs (drop 1 args)
  srand cfg.seed
  tsetInitSeed {ex = ExampleExecutor} cfg.seed

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
  logStdV <- the (IO (Tensor [] ExampleExecutor ExampleDType WithGrad)) (tparamScalar "actor_log_std" 0.0)

  -- Hard-copy online → target at init.
  _ <- polyakUpdate {ex=ExampleExecutor} 1.0 "q1_" "q1tgt_"
  _ <- polyakUpdate {ex=ExampleExecutor} 1.0 "q2_" "q2tgt_"

  buffer <- mkBuffer {obsDim=ObsDim, actDim=ActDim} cfg.bufferCap
  stepRef <- newIORef (the Nat 0)
  resetSeedI <- randomInt32
  let initEnvs : VecEnv NumEnvs PState
      initEnvs = fst (resetAll {state=PState} {action=Double} {obs=Vect 3 Double}
                              (cast resetSeedI))
  envRef <- newIORef initEnvs
  epLenRef <- newIORef (the (Vect NumEnvs Nat) (replicate NumEnvs 0))
  retRef <- newIORef (the (Vect NumEnvs Double) (replicate NumEnvs 0.0))
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

  metrics <- newRLMetricsState 20
  let trainCfg : TrainConfig SACState
      trainCfg = mkTrainConfig cfg.epochs 2000
                            (WindowedAvg cfg.esThreshold cfg.esWindow cfg.esPatience)
                            (\_ => readRLMetrics "recent_20" metrics) (\_ => pure ())
  (trained, epochsDone, _) <- runTrainingIO {ex=ExampleExecutor}
    (\s, _ => do
       (s', loss) <- sacStepBatched q1Opt q2Opt actorOpt cfg s
       recordReturn metrics (negate loss)
       pure (s', loss))
    (pure ())
    trainCfg st0

  putStrLn ""
  let nEval = the Nat 20
  evalSum <- withNoGrad {ex=ExampleExecutor} (evalN trained.actor nEval 0.0)
  let avgReturn = evalSum / cast (natToInteger nEval)
  putStrLn $ "Eval (" ++ show nEval ++ " episodes, greedy): avg_return=" ++ show avgReturn
  putStrLn ""
  putStrLn $ formatResult [("avg_return", show avgReturn),
                            ("epochs", show epochsDone),
                            ("seed", show cfg.seed)]
