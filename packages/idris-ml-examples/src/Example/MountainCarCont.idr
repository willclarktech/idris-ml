module Example.MountainCarCont

import Data.List
import Data.Vect
import Data.IORef
import System
import Compat.Random

import ML.Simple
import Array            -- Vector / VArray / SArray
import Floating         -- Math.tanh's Floating interface
import Gym.ClassicControl.MountainCarCont
import Gym.Env
import Gym.Vector
import Math
import RL.ReplayBuffer
import Sampler
import Train
import BuildConfig

----------------------------------------------------------------------
-- SAC on MountainCarContinuous-v0 with velocity-magnitude reward shaping.
--
-- MountainCarCont's reward is sparse (terminal +100, per-step -0.1·a²).
-- Shape the *training* reward with `r_shaped = r_raw + shaping·|v_next|`.
-- Eval reports the *raw* return.
--
-- Architecture: separate actor + Q1 + Q2 + 2 target Q nets, scoped
-- paramIds, three Adam optimizers, polyak τ-soft target update. The
-- scope segments carry a trailing underscore so "q1_" is not a substring
-- of "q1tgt_" (else the q1 optimizer would also step the q1-target net).
----------------------------------------------------------------------

ObsDim : Nat; ObsDim = 2
ActDim : Nat; ActDim = 1
QInputDim : Nat; QInputDim = 3          -- ObsDim + ActDim
Hidden : Nat; Hidden = 64
EpisodeLen : Nat; EpisodeLen = 999
MaxAct : Double; MaxAct = 1.0

||| Parallel envs collecting transitions in lockstep.
NumEnvs : Nat; NumEnvs = 4

-- --- Architectures --------------------------------------------------

ActorNet : Type
ActorNet = Seq ObsDim 1 Ex F WithGrad

QNet : Type
QNet = Seq QInputDim 1 Ex F WithGrad

mkActor : IO ActorNet
mkActor = runInit $ scoped "actor_" $ do
  l1 <- linear {i=ObsDim} {o=Hidden}
  l2 <- linear {i=Hidden} {o=Hidden}
  l3 <- linear {i=Hidden} {o=1}
  pure (l1 ~~> reluA ~~> l2 ~~> reluA ~~> l3 ~~> Nil)

mkQ : (scope : String) -> IO QNet
mkQ scope = runInit $ scoped scope $ do
  l1 <- linear {i=QInputDim} {o=Hidden}
  l2 <- linear {i=Hidden} {o=Hidden}
  l3 <- linear {i=Hidden} {o=1}
  pure (l1 ~~> reluA ~~> l2 ~~> reluA ~~> l3 ~~> Nil)

-- --- Observation helpers --------------------------------------------

observeVec : MCCState -> Vect ObsDim Double
observeVec s = mccObserve s

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
  in Prelude.log (1.0 - tu * tu + 1.0e-6) + Prelude.log MaxAct

-- --- Forward helpers (single-sample, via [1, n] batched forward) -----

actorMean : ActorNet -> Vect ObsDim Double -> IO Double
actorMean actor obs = do
  let stateV = the (Tensor [1, ObsDim] Ex F WithGrad)
                 (MkTensor (bulkToTensor2d {ex=Ex} {dt=F} [obsTensor obs]) Nothing)
  outV <- forwardSeq {b=1} actor stateV
  pure (primItem2d {ex=Ex} outV.tensorPtr 0 0)

qValue : QNet -> Vect ObsDim Double -> Double -> IO Double
qValue q obs action = do
  let inV = the (Tensor [1, QInputDim] Ex F WithGrad)
                (MkTensor (bulkToTensor2d {ex=Ex} {dt=F} [qInputTensor (qInput obs action)]) Nothing)
  outV <- forwardSeq {b=1} q inV
  pure (primItem2d {ex=Ex} outV.tensorPtr 0 0)

sampleActionIO : ActorNet -> Tensor [] Ex F WithGrad -> Vect ObsDim Double ->
                 IO (Double, Double)
sampleActionIO actor logStdV obs = do
  mean <- actorMean actor obs
  let logStd = primItem {ex=Ex} logStdV.tensorPtr
      std = Prelude.exp logStd
  eps <- normalSample
  let u = mean + std * eps
      action = Math.tanh u * MaxAct
      lp_u = -0.5 * ((u - mean) / std) * ((u - mean) / std) - logStd - logTwoPiHalf
      lp = lp_u - squashCorrection u
  pure (action, lp)

-- Batched action sampling across NumEnvs envs: one batched actor forward
-- (mean per env), then N independent eps draws. Shared logStd scalar.
sampleActionsBatched : {n : Nat} -> ActorNet -> Tensor [] Ex F WithGrad ->
                       Vect n MCCState -> IO (Vect n Double)
sampleActionsBatched actor logStdV envs = do
  let obsRows : Vect n (Vector ObsDim Double)
      obsRows = map (\s => obsTensor (observeVec s)) envs
      batchPtr = bulkToTensor2d {ex=Ex} {dt=F} obsRows
      stateV : Tensor [n, ObsDim] Ex F WithGrad
      stateV = MkTensor batchPtr Nothing
  meanB <- forwardSeq {b=n} actor stateV
  let logStd = primItem {ex=Ex} logStdV.tensorPtr
      std = Prelude.exp logStd
  go meanB std 0 envs
  where
    go : {n : Nat} -> Tensor [n, 1] Ex F WithGrad ->
         Double -> Int -> Vect k MCCState -> IO (Vect k Double)
    go _ _ _ [] = pure []
    go meanB std i (_ :: rest) = do
      let mean = primItem2d {ex=Ex} meanB.tensorPtr i 0
      eps <- normalSample
      let u = mean + std * eps
          action = Math.tanh u * MaxAct
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
  logStdV : Tensor [] Ex F WithGrad
  buffer  : ReplayBuffer ObsDim ActDim
  stepRef : IORef Nat
  envRef  : IORef (VecEnv NumEnvs MCCState)
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

computeTargetVal : QNet -> QNet -> ActorNet -> Tensor [] Ex F WithGrad ->
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

perSampleQLoss : {n : Nat} -> (qOutB : Tensor [n, 1] Ex F WithGrad) -> Double ->
                 Int -> IO (Tensor [] Ex F WithGrad)
perSampleQLoss qOutB tv k = do
  qRow    <- trowSelect qOutB k
  qScalar <- telemSelect qRow 0
  targetT <- tconstScalar tv
  diff    <- tsub qScalar targetT
  tmul diff diff

meanScalarLoss : (n : Nat) -> List (Tensor [] Ex F WithGrad) -> IO (Tensor [] Ex F WithGrad)
meanScalarLoss n losses = do
  zero <- tconstScalar 0.0
  let summed = foldl (\a, b => MkTensor (primAdd {ex=Ex} a.tensorPtr b.tensorPtr) Nothing) zero losses
  tmulScalar summed (1.0 / cast n)

qLossBatch : (n : Nat) -> QNet -> QNet -> QNet -> ActorNet -> Tensor [] Ex F WithGrad ->
             Double -> Double -> Vect n (Transition ObsDim ActDim) ->
             IO (Tensor [] Ex F WithGrad)
qLossBatch n qOnline q1Tgt q2Tgt actor logStdV gamma alpha batch = do
  targetVals <- traverse (computeTargetVal q1Tgt q2Tgt actor logStdV gamma alpha) batch
  let qInputs = the (Vect n (Vector QInputDim Double))
                    (map (\t => qInputTensor (qInput t.obs (oneAct t.action))) batch)
      qInputBT = bulkToTensor2d {ex=Ex} {dt=F} qInputs
      qInputV = the (Tensor [n, QInputDim] Ex F WithGrad) (MkTensor qInputBT Nothing)
  qOutB <- forwardSeq {b=n} qOnline qInputV
  losses <- go qOutB (toList targetVals) 0
  meanScalarLoss n losses
  where
    oneAct : Vect ActDim Double -> Double
    oneAct [a] = a
    go : {n : Nat} -> Tensor [n, 1] Ex F WithGrad -> List Double -> Int -> IO (List (Tensor [] Ex F WithGrad))
    go _ [] _ = pure []
    go qOutB (tv :: rest) k = do
      l <- perSampleQLoss qOutB tv k
      ls <- go qOutB rest (k + 1)
      pure (l :: ls)

-- --- Actor loss with reparameterization -----------------------------

buildScalarColumn : {n : Nat} -> Vect n Double -> Tensor [n, 1] Ex F WithGrad
buildScalarColumn {n} xs =
  let rows = the (Vect n (Vector 1 Double)) (map (\x => VArray [SArray x]) xs)
      ptr = bulkToTensor2d {ex=Ex} {dt=F} rows
  in MkTensor ptr Nothing

actorPerStepLoss : {n : Nat} ->
                   Tensor [n, 1] Ex F WithGrad -> Tensor [n, 1] Ex F WithGrad ->
                   Tensor [n, 1] Ex F WithGrad -> Tensor [n, 1] Ex F WithGrad ->
                   Tensor [] Ex F WithGrad -> Double ->
                   Int -> IO (Tensor [] Ex F WithGrad)
actorPerStepLoss meanB uBT q1B q2B logStdV alpha rowIdx = do
  q1Row <- trowSelect q1B rowIdx
  q1S   <- telemSelect q1Row 0
  let q1Val = primItem1d {ex=Ex} q1Row.tensorPtr 0
  q2Row <- trowSelect q2B rowIdx
  q2S   <- telemSelect q2Row 0
  let q2Val = primItem1d {ex=Ex} q2Row.tensorPtr 0
      minQS = if q1Val <= q2Val then q1S else q2S
  meanRow <- trowSelect meanB rowIdx
  meanS   <- telemSelect meanRow 0
  uRow    <- trowSelect uBT rowIdx
  uS      <- telemSelect uRow 0
  let uVal = primItem1d {ex=Ex} uRow.tensorPtr 0
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

actorLossBatch : (n : Nat) -> ActorNet -> QNet -> QNet -> Tensor [] Ex F WithGrad ->
                 Double -> Vect n (Vect ObsDim Double) -> IO (Tensor [] Ex F WithGrad)
actorLossBatch n actor q1 q2 logStdV alpha obsBatch = do
  let logStd = primItem {ex=Ex} logStdV.tensorPtr
      stdVal = Prelude.exp logStd
  epses <- traverse (\_ => normalSample) obsBatch
  let obsTensors = the (Vect n (Vector ObsDim Double)) (map obsTensor obsBatch)
      obsBT = bulkToTensor2d {ex=Ex} {dt=F} obsTensors
      obsBV = the (Tensor [n, ObsDim] Ex F WithGrad) (MkTensor obsBT Nothing)
  meanB <- forwardSeq {b=n} actor obsBV
  let epsScales = map (\e => stdVal * e) epses
      epsBV = buildScalarColumn epsScales
  uBT         <- tadd meanB epsBV
  aSquashedBT <- ttanh uBT
  aReparamBT  <- tmulScalar aSquashedBT MaxAct
  qInputBT    <- tconcat2dAxis1 obsBV aReparamBT
  q1B    <- forwardSeq {b=n} q1 qInputBT
  q2B    <- forwardSeq {b=n} q2 qInputBT
  losses <- go meanB uBT q1B q2B (toList epses) 0
  meanScalarLoss n losses
  where
    go : {n : Nat} ->
         Tensor [n, 1] Ex F WithGrad -> Tensor [n, 1] Ex F WithGrad ->
         Tensor [n, 1] Ex F WithGrad -> Tensor [n, 1] Ex F WithGrad ->
         List Double -> Int -> IO (List (Tensor [] Ex F WithGrad))
    go _ _ _ _ [] _ = pure []
    go meanB uBT q1B q2B (_ :: rest) k = do
      l <- actorPerStepLoss meanB uBT q1B q2B logStdV alpha k
      ls <- go meanB uBT q1B q2B rest (k + 1)
      pure (l :: ls)

-- --- Batch update ---------------------------------------------------

runBatchUpdate : Optimizer Ex -> Optimizer Ex -> Optimizer Ex ->
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

-- --- Main loop ------------------------------------------------------

-- Step every env with its action; auto-reset on Terminated OR per-env
-- EpisodeLen truncation. Bootstrap-done uses Terminated only.
stepAllAutoResetMCC : Vect n MCCState -> Vect n Double -> Vect n Nat ->
                      (Vect n MCCState, Vect n Double, Vect n Bool,
                       Vect n Bool, Vect n Nat)
stepAllAutoResetMCC [] [] [] = ([], [], [], [], [])
stepAllAutoResetMCC (s :: ss) (a :: as) (l :: ls) =
  case mccStep s a of
    (r, s', outcome, _) =>
      let terminated = case outcome of
                         Terminated => True
                         _          => False
          truncated = (l + 1) >= EpisodeLen
          isDone = terminated || truncated
          nextS  = if isDone then MkMCC (-0.5) 0.0 else s'
          nextL  = the Nat (if isDone then 0 else l + 1)
      in case stepAllAutoResetMCC ss as ls of
           (rest, rs, bds, ds, restL) =>
             (nextS :: rest, r :: rs, terminated :: bds, isDone :: ds, nextL :: restL)

sacStepBatched : Optimizer Ex -> Optimizer Ex -> Optimizer Ex ->
                 Config -> SACState -> IO (SACState, Double)
sacStepBatched q1Opt q2Opt actorOpt cfg st = do
  stepCount <- readIORef st.stepRef
  envs0 <- readIORef st.envRef
  epLens <- readIORef st.epLenRef
  oldRets <- readIORef st.retRef

  -- Action selection: warmup uses N uniform-random samples, post-warmup
  -- uses one batched actor forward → N tanh-squashed Gaussian samples.
  actions <- if stepCount < cfg.warmupSteps
               then traverse (\_ => randomRIO (the Double (negate MaxAct), MaxAct)) envs0.envs
               else withNoGrad {ex=Ex} (sampleActionsBatched st.actor st.logStdV envs0.envs)

  case stepAllAutoResetMCC envs0.envs actions epLens of
    (envs', rewards, bufferDones, isDones, newEpLens) => do
      pushAll envs0.envs actions rewards envs' bufferDones cfg.shaping
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
                   _ <- polyakUpdate {ex=Ex} cfg.tau "q1_" "q1tgt_"
                   _ <- polyakUpdate {ex=Ex} cfg.tau "q2_" "q2tgt_"
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

    pushAll : Vect n MCCState -> Vect n Double -> Vect n Double ->
              Vect n MCCState -> Vect n Bool -> Double -> IO ()
    pushAll [] [] [] [] [] _ = pure ()
    pushAll (s :: ss) (a :: as) (r :: rs) (s' :: ss') (bd :: bds) shaping = do
      let shapedR = r + shaping * abs s'.mccVel
      push st.buffer (MkTransition (observeVec s) [a] shapedR (observeVec s') bd)
      pushAll ss as rs ss' bds shaping

-- --- Greedy evaluation ----------------------------------------------

greedyAct : ActorNet -> Vect ObsDim Double -> IO Double
greedyAct actor obs = do
  mean <- actorMean actor obs
  pure (Math.tanh mean * MaxAct)

evalEp : ActorNet -> MCCState -> Nat -> Double -> IO Double
evalEp _ _ Z acc = pure acc
evalEp actor st (S k) acc = do
  a <- greedyAct actor (mccObserve st)
  case mccStep st a of
    (r, st', outcome, _) =>
      case outcome of
        Terminated => pure (acc + r)
        _          => evalEp actor st' k (acc + r)

evalN : ActorNet -> Nat -> Double -> IO Double
evalN _ Z acc = pure acc
evalN actor (S k) acc = do
  v <- withNoGrad {ex=Ex} (evalEp actor (MkMCC (-0.5) 0.0) EpisodeLen 0.0)
  evalN actor k (acc + v)

-- --- Main -----------------------------------------------------------

%default partial

main : IO ()
main = do
  requireMachine {m = ChosenMachine}
  args <- getArgs
  let cfg = parseArgs defaultConfig specs (drop 1 args)
  srand cfg.seed
  tsetInitSeed {ex = Ex} cfg.seed

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
  logStdV <- the (IO (Tensor [] Ex F WithGrad)) (tparamScalar "actor_log_std" 0.0)

  _ <- polyakUpdate {ex=Ex} 1.0 "q1_" "q1tgt_"
  _ <- polyakUpdate {ex=Ex} 1.0 "q2_" "q2tgt_"

  buffer <- mkBuffer {obsDim=ObsDim, actDim=ActDim} cfg.bufferCap
  stepRef <- newIORef (the Nat 0)
  resetSeedI <- randomInt32
  let initEnvs : VecEnv NumEnvs MCCState
      initEnvs = fst (resetAll {state=MCCState} {action=Double} {obs=Vect 2 Double}
                              (cast resetSeedI))
  envRef <- newIORef initEnvs
  epLenRef <- newIORef (the (Vect NumEnvs Nat) (replicate NumEnvs 0))
  retRef <- newIORef (the (Vect NumEnvs Double) (replicate NumEnvs 0.0))
  lastEpRef <- newIORef (the Double 0.0)

  let st0 = MkSAC actor q1 q2 q1Tgt q2Tgt logStdV buffer stepRef envRef epLenRef retRef lastEpRef
  -- Three Adams, each scoped to one network (trailing "_" keeps "q1_"
  -- distinct from "q1tgt_"). The actor opt also covers "actor_log_std".
  actorOpt <- adam {scope="actor_"} cfg.lr ({ clip := NormClip cfg.clipNorm } defaultOpts)
  q1Opt    <- adam {scope="q1_"}    cfg.lr ({ clip := NormClip cfg.clipNorm } defaultOpts)
  q2Opt    <- adam {scope="q2_"}    cfg.lr ({ clip := NormClip cfg.clipNorm } defaultOpts)

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
  (trained, epochsDone, _) <- fit {batch = ()}
    (\s, _ => do
       (s', loss) <- sacStepBatched q1Opt q2Opt actorOpt cfg s
       recordReturn metrics (negate loss)
       pure (s', loss))
    actorOpt (generate (pure ()))
    trainCfg st0

  putStrLn ""
  let nEval = the Nat 20
  evalSum <- evalN trained.actor nEval 0.0
  let avgReturn = evalSum / cast (natToInteger nEval)
  putStrLn $ "Eval (" ++ show nEval ++ " episodes, greedy): avg_return=" ++ show avgReturn
  putStrLn ""
  putStrLn $ formatResult [("avg_return", show avgReturn),
                            ("epochs", show epochsDone),
                            ("seed", show cfg.seed)]
