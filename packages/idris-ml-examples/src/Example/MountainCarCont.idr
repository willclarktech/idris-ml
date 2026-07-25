module Example.MountainCarCont

import Control.Linear.LIO
import Data.IORef
import Data.Linear.Notation
import Data.List
import Data.String
import Data.Vect
import System

import Gym.ClassicControl.MountainCarCont
import Gym.Env
import Gym.Vector
import Ml.Array
import Ml.Checkpoint
import Ml.Compat.Random
import Ml.Fit
import Ml.Floating
import Ml.Math
import Ml.RL.ReplayBuffer
import Ml.Sampler
import Ml.Simple
import Ml.Train
import Ml.Train.Freeze
import Random.Source

import BuildConfig

-- Actor + Q-nets are linear `Seq`s; hide the IO `Nn.Seq` constructors.

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

ObsDim     : Nat; ObsDim = 2
ActDim     : Nat; ActDim = 1
QInputDim  : Nat; QInputDim = 3          -- ObsDim + ActDim
Hidden     : Nat; Hidden = 64
EpisodeLen : Nat; EpisodeLen = 999
MaxAct     : Double; MaxAct = 1.0

||| Parallel envs collecting transitions in lockstep.
NumEnvs : Nat; NumEnvs = 4

-- --- Architectures --------------------------------------------------

ActorNet : Type
ActorNet = Seq ObsDim 1 Ex F WithGrad

QNet : Type
QNet = Seq QInputDim 1 Ex F WithGrad

mkActor : Init ActorNet
mkActor = scoped "actor_" $ do
  l1 <- linear {i=ObsDim} {o=Hidden}
  l2 <- linear {i=Hidden} {o=Hidden}
  l3 <- linear {i=Hidden} {o=1}
  pure (l1 ~~> reluA ~~> l2 ~~> reluA ~~> l3 ~~> Nil)

mkQ : (scope : String) -> Init QNet
mkQ scope = scoped scope $ do
  l1 <- linear {i=QInputDim} {o=Hidden}
  l2 <- linear {i=Hidden} {o=Hidden}
  l3 <- linear {i=Hidden} {o=1}
  pure (l1 ~~> reluA ~~> l2 ~~> reluA ~~> l3 ~~> Nil)

-- The five nets bundled as one linear value (see Example.Sac).
record Nets where
  constructor MkNets
  1 actor : ActorNet
  1 q1    : QNet
  1 q2    : QNet
  1 q1Tgt : QNet
  1 q2Tgt : QNet

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

actorMeanL : (1 _ : ActorNet) -> Vect ObsDim Double -> L IO {use = 1} (LPair (!* Double) ActorNet)
actorMeanL actor obs = do
  stateV <- liftIO1 (ioRerun (\_ =>
    the (Tensor [1, ObsDim] Ex F WithGrad)
        (MkTensor (bulkToTensor2d {ex=Ex} {dt=F} [obsTensor obs]) Nothing)))
  (MkBang outV # actor') <- forwardSeq {b=1} actor stateV
  pure1 (MkBang (primItem2d {ex=Ex} outV.tensorPtr 0 0) # actor')

qValueL : (1 _ : QNet) -> Vect ObsDim Double -> Double -> L IO {use = 1} (LPair (!* Double) QNet)
qValueL q obs action = do
  inV <- liftIO1 (ioRerun (\_ =>
    the (Tensor [1, QInputDim] Ex F WithGrad)
        (MkTensor (bulkToTensor2d {ex=Ex} {dt=F} [qInputTensor (qInput obs action)]) Nothing)))
  (MkBang outV # q') <- forwardSeq {b=1} q inV
  pure1 (MkBang (primItem2d {ex=Ex} outV.tensorPtr 0 0) # q')

sampleActionIOL : (1 _ : ActorNet) -> Tensor [] Ex F WithGrad -> Vect ObsDim Double ->
                  L IO {use = 1} (LPair (!* (Double, Double)) ActorNet)
sampleActionIOL actor logStdV obs = do
  (MkBang mean # actor') <- actorMeanL actor obs
  let logStd = primItem {ex=Ex} logStdV.tensorPtr
      std = Prelude.exp logStd
  eps <- liftIO1 normalSample
  let u = mean + std * eps
      action = Math.tanh u * MaxAct
      lp_u   = -0.5 * ((u - mean) / std) * ((u - mean) / std) - logStd - logTwoPiHalf
      lp     = lp_u - squashCorrection u
  pure1 (MkBang (action, lp) # actor')

-- Batched action sampling across NumEnvs envs: one batched actor forward
-- (mean per env), then N independent eps draws. Shared logStd scalar.
sampleActionsBatchedL : {n : Nat} -> (1 _ : ActorNet) -> Tensor [] Ex F WithGrad ->
                        Vect n MCCState -> L IO {use = 1} (LPair (!* (Vect n Double)) ActorNet)
sampleActionsBatchedL actor logStdV envs = do
  stateV <- liftIO1 (ioRerun (\_ =>
    the (Tensor [n, ObsDim] Ex F WithGrad)
        (MkTensor (bulkToTensor2d {ex=Ex} {dt=F} (map (\s => obsTensor (observeVec s)) envs)) Nothing)))
  (MkBang meanB # actor') <- forwardSeq {b=n} actor stateV
  let logStd = primItem {ex=Ex} logStdV.tensorPtr
      std = Prelude.exp logStd
  acts <- liftIO1 (go meanB std 0 envs)
  pure1 (MkBang acts # actor')
  where
    go : {n : Nat} -> Tensor [n, 1] Ex F WithGrad ->
         Double -> Int -> Vect k MCCState -> IO (Vect k Double)
    go _ _ _ []                = pure []
    go meanB std i (_ :: rest) = do
      let mean = primItem2d {ex=Ex} meanB.tensorPtr i 0
      eps <- normalSample
      let u = mean + std * eps
          action = Math.tanh u * MaxAct
      as <- go meanB std (i + 1) rest
      pure (action :: as)

-- --- SAC state -------------------------------------------------------

-- The five nets are a single **linear** bundle; logStdV is a Tensor (ω); the
-- buffer + IORefs are ω.
record SACState where
  constructor MkSAC
  1 nets    : Nets
  logStdV   : Tensor [] Ex F WithGrad
  buffer    : ReplayBuffer ObsDim ActDim
  stepRef   : IORef Nat
  envRef    : IORef (VecEnv NumEnvs MCCState)
  epLenRef  : IORef (Vect NumEnvs Nat)
  retRef    : IORef (Vect NumEnvs Double)
  lastEpRef : IORef Double
  -- exact registry names of the twin Q-nets + their targets (from reflectNames),
  -- paired positionally by polyakUpdatePaired — no string-prefix scoping.
  q1Names    : List String
  q1TgtNames : List String
  q2Names    : List String
  q2TgtNames : List String

sampleActionsNetsL : {n : Nat} -> (1 _ : Nets) -> Tensor [] Ex F WithGrad ->
                     Vect n MCCState -> L IO {use = 1} (LPair (!* (Vect n Double)) Nets)
sampleActionsNetsL (MkNets actor q1 q2 q1Tgt q2Tgt) logStdV envs = do
  (MkBang acts # actor') <- sampleActionsBatchedL actor logStdV envs
  pure1 (MkBang acts # MkNets actor' q1 q2 q1Tgt q2Tgt)

record Config where
  constructor MkConfig
  lr          : Double
  epochs      : Nat
  gamma       : Double
  alpha       : Double
  bufferCap   : Nat
  batchSize   : Nat
  warmupSteps : Nat
  tau         : Double
  shaping     : Double
  clipNorm    : Double
  seed        : Bits64
  esThreshold : Double
  esWindow    : Nat
  esPatience  : Nat
  lrFind      : Bool

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

computeTargetValL : (1 _ : Nets) -> Tensor [] Ex F WithGrad ->
                    Double -> Double -> Transition ObsDim ActDim ->
                    L IO {use = 1} (LPair (!* Double) Nets)
computeTargetValL (MkNets actor q1 q2 q1Tgt q2Tgt) logStdV gamma alpha t = do
  (MkBang nextPair # actor') <- sampleActionIOL actor logStdV t.nextObs
  let (nextAction, nextLogP) = nextPair
  (MkBang q1NextD # q1Tgt') <- qValueL q1Tgt t.nextObs nextAction
  (MkBang q2NextD # q2Tgt') <- qValueL q2Tgt t.nextObs nextAction
  let minQNextD = if q1NextD <= q2NextD then q1NextD else q2NextD
      doneMask = if t.done then 0.0 else 1.0
  pure1 (MkBang (t.reward + gamma * doneMask * (minQNextD - alpha * nextLogP))
         # MkNets actor' q1 q2 q1Tgt' q2Tgt')

foldTargetValsL : (1 _ : Nets) -> Tensor [] Ex F WithGrad -> Double -> Double ->
                  List (Transition ObsDim ActDim) -> List Double ->
                  L IO {use = 1} (LPair (!* (List Double)) Nets)
foldTargetValsL nets _ _ _ [] acc                        = pure1 (MkBang (reverse acc) # nets)
foldTargetValsL nets logStdV gamma alpha (t :: rest) acc = do
  (MkBang tv # nets') <- computeTargetValL nets logStdV gamma alpha t
  foldTargetValsL nets' logStdV gamma alpha rest (tv :: acc)

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

qOnlineForwardLossL : (n : Nat) -> (1 _ : QNet) -> Vect n (Transition ObsDim ActDim) ->
                      List Double -> L IO {use = 1} (LPair (!* (Tensor [] Ex F WithGrad)) QNet)
qOnlineForwardLossL n qOnline batch targetVals = do
  qInputV <- liftIO1 (ioRerun (\_ =>
    the (Tensor [n, QInputDim] Ex F WithGrad)
        (MkTensor (bulkToTensor2d {ex=Ex} {dt=F}
                    (map (\t => qInputTensor (qInput t.obs (oneAct t.action))) batch)) Nothing)))
  (MkBang qOutB # qOnline') <- forwardSeq {b=n} qOnline qInputV
  loss <- liftIO1 (do losses <- go qOutB targetVals 0; meanScalarLoss n losses)
  pure1 (MkBang loss # qOnline')
  where
    oneAct : Vect ActDim Double -> Double
    oneAct [a] = a
    go : {n : Nat} -> Tensor [n, 1] Ex F WithGrad -> List Double -> Int -> IO (List (Tensor [] Ex F WithGrad))
    go _ [] _               = pure []
    go qOutB (tv :: rest) k = do
      l <- perSampleQLoss qOutB tv k
      ls <- go qOutB rest (k + 1)
      pure (l :: ls)

q1LossL : (n : Nat) -> (1 _ : Nets) -> Vect n (Transition ObsDim ActDim) -> List Double ->
          L IO {use = 1} (LPair (!* (Tensor [] Ex F WithGrad)) Nets)
q1LossL n (MkNets actor q1 q2 q1Tgt q2Tgt) batch tvs = do
  (MkBang loss # q1') <- qOnlineForwardLossL n q1 batch tvs
  pure1 (MkBang loss # MkNets actor q1' q2 q1Tgt q2Tgt)

q2LossL : (n : Nat) -> (1 _ : Nets) -> Vect n (Transition ObsDim ActDim) -> List Double ->
          L IO {use = 1} (LPair (!* (Tensor [] Ex F WithGrad)) Nets)
q2LossL n (MkNets actor q1 q2 q1Tgt q2Tgt) batch tvs = do
  (MkBang loss # q2') <- qOnlineForwardLossL n q2 batch tvs
  pure1 (MkBang loss # MkNets actor q1 q2' q1Tgt q2Tgt)

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

actorLossBatchL : (n : Nat) -> (1 _ : Nets) -> Tensor [] Ex F WithGrad ->
                  Double -> Vect n (Vect ObsDim Double) ->
                  L IO {use = 1} (LPair (!* (Tensor [] Ex F WithGrad)) Nets)
actorLossBatchL n (MkNets actor q1 q2 q1Tgt q2Tgt) logStdV alpha obsBatch = do
  let logStd = primItem {ex=Ex} logStdV.tensorPtr
      stdVal = Prelude.exp logStd
  epses <- liftIO1 (traverse (\_ => normalSample) obsBatch)
  obsBV <- liftIO1 (ioRerun (\_ =>
    the (Tensor [n, ObsDim] Ex F WithGrad)
        (MkTensor (bulkToTensor2d {ex=Ex} {dt=F} (map obsTensor obsBatch)) Nothing)))
  (MkBang meanB # actor') <- forwardSeq {b=n} actor obsBV
  (uBT, qInputBT) <- liftIO1 $ do
    let epsScales = map (\e => stdVal * e) epses
        epsBV = buildScalarColumn epsScales
    uBT         <- tadd meanB epsBV
    aSquashedBT <- ttanh uBT
    aReparamBT  <- tmulScalar aSquashedBT MaxAct
    qInputBT    <- tconcat2dAxis1 obsBV aReparamBT
    pure (uBT, qInputBT)
  (MkBang q1B # q1') <- forwardSeq {b=n} q1 qInputBT
  (MkBang q2B # q2') <- forwardSeq {b=n} q2 qInputBT
  loss <- liftIO1 (do losses <- go meanB uBT q1B q2B (toList epses) 0; meanScalarLoss n losses)
  pure1 (MkBang loss # MkNets actor' q1' q2' q1Tgt q2Tgt)
  where
    go : {n : Nat} ->
         Tensor [n, 1] Ex F WithGrad -> Tensor [n, 1] Ex F WithGrad ->
         Tensor [n, 1] Ex F WithGrad -> Tensor [n, 1] Ex F WithGrad ->
         List Double -> Int -> IO (List (Tensor [] Ex F WithGrad))
    go _ _ _ _ [] _                    = pure []
    go meanB uBT q1B q2B (_ :: rest) k = do
      l <- actorPerStepLoss meanB uBT q1B q2B logStdV alpha k
      ls <- go meanB uBT q1B q2B rest (k + 1)
      pure (l :: ls)

-- --- Batch update ---------------------------------------------------

runBatchUpdateL : Optimizer Ex -> Optimizer Ex -> Optimizer Ex ->
                  (1 _ : Nets) -> Tensor [] Ex F WithGrad -> Config -> {n : Nat} ->
                  Vect n (Transition ObsDim ActDim) -> L IO {use = 1} Nets
runBatchUpdateL q1Opt q2Opt actorOpt nets logStdV cfg {n} batch = do
  (MkBang tvs1 # nets1) <- foldTargetValsL nets logStdV cfg.gamma cfg.alpha (toList batch) []
  (MkBang q1LossV # nets2) <- q1LossL n nets1 batch tvs1
  _ <- liftIO1 (trainStep q1Opt q1LossV)
  (MkBang tvs2 # nets3) <- foldTargetValsL nets2 logStdV cfg.gamma cfg.alpha (toList batch) []
  (MkBang q2LossV # nets4) <- q2LossL n nets3 batch tvs2
  _ <- liftIO1 (trainStep q2Opt q2LossV)
  let obsVec = the (Vect n (Vect ObsDim Double)) (map (\t => t.obs) batch)
  (MkBang aLossV # nets5) <- actorLossBatchL n nets4 logStdV cfg.alpha obsVec
  _ <- liftIO1 (trainStep actorOpt aLossV)
  pure1 nets5

-- --- Main loop ------------------------------------------------------

-- Step every env with its action; auto-reset on Terminated OR per-env
-- EpisodeLen truncation. Bootstrap-done uses Terminated only.
--
-- Not `Gym.Vector.stepAutoReset`: that resets on the env's own Outcome, and
-- this loop also truncates on a per-env step count the env does not know
-- about. The reset itself still goes through `Env.reset`, so a restarted env
-- draws from MountainCar's U(-0.6, -0.4) start distribution.
stepAllAutoResetMCC : Source -> Vect n MCCState -> Vect n Double -> Vect n Nat ->
                      (Vect n MCCState, Vect n Double, Vect n Bool,
                       Vect n Bool, Vect n Nat, Source)
stepAllAutoResetMCC seed [] [] []                      = ([], [], [], [], [], seed)
stepAllAutoResetMCC seed (s :: ss) (a :: as) (l :: ls) =
  case mccStep s a of
    (r, s', outcome, _) =>
      let terminated = case outcome of
                         Terminated => True
                         _          => False
          truncated = (l + 1) >= EpisodeLen
          isDone    = terminated || truncated
          -- `the` pins the pair type: an `if` in a let-pattern binding leaves
          -- the scrutinee's type unsolved otherwise (docs/develop/gotchas.md).
          (nextS, seedNext) = the (MCCState, Source)
            (if isDone
               then reset {state=MCCState} {action=Double} {obs=Vect ObsDim Double} seed
               else (s', seed))
          nextL     = the Nat (if isDone then 0 else l + 1)
      in case stepAllAutoResetMCC seedNext ss as ls of
           (rest, rs, bds, ds, restL, seedEnd) =>
             (nextS :: rest, r :: rs, terminated :: bds, isDone :: ds, nextL :: restL, seedEnd)

-- Extracted so the call sites bind a function result (a `<- (inline case)` of
-- a linear result trips the L IO bind elaborator).
selectActionsL : Config -> Tensor [] Ex F WithGrad -> Nat -> Vect NumEnvs MCCState ->
                 (1 _ : Nets) -> L IO {use = 1} (LPair (!* (Vect NumEnvs Double)) Nets)
selectActionsL cfg logStdV stepCount envs nets =
  case stepCount < cfg.warmupSteps of
    True => do
      acts <- liftIO1 (traverse (\_ => randomRIO (the Double (negate MaxAct), MaxAct)) envs)
      pure1 (MkBang acts # nets)
    False => withNoGradL {ex=Ex} (sampleActionsNetsL nets logStdV envs)

maybeUpdateL : Optimizer Ex -> Optimizer Ex -> Optimizer Ex -> Config ->
               Tensor [] Ex F WithGrad -> ReplayBuffer ObsDim ActDim ->
               (q1Names : List String) -> (q1TgtNames : List String) ->
               (q2Names : List String) -> (q2TgtNames : List String) ->
               (bufSz : Nat) -> (stepCount : Nat) -> (1 _ : Nets) -> L IO {use = 1} Nets
maybeUpdateL q1Opt q2Opt actorOpt cfg logStdV buffer
             q1Names q1TgtNames q2Names q2TgtNames bufSz stepCount nets =
  case bufSz >= cfg.batchSize of
    False => pure1 nets
    True  => case stepCount >= cfg.warmupSteps of
      False => pure1 nets
      True  => do
        mBatch <- liftIO1 (sampleN cfg.batchSize buffer)
        case mBatch of
          Nothing    => pure1 nets
          Just batch => do
            nets' <- runBatchUpdateL q1Opt q2Opt actorOpt nets logStdV cfg batch
            liftIO1 $ do
              _ <- polyakUpdatePaired {ex=Ex} q1Names q1TgtNames cfg.tau
              _ <- polyakUpdatePaired {ex=Ex} q2Names q2TgtNames cfg.tau
              pure ()
            pure1 nets'

sacStepBatchedL : Optimizer Ex -> Optimizer Ex -> Optimizer Ex ->
                  Config -> (1 _ : SACState) -> L IO {use = 1} (LPair (!* Double) SACState)
sacStepBatchedL q1Opt q2Opt actorOpt cfg
                (MkSAC nets logStdV buffer stepRef envRef epLenRef retRef lastEpRef
                       q1Names q1TgtNames q2Names q2TgtNames) = do
  stepCount <- liftIO1 (readIORef stepRef)
  envs0 <- liftIO1 (readIORef envRef)
  epLens <- liftIO1 (readIORef epLenRef)
  oldRets <- liftIO1 (readIORef retRef)

  (MkBang actions # nets1) <- selectActionsL cfg logStdV stepCount envs0.envs nets

  -- Seeded by `srand cfg.seed` in main, so auto-resets stay reproducible.
  stepSeedI <- liftIO1 randomInt32
  case stepAllAutoResetMCC (Seeded (cast stepSeedI)) envs0.envs actions epLens of
    (envs', rewards, bufferDones, isDones, newEpLens, _) => do
      liftIO1 $ do
        pushAll buffer cfg.shaping envs0.envs actions rewards envs' bufferDones
        writeIORef envRef (MkVecEnv envs')
        writeIORef stepRef (stepCount + 1)
        writeIORef epLenRef newEpLens
        let newRets : Vect NumEnvs Double
            newRets = zipWith3 (\old, r, d => if d then 0.0 else old + r) oldRets rewards isDones
            completed : List Double
            completed = getCompleted (toList oldRets) (toList rewards) (toList isDones)
        writeIORef retRef newRets
        case completed of
          []        => pure ()
          (e :: es) => writeIORef lastEpRef (last (e :: es))

      bufSz <- liftIO1 (bufferSize buffer)
      nets2 <- maybeUpdateL q1Opt q2Opt actorOpt cfg logStdV buffer
                            q1Names q1TgtNames q2Names q2TgtNames bufSz stepCount nets1
      lastEp <- liftIO1 (readIORef lastEpRef)
      pure1 (MkBang (negate lastEp)
             # MkSAC nets2 logStdV buffer stepRef envRef epLenRef retRef lastEpRef
                     q1Names q1TgtNames q2Names q2TgtNames)
  where
    zipWith3 : (a -> b -> c -> d) -> Vect n a -> Vect n b -> Vect n c -> Vect n d
    zipWith3 _ [] [] []                      = []
    zipWith3 f (x :: xs) (y :: ys) (z :: zs) = f x y z :: zipWith3 f xs ys zs

    getCompleted : List Double -> List Double -> List Bool -> List Double
    getCompleted [] _ _                            = []
    getCompleted _ [] _                            = []
    getCompleted _ _ []                            = []
    getCompleted (run :: rs) (rw :: rws) (d :: ds) =
      let recur = getCompleted rs rws ds
      in if d then (run + rw) :: recur else recur

    pushAll : ReplayBuffer ObsDim ActDim -> Double -> Vect n MCCState -> Vect n Double ->
              Vect n Double -> Vect n MCCState -> Vect n Bool -> IO ()
    pushAll _   _       []         []         []         []          []       = pure ()
    pushAll buf shaping (s :: ss) (a :: as) (r :: rs) (s' :: ss') (bd :: bds) = do
      let shapedR = r + shaping * abs s'.mccVel
      push buf (MkTransition (observeVec s) [a] shapedR (observeVec s') bd)
      pushAll buf shaping ss as rs ss' bds

-- --- Greedy evaluation ----------------------------------------------

greedyActL : (1 _ : ActorNet) -> Vect ObsDim Double -> L IO {use = 1} (LPair (!* Double) ActorNet)
greedyActL actor obs = do
  (MkBang mean # actor') <- actorMeanL actor obs
  pure1 (MkBang (Math.tanh mean * MaxAct) # actor')

evalEpL : (1 _ : ActorNet) -> MCCState -> Nat -> Double -> L IO {use = 1} (LPair (!* Double) ActorNet)
evalEpL actor _ Z acc      = pure1 (MkBang acc # actor)
evalEpL actor st (S k) acc = do
  (MkBang a # actor') <- greedyActL actor (mccObserve st)
  case mccStep st a of
    (r, st', outcome, _) =>
      case outcome of
        Terminated => pure1 (MkBang (acc + r) # actor')
        _          => evalEpL actor' st' k (acc + r)

-- Each episode starts from a fresh `reset` draw, as the reference's
-- `env.reset()` does. A fixed start would make every greedy episode the same
-- trajectory, so the mean over N of them would carry one sample's worth of
-- information.
evalNL : (1 _ : ActorNet) -> Nat -> Double -> L IO {use = 1} (LPair (!* Double) ActorNet)
evalNL actor Z acc     = pure1 (MkBang acc # actor)
evalNL actor (S k) acc = do
  resetSeedI <- liftIO1 randomInt32
  let (st0, _) = reset {state=MCCState} {action=Double} {obs=Vect ObsDim Double}
                       (Seeded (cast resetSeedI))
  (MkBang v # actor') <- withNoGradL {ex=Ex} (evalEpL actor st0 EpisodeLen 0.0)
  evalNL actor' k (acc + v)

----------------------------------------------------------------------
-- State construction / eval / discard (linear)
----------------------------------------------------------------------

buildStateL : Config -> L IO {use = 1} SACState
buildStateL cfg = do
  actor <- runInitL mkActor
  q1    <- runInitL (mkQ "q1_")
  q2    <- runInitL (mkQ "q2_")
  q1Tgt <- runInitL (mkQ "q1tgt_")
  q2Tgt <- runInitL (mkQ "q2tgt_")
  -- Capture each net's exact param names once (static for the run); pair them
  -- positionally for every target sync, so no naming convention is load-bearing.
  let (MkBang q1Names # q1)       = reflectNames q1
  let (MkBang q2Names # q2)       = reflectNames q2
  let (MkBang q1TgtNames # q1Tgt) = reflectNames q1Tgt
  let (MkBang q2TgtNames # q2Tgt) = reflectNames q2Tgt
  logStdV <- liftIO1 (the (IO (Tensor [] Ex F WithGrad)) (tparamScalar "actor_log_std" 0.0))
  -- After actor_log_std: it is a registered param too, so the dump has to
  -- see it alongside the five networks.
  liftIO1 (maybeDumpInit {ex = ExampleExecutor})
  liftIO1 $ do
    _ <- polyakUpdatePaired {ex=Ex} q1Names q1TgtNames 1.0
    _ <- polyakUpdatePaired {ex=Ex} q2Names q2TgtNames 1.0
    pure ()
  buffer  <- liftIO1 (mkBuffer {obsDim=ObsDim, actDim=ActDim} cfg.bufferCap)
  stepRef <- liftIO1 (newIORef (the Nat 0))
  resetSeedI <- liftIO1 randomInt32
  let initEnvs : VecEnv NumEnvs MCCState
      initEnvs = fst (resetAll {state=MCCState} {action=Double} {obs=Vect 2 Double}
                              (Seeded (cast resetSeedI)))
  envRef    <- liftIO1 (newIORef initEnvs)
  epLenRef  <- liftIO1 (newIORef (the (Vect NumEnvs Nat) (replicate NumEnvs 0)))
  retRef    <- liftIO1 (newIORef (the (Vect NumEnvs Double) (replicate NumEnvs 0.0)))
  lastEpRef <- liftIO1 (newIORef (the Double 0.0))
  pure1 (MkSAC (MkNets actor q1 q2 q1Tgt q2Tgt) logStdV buffer stepRef envRef epLenRef retRef lastEpRef
               q1Names q1TgtNames q2Names q2TgtNames)

finalReportL : Config -> Nat -> (1 _ : SACState) -> L IO ()
finalReportL cfg epochsDone (MkSAC (MkNets actor q1 q2 q1Tgt q2Tgt) _ _ _ _ _ _ _ _ _ _ _) = do
  let nEval = the Nat 20
  (MkBang evalSum # actor') <- evalNL actor nEval 0.0
  discard actor'
  discard q1
  discard q2
  discard q1Tgt
  discard q2Tgt
  liftIO1 $ do
    let avgReturn = evalSum / cast (natToInteger nEval)
    putStrLn ""
    putStrLn $ "Eval (" ++ show nEval ++ " episodes, greedy): avg_return=" ++ show avgReturn
    putStrLn ""
    putStrLn $ formatResult [("avg_return", show avgReturn),
                              ("epochs", show epochsDone),
                              ("seed", show cfg.seed)]

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

  putStrLn ""

  if cfg.lrFind
    then do
      putStrLn "lr_find skipped for SAC: per-step epochs + warmup don't fit"
      putStrLn "the LR-range-test pattern. See docs/develop/hyperparameter-tuning-2026.md."
    else Control.Linear.LIO.run $ do
      st0 <- buildStateL cfg
      -- Three Adams, each owning one network's params (typed ownership via
      -- restrictTo: LR 0 outside the net's registry names, so no cross-net
      -- update leak). The "actor_" set also covers log_std.
      actorOpt <- liftIO1 (adam cfg.lr ({ clip := NormClip cfg.clipNorm } defaultOpts))
      q1Opt    <- liftIO1 (adam cfg.lr ({ clip := NormClip cfg.clipNorm } defaultOpts))
      q2Opt    <- liftIO1 (adam cfg.lr ({ clip := NormClip cfg.clipNorm } defaultOpts))
      liftIO1 $ do
        restrictTo actorOpt !(namesMatching {ex=Ex} (isPrefixOf "actor_"))
        restrictTo q1Opt    !(namesMatching {ex=Ex} (isPrefixOf "q1_"))
        restrictTo q2Opt    !(namesMatching {ex=Ex} (isPrefixOf "q2_"))
      metrics  <- liftIO1 (newRLMetricsState 20)
      let trainCfg : TrainConfig SACState
          trainCfg = { metricsL := readRLMetrics "recent_20" metrics }
                       (mkTrainConfig cfg.epochs 2000
                          (WindowedAvg cfg.esThreshold cfg.esWindow cfg.esPatience)
                          (const (pure (the (List (String, String)) []))) (\_ => pure ()))
      (MkBang (epochsDone, _) # trained) <- fit {batch = ()}
        (\s, _ => do
           (MkBang loss # s') <- sacStepBatchedL q1Opt q2Opt actorOpt cfg s
           dd <- liftIO1 (do recordReturn metrics (negate loss); pure loss)
           pure1 (MkBang dd # s'))
        actorOpt (generate (pure ())) trainCfg st0
      finalReportL cfg epochsDone trained
