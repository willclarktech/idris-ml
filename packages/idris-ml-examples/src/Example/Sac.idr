module Example.Sac

import Control.Linear.LIO
import Data.IORef
import Data.Linear.Notation
import Data.List
import Data.String
import Data.Vect
import System

import Array
import BuildConfig
import Compat.Random
import Fit
import Floating
import Gym.ClassicControl.Pendulum
import Gym.Env
import Gym.Vector
import ML.Simple
import Math
import RL.ReplayBuffer
import Sampler
import Train
import Train.Freeze

-- Actor + Q-nets are linear `Seq`s; hide the IO `Nn.Seq` constructors.

----------------------------------------------------------------------
-- SAC on Pendulum-v1 (aligned with `torch_ref/models/sac.py`):
--   Actor : Linear(3,64) → ReLU → Linear(64,64) → ReLU → Linear(64,1) = mean
--   Q1/Q2 : Linear(4,64) → ReLU → Linear(64,64) → ReLU → Linear(64,1) = value
--           (input is obs ++ action)
--   log_std : standalone learnable Tensor [] under "actor_log_std".
--   Target Q nets: scoped "q1tgt_" / "q2tgt_", moved via polyak soft
--                  update — no optimizer owns them. Trailing "_" keeps
--                  "q1_" out of "q1tgt_" so the q1 Adam can't step the
--                  q1-target net.
----------------------------------------------------------------------

ObsDim     : Nat; ObsDim = 3
ActDim     : Nat; ActDim = 1
QInputDim  : Nat; QInputDim = 4          -- ObsDim + ActDim
Hidden     : Nat; Hidden = 64
EpisodeLen : Nat; EpisodeLen = 200
MaxAction  : Double; MaxAction = 2.0

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

-- The five SAC nets bundled as one linear value (named-field access avoids a
-- 5-deep nested LPair). Threaded single-owner through every forward.
record Nets where
  constructor MkNets
  1 actor : ActorNet
  1 q1    : QNet
  1 q2    : QNet
  1 q1Tgt : QNet
  1 q2Tgt : QNet

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
      action = Math.tanh u * MaxAction
      lp_u   = -0.5 * ((u - mean) / std) * ((u - mean) / std) - logStd - logTwoPiHalf
      lp     = lp_u - squashCorrection u
  pure1 (MkBang (action, lp) # actor')

sampleActionsBatchedL : {n : Nat} -> (1 _ : ActorNet) -> Tensor [] Ex F WithGrad ->
                        Vect n PState -> L IO {use = 1} (LPair (!* (Vect n Double)) ActorNet)
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
         Double -> Int -> Vect k PState -> IO (Vect k Double)
    go _ _ _ []                = pure []
    go meanB std i (_ :: rest) = do
      let mean = primItem2d {ex=Ex} meanB.tensorPtr i 0
      eps <- normalSample
      let u = mean + std * eps
          action = Math.tanh u * MaxAction
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
  envRef    : IORef (VecEnv NumEnvs PState)
  epLenRef  : IORef (Vect NumEnvs Nat)
  retRef    : IORef (Vect NumEnvs Double)
  lastEpRef : IORef Double

-- Batched action sampling within the Nets bundle (uses actor; rest pass through).
sampleActionsNetsL : {n : Nat} -> (1 _ : Nets) -> Tensor [] Ex F WithGrad ->
                     Vect n PState -> L IO {use = 1} (LPair (!* (Vect n Double)) Nets)
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
  clipNorm    : Double
  seed        : Bits64
  esThreshold : Double
  esWindow    : Nat
  esPatience  : Nat
  lrFind      : Bool

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

-- Bootstrap target for one transition, threading the Nets (uses actor +
-- q1Tgt + q2Tgt; q1/q2 pass through). logStdV is a Tensor (ω).
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

-- Forward the given (linear) online Q net on the batch's q-inputs, then the
-- per-sample TD loss vs the pre-computed target values. Threads the one net.
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

-- q1 / q2 online-loss within the Nets bundle (forward the named online net,
-- the other four pass through).
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
    aReparamBT  <- tmulScalar aSquashedBT MaxAction
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

-- --- One batch update: three group-scoped optimizer steps ------------

-- One batch update threading the Nets bundle: q1 loss + step, q2 loss + step,
-- actor loss + step. logStdV (ω) feeds the target + actor losses.
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

-- --- Main loop -------------------------------------------------------

-- Step every env with its action; auto-reset on per-env EpisodeLen.
-- Pendulum has no termination (continuous task), so isDone is purely the
-- per-env step counter hitting EpisodeLen.
stepAllAutoResetP : Vect n PState -> Vect n Double -> Vect n Nat ->
                    (Vect n PState, Vect n Double, Vect n Bool, Vect n Nat)
stepAllAutoResetP [] [] []                      = ([], [], [], [])
stepAllAutoResetP (s :: ss) (a :: as) (l :: ls) =
  case pStep s a of
    (r, s', _, _) =>
      let isDone = (l + 1) >= EpisodeLen
          nextS = if isDone then MkP 3.141592653589793 0.0 else s'
          nextL = the Nat (if isDone then 0 else l + 1)
      in case stepAllAutoResetP ss as ls of
           (rest, rs, ds, restL) =>
             (nextS :: rest, r :: rs, isDone :: ds, nextL :: restL)

-- Action selection (warmup random vs policy sample), threading Nets. Extracted
-- so the call site binds a function result, not an inline `case` (a `<- case`
-- of a linear result trips the L IO bind elaborator's Lazy handling).
selectActionsL : Config -> Tensor [] Ex F WithGrad -> Nat -> Vect NumEnvs PState ->
                 (1 _ : Nets) -> L IO {use = 1} (LPair (!* (Vect NumEnvs Double)) Nets)
selectActionsL cfg logStdV stepCount envs nets =
  case stepCount < cfg.warmupSteps of
    True => do
      acts <- liftIO1 (traverse (\_ => randomRIO (the Double (negate MaxAction), MaxAction)) envs)
      pure1 (MkBang acts # nets)
    False => withNoGradL {ex=Ex} (sampleActionsNetsL nets logStdV envs)

-- Maybe run a batch update + target polyak (once warm + buffer full), threading
-- Nets. Extracted for the same reason as selectActionsL.
maybeUpdateL : Optimizer Ex -> Optimizer Ex -> Optimizer Ex -> Config ->
               ReplayBuffer ObsDim ActDim -> Tensor [] Ex F WithGrad ->
               (bufSz : Nat) -> (stepCount : Nat) -> (1 _ : Nets) -> L IO {use = 1} Nets
maybeUpdateL q1Opt q2Opt actorOpt cfg buffer logStdV bufSz stepCount nets =
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
              _ <- polyakUpdate {ex=Ex} cfg.tau "q1_" "q1tgt_"
              _ <- polyakUpdate {ex=Ex} cfg.tau "q2_" "q2tgt_"
              pure ()
            pure1 nets'

sacStepBatchedL : Optimizer Ex -> Optimizer Ex -> Optimizer Ex ->
                  Config -> (1 _ : SACState) -> L IO {use = 1} (LPair (!* Double) SACState)
sacStepBatchedL q1Opt q2Opt actorOpt cfg
                (MkSAC nets logStdV buffer stepRef envRef epLenRef retRef lastEpRef) = do
  stepCount <- liftIO1 (readIORef stepRef)
  envs0 <- liftIO1 (readIORef envRef)
  epLens <- liftIO1 (readIORef epLenRef)
  oldRets <- liftIO1 (readIORef retRef)

  (MkBang actions # nets1) <- selectActionsL cfg logStdV stepCount envs0.envs nets

  case stepAllAutoResetP envs0.envs actions epLens of
    (envs', rewards, isDones, newEpLens) => do
      liftIO1 $ do
        pushAll buffer envs0.envs actions rewards envs' isDones
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
      nets2 <- maybeUpdateL q1Opt q2Opt actorOpt cfg buffer logStdV bufSz stepCount nets1
      lastEp <- liftIO1 (readIORef lastEpRef)
      pure1 (MkBang (negate lastEp)
             # MkSAC nets2 logStdV buffer stepRef envRef epLenRef retRef lastEpRef)
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

    pushAll : ReplayBuffer ObsDim ActDim -> Vect n PState -> Vect n Double -> Vect n Double ->
              Vect n PState -> Vect n Bool -> IO ()
    pushAll _   []         []         []         []           []    = pure ()
    pushAll buf (s :: ss) (a :: as) (r :: rs) (s' :: ss') (d :: ds) = do
      push buf (MkTransition (observeVec s) [a] r (observeVec s') d)
      pushAll buf ss as rs ss' ds

-- --- Greedy evaluation ----------------------------------------------

greedyActL : (1 _ : ActorNet) -> Vect ObsDim Double -> L IO {use = 1} (LPair (!* Double) ActorNet)
greedyActL actor obs = do
  (MkBang mean # actor') <- actorMeanL actor obs
  pure1 (MkBang (Math.tanh mean * MaxAction) # actor')

evalEpL : (1 _ : ActorNet) -> PState -> Nat -> Double -> L IO {use = 1} (LPair (!* Double) ActorNet)
evalEpL actor _ Z acc      = pure1 (MkBang acc # actor)
evalEpL actor st (S k) acc = do
  (MkBang a # actor') <- greedyActL actor (observeVec st)
  case pStep st a of
    (r, st', _, _) => evalEpL actor' st' k (acc + r)

evalNL : (1 _ : ActorNet) -> Nat -> Double -> L IO {use = 1} (LPair (!* Double) ActorNet)
evalNL actor Z acc     = pure1 (MkBang acc # actor)
evalNL actor (S k) acc = do
  (MkBang v # actor') <- withNoGradL {ex=Ex} (evalEpL actor (MkP 3.141592653589793 0.0) EpisodeLen 0.0)
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
  logStdV <- liftIO1 (the (IO (Tensor [] Ex F WithGrad)) (tparamScalar "actor_log_std" 0.0))
  liftIO1 $ do
    _ <- polyakUpdate {ex=Ex} 1.0 "q1_" "q1tgt_"
    _ <- polyakUpdate {ex=Ex} 1.0 "q2_" "q2tgt_"
    pure ()
  buffer  <- liftIO1 (mkBuffer {obsDim=ObsDim, actDim=ActDim} cfg.bufferCap)
  stepRef <- liftIO1 (newIORef (the Nat 0))
  resetSeedI <- liftIO1 randomInt32
  let initEnvs : VecEnv NumEnvs PState
      initEnvs = fst (resetAll {state=PState} {action=Double} {obs=Vect 3 Double}
                              (cast resetSeedI))
  envRef    <- liftIO1 (newIORef initEnvs)
  epLenRef  <- liftIO1 (newIORef (the (Vect NumEnvs Nat) (replicate NumEnvs 0)))
  retRef    <- liftIO1 (newIORef (the (Vect NumEnvs Double) (replicate NumEnvs 0.0)))
  lastEpRef <- liftIO1 (newIORef (the Double 0.0))
  pure1 (MkSAC (MkNets actor q1 q2 q1Tgt q2Tgt) logStdV buffer stepRef envRef epLenRef retRef lastEpRef)

finalReportL : Config -> Nat -> (1 _ : SACState) -> L IO ()
finalReportL cfg epochsDone (MkSAC (MkNets actor q1 q2 q1Tgt q2Tgt) _ _ _ _ _ _ _) = do
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

  putStrLn "=== SAC on Pendulum ==="
  putStrLn $ "Config: lr=" ++ show cfg.lr
           ++ " steps=" ++ show cfg.epochs
           ++ " gamma=" ++ show cfg.gamma
           ++ " alpha=" ++ show cfg.alpha
           ++ " batch=" ++ show cfg.batchSize
           ++ " warmup=" ++ show cfg.warmupSteps
           ++ " tau=" ++ show cfg.tau
           ++ " seed=" ++ show cfg.seed

  putStrLn ""

  if cfg.lrFind
    then do
      putStrLn "lr_find skipped for SAC: per-step epochs + warmup don't fit"
      putStrLn "the LR-range-test pattern. See docs/develop/hyperparameter-tuning-2026.md."
    else Control.Linear.LIO.run $ do
      st0 <- buildStateL cfg
      -- Three Adams, each owning one network's params (typed ownership:
      -- restrictTo zeroes LR on everything outside the net's registry names,
      -- so one net's loss can't leak updates into another's weights). The
      -- "actor_" set also covers "actor_log_std"; the explicit isPrefixOf is
      -- leak-safe here ("q1_" is not a prefix of "q1tgt_").
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
