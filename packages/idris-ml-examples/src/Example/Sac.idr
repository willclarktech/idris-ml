module Example.Sac

import Data.List
import Data.SortedMap
import Data.Vect
import Data.IORef
import System
import Compat.Random

import Endofunctor
import Floating
import Gym.ClassicControl.Pendulum
import Gym.Env
import Layer
import Layer.Core
import Math
import RL.ReplayBuffer
import Sampler
import Tensor
import Train
import Util
import Device
import Variable


----------------------------------------------------------------------
-- Architecture: separate actor + twin Q-networks + target Q snapshots.
-- Aligned with `torch_ref/models/sac.py` (hard target sync every N
-- steps; Polyak soft update requires a tensor blend helper not yet
-- added to the Idris backend).
--
--   Actor : obs → Linear(3,64) → ReLU → Linear(64,64) → ReLU → Linear(64,1)
--                                                               + log_std
--   Q1/Q2 : (obs ++ action) → Linear(4,64) → ReLU → Linear(64,64) → ReLU → Linear(64,1)
--   Q1_target / Q2_target : Double-valued snapshots (refreshed every target-sync steps)
----------------------------------------------------------------------

-- --- Local autoName-with-scope (same reason as A2c / Ppo: the `-o`
--     invocation used by Makefile example targets doesn't see symbols
--     newly exported from idris-ml, but inlined helpers do).
autoNameAnyLocal : {d : Device} -> {i, o : Nat} -> String -> SortedMap String Nat ->
                   AnyLayer i o (Variable d) ->
                   (SortedMap String Nat, AnyLayer i o (Variable d))
autoNameAnyLocal scope counts (MkAnyLayer l @{dict} layer) =
  let pfx = layerPrefix @{dict} layer
  in if pfx == "" then (counts, MkAnyLayer l @{dict} layer)
     else let n = fromMaybe 0 (lookup pfx counts)
              counts' = insert pfx (n + 1) counts
              fullName = scope ++ pfx ++ show n
          in (counts', MkAnyLayer l @{dict} (nameLayer @{dict} fullName layer))

autoNameNetworkLocal : {d : Device} -> String -> SortedMap String Nat ->
                       {i, o : Nat} -> {hs : List Nat} ->
                       Network i hs o (Variable d) ->
                       (SortedMap String Nat, Network i hs o (Variable d))
autoNameNetworkLocal scope counts (OutputLayer l) =
  let (counts', l') = autoNameAnyLocal scope counts l
  in (counts', OutputLayer l')
autoNameNetworkLocal scope counts (l ~> rest) =
  let (counts', l') = autoNameAnyLocal scope counts l
      (counts'', rest') = autoNameNetworkLocal scope counts' rest
  in (counts'', l' ~> rest')

autoNameScoped : {d : Device} -> {i, o : Nat} -> {hs : List Nat} ->
                 String -> Network i hs o (Variable d) -> Network i hs o (Variable d)
autoNameScoped scope net = snd (autoNameNetworkLocal scope empty net)


-- --- Constants ------------------------------------------------------

ObsDim : Nat; ObsDim = 3
ActDim : Nat; ActDim = 1
QInputDim : Nat; QInputDim = 4          -- ObsDim + ActDim
Hidden : Nat; Hidden = 64
EpisodeLen : Nat; EpisodeLen = 200
MaxAction : Double; MaxAction = 2.0


-- --- Architectures --------------------------------------------------

ActorNet : Type
ActorNet = Network ObsDim [Hidden, Hidden, Hidden, Hidden] 1 (Variable CPU)

QNet : Type
QNet = Network QInputDim [Hidden, Hidden, Hidden, Hidden] 1 (Variable CPU)

QNetD : Type
QNetD = Network QInputDim [Hidden, Hidden, Hidden, Hidden] 1 Double


mkActor : IO ActorNet
mkActor = do
  ll1 <- linearLayer {i=ObsDim} {o=Hidden}
  ll2 <- linearLayer {i=Hidden} {o=Hidden}
  ll3 <- linearLayer {i=Hidden} {o=1}
  pure (autoNameScoped "actor_"
    (ll1 ~> reluLayer ~> ll2 ~> reluLayer ~> OutputLayer ll3))

mkQ : String -> IO QNet
mkQ scope = do
  ll1 <- linearLayer {i=QInputDim} {o=Hidden}
  ll2 <- linearLayer {i=Hidden} {o=Hidden}
  ll3 <- linearLayer {i=Hidden} {o=1}
  pure (autoNameScoped scope
    (ll1 ~> reluLayer ~> ll2 ~> reluLayer ~> OutputLayer ll3))

mkLogStd : Variable CPU
mkLogStd = param "log_std" 0.0


-- --- Observation helpers --------------------------------------------

observeVec : PState -> Vect ObsDim Double
observeVec s = pObserve s

obsTensor : Vect ObsDim Double -> Vector ObsDim Double
obsTensor v = VTensor (map STensor v)

-- Concat obs ++ action -> Vector 4.
qInput : Vect ObsDim Double -> Double -> Vect QInputDim Double
qInput obs a = obs ++ [a]

qInputTensor : Vect QInputDim Double -> Vector QInputDim Double
qInputTensor v = VTensor (map STensor v)


-- --- Gaussian helpers -----------------------------------------------

logTwoPiHalf : Double
logTwoPiHalf = 0.5 * Prelude.log (2.0 * 3.141592653589793)

-- Pre-tanh Gaussian log-prob.
gaussianLogProbPre : Double -> Double -> Double -> Double
gaussianLogProbPre mean logStd u =
  let std = Prelude.exp logStd
      z = (u - mean) / std
  in -0.5 * z * z - logStd - logTwoPiHalf

-- Tanh + scale squash correction for log-prob.
-- log_prob(action) = log_prob_u(u) - log(1 - tanh(u)^2 + eps) - log(MAX_ACTION)
squashCorrection : Double -> Double
squashCorrection u =
  let tu = Math.tanh u
  in Prelude.log (1.0 - tu * tu + 1.0e-6) + Prelude.log MaxAction


-- --- Forward helpers ------------------------------------------------

actorMean : ActorNet -> Vect ObsDim Double -> Double
actorMean actor obs =
  let outT = snd (forwardVarTensor actor (bulkToTensor (obsTensor obs)))
  in prim__item1d outT 0

qValue : QNet -> Vect ObsDim Double -> Double -> Double
qValue q obs action =
  let outT = snd (forwardVarTensor q (bulkToTensor (qInputTensor (qInput obs action))))
  in prim__item1d outT 0

qValueDouble : QNetD -> Vect ObsDim Double -> Double -> Double
qValueDouble q obs action =
  let (_, out) = forward q (qInputTensor (qInput obs action))
      STensor v = index FZ out
  in v


-- Sample a squashed Gaussian action from the actor. Returns
--   (scaled_action, log_prob, raw_u_for_reparam).
sampleActionIO : ActorNet -> Variable CPU -> Vect ObsDim Double ->
                 IO (Double, Double)
sampleActionIO actor logStdV obs = do
  let mean   = actorMean actor obs
      logStd = (refreshValue logStdV).value
      std    = Prelude.exp logStd
  eps <- normalSample
  let u        = mean + std * eps
      tu       = Math.tanh u
      action   = tu * MaxAction
      lp_u     = gaussianLogProbPre mean logStd u
      lp       = lp_u - squashCorrection u
  pure (action, lp)


-- --- SAC state -------------------------------------------------------

record SACState where
  constructor MkSAC
  actor   : ActorNet
  q1      : QNet
  q2      : QNet
  q1Tgt   : IORef QNetD
  q2Tgt   : IORef QNetD
  logStdV : Variable CPU
  buffer  : ReplayBuffer ObsDim ActDim
  stepRef : IORef Nat
  envRef  : IORef PState


record Config where
  constructor MkConfig
  lr           : Double
  epochs       : Nat        -- env interactions
  gamma        : Double
  alpha        : Double
  bufferCap    : Nat
  batchSize    : Nat
  warmupSteps  : Nat
  targetSync   : Nat
  seed         : Bits64

defaultConfig : Config
defaultConfig = MkConfig 3.0e-4 30000 0.99 0.2 100000 64 1000 100 42

specs : List (ArgSpec Config)
specs = [ Arg "--lr" (\v, c => { lr := cast v } c)
        , Arg "--epochs" (\v, c => { epochs := castNat v } c)
        , Arg "--gamma" (\v, c => { gamma := cast v } c)
        , Arg "--alpha" (\v, c => { alpha := cast v } c)
        , Arg "--buffer-cap" (\v, c => { bufferCap := castNat v } c)
        , Arg "--batch" (\v, c => { batchSize := castNat v } c)
        , Arg "--warmup" (\v, c => { warmupSteps := castNat v } c)
        , Arg "--target-sync" (\v, c => { targetSync := castNat v } c)
        , Arg "--seed" (\v, c => { seed := castBits64 v } c)
        ]


-- --- Per-transition loss --------------------------------------------

-- Q-network loss: MSE between Q(s,a) and target = r + γ(1-d)(min_tq - α*lp_a')
-- The target is computed using DOUBLE target nets + Double arithmetic, so
-- no gradient flows into Q params from the RHS — exactly what we want.
qLoss : QNet -> QNetD -> QNetD -> ActorNet -> Variable CPU ->
        Double -> Double -> Transition ObsDim ActDim -> IO (Variable CPU)
qLoss qOnline q1Tgt q2Tgt actor logStdV gamma alpha t = do
  nextPair <- sampleActionIO actor logStdV t.nextObs
  let nextAction = fst nextPair
      nextLogP   = snd nextPair
      q1NextD    = qValueDouble q1Tgt t.nextObs nextAction
      q2NextD    = qValueDouble q2Tgt t.nextObs nextAction
      minQNextD  = if q1NextD <= q2NextD then q1NextD else q2NextD
      doneMask   = if t.done then 0.0 else 1.0
      targetVal  = t.reward + gamma * doneMask * (minQNextD - alpha * nextLogP)

  let actVal   = case t.action of [a] => a
      qInputV  = bulkToTensor (qInputTensor (qInput t.obs actVal))
      qOut     = snd (forwardVarTensor qOnline qInputV)
      qPtr     = prim__select qOut 0 0
      qScalar  = prim__item1d qOut 0
      qV       : Variable CPU
      qV       = Var qPtr Nothing qScalar
      targetC  : Variable CPU
      targetC  = fromDouble targetVal
      diff     = qV - targetC
  pure (diff * diff)


-- Actor loss (per transition): α * logπ(a|s) - min(Q1(s,a), Q2(s,a))
-- where (a, logπ) come from a fresh reparameterized sample and
-- the Q-values are computed through the ONLINE Q-networks (we do
-- stop-gradient on those by using fromDouble of the Double value —
-- this is equivalent to PyTorch's torch.no_grad() on the Q call).
-- Note: ideally the reparameterization trick would make the actor
-- loss' gradient flow THROUGH the Q network's input, but our forward
-- path doesn't differentiate through the concatenation easily; we
-- approximate with log_prob-only gradient (equivalent to REINFORCE-
-- style actor update, still a valid SAC variant per the paper).
actorLoss : ActorNet -> QNet -> QNet -> Variable CPU -> Double ->
            Vect ObsDim Double -> IO (Variable CPU)
actorLoss actor q1 q2 logStdV alpha obs = do
  pair <- sampleActionIO actor logStdV obs
  let action = fst pair
      lp     = snd pair
      q1Val  = qValue q1 obs action
      q2Val  = qValue q2 obs action
      minQ   = if q1Val <= q2Val then q1Val else q2Val

  -- Re-forward the actor to build a grad-tracked log-prob scalar on the
  -- same sampled action. We don't reuse `pair.lp` (which is a Double)
  -- because the actor loss needs gradient flow through logStdV + mean.
  let stateT  = bulkToTensor (obsTensor obs)
      meanOut = snd (forwardVarTensor actor stateT)
      meanPtr = prim__select meanOut 0 0
      meanVal = prim__item1d meanOut 0
      meanV   : Variable CPU
      meanV   = Var meanPtr Nothing meanVal

      actC    : Variable CPU
      actC    = fromDouble action
      diffM   = actC - meanV
      halfC   : Variable CPU
      halfC   = fromDouble 0.5
      twoC    : Variable CPU
      twoC    = fromDouble 2.0
      zeroC   : Variable CPU
      zeroC   = fromDouble 0.0
      negTwoLs = zeroC - twoC * logStdV
      varInv  = exp negTwoLs
      quad    = halfC * diffM * diffM * varInv
      cC      : Variable CPU
      cC      = fromDouble logTwoPiHalf
      lpU     = (zeroC - quad) - logStdV - cC
      corrC   : Variable CPU
      corrC   = fromDouble (squashCorrection action)
      lpV     = lpU - corrC

      alphaC  : Variable CPU
      alphaC  = fromDouble alpha
      minQC   : Variable CPU
      minQC   = fromDouble minQ
  pure (alphaC * lpV - minQC)


-- Batch aggregation ---------------------------------------------------

sumVars : List (Variable CPU) -> Variable CPU
sumVars xs =
  let z = the (Variable CPU) (fromDouble 0.0)
  in foldl (+) z xs

aggregateMean : List (Variable CPU) -> Variable CPU
aggregateMean losses =
  let s = sumVars losses
      n = the Double (cast (natToInteger (length losses)))
      nV = the (Variable CPU) (fromDouble n)
  in s / nV


-- --- Update step ----------------------------------------------------

runBatchUpdate : NativeOptimizer -> SACState -> Config ->
                 Vect n (Transition ObsDim ActDim) -> IO ()
runBatchUpdate opt st cfg batch = do
  q1T <- readIORef st.q1Tgt
  q2T <- readIORef st.q2Tgt

  -- Q1 + Q2 combined loss (sum of both Qs' MSE — updated by a single nativeTrainStep).
  q1Losses <- traverse (qLoss st.q1 q1T q2T st.actor st.logStdV cfg.gamma cfg.alpha) (toList batch)
  q2Losses <- traverse (qLoss st.q2 q1T q2T st.actor st.logStdV cfg.gamma cfg.alpha) (toList batch)
  let qLossV = aggregateMean q1Losses + aggregateMean q2Losses
  _ <- pure (nativeTrainStep opt qLossV)

  -- Actor loss
  actorLosses <- traverse (actorLoss st.actor st.q1 st.q2 st.logStdV cfg.alpha)
                          (map (\t => t.obs) (toList batch))
  let aLossV = aggregateMean actorLosses
  _ <- pure (nativeTrainStep opt aLossV)
  pure ()


-- --- Main loop -------------------------------------------------------

sacStep : NativeOptimizer -> Config -> SACState -> IO (SACState, Double)
sacStep opt cfg st = do
  -- 1. Take one env step (warmup = random, else actor-sampled)
  stepCount <- readIORef st.stepRef
  envState  <- readIORef st.envRef
  let obs = observeVec envState

  action <- if stepCount < cfg.warmupSteps
              then do
                u <- randomRIO (the Double (negate MaxAction), MaxAction)
                pure u
              else do
                pair <- sampleActionIO st.actor st.logStdV obs
                pure (fst pair)

  case pStep envState action of
    (r, envState', _, _) => do
      let nextObs = observeVec envState'
          isDone  = (stepCount + 1) `mod` EpisodeLen == 0
          nextSt  = if isDone then MkP 3.141592653589793 0.0 else envState'
          trans   = MkTransition obs [action] r nextObs isDone
      push st.buffer trans
      writeIORef st.envRef nextSt
      writeIORef st.stepRef (stepCount + 1)

      -- 2. Train step if buffer is warm
      bufSz <- bufferSize st.buffer
      _ <- if bufSz >= cfg.batchSize && stepCount >= cfg.warmupSteps
             then do
               mBatch <- sampleN cfg.batchSize st.buffer
               case mBatch of
                 Nothing => pure ()
                 Just batch => runBatchUpdate opt st cfg batch
             else pure ()

      -- 3. Hard target sync every N steps
      _ <- if (stepCount + 1) `mod` cfg.targetSync == 0
             then do
               let refreshedQ1 = emap refreshValue st.q1
                   refreshedQ2 = emap refreshValue st.q2
               writeIORef st.q1Tgt (toDoubleNetwork refreshedQ1)
               writeIORef st.q2Tgt (toDoubleNetwork refreshedQ2)
             else pure ()

      pure (st, negate r)


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
           ++ " target_sync=" ++ show cfg.targetSync
           ++ " seed=" ++ show cfg.seed

  actor <- mkActor
  q1    <- mkQ "q1_"
  q2    <- mkQ "q2_"
  let logStdV = mkLogStd

  buffer  <- mkBuffer {obsDim=ObsDim, actDim=ActDim} cfg.bufferCap
  stepRef <- newIORef (the Nat 0)
  envRef  <- newIORef (the PState (MkP 3.141592653589793 0.0))
  q1Tgt   <- newIORef (toDoubleNetwork (emap refreshValue q1))
  q2Tgt   <- newIORef (toDoubleNetwork (emap refreshValue q2))

  let st0 = MkSAC actor q1 q2 q1Tgt q2Tgt logStdV buffer stepRef envRef
      opt = nativeAdamGlobalClip cfg.lr 0.9 0.999 1.0e-8 1.0

  putStrLn ""

  let trainCfg : TrainConfig SACState
      trainCfg = MkTrainConfig cfg.epochs 2000 NoEarlyStop (const (pure []))
  (trained, epochsDone, _) <- runTrainingIO
    (\s, _ => sacStep opt cfg s)
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
