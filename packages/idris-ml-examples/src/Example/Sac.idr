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
-- Architecture (aligned with `torch_ref/models/sac.py`):
--   Actor : Linear(3,64) → ReLU → Linear(64,64) → ReLU → Linear(64,1)   = mean
--   Q1/Q2 : Linear(4,64) → ReLU → Linear(64,64) → ReLU → Linear(64,1)   = value
--           (input is obs ++ action)
--   log_std : standalone learnable Variable (scope "actor_log_std")
--   Target Q nets: same architecture as Q1/Q2, registered under
--                  "q1tgt_" / "q2tgt_" scope. No optimizer owns them;
--                  they move via polyak soft update.
----------------------------------------------------------------------

-- --- Inline autoName-with-scope (single-file invocation quirk — see A2C/PPO).

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

-- Inline FFI wrappers (single-file invocation quirk — see above).

%foreign "C:optimizer_create_adam_group,libidrisml"
prim__mkAdamGroupLocal : Double -> Double -> Double -> Double -> String -> AnyPtr

mkAdamGroup : String -> Double -> Double -> NativeOptimizer
mkAdamGroup scope lr clip =
  MkNativeOptimizer (prim__mkAdamGroupLocal lr 0.9 0.999 1.0e-8 scope) (NormClip clip)

%foreign "C:polyak_blend,libidrisml"
prim__polyakLocal : Double -> String -> String -> Int

polyakBlend : Double -> String -> String -> IO Int
polyakBlend tau on tg = pure (prim__polyakLocal tau on tg)


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

-- log_std is scoped under "actor_" so it's updated by the actor optimizer.
mkLogStd : Variable CPU
mkLogStd = param "actor_log_std" 0.0


-- --- Observation helpers --------------------------------------------

observeVec : PState -> Vect ObsDim Double
observeVec s = pObserve s

obsTensor : Vect ObsDim Double -> Vector ObsDim Double
obsTensor v = VTensor (map STensor v)

qInput : Vect ObsDim Double -> Double -> Vect QInputDim Double
qInput obs a = obs ++ [a]

qInputTensor : Vect QInputDim Double -> Vector QInputDim Double
qInputTensor v = VTensor (map STensor v)


-- --- Gaussian / squash helpers --------------------------------------

logTwoPiHalf : Double
logTwoPiHalf = 0.5 * Prelude.log (2.0 * 3.141592653589793)

-- Tanh squash correction for log-prob, as a plain Double.
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

-- Evaluate a Q-net at (obs, action), return the scalar as a plain Double.
-- Used for values fed into Q-loss target (no gradient needed) and for
-- min(Q1, Q2) comparisons in actor sampling.
qValue : QNet -> Vect ObsDim Double -> Double -> Double
qValue q obs action =
  let outT = snd (forwardVarTensor q (bulkToTensor (qInputTensor (qInput obs action))))
  in prim__item1d outT 0


-- Sample a squashed Gaussian action. Pure-Double outputs for rollout use.
sampleActionIO : ActorNet -> Variable CPU -> Vect ObsDim Double ->
                 IO (Double, Double)
sampleActionIO actor logStdV obs = do
  let mean   = actorMean actor obs
      logStd = (refreshValue logStdV).value
      std    = Prelude.exp logStd
  eps <- normalSample
  let u        = mean + std * eps
      action   = Math.tanh u * MaxAction
      lp_u     = -0.5 * ((u - mean) / std) * ((u - mean) / std) - logStd - logTwoPiHalf
      lp       = lp_u - squashCorrection u
  pure (action, lp)


-- --- SAC state -------------------------------------------------------

record SACState where
  constructor MkSAC
  actor   : ActorNet
  q1      : QNet
  q2      : QNet
  q1Tgt   : QNet              -- Variable net, scope "q1tgt_", not owned by any optimizer
  q2Tgt   : QNet              -- Variable net, scope "q2tgt_"
  logStdV : Variable CPU
  buffer  : ReplayBuffer ObsDim ActDim
  stepRef : IORef Nat
  envRef  : IORef PState


record Config where
  constructor MkConfig
  lr           : Double
  epochs       : Nat          -- env interactions
  gamma        : Double
  alpha        : Double
  bufferCap    : Nat
  batchSize    : Nat
  warmupSteps  : Nat
  tau          : Double       -- Polyak soft-target coefficient
  clipNorm     : Double
  seed         : Bits64

||| Defaults aligned with `torch_ref/models/sac.py`.
defaultConfig : Config
defaultConfig = MkConfig 3.0e-4 30000 0.99 0.2 100000 64 1000 0.005 1.0 42

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
        ]


-- --- Q-network loss (batched) ---------------------------------------

-- Compute the per-transition target value as a plain Double. No
-- gradient flows from this side; q1Tgt/q2Tgt forwards happen via
-- `qValue` on a fresh Variable graph that we discard.
computeTargetVal : QNet -> QNet -> ActorNet -> Variable CPU ->
                   Double -> Double -> Transition ObsDim ActDim -> IO Double
computeTargetVal q1Tgt q2Tgt actor logStdV gamma alpha t = do
  nextPair <- sampleActionIO actor logStdV t.nextObs
  let nextAction = fst nextPair
      nextLogP   = snd nextPair
      q1NextD    = qValue q1Tgt t.nextObs nextAction
      q2NextD    = qValue q2Tgt t.nextObs nextAction
      minQNextD  = if q1NextD <= q2NextD then q1NextD else q2NextD
      doneMask   = if t.done then 0.0 else 1.0
  pure (t.reward + gamma * doneMask * (minQNextD - alpha * nextLogP))

-- Batched Q-loss: MSE(Q(s,a) - target) summed over B transitions.
-- Stack qInputs ([obs ++ action]) into [B, QInputDim], do one batched
-- Q-online forward, then assemble per-sample losses as Variables.
qLossBatch : (n : Nat) -> QNet -> QNet -> QNet -> ActorNet -> Variable CPU ->
             Double -> Double -> Vect n (Transition ObsDim ActDim) ->
             IO (Variable CPU)
qLossBatch n qOnline q1Tgt q2Tgt actor logStdV gamma alpha batch = do
  targetVals <- traverse (computeTargetVal q1Tgt q2Tgt actor logStdV gamma alpha) batch
  let qInputs : Vect n (Vector QInputDim Double)
      qInputs   = map (\t => qInputTensor (qInput t.obs (oneAct t.action))) batch
      qInputBT  = bulkToTensor2d qInputs
      qOutB     = snd (forwardVarTensorBatch qOnline n qInputBT)  -- [B, 1]
      losses    = perSampleLosses qOutB targetVals 0
      n_d       = the Double (cast (natToInteger n))
      sumV      = foldl (+) (the (Variable CPU) (fromDouble 0.0)) losses
      nV        = the (Variable CPU) (fromDouble n_d)
  pure (sumV / nV)
  where
    oneAct : Vect ActDim Double -> Double
    oneAct [a] = a
    perSampleLosses : (qOutB : AnyPtr) -> Vect k Double -> Int -> List (Variable CPU)
    perSampleLosses _ [] _ = []
    perSampleLosses qOutB (tv :: rest) k =
      let qRow = prim__select qOutB 0 k
          qScalar = prim__item1d qRow 0
          qV : Variable CPU
          qV = Var (prim__select qRow 0 0) Nothing qScalar
          targetC : Variable CPU
          targetC = fromDouble tv
          diff = qV - targetC
      in (diff * diff) :: perSampleLosses qOutB rest (k + 1)


-- --- Actor loss with reparameterization -----------------------------

-- The key change from the earlier (log-prob-only) version: we build the
-- reparameterized action AS A GRAD-TRACKED TENSOR using Variable-level
-- operations on the actor's mean output and logStdV, concatenate with
-- the obs tensor, forward Q1 / Q2 through it, and let gradient flow
-- all the way back to the actor via the action input. The Q-net params
-- also get gradients (they're in the graph) but actor_opt is
-- prefix-scoped so it only applies actor grads; the leaked Q grads
-- are zeroed next iteration by the respective Q optimizers.
actorLoss : ActorNet -> QNet -> QNet -> Variable CPU -> Double ->
            Vect ObsDim Double -> IO (Variable CPU)
actorLoss actor q1 q2 logStdV alpha obs = do
  let stateT  = bulkToTensor (obsTensor obs)
      meanOut = snd (forwardVarTensor actor stateT)    -- grad-tracked [1]
      meanVal = prim__item1d meanOut 0
      logStd  = (refreshValue logStdV).value
      stdVal  = Prelude.exp logStd

  eps <- normalSample
  let -- u = mean + std*eps as a grad-tracked [1] tensor.
      -- `prim__addScalar` preserves autograd; the `std*eps` product is a
      -- constant Double.
      uT          = prim__addScalar meanOut (stdVal * eps)
      -- Grad-tracked tanh, then scale to [-MaxAction, MaxAction].
      aSquashedT  = prim__tanh uT
      aReparamT   = prim__mulScalar aSquashedT MaxAction

      -- Concatenate obs (constant) with reparam action (grad-tracked) → [4].
      obsT        = bulkToTensor (obsTensor obs)
      qInputT     = prim__cat2 obsT aReparamT

      -- Q1 / Q2 forward on the grad-tracked input → grad-tracked scalars.
      q1OutT  = snd (forwardVarTensor q1 qInputT)
      q1Val   = prim__item1d q1OutT 0
      q1V     : Variable CPU
      q1V     = Var (prim__select q1OutT 0 0) Nothing q1Val

      q2OutT  = snd (forwardVarTensor q2 qInputT)
      q2Val   = prim__item1d q2OutT 0
      q2V     : Variable CPU
      q2V     = Var (prim__select q2OutT 0 0) Nothing q2Val

      -- Pick the smaller as a Variable (cached values drive the branch).
      minQV   = if q1Val <= q2Val then q1V else q2V

      -- Grad-tracked log-prob via Variable arithmetic, matching the
      -- reparameterized sample. logπ(a) = logπ_u(u) − correction(u).
      actVal  = Math.tanh (meanVal + stdVal * eps) * MaxAction
      uVal    = meanVal + stdVal * eps
      actC    : Variable CPU
      actC    = fromDouble actVal
      -- u-Variable: reconstruct u as a Variable from meanOut + stdVal*eps constant.
      uV      : Variable CPU
      uV      = Var (prim__select uT 0 0) Nothing uVal

      halfC   : Variable CPU
      halfC   = fromDouble 0.5
      twoC    : Variable CPU
      twoC    = fromDouble 2.0
      zeroC   : Variable CPU
      zeroC   = fromDouble 0.0

      meanV   : Variable CPU
      meanV   = Var (prim__select meanOut 0 0) Nothing meanVal
      diffM   = uV - meanV
      negTwoLs = zeroC - twoC * logStdV
      varInv  = exp negTwoLs
      quad    = halfC * diffM * diffM * varInv
      cC      : Variable CPU
      cC      = fromDouble logTwoPiHalf
      lpU     = (zeroC - quad) - logStdV - cC
      corrC   : Variable CPU
      corrC   = fromDouble (squashCorrection uVal)
      lpV     = lpU - corrC

      alphaC  : Variable CPU
      alphaC  = fromDouble alpha
  pure (alphaC * lpV - minQV)


-- --- Batch aggregation ----------------------------------------------

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


-- --- One batch update: three group-scoped optimizer steps -----------

runBatchUpdate : NativeOptimizer -> NativeOptimizer -> NativeOptimizer ->
                 SACState -> Config -> {n : Nat} ->
                 Vect n (Transition ObsDim ActDim) -> IO ()
runBatchUpdate q1Opt q2Opt actorOpt st cfg {n} batch = do
  -- Q1 loss (owned by q1Opt which scopes to "q1_" — won't touch q2 or actor).
  -- Batched: one Q-online forward across the whole minibatch.
  q1LossV <- qLossBatch n st.q1 st.q1Tgt st.q2Tgt st.actor st.logStdV
                        cfg.gamma cfg.alpha batch
  _ <- pure (nativeTrainStep q1Opt q1LossV)

  -- Q2 loss.
  q2LossV <- qLossBatch n st.q2 st.q1Tgt st.q2Tgt st.actor st.logStdV
                        cfg.gamma cfg.alpha batch
  _ <- pure (nativeTrainStep q2Opt q2LossV)

  -- Actor loss (reparameterized). Per-sample because each transition
  -- needs its own random eps for sampling. Future work: batch by
  -- pre-sampling eps as a [B] vector and stacking grad-tracked qInputs.
  actorLosses <- traverse (actorLoss st.actor st.q1 st.q2 st.logStdV cfg.alpha)
                          (map (\t => t.obs) (toList batch))
  let aLossV = aggregateMean actorLosses
  _ <- pure (nativeTrainStep actorOpt aLossV)
  pure ()


-- --- Main loop -------------------------------------------------------

sacStep : NativeOptimizer -> NativeOptimizer -> NativeOptimizer ->
          Config -> SACState -> IO (SACState, Double)
sacStep q1Opt q2Opt actorOpt cfg st = do
  stepCount <- readIORef st.stepRef
  envState  <- readIORef st.envRef
  let obs = observeVec envState

  action <- if stepCount < cfg.warmupSteps
              then randomRIO (the Double (negate MaxAction), MaxAction)
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

      bufSz <- bufferSize st.buffer
      _ <- if bufSz >= cfg.batchSize && stepCount >= cfg.warmupSteps
             then do
               mBatch <- sampleN cfg.batchSize st.buffer
               case mBatch of
                 Nothing    => pure ()
                 Just batch => do
                   runBatchUpdate q1Opt q2Opt actorOpt st cfg batch
                   -- Polyak soft-update target Q-nets every step.
                   _ <- polyakBlend cfg.tau "q1_" "q1tgt_"
                   _ <- polyakBlend cfg.tau "q2_" "q2tgt_"
                   pure ()
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
           ++ " tau=" ++ show cfg.tau
           ++ " seed=" ++ show cfg.seed

  actor <- mkActor
  q1    <- mkQ "q1_"
  q2    <- mkQ "q2_"
  q1Tgt <- mkQ "q1tgt_"
  q2Tgt <- mkQ "q2tgt_"
  let logStdV = mkLogStd

  -- Hard-copy online → target at init (tau=1).
  _ <- polyakBlend 1.0 "q1_" "q1tgt_"
  _ <- polyakBlend 1.0 "q2_" "q2tgt_"

  buffer  <- mkBuffer {obsDim=ObsDim, actDim=ActDim} cfg.bufferCap
  stepRef <- newIORef (the Nat 0)
  envRef  <- newIORef (the PState (MkP 3.141592653589793 0.0))

  let st0 = MkSAC actor q1 q2 q1Tgt q2Tgt logStdV buffer stepRef envRef
      actorOpt = mkAdamGroup "actor_" cfg.lr cfg.clipNorm
      q1Opt    = mkAdamGroup "q1_"    cfg.lr cfg.clipNorm
      q2Opt    = mkAdamGroup "q2_"    cfg.lr cfg.clipNorm

  putStrLn ""

  let trainCfg : TrainConfig SACState
      trainCfg = MkTrainConfig cfg.epochs 2000 NoEarlyStop (const (pure []))
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
