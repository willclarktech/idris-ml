module Example.Ppo

import Data.List
import Data.Vect
import Data.IORef
import System
import Compat.Random

import Floating
import Gym.ClassicControl.Acrobot
import Gym.Env
import Gym.Vector
import Hpo.LrFinder
import Layer.Activation
import Layer.Core
import Layer.Linear
import Math
import RL.Gae
import Sampler
import Array
import Train
import Util
import Executor
import Tensor
import BuildConfig


----------------------------------------------------------------------
-- Architecture: separate actor and critic MLPs with discrete-action
-- (categorical) policy on Acrobot. Pendulum (continuous-action Gaussian)
-- was the original env but didn't converge at CPU-feasible rollout
-- sizes; Acrobot is the canonical "PPO clipped-surrogate demonstrates"
-- benchmark — discrete actions, sparse reward, longer horizon.
--
-- Actor:  Linear(6→64) → tanh → Linear(64→64) → tanh → Linear(64→3) = action logits
-- Critic: Linear(6→64) → tanh → Linear(64→64) → tanh → Linear(64→1) = value
--
-- Actor and critic params are scoped via constructor-time paramPrefix
-- ("actor_..." / "critic_..."), no autoNameScoped indirection.
--
-- ⚠ MLX `*>` quirk: iterating mini-batches via `traverse_` (which is
-- defined as `foldr ((*>) . f) (pure ())`) crashes on the MLX backend
-- with "invalid memory reference" the second runBatch invocation
-- onward. The Idris-side `*>` for IO unrolls into something that
-- interacts poorly with 's tape-tracked intermediates after a
-- tape_reset. Workaround: iterate via a do-block recursive helper
-- (`runBatches`), which desugars to `>>=` and works cleanly.
----------------------------------------------------------------------

ObsDim : Nat; ObsDim = 6
Hidden : Nat; Hidden = 64
NumActions : Nat; NumActions = 3
EpisodeLen : Nat; EpisodeLen = 500   -- Acrobot defaultTimeLimit

||| Number of parallel envs run per ppoEpoch. Matches Idris-side
||| `Example.A2c.NumEnvs` and PyTorch `torch_ref.models.ppo.NUM_ENVS`.
||| Compile-time because the batched-forward shape `[NumEnvs, ObsDim]` is
||| part of the autograd graph.
NumEnvs : Nat; NumEnvs = 4

||| Per-env rollout length. Total samples per ppoEpoch is
||| `NumEnvs * RolloutLen`. Pre-batched used `RolloutLen=1024` with one
||| env (1024 samples); we keep the same per-update sample budget across
||| 4 parallel envs (256 per env × 4 envs = 1024 total).
RolloutLen : Nat; RolloutLen = 256
BatchSize : Nat; BatchSize = 64

Actor : Type
Actor = Network ObsDim [Hidden, Hidden, Hidden, Hidden] NumActions ExampleExecutor ExampleDType WithGrad

Critic : Type
Critic = Network ObsDim [Hidden, Hidden, Hidden, Hidden] 1 ExampleExecutor ExampleDType WithGrad

mkActor : IO Actor
mkActor = do
  ll1 <- linearLayerAny {i=ObsDim} {o=Hidden}     "actor_ll1"
  ll2 <- linearLayerAny {i=Hidden} {o=Hidden}     "actor_ll2"
  ll3 <- linearLayerAny {i=Hidden} {o=NumActions} "actor_ll3"
  pure (ll1 ~~> tanhLayerAny ~~> ll2 ~~> tanhLayerAny ~~> OutputLayer ll3)

mkCritic : IO Critic
mkCritic = do
  ll1 <- linearLayerAny {i=ObsDim} {o=Hidden} "critic_ll1"
  ll2 <- linearLayerAny {i=Hidden} {o=Hidden} "critic_ll2"
  ll3 <- linearLayerAny {i=Hidden} {o=1}      "critic_ll3"
  pure (ll1 ~~> tanhLayerAny ~~> ll2 ~~> tanhLayerAny ~~> OutputLayer ll3)


----------------------------------------------------------------------
-- Observation helpers
----------------------------------------------------------------------

observeVec : AState -> Vect ObsDim Double
observeVec s = aObserve s

obsTensor : Vect ObsDim Double -> Vector ObsDim Double
obsTensor v = VArray (map SArray v)


----------------------------------------------------------------------
-- Rollout record
----------------------------------------------------------------------

record RollStep where
  constructor MkRS
  obs        : Vect ObsDim Double
  action     : Nat
  oldLogProb : Double
  value      : Double
  reward     : Double
  isDone     : Bool


----------------------------------------------------------------------
-- Categorical policy helpers
----------------------------------------------------------------------

criticValue : Critic -> Vect ObsDim Double -> IO Double
criticValue critic obs = do
  let stateV = the (TVec ObsDim ExampleExecutor ExampleDType WithGrad) (MkTensor (bulkToTensor {ex=ExampleExecutor} {dt=ExampleDType} (obsTensor obs)) Nothing)
  (_, outV) <- forwardVar critic stateV
  pure (primItem1d {ex=ExampleExecutor} outV.tensorPtr 0)

sampleActionIO : Actor -> Critic -> Vect ObsDim Double -> IO (Nat, Double, Double)
sampleActionIO actor critic obs = do
  let stateV  = the (TVec ObsDim ExampleExecutor ExampleDType WithGrad) (MkTensor (bulkToTensor {ex=ExampleExecutor} {dt=ExampleDType} (obsTensor obs)) Nothing)
  (_, logitsV) <- forwardVar actor stateV
  let logPT   = primLogSoftmax {ex=ExampleExecutor} logitsV.tensorPtr 0
      lp0     = primItem1d {ex=ExampleExecutor} logPT 0
      lp1     = primItem1d {ex=ExampleExecutor} logPT 1
      lp2     = primItem1d {ex=ExampleExecutor} logPT 2
  v <- criticValue critic obs
  u <- randomRIO (the Double 0.0, 1.0)
  let a = categoricalSample [Prelude.exp lp0, Prelude.exp lp1, Prelude.exp lp2] u
      lp = case a of
             0 => lp0
             1 => lp1
             _ => lp2
  pure (a, lp, v)


----------------------------------------------------------------------
-- Rollout
----------------------------------------------------------------------

rollout : Actor -> Critic -> AState -> Nat -> Nat ->
          IO (List RollStep, AState, List Double)
rollout _ _ st _ Z = pure ([], st, [])
rollout actor critic st stepsLeft (S k) = do
  let obs = observeVec st
  -- Per-step no-grad bracket: free this step's forward intermediates
  -- immediately. The whole rollout is RolloutLen (e.g. 1024) steps; a
  -- single outer withNoGrad would accumulate all of them past the
  -- paravirt-Metal ceiling. The step result is plain data (Nat/Double).
  triple <- withNoGrad {ex=ExampleExecutor} (sampleActionIO actor critic obs)
  let a  = fst triple
      lp = fst (snd triple)
      v  = snd (snd triple)
  case aStep st a of
    (r, st', outcome, _) => do
      let natTerm = case outcome of
                      Terminated => True
                      _          => False
          truncate = stepsLeft == 1
          isDone = natTerm || truncate
          stepRec = MkRS obs a lp v r isDone
          nextSt = if isDone then MkA 0.0 0.0 0.0 0.0 else st'
          nextStepsLeft = if isDone then EpisodeLen else stepsLeft `minus` 1
      recur <- rollout actor critic nextSt nextStepsLeft k
      let (stepsRest, finalSt, retsRest) = recur
          retsCarry = if isDone then 0.0 :: retsRest else retsRest
      pure (stepRec :: stepsRest, finalSt, retsCarry)


----------------------------------------------------------------------
-- Batched rollout (NumEnvs parallel envs, one batched forward per step)
----------------------------------------------------------------------

-- Sample one action per env from a [n, NumActions] batched log-prob
-- tensor. Records the log-prob of the chosen action per env (PPO needs
-- this for the importance ratio later).
sampleActionFromBatch : {n : Nat} -> Tensor [n, NumActions] ExampleExecutor ExampleDType g ->
                        Vect n AState -> IO (Vect n Nat, Vect n Double)
sampleActionFromBatch logProbsB envs = go 0 envs
  where
    go : Int -> Vect k AState -> IO (Vect k Nat, Vect k Double)
    go _ [] = pure ([], [])
    go i (_ :: rest) = do
      let lp0 = primItem2d {ex=ExampleExecutor} logProbsB.tensorPtr i 0
          lp1 = primItem2d {ex=ExampleExecutor} logProbsB.tensorPtr i 1
          lp2 = primItem2d {ex=ExampleExecutor} logProbsB.tensorPtr i 2
      u <- randomRIO (the Double 0.0, 1.0)
      let a = categoricalSample [Prelude.exp lp0, Prelude.exp lp1, Prelude.exp lp2] u
          lp = case a of
                 0 => lp0
                 1 => lp1
                 _ => lp2
      (acts, lpVals) <- go (i + 1) rest
      pure (a :: acts, lp :: lpVals)

-- Step every env with its action; auto-reset envs that terminate OR hit
-- the EpisodeLen truncation cap. Returns next states, rewards, done
-- flags, and updated per-env stepsLeft counters.
stepAllAutoResetTrunc : Vect n AState -> Vect n Nat -> Vect n Nat ->
                        (Vect n AState, Vect n Double, Vect n Bool, Vect n Nat)
stepAllAutoResetTrunc [] [] [] = ([], [], [], [])
stepAllAutoResetTrunc (s :: ss) (a :: as) (sl :: sls) =
  case aStep s a of
    (r, s', outcome, _) =>
      let natTerm = case outcome of
                      Terminated => True
                      _          => False
          truncate = sl == 1
          isDone = natTerm || truncate
          nextS  = if isDone then MkA 0.0 0.0 0.0 0.0 else s'
          nextSl = if isDone then EpisodeLen else sl `minus` 1
      in case stepAllAutoResetTrunc ss as sls of
           (rest, rs, ds, sls') => (nextS :: rest, r :: rs, isDone :: ds, nextSl :: sls')

-- Build per-env RollStep records.
mkRollSteps : Vect n (Vect ObsDim Double) -> Vect n Nat -> Vect n Double ->
              Vect n Double -> Vect n Double -> Vect n Bool ->
              Vect n RollStep
mkRollSteps [] [] [] [] [] [] = []
mkRollSteps (o :: os) (a :: as) (lp :: lps) (v :: vs) (r :: rs) (d :: ds) =
  MkRS o a lp v r d :: mkRollSteps os as lps vs rs ds

||| Batched per-env rollout. Each env steps RolloutLen times in lockstep;
||| one batched (actor, critic) forward per timestep amortises Idris-side
||| per-op overhead across NumEnvs samples. Threads per-env stepsLeft for
||| truncation. Done envs auto-reset to `MkA 0 0 0 0` with stepsLeft
||| restored to EpisodeLen.
rolloutBatched : {n : Nat} -> Actor -> Critic ->
                 VecEnv n AState -> Vect n Nat -> Nat ->
                 IO (Vect n (List RollStep), VecEnv n AState, Vect n Nat)
rolloutBatched actor critic v0 sl0 rolloutLen = do
  (envs', sls', stepLists) <- go rolloutLen v0.envs sl0 (replicate n [])
  pure (map reverse stepLists, MkVecEnv envs', sls')
  where
    -- Helper: map with index over a Vect.
    mapIdx : (Nat -> a -> b) -> Vect k a -> Vect k b
    mapIdx _ [] = []
    mapIdx f (x :: xs) = f 0 x :: mapIdx (\i, v => f (S i) v) xs

    go : Nat -> Vect n AState -> Vect n Nat -> Vect n (List RollStep) ->
         IO (Vect n AState, Vect n Nat, Vect n (List RollStep))
    go Z envs sls accs = pure (envs, sls, accs)
    go (S k) envs sls accs = withNoGrad {ex=ExampleExecutor} $ do
      -- Per-step no-grad bracket (matches sequential rollout's hygiene):
      -- free this step's forward intermediates immediately. With
      -- RolloutLen=256 per env and N=4 envs, accumulating all forward
      -- handles in one outer bracket would push past the paravirt-Metal
      -- ceiling.
      let obsRows : Vect n (Vector ObsDim Double)
          obsRows = map (\s => obsTensor (observeVec s)) envs
          batchPtr = bulkToTensor2d {ex=ExampleExecutor} {dt=ExampleDType} obsRows
          stateV : Tensor [n, ObsDim] ExampleExecutor ExampleDType WithGrad
          stateV = MkTensor batchPtr Nothing
      (_, logitsV) <- forwardVarBatch actor stateV
      let logProbsV = the (Tensor [n, NumActions] ExampleExecutor ExampleDType WithGrad)
                        (MkTensor (primLogSoftmax2d {ex=ExampleExecutor} logitsV.tensorPtr) Nothing)
      (_, valuesV) <- forwardVarBatch critic stateV
      (acts, lps) <- sampleActionFromBatch logProbsV envs
      let valueRows : Vect n Double
          valueRows = mapIdx (\i, _ => primItem2d {ex=ExampleExecutor} valuesV.tensorPtr (cast i) 0) envs
          obsVects : Vect n (Vect ObsDim Double)
          obsVects = map observeVec envs
      case stepAllAutoResetTrunc envs acts sls of
        (envs', rewards, dones, sls') =>
          let newSteps = mkRollSteps obsVects acts lps valueRows rewards dones
              accs' = zipWith (\acc, s => s :: acc) accs newSteps
          in go k envs' sls' accs'


----------------------------------------------------------------------
-- GAE + advantage normalization
----------------------------------------------------------------------

bootstrapV : Critic -> Vect ObsDim Double -> IO Double
bootstrapV critic obs = criticValue critic obs

computeBootstrap : Critic -> List RollStep -> AState -> IO Double
computeBootstrap _ [] _ = pure 0.0
computeBootstrap critic steps finalSt =
  case last' steps of
    Nothing => pure 0.0
    Just ls => if ls.isDone then pure 0.0 else bootstrapV critic (observeVec finalSt)

stepTriple : RollStep -> (Double, Double, Bool)
stepTriple s = (s.reward, s.value, s.isDone)

flattenTriple : (RollStep, (Double, Double)) -> (RollStep, Double, Double)
flattenTriple (sRec, (a, r)) = (sRec, a, r)

tripleAdv : (RollStep, Double, Double) -> Double
tripleAdv (_, a, _) = a

normAdvs : List (RollStep, Double, Double) -> List (RollStep, Double, Double)
normAdvs triples =
  let advs   = map tripleAdv triples
      nN     = the Double (cast (natToInteger (length advs)))
      mu     = if nN > 0.0 then sum advs / nN else 0.0
      sqDevs = map (\a => (a - mu) * (a - mu)) advs
      vr     = if nN > 0.0 then sum sqDevs / nN else 1.0
      sd     = sqrt (vr + 1.0e-8)
      renorm : (RollStep, Double, Double) -> (RollStep, Double, Double)
      renorm (s, a, r) = (s, (a - mu) / sd, r)
  in map renorm triples


----------------------------------------------------------------------
-- Per-step PPO loss (categorical policy,  typed-surface)
----------------------------------------------------------------------

clipScalar : Double -> Double -> Double -> Double
clipScalar lo hi x = if x < lo then lo else if x > hi then hi else x

perStepLoss : {n : Nat} -> (logitsB : Tensor [n, NumActions] ExampleExecutor ExampleDType WithGrad) ->
              (valueB : Tensor [n, 1] ExampleExecutor ExampleDType WithGrad) -> (rowIdx : Int) ->
              Double -> Double -> Double ->
              (RollStep, Double, Double) -> IO (Tensor [] ExampleExecutor ExampleDType WithGrad)
perStepLoss logitsB valueB rowIdx clipEps entropyCoef valueCoef (step, adv, retT) = do
  logitsRow <- trowSelect logitsB rowIdx
  let logPT = the (Tensor [NumActions] ExampleExecutor ExampleDType WithGrad)
                  (MkTensor (primLogSoftmax {ex=ExampleExecutor} logitsRow.tensorPtr 0) Nothing)
      aIdx : Int
      aIdx = cast {to=Int} (cast {to=Integer} step.action)
  lpNew <- telemSelect logPT aIdx
  let lpVal = primItem1d {ex=ExampleExecutor} logPT.tensorPtr aIdx
  valueRow <- trowSelect valueB rowIdx
  valueV   <- telemSelect valueRow 0
  oldLPT   <- tconstScalar step.oldLogProb
  diffLP   <- tsub lpNew oldLPT
  let ratioVal = Prelude.exp (lpVal - step.oldLogProb)
  ratioV   <- texp diffLP
  surr1    <- tmulScalar ratioV adv
  let surr1Val = ratioVal * adv
      clipped = clipScalar (1.0 - clipEps) (1.0 + clipEps) ratioVal
      surr2Val = clipped * adv
  policyT <- if surr1Val <= surr2Val
               then tneg surr1
               else tconstScalar (negate surr2Val)
  retC      <- tconstScalar retT
  diffV     <- tsub valueV retC
  diffVsq   <- tmul diffV diffV
  valueTerm <- tmulScalar diffVsq (0.5 * valueCoef)
  lp0V <- telemSelect logPT 0
  lp1V <- telemSelect logPT 1
  lp2V <- telemSelect logPT 2
  p0V  <- texp lp0V
  p1V  <- texp lp1V
  p2V  <- texp lp2V
  let negEntV = the (Tensor [] ExampleExecutor ExampleDType WithGrad)
                    (MkTensor (primAdd {ex=ExampleExecutor}
                              (primAdd {ex=ExampleExecutor} (primMul {ex=ExampleExecutor} p0V.tensorPtr lp0V.tensorPtr)
                                         (primMul {ex=ExampleExecutor} p1V.tensorPtr lp1V.tensorPtr))
                              (primMul {ex=ExampleExecutor} p2V.tensorPtr lp2V.tensorPtr))
                            Nothing)
  entTerm <- tmulScalar negEntV entropyCoef
  pure (MkTensor (primAdd {ex=ExampleExecutor} (primAdd {ex=ExampleExecutor} policyT.tensorPtr valueTerm.tensorPtr)
                       entTerm.tensorPtr) Nothing)


meanScalarLoss : (n : Nat) -> List (Tensor [] ExampleExecutor ExampleDType WithGrad) -> IO (Tensor [] ExampleExecutor ExampleDType WithGrad)
meanScalarLoss n losses = do
  zero <- tconstScalar 0.0
  let summed = foldl (\a, b => MkTensor (primAdd {ex=ExampleExecutor} a.tensorPtr b.tensorPtr) Nothing) zero losses
  tmulScalar summed (1.0 / cast n)


----------------------------------------------------------------------
-- Mini-batch shuffling
----------------------------------------------------------------------

shuffleIO : List a -> IO (List a)
shuffleIO xs = do
  tags <- traverse (\_ => randomRIO (the Double 0.0, 1.0)) xs
  pure (map snd (sortBy (\a, b => compare (fst a) (fst b)) (zip tags xs)))

chunksOf : Nat -> List a -> List (List a)
chunksOf _ [] = []
chunksOf Z xs = [xs]
chunksOf n xs = take n xs :: chunksOf n (drop n xs)


----------------------------------------------------------------------
-- Config + update loop
----------------------------------------------------------------------

record Config where
  constructor MkConfig
  lr          : Double
  epochs      : Nat
  gamma       : Double
  lam         : Double
  clipEps     : Double
  kEpochs     : Nat
  entropyCoef : Double
  valueCoef   : Double
  seed        : Bits64
  lrFind      : Bool

defaultConfig : Config
defaultConfig = MkConfig 3.0e-4 100 0.99 0.95 0.2 10 0.01 0.5 42 False

specs : List (ArgSpec Config)
specs = [ Arg "--lr" (\v, c => { lr := cast v } c)
        , Arg "--epochs" (\v, c => { epochs := castNat v } c)
        , Arg "--gamma" (\v, c => { gamma := cast v } c)
        , Arg "--lambda" (\v, c => { lam := cast v } c)
        , Arg "--clip-eps" (\v, c => { clipEps := cast v } c)
        , Arg "--k-epochs" (\v, c => { kEpochs := castNat v } c)
        , Arg "--entropy" (\v, c => { entropyCoef := cast v } c)
        , Arg "--value-coef" (\v, c => { valueCoef := cast v } c)
        , Arg "--seed" (\v, c => { seed := castBits64 v } c)
        , Arg "--lr-find" (\v, c => { lrFind := (v == "1" || v == "true") } c)
        ]

record PPOState where
  constructor MkPPO
  actor    : Actor
  critic   : Critic
  envRef   : IORef (VecEnv NumEnvs AState)
  stepsRef : IORef (Vect NumEnvs Nat)


prepareRollout : Critic -> Config -> List RollStep -> AState ->
                 IO (List (RollStep, Double, Double))
prepareRollout critic cfg steps finalSt = do
  bootstrap <- computeBootstrap critic steps finalSt
  let triples   = map stepTriple steps
      gaeOut    = gae cfg.gamma cfg.lam bootstrap triples
      merged    = map flattenTriple (zip steps gaeOut)
  pure (normAdvs merged)


-- Per-env bootstrap: critic value at each env's final state, zeroed for
-- envs whose last step terminated (matches sequential `computeBootstrap`).
computeBootstrapsBatched : Critic -> Vect n (List RollStep) -> VecEnv n AState ->
                           IO (Vect n Double)
computeBootstrapsBatched critic stepLists v = batchOver stepLists v.envs
  where
    batchOver : Vect k (List RollStep) -> Vect k AState -> IO (Vect k Double)
    batchOver [] [] = pure []
    batchOver (steps :: rest) (s :: ss) = do
      b <- computeBootstrap critic steps s
      bs <- batchOver rest ss
      pure (b :: bs)


-- Batched prepareRollout: per-env GAE → concat → normalize advantages
-- across the whole flat batch (matches PyTorch SyncVectorEnv updates,
-- where advantages are normalized over T*N samples).
prepareRolloutBatched : Critic -> Config -> Vect n (List RollStep) ->
                        VecEnv n AState -> IO (List (RollStep, Double, Double))
prepareRolloutBatched critic cfg stepLists finalEnvs = do
  bootstraps <- computeBootstrapsBatched critic stepLists finalEnvs
  let mergedPerEnv : Vect n (List (RollStep, Double, Double))
      mergedPerEnv = zipWith
        (\steps, boot =>
          let triples = map stepTriple steps
              gaeOut = gae cfg.gamma cfg.lam boot triples
          in map flattenTriple (zip steps gaeOut))
        stepLists bootstraps
      flatMerged = concat (toList mergedPerEnv)
  pure (normAdvs flatMerged)


-- Stack mini-batch obs into [B, ObsDim], do one batched actor + critic
-- forward each, then build per-sample loss expressions by indexing into
-- the [B, NumActions] / [B, 1] tensors.
runBatch : NativeOptimizer ExampleExecutor -> Actor -> Critic -> Config ->
           List (RollStep, Double, Double) -> IO ()
runBatch opt actor critic cfg batch = withGenFree {ex=ExampleExecutor} $ do
  -- Per-minibatch generation bracket: free this update's grad
  -- intermediates immediately. PPO runs K (e.g. 10) epochs × minibatches
  -- of batched forward+loss over the whole rollout; without per-step
  -- freeing the within-epoch handle count bursts past the paravirt-Metal
  -- ceiling. Params update in-place via the registry (rc>1, spared).
  let batchVec = Data.Vect.fromList batch
      n = length batch
      obsBatch = the (Vect (length batch) (Vector ObsDim Double))
                     (map (\(s, _, _) => obsTensor s.obs) batchVec)
      stackedT = bulkToTensor2d {ex=ExampleExecutor} {dt=ExampleDType} obsBatch
      stackedV = the (Tensor [n, ObsDim] ExampleExecutor ExampleDType WithGrad) (MkTensor stackedT Nothing)
  (_, logitsB) <- forwardVarBatch actor stackedV
  (_, valueB)  <- forwardVarBatch critic stackedV
  losses <- enumeratedLosses logitsB valueB batchVec 0
  loss <- meanScalarLoss n losses
  _ <- nativeTrainStep opt loss
  pure ()
  where
    enumeratedLosses : {n : Nat} -> Tensor [n, NumActions] ExampleExecutor ExampleDType WithGrad ->
                       Tensor [n, 1] ExampleExecutor ExampleDType WithGrad ->
                       Vect k (RollStep, Double, Double) -> Int ->
                       IO (List (Tensor [] ExampleExecutor ExampleDType WithGrad))
    enumeratedLosses _ _ [] _ = pure []
    enumeratedLosses lB vB (t :: rest) k = do
      l  <- perStepLoss lB vB k cfg.clipEps cfg.entropyCoef cfg.valueCoef t
      ls <- enumeratedLosses lB vB rest (k + 1)
      pure (l :: ls)


-- Iterate runBatch over a list of mini-batches via do-block recursion.
-- ⚠ Do NOT use `traverse_` here: its `foldr ((*>) . f) (pure ())`
-- desugaring crashes  PPO on MLX with "invalid memory reference"
-- on the second runBatch onwards. `>>=`-style sequencing (do-block)
-- works fine. See the module-header comment.
runBatches : NativeOptimizer ExampleExecutor -> Actor -> Critic -> Config ->
             List (List (RollStep, Double, Double)) -> IO ()
runBatches _ _ _ _ [] = pure ()
runBatches opt actor critic cfg (b :: rest) = do
  runBatch opt actor critic cfg b
  runBatches opt actor critic cfg rest


kEpochUpdate : NativeOptimizer ExampleExecutor -> Actor -> Critic -> Config ->
               List (RollStep, Double, Double) -> Nat -> IO ()
kEpochUpdate _ _ _ _ _ Z = pure ()
kEpochUpdate opt actor critic cfg prepped (S k) = do
  shuffled <- shuffleIO prepped
  let batches = chunksOf BatchSize shuffled
  runBatches opt actor critic cfg batches
  kEpochUpdate opt actor critic cfg prepped k


ppoEpoch : NativeOptimizer ExampleExecutor -> Config -> PPOState -> IO (PPOState, Double)
ppoEpoch opt cfg st = do
  startEnvs <- readIORef st.envRef
  startSls  <- readIORef st.stepsRef
  -- rolloutBatched brackets each timestep's batched forward in
  -- withNoGrad itself; gradients come from kEpochUpdate's separate
  -- batched forward (PPO recomputes log-probs over the rollout for each
  -- inner epoch). No outer withNoGrad here — would pin handles for the
  -- full RolloutLen and pressure paravirt-Metal.
  rolled <- rolloutBatched st.actor st.critic startEnvs startSls RolloutLen
  let stepLists = fst rolled
      finalEnvs = fst (snd rolled)
      finalSls  = snd (snd rolled)
  writeIORef st.envRef finalEnvs
  writeIORef st.stepsRef finalSls

  prepped <- withNoGrad {ex=ExampleExecutor} (prepareRolloutBatched st.critic cfg stepLists finalEnvs)
  kEpochUpdate opt st.actor st.critic cfg prepped cfg.kEpochs

  -- Episode returns: walk each env's step list separately, summing
  -- per-completed-episode returns. Matches PyTorch SyncVectorEnv where
  -- any env's done counts once.
  let allReturns = concat (toList (map computeEpisodeReturns stepLists))
      nEp = length allReturns
      sumEp = sum allReturns
      sumRew = sum (map (\steps => sum (map (\s => s.reward) steps)) (toList stepLists))
      avgEp = if nEp > 0
              then sumEp / cast (natToInteger nEp)
              else sumRew / cast (natToInteger NumEnvs)
  pure (st, negate avgEp)
  where
    computeEpisodeReturns : List RollStep -> List Double
    computeEpisodeReturns = go 0.0 []
      where
        go : Double -> List Double -> List RollStep -> List Double
        go _ acc [] = reverse acc
        go run acc (s :: rest) =
          if s.isDone
            then go 0.0 ((run + s.reward) :: acc) rest
            else go (run + s.reward) acc rest


----------------------------------------------------------------------
-- Greedy evaluation
----------------------------------------------------------------------

greedyAct : Actor -> Vect ObsDim Double -> IO Nat
greedyAct actor obs = do
  let stateV = the (TVec ObsDim ExampleExecutor ExampleDType WithGrad) (MkTensor (bulkToTensor {ex=ExampleExecutor} {dt=ExampleDType} (obsTensor obs)) Nothing)
  (_, logits) <- forwardVar actor stateV
  let l0 = primItem1d {ex=ExampleExecutor} logits.tensorPtr 0
      l1 = primItem1d {ex=ExampleExecutor} logits.tensorPtr 1
      l2 = primItem1d {ex=ExampleExecutor} logits.tensorPtr 2
  pure (if l0 >= l1 && l0 >= l2 then 0
        else if l1 >= l2 then 1
        else 2)

evalEp : Actor -> AState -> Nat -> Double -> IO Double
evalEp _ _ Z acc = pure acc
evalEp actor st (S k) acc = do
  a <- greedyAct actor (observeVec st)
  case aStep st a of
    (r, st', outcome, _) =>
      case outcome of
        Terminated => pure (acc + r)
        _          => evalEp actor st' k (acc + r)

evalN : Actor -> Nat -> Double -> IO Double
evalN _ Z acc = pure acc
evalN actor (S k) acc = do
  v <- evalEp actor (MkA 0.0 0.0 0.0 0.0) EpisodeLen 0.0
  evalN actor k (acc + v)


----------------------------------------------------------------------
-- Main
----------------------------------------------------------------------

main : IO ()
main = do
  requireMachine {m = ChosenMachine}
  args <- getArgs
  let cfg = parseArgs defaultConfig specs (drop 1 args)
  srand cfg.seed

  putStrLn "=== PPO on Acrobot (separate actor + critic, categorical policy) ==="
  putStrLn $ "Config: lr=" ++ show cfg.lr
           ++ " epochs=" ++ show cfg.epochs
           ++ " rollout=" ++ show RolloutLen
           ++ " batch=" ++ show BatchSize
           ++ " gamma=" ++ show cfg.gamma
           ++ " lambda=" ++ show cfg.lam
           ++ " clip=" ++ show cfg.clipEps
           ++ " K=" ++ show cfg.kEpochs
           ++ " entropy=" ++ show cfg.entropyCoef
           ++ " seed=" ++ show cfg.seed

  actor  <- mkActor
  critic <- mkCritic
  resetSeedI <- randomInt32
  let initEnvs : VecEnv NumEnvs AState
      initEnvs = fst (resetAll {state=AState} {action=Nat} {obs=Vect 6 Double}
                              (cast resetSeedI))
  envRef   <- newIORef initEnvs
  stepsRef <- newIORef (the (Vect NumEnvs Nat) (replicate NumEnvs EpisodeLen))
  let st0 = MkPPO actor critic envRef stepsRef
      opt = nativeAdamGlobalClip cfg.lr 0.9 0.999 1.0e-8 0.5

  putStrLn ""

  when cfg.lrFind $ do
    let lrCfg : LrFindConfig
        lrCfg = { numIters := 30 } defaultLrFindConfig
    _ <- lrFind lrCfg
      (\s, _ => ppoEpoch opt cfg s)
      (pure ()) opt st0
    putStrLn ""
    putStrLn "Done — re-run without --lr-find at the recommended LR."
    exitSuccess

  metrics <- newRLMetricsState 50
  let trainCfg : TrainConfig PPOState
      trainCfg = mkTrainConfig cfg.epochs 10 NoEarlyStop
                   (\_ => readRLMetrics "recent_50" metrics) (\_ => pure ())
  (trained, epochsDone, _) <- runTrainingIO {ex=ExampleExecutor}
    (\s, _ => do
       (s', loss) <- ppoEpoch opt cfg s
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
