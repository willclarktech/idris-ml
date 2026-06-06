module Example.A2c

import Data.List
import Data.Vect
import Data.IORef
import System
import Compat.Random

import Floating
import Gym.ClassicControl.CartPole
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
-- Architecture: separate actor and critic MLPs (aligned with PyTorch
-- reference `a2c.py`).  layer constructors take a paramPrefix
-- directly, so each net's params land under "actor_..." / "critic_..."
-- without the V1 autoNameScoped indirection.
--   Actor  : 4 -> 64 -> 64 -> 2 (action logits)
--   Critic : 4 -> 64 -> 64 -> 1 (state value)
----------------------------------------------------------------------

ObsDim : Nat; ObsDim = 4
Hidden : Nat; Hidden = 64
NumActions : Nat; NumActions = 2
MaxSteps : Nat; MaxSteps = cartPoleMaxSteps
RolloutLen : Nat; RolloutLen = 20

||| Number of parallel envs run per a2cEpoch. Compile-time because the
||| batched-forward shape `[NumEnvs, ObsDim]` is part of the autograd
||| graph. PyTorch's `gym.vector.SyncVectorEnv` and our `Gym.Vector.VecEnv`
||| share this semantic — N independent envs stepped in lockstep — and we
||| use the same N on both sides to keep the cross-backend reference
||| comparable.
NumEnvs : Nat; NumEnvs = 4

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

observeVec : CPState -> Vect ObsDim Double
observeVec s = cpObserve s

obsTensor : Vect ObsDim Double -> Vector ObsDim Double
obsTensor v = VArray (map SArray v)


----------------------------------------------------------------------
-- Rollout record
----------------------------------------------------------------------

record RollStep where
  constructor MkRS
  obs     : Vect ObsDim Double
  action  : Nat
  reward  : Double
  value   : Double
  isDone  : Bool


----------------------------------------------------------------------
-- Sampling + rollout
----------------------------------------------------------------------

sampleActionIO : Actor -> Critic -> Vect ObsDim Double -> IO (Nat, Double)
sampleActionIO actor critic obs = do
  let stateT  = bulkToTensor {ex=ExampleExecutor} {dt=ExampleDType} (obsTensor obs)
      stateV  = the (TVec ObsDim ExampleExecutor ExampleDType WithGrad) (MkTensor stateT Nothing)
  (_, logitsV) <- forwardVar actor stateV
  let logPT   = primLogSoftmax {ex=ExampleExecutor} logitsV.tensorPtr 0
      lp0     = primItem1d {ex=ExampleExecutor} logPT 0
      lp1     = primItem1d {ex=ExampleExecutor} logPT 1
  (_, valueV) <- forwardVar critic stateV
  let v       = primItem1d {ex=ExampleExecutor} valueV.tensorPtr 0
  u <- randomRIO (the Double 0.0, 1.0)
  let a = categoricalSample [Prelude.exp lp0, Prelude.exp lp1] u
  pure (a, v)

rollout : Actor -> Critic -> CPState -> Nat -> IO (List RollStep, CPState)
rollout _ _ st Z = pure ([], st)
rollout actor critic st (S k) = do
  let obs = observeVec st
  pair <- sampleActionIO actor critic obs
  let a = fst pair
      v = snd pair
  case cpStep st a of
    (r, st', outcome, _) => do
      let isDone = done outcome
          stepRec = MkRS obs a r v isDone
          nextSt = if isDone then MkCP 0 0 0 0 else st'
      recur <- rollout actor critic nextSt k
      pure (stepRec :: fst recur, snd recur)


----------------------------------------------------------------------
-- Batched rollout (NumEnvs parallel envs, one batched forward per step)
----------------------------------------------------------------------

-- Sample one action per env from a [n, NumActions] batched log-prob tensor.
-- Threads the env index through and consumes one randomRIO call per env
-- per timestep (matches the sequential rollout's one-rand-per-step rule).
sampleActionFromBatch : {n : Nat} -> Tensor [n, NumActions] ExampleExecutor ExampleDType g ->
                        Vect n CPState -> IO (Vect n Nat, Vect n Double)
sampleActionFromBatch logProbsB envs = go 0 envs
  where
    go : Int -> Vect k CPState -> IO (Vect k Nat, Vect k Double)
    go _ [] = pure ([], [])
    go i (_ :: rest) = do
      let lp0 = primItem2d {ex=ExampleExecutor} logProbsB.tensorPtr i 0
          lp1 = primItem2d {ex=ExampleExecutor} logProbsB.tensorPtr i 1
      u <- randomRIO (the Double 0.0, 1.0)
      let a = categoricalSample [Prelude.exp lp0, Prelude.exp lp1] u
      (acts, lpVals) <- go (i + 1) rest
      pure (a :: acts, (if a == 0 then lp0 else lp1) :: lpVals)

-- Step every env with its action; auto-reset envs that terminate (matches
-- the sequential `rollout`'s nextSt = if isDone then MkCP 0 0 0 0 else st').
stepAllAutoReset : Vect n CPState -> Vect n Nat ->
                   (Vect n CPState, Vect n Double, Vect n Bool)
stepAllAutoReset [] [] = ([], [], [])
stepAllAutoReset (s :: ss) (a :: as) =
  case cpStep s a of
    (r, s', outcome, _) =>
      let isDone = done outcome
          nextS  = if isDone then MkCP 0 0 0 0 else s'
          (rest, rs, ds) = stepAllAutoReset ss as
      in (nextS :: rest, r :: rs, isDone :: ds)

-- Build per-env RollStep records given the rollout-step values.
mkRollSteps : Vect n (Vect ObsDim Double) -> Vect n Nat -> Vect n Double ->
              Vect n Double -> Vect n Bool -> Vect n RollStep
mkRollSteps [] [] [] [] [] = []
mkRollSteps (o :: os) (a :: as) (r :: rs) (v :: vs) (d :: ds) =
  MkRS o a r v d :: mkRollSteps os as rs vs ds

||| Batched per-env rollout. Each env steps RolloutLen times in lockstep;
||| one batched (actor, critic) forward per timestep amortises the
||| Idris-side per-op overhead across NumEnvs samples.
|||
||| Done envs auto-reset to `MkCP 0 0 0 0` so the [NumEnvs, ObsDim]
||| observation shape stays constant timestep-to-timestep (mirrors
||| `stepAutoReset` from `Gym.Vector` and PyTorch's `SyncVectorEnv`).
rolloutBatched : {n : Nat} -> Actor -> Critic -> VecEnv n CPState ->
                 Nat -> IO (Vect n (List RollStep), VecEnv n CPState)
rolloutBatched actor critic v0 rolloutLen = do
  (envs', stepLists) <- go rolloutLen v0.envs (replicate n [])
  pure (map reverse stepLists, MkVecEnv envs')
  where
    -- Helper: map with index over a Vect.
    mapIdx : (Nat -> a -> b) -> Vect k a -> Vect k b
    mapIdx _ [] = []
    mapIdx f (x :: xs) = f 0 x :: mapIdx (\i, v => f (S i) v) xs

    go : Nat -> Vect n CPState -> Vect n (List RollStep) ->
         IO (Vect n CPState, Vect n (List RollStep))
    go Z envs accs = pure (envs, accs)
    go (S k) envs accs = do
      let obsRows : Vect n (Vector ObsDim Double)
          obsRows = map (\s => obsTensor (observeVec s)) envs
          batchPtr = bulkToTensor2d {ex=ExampleExecutor} {dt=ExampleDType} obsRows
          stateV : Tensor [n, ObsDim] ExampleExecutor ExampleDType WithGrad
          stateV = MkTensor batchPtr Nothing
      (_, logitsV) <- forwardVarBatch actor stateV
      let logProbsV = the (Tensor [n, NumActions] ExampleExecutor ExampleDType WithGrad)
                        (MkTensor (primLogSoftmax2d {ex=ExampleExecutor} logitsV.tensorPtr) Nothing)
      (_, valuesV) <- forwardVarBatch critic stateV
      (acts, _) <- sampleActionFromBatch logProbsV envs
      let valueRows : Vect n Double
          valueRows = mapIdx (\i, _ => primItem2d {ex=ExampleExecutor} valuesV.tensorPtr (cast i) 0) envs
          obsVects : Vect n (Vect ObsDim Double)
          obsVects = map observeVec envs
      case stepAllAutoReset envs acts of
        (envs', rewards, dones) =>
          let newSteps = mkRollSteps obsVects acts rewards valueRows dones
              accs' = zipWith (\acc, s => s :: acc) accs newSteps
          in go k envs' accs'


----------------------------------------------------------------------
-- GAE helpers (pure Double — no autograd)
----------------------------------------------------------------------

bootstrapV : Critic -> Vect ObsDim Double -> IO Double
bootstrapV critic obs = do
  let stateV = the (TVec ObsDim ExampleExecutor ExampleDType WithGrad) (MkTensor (bulkToTensor {ex=ExampleExecutor} {dt=ExampleDType} (obsTensor obs)) Nothing)
  (_, valueV) <- forwardVar critic stateV
  pure (primItem1d {ex=ExampleExecutor} valueV.tensorPtr 0)

computeBootstrap : Critic -> List RollStep -> CPState -> IO Double
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
-- Per-step A2C loss ( typed-surface, autograd-tracked)
----------------------------------------------------------------------

perStepLoss : {n : Nat} -> (logitsB : Tensor [n, NumActions] ExampleExecutor ExampleDType WithGrad) ->
              (valuesB : Tensor [n, 1] ExampleExecutor ExampleDType WithGrad) -> (rowIdx : Int) ->
              Double -> Double ->
              (RollStep, Double, Double) -> IO (Tensor [] ExampleExecutor ExampleDType WithGrad)
perStepLoss logitsB valuesB rowIdx entropyCoef valueCoef (step, adv, retT) = do
  logitsRow <- trowSelect logitsB rowIdx
  let logPT = the (Tensor [NumActions] ExampleExecutor ExampleDType WithGrad)
                 (MkTensor (primLogSoftmax {ex=ExampleExecutor} logitsRow.tensorPtr 0) Nothing)
      aIdx : Int
      aIdx = cast {to=Int} (cast {to=Integer} step.action)
  logProbV <- telemSelect logPT aIdx

  valueRow <- trowSelect valuesB rowIdx
  valueV   <- telemSelect valueRow 0

  retC     <- tconstScalar retT

  policyT  <- tmulScalar logProbV (negate adv)

  diff     <- tsub valueV retC
  sq       <- tmul diff diff
  valueTerm <- tmulScalar sq valueCoef

  lp0V <- telemSelect logPT 0
  lp1V <- telemSelect logPT 1
  p0V  <- texp lp0V
  p1V  <- texp lp1V
  let negEntV = the (Tensor [] ExampleExecutor ExampleDType WithGrad) (MkTensor
                  (primAdd {ex=ExampleExecutor} (primMul {ex=ExampleExecutor} p0V.tensorPtr lp0V.tensorPtr)
                             (primMul {ex=ExampleExecutor} p1V.tensorPtr lp1V.tensorPtr))
                  Nothing)
  entTerm <- tmulScalar negEntV entropyCoef
  pure (MkTensor (primAdd {ex=ExampleExecutor} (primAdd {ex=ExampleExecutor} policyT.tensorPtr valueTerm.tensorPtr)
                          entTerm.tensorPtr) Nothing)


aggregateLoss : List (Tensor [] ExampleExecutor ExampleDType WithGrad) -> IO (Tensor [] ExampleExecutor ExampleDType WithGrad)
aggregateLoss losses = do
  zero <- tconstScalar 0.0
  let summed = foldl (\a, b => MkTensor (primAdd {ex=ExampleExecutor} a.tensorPtr b.tensorPtr) Nothing) zero losses
      n = the Double (cast (natToInteger (length losses)))
  tmulScalar summed (1.0 / n)


-- Build the batched loss tensor from pre-normalized (RollStep, adv, ret)
-- triples. One batched actor + critic forward over the flat batch, then
-- per-sample policy/value/entropy losses indexed into the [B, NumActions]
-- / [B, 1] outputs. Sequential and batched paths share this.
buildLossFromMerged : Actor -> Critic -> Double -> Double ->
                      List (RollStep, Double, Double) ->
                      IO (Tensor [] ExampleExecutor ExampleDType WithGrad)
buildLossFromMerged actor critic entropyCoef valueCoef merged = do
  let normalized = normAdvs merged
      normVec = Data.Vect.fromList normalized
      n = length normalized
      obsBatch = the (Vect (length normalized) (Vector ObsDim Double))
                     (map (\(s, _, _) => obsTensor s.obs) normVec)
      stackedT = bulkToTensor2d {ex=ExampleExecutor} {dt=ExampleDType} obsBatch
      stackedV = the (Tensor [n, ObsDim] ExampleExecutor ExampleDType WithGrad) (MkTensor stackedT Nothing)
  (_, logitsB) <- forwardVarBatch actor stackedV
  (_, valuesB) <- forwardVarBatch critic stackedV
  losses <- enumeratedLosses logitsB valuesB normVec 0
  aggregateLoss losses
  where
    enumeratedLosses : {n : Nat} -> Tensor [n, NumActions] ExampleExecutor ExampleDType WithGrad ->
                       Tensor [n, 1] ExampleExecutor ExampleDType WithGrad ->
                       Vect k (RollStep, Double, Double) -> Int ->
                       IO (List (Tensor [] ExampleExecutor ExampleDType WithGrad))
    enumeratedLosses _ _ [] _ = pure []
    enumeratedLosses lB vB (t :: rest) k = do
      l <- perStepLoss lB vB k entropyCoef valueCoef t
      ls <- enumeratedLosses lB vB rest (k + 1)
      pure (l :: ls)

-- Sequential rollout's loss: one GAE chain off `bootstrap`, then the
-- shared post-GAE machinery.
buildLoss : Actor -> Critic -> Double -> Double -> Double -> Double ->
            Double -> List RollStep -> IO (Tensor [] ExampleExecutor ExampleDType WithGrad)
buildLoss actor critic gamma lam entropyCoef valueCoef bootstrap steps =
  let triples = map stepTriple steps
      gaeOut = gae gamma lam bootstrap triples
      merged = map flattenTriple (zip steps gaeOut)
  in buildLossFromMerged actor critic entropyCoef valueCoef merged

-- Batched rollout's loss: per-env GAE off the env's own bootstrap, then
-- concat into one flat triples list and normalize across the whole batch
-- (matches PyTorch's SyncVectorEnv update where advantages are normalized
-- over T*N samples, not per-env). Shared post-GAE machinery.
buildLossBatched : {n : Nat} -> Actor -> Critic ->
                   Double -> Double -> Double -> Double ->
                   Vect n (List RollStep) -> Vect n Double ->
                   IO (Tensor [] ExampleExecutor ExampleDType WithGrad)
buildLossBatched actor critic gamma lam entropyCoef valueCoef stepLists bootstraps =
  let mergedPerEnv : Vect n (List (RollStep, Double, Double))
      mergedPerEnv = zipWith
        (\steps, boot =>
          let triples = map stepTriple steps
              gaeOut = gae gamma lam boot triples
          in map flattenTriple (zip steps gaeOut))
        stepLists bootstraps
      flatMerged = concat (toList mergedPerEnv)
  in buildLossFromMerged actor critic entropyCoef valueCoef flatMerged


----------------------------------------------------------------------
-- Config + epoch
----------------------------------------------------------------------

record A2CState where
  constructor MkA2C
  actor  : Actor
  critic : Critic
  envRef : IORef (VecEnv NumEnvs CPState)
  retRef : IORef (Vect NumEnvs Double)

record Config where
  constructor MkConfig
  lr          : Double
  epochs      : Nat
  gamma       : Double
  lam         : Double
  entropyCoef : Double
  valueCoef   : Double
  seed        : Bits64
  lrFind      : Bool

defaultConfig : Config
defaultConfig = MkConfig 7.0e-4 5000 0.99 0.95 0.01 0.5 42 False

specs : List (ArgSpec Config)
specs = [ Arg "--lr" (\v, c => { lr := cast v } c)
        , Arg "--epochs" (\v, c => { epochs := castNat v } c)
        , Arg "--gamma" (\v, c => { gamma := cast v } c)
        , Arg "--lambda" (\v, c => { lam := cast v } c)
        , Arg "--entropy" (\v, c => { entropyCoef := cast v } c)
        , Arg "--value-coef" (\v, c => { valueCoef := cast v } c)
        , Arg "--seed" (\v, c => { seed := castBits64 v } c)
        , Arg "--lr-find" (\v, c => { lrFind := (v == "1" || v == "true") } c)
        ]

lastTerminated : List RollStep -> Bool
lastTerminated steps = case last' steps of
  Just ls => ls.isDone
  Nothing => False

-- Per-env bootstrap: critic forward on each env's final state. Skips
-- envs whose last step terminated (those contribute 0 to GAE).
computeBootstrapsBatched : Critic -> Vect n (List RollStep) -> VecEnv n CPState ->
                           IO (Vect n Double)
computeBootstrapsBatched critic stepLists v = batchOver stepLists v.envs
  where
    batchOver : Vect k (List RollStep) -> Vect k CPState -> IO (Vect k Double)
    batchOver [] [] = pure []
    batchOver (steps :: rest) (s :: ss) = do
      b <- computeBootstrap critic steps s
      bs <- batchOver rest ss
      pure (b :: bs)

a2cEpoch : NativeOptimizer ExampleExecutor -> Config -> A2CState -> IO (A2CState, Double)
a2cEpoch opt cfg st = do
  startEnvs <- readIORef st.envRef
  -- Rollout-phase forward only extracts logits/values as Doubles (for
  -- sampling + bootstrap). The grad path is rebuilt fresh in
  -- buildLossBatched's batched forward, so the rollout's per-step
  -- forward doesn't need autograd tracking. withNoGrad skips tape
  -- append (tape/mlx) and disables libtorch's autograd graph (torch).
  rolled <- withNoGrad {ex=ExampleExecutor} (rolloutBatched st.actor st.critic startEnvs RolloutLen)
  let stepLists = fst rolled
      finalEnvs = snd rolled
  writeIORef st.envRef finalEnvs
  bootstraps <- withNoGrad {ex=ExampleExecutor} (computeBootstrapsBatched st.critic stepLists finalEnvs)
  loss <- buildLossBatched st.actor st.critic cfg.gamma cfg.lam
                              cfg.entropyCoef cfg.valueCoef stepLists bootstraps
  _ <- nativeTrainStep opt loss

  -- Per-env running returns: each env independently accumulates its own
  -- episodic return; on termination it adds to the reported list and
  -- resets to 0. Reporting averages over completed episodes this epoch
  -- (matches PyTorch SyncVectorEnv: any-env termination counts once).
  oldRunRets <- readIORef st.retRef
  case updateRetVect oldRunRets stepLists of
    (newRuns, epReturns) => do
      writeIORef st.retRef newRuns
      let nEp = length epReturns
          sumRew : Double
          sumRew = sum (map (\steps => sum (map (\s => s.reward) steps)) stepLists)
          reported = if nEp > 0
                     then sum epReturns / cast (natToInteger nEp)
                     else sumRew / cast (natToInteger NumEnvs)
      pure (st, negate reported)
  where
    -- Walk one env's running return through its rollout, emitting any
    -- completed episode returns and updating the running tally.
    walkOne : Double -> List RollStep -> (Double, List Double)
    walkOne run [] = (run, [])
    walkOne run (s :: ss) =
      let r' = run + s.reward
      in case walkOne (if s.isDone then 0.0 else r') ss of
           (final, eps) =>
             if s.isDone then (final, r' :: eps) else (final, eps)

    updateRetVect : Vect k Double -> Vect k (List RollStep) ->
                    (Vect k Double, List Double)
    updateRetVect [] [] = ([], [])
    updateRetVect (r :: rs) (steps :: rest) =
      case walkOne r steps of
        (r', eps) => case updateRetVect rs rest of
          (rs', restEps) => (r' :: rs', eps ++ restEps)


----------------------------------------------------------------------
-- Greedy evaluation
----------------------------------------------------------------------

greedyAct : Actor -> Vect ObsDim Double -> IO Nat
greedyAct actor obs = do
  let stateV = the (TVec ObsDim ExampleExecutor ExampleDType WithGrad) (MkTensor (bulkToTensor {ex=ExampleExecutor} {dt=ExampleDType} (obsTensor obs)) Nothing)
  (_, logits) <- forwardVar actor stateV
  let l0 = primItem1d {ex=ExampleExecutor} logits.tensorPtr 0
      l1 = primItem1d {ex=ExampleExecutor} logits.tensorPtr 1
  pure (if l0 >= l1 then 0 else 1)

evalEp : Actor -> CPState -> Nat -> Double -> IO Double
evalEp _ _ Z acc = pure acc
evalEp actor st (S k) acc = do
  a <- greedyAct actor (observeVec st)
  case cpStep st a of
    (r, st', outcome, _) =>
      if done outcome then pure (acc + r)
      else evalEp actor st' k (acc + r)

evalN : Actor -> Nat -> Double -> IO Double
evalN _ Z acc = pure acc
evalN actor (S k) acc = do
  ep <- evalEp actor (MkCP 0 0 0 0) MaxSteps 0.0
  evalN actor k (acc + ep)


----------------------------------------------------------------------
-- Main
----------------------------------------------------------------------

main : IO ()
main = do
  requireMachine {m = ChosenMachine}
  args <- getArgs
  let cfg = parseArgs defaultConfig specs (drop 1 args)
  srand cfg.seed

  putStrLn "=== A2C on CartPole (separate actor + critic) ==="
  putStrLn $ "Config: lr=" ++ show cfg.lr
           ++ " epochs=" ++ show cfg.epochs
           ++ " rollout=" ++ show RolloutLen
           ++ " gamma=" ++ show cfg.gamma
           ++ " lambda=" ++ show cfg.lam
           ++ " entropy=" ++ show cfg.entropyCoef
           ++ " seed=" ++ show cfg.seed

  actor <- mkActor
  critic <- mkCritic
  let initEnvs : VecEnv NumEnvs CPState
      initEnvs = resetAll {state=CPState} {action=Nat} {obs=Vect 4 Double}
  envRef <- newIORef initEnvs
  retRef <- newIORef (the (Vect NumEnvs Double) (replicate NumEnvs 0.0))
  let st0 = MkA2C actor critic envRef retRef
      opt = nativeAdamGlobalClip cfg.lr 0.9 0.999 1.0e-8 0.5

  putStrLn ""

  when cfg.lrFind $ do
    let lrCfg : LrFindConfig
        lrCfg = { numIters := 100 } defaultLrFindConfig
    _ <- lrFind lrCfg
      (\s, _ => a2cEpoch opt cfg s)
      (pure ()) opt st0
    putStrLn ""
    putStrLn "Done — re-run without --lr-find at the recommended LR."
    exitSuccess

  metrics <- newRLMetricsState 50
  let trainCfg : TrainConfig A2CState
      trainCfg = mkTrainConfig cfg.epochs 500 NoEarlyStop
                   (\_ => readRLMetrics "recent_50" metrics) (\_ => pure ())
  (trained, epochsDone, _) <- runTrainingIO {ex=ExampleExecutor}
    (\s, _ => do
       (s', loss) <- a2cEpoch opt cfg s
       recordReturn metrics (negate loss)
       pure (s', loss))
    (pure ())
    trainCfg st0

  putStrLn ""
  let nEval = the Nat 30
  -- Greedy eval doesn't need gradients — disable autograd graph
  -- construction for the 30 × 200 forward passes.
  evalSum <- withNoGrad {ex=ExampleExecutor} (evalN trained.actor nEval 0.0)
  let avgReturn = evalSum / cast (natToInteger nEval)
  putStrLn $ "Eval (" ++ show nEval ++ " episodes, greedy): avg_return=" ++ show avgReturn
  putStrLn ""
  putStrLn $ formatResult [("avg_return", show avgReturn),
                            ("epochs", show epochsDone),
                            ("seed", show cfg.seed)]
