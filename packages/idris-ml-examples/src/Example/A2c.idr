module Example.A2c

import Data.List
import Data.Vect
import Data.IORef
import System
import Compat.Random

import ML.Simple
import Array            -- Vector / VArray / SArray
import Gym.ClassicControl.CartPole
import Gym.Env
import Gym.Vector
import Hpo.LrFinder
import RL.Gae
import Sampler
import Train
import BuildConfig

----------------------------------------------------------------------
-- Architecture: separate actor and critic MLPs (aligned with PyTorch
-- reference `a2c.py`). Both register into the C param registry under
-- distinct `scoped` prefixes (actor./critic.), so a single Adam steps
-- both networks off the combined policy+value+entropy loss.
--   Actor  : 4 -> 64 -> tanh -> 64 -> tanh -> 2 (action logits)
--   Critic : 4 -> 64 -> tanh -> 64 -> tanh -> 1 (state value)
----------------------------------------------------------------------

ObsDim     : Nat; ObsDim = 4
Hidden     : Nat; Hidden = 64
NumActions : Nat; NumActions = 2
MaxSteps   : Nat; MaxSteps = cartPoleMaxSteps
RolloutLen : Nat; RolloutLen = 20

||| Number of parallel envs run per a2cEpoch. Compile-time because the
||| batched-forward shape `[NumEnvs, ObsDim]` is part of the autograd graph.
NumEnvs : Nat; NumEnvs = 4

Actor : Type
Actor = Seq ObsDim NumActions Ex F WithGrad

Critic : Type
Critic = Seq ObsDim 1 Ex F WithGrad

mkActor : IO Actor
mkActor = runInit $ scoped "actor" $ do
  l1 <- linear {i=ObsDim} {o=Hidden}
  l2 <- linear {i=Hidden} {o=Hidden}
  l3 <- linear {i=Hidden} {o=NumActions}
  pure (l1 ~~> tanhA ~~> l2 ~~> tanhA ~~> l3 ~~> Nil)

mkCritic : IO Critic
mkCritic = runInit $ scoped "critic" $ do
  l1 <- linear {i=ObsDim} {o=Hidden}
  l2 <- linear {i=Hidden} {o=Hidden}
  l3 <- linear {i=Hidden} {o=1}
  pure (l1 ~~> tanhA ~~> l2 ~~> tanhA ~~> l3 ~~> Nil)

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
  obs    : Vect ObsDim Double
  action : Nat
  reward : Double
  value  : Double
  isDone : Bool

----------------------------------------------------------------------
-- Batched rollout (NumEnvs parallel envs, one batched forward per step)
----------------------------------------------------------------------

-- Sample one action per env from a [n, NumActions] batched log-prob tensor.
sampleActionFromBatch : {n : Nat} -> Tensor [n, NumActions] Ex F g ->
                        Vect n CPState -> IO (Vect n Nat, Vect n Double)
sampleActionFromBatch logProbsB envs = go 0 envs
  where
    go : Int -> Vect k CPState -> IO (Vect k Nat, Vect k Double)
    go _ [] = pure ([], [])
    go i (_ :: rest) = do
      let lp0 = primItem2d {ex=Ex} logProbsB.tensorPtr i 0
          lp1 = primItem2d {ex=Ex} logProbsB.tensorPtr i 1
      u <- randomRIO (the Double 0.0, 1.0)
      let a = categoricalSample [Prelude.exp lp0, Prelude.exp lp1] u
      (acts, lpVals) <- go (i + 1) rest
      pure (a :: acts, (if a == 0 then lp0 else lp1) :: lpVals)

-- Step every env with its action; auto-reset envs that terminate.
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

mkRollSteps : Vect n (Vect ObsDim Double) -> Vect n Nat -> Vect n Double ->
              Vect n Double -> Vect n Bool -> Vect n RollStep
mkRollSteps [] [] [] [] [] = []
mkRollSteps (o :: os) (a :: as) (r :: rs) (v :: vs) (d :: ds) =
  MkRS o a r v d :: mkRollSteps os as rs vs ds

||| Batched per-env rollout. Each env steps RolloutLen times in lockstep;
||| one batched (actor, critic) forward per timestep amortises the
||| Idris-side per-op overhead across NumEnvs samples. Done envs
||| auto-reset so the [NumEnvs, ObsDim] shape stays constant.
rolloutBatched : {n : Nat} -> Actor -> Critic -> VecEnv n CPState ->
                 Nat -> IO (Vect n (List RollStep), VecEnv n CPState)
rolloutBatched actor critic v0 rolloutLen = do
  (envs', stepLists) <- go rolloutLen v0.envs (replicate n [])
  pure (map reverse stepLists, MkVecEnv envs')
  where
    mapIdx : (Nat -> a -> b) -> Vect k a -> Vect k b
    mapIdx _ [] = []
    mapIdx f (x :: xs) = f 0 x :: mapIdx (\i, v => f (S i) v) xs

    go : Nat -> Vect n CPState -> Vect n (List RollStep) ->
         IO (Vect n CPState, Vect n (List RollStep))
    go Z envs accs = pure (envs, accs)
    go (S k) envs accs = do
      let obsRows : Vect n (Vector ObsDim Double)
          obsRows = map (\s => obsTensor (observeVec s)) envs
          batchPtr = bulkToTensor2d {ex=Ex} {dt=F} obsRows
          stateV : Tensor [n, ObsDim] Ex F WithGrad
          stateV = MkTensor batchPtr Nothing
      logitsV <- forwardSeq {b=n} actor stateV
      let logProbsV = the (Tensor [n, NumActions] Ex F WithGrad)
                        (MkTensor (primLogSoftmax2d {ex=Ex} logitsV.tensorPtr) Nothing)
      valuesV <- forwardSeq {b=n} critic stateV
      (acts, _) <- sampleActionFromBatch logProbsV envs
      let valueRows : Vect n Double
          valueRows = mapIdx (\i, _ => primItem2d {ex=Ex} valuesV.tensorPtr (cast i) 0) envs
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
  let stateV = the (Tensor [1, ObsDim] Ex F WithGrad)
                 (MkTensor (bulkToTensor2d {ex=Ex} {dt=F} [obsTensor obs]) Nothing)
  valueV <- forwardSeq {b=1} critic stateV
  pure (primItem2d {ex=Ex} valueV.tensorPtr 0 0)

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
-- Per-step A2C loss (autograd-tracked)
----------------------------------------------------------------------

perStepLoss : {n : Nat} -> (logitsB : Tensor [n, NumActions] Ex F WithGrad) ->
              (valuesB : Tensor [n, 1] Ex F WithGrad) -> (rowIdx : Int) ->
              Double -> Double ->
              (RollStep, Double, Double) -> IO (Tensor [] Ex F WithGrad)
perStepLoss logitsB valuesB rowIdx entropyCoef valueCoef (step, adv, retT) = do
  logitsRow <- trowSelect logitsB rowIdx
  let logPT = the (Tensor [NumActions] Ex F WithGrad)
                 (MkTensor (primLogSoftmax {ex=Ex} logitsRow.tensorPtr 0) Nothing)
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
  let negEntV = the (Tensor [] Ex F WithGrad) (MkTensor
                  (primAdd {ex=Ex} (primMul {ex=Ex} p0V.tensorPtr lp0V.tensorPtr)
                             (primMul {ex=Ex} p1V.tensorPtr lp1V.tensorPtr))
                  Nothing)
  entTerm <- tmulScalar negEntV entropyCoef
  pure (MkTensor (primAdd {ex=Ex} (primAdd {ex=Ex} policyT.tensorPtr valueTerm.tensorPtr)
                          entTerm.tensorPtr) Nothing)

aggregateLoss : List (Tensor [] Ex F WithGrad) -> IO (Tensor [] Ex F WithGrad)
aggregateLoss losses = do
  zero <- tconstScalar 0.0
  let summed = foldl (\a, b => MkTensor (primAdd {ex=Ex} a.tensorPtr b.tensorPtr) Nothing) zero losses
      n = the Double (cast (natToInteger (length losses)))
  tmulScalar summed (1.0 / n)

-- Build the batched loss tensor from pre-normalized (RollStep, adv, ret)
-- triples. One batched actor + critic forward over the flat batch, then
-- per-sample policy/value/entropy losses indexed into the [B, NumActions]
-- / [B, 1] outputs.
buildLossFromMerged : Actor -> Critic -> Double -> Double ->
                      List (RollStep, Double, Double) ->
                      IO (Tensor [] Ex F WithGrad)
buildLossFromMerged actor critic entropyCoef valueCoef merged = do
  let normalized = normAdvs merged
      normVec = Data.Vect.fromList normalized
      n = length normalized
      obsBatch = the (Vect (length normalized) (Vector ObsDim Double))
                     (map (\(s, _, _) => obsTensor s.obs) normVec)
      stackedT = bulkToTensor2d {ex=Ex} {dt=F} obsBatch
      stackedV = the (Tensor [n, ObsDim] Ex F WithGrad) (MkTensor stackedT Nothing)
  logitsB <- forwardSeq {b=n} actor stackedV
  valuesB <- forwardSeq {b=n} critic stackedV
  losses <- enumeratedLosses logitsB valuesB normVec 0
  aggregateLoss losses
  where
    enumeratedLosses : {n : Nat} -> Tensor [n, NumActions] Ex F WithGrad ->
                       Tensor [n, 1] Ex F WithGrad ->
                       Vect k (RollStep, Double, Double) -> Int ->
                       IO (List (Tensor [] Ex F WithGrad))
    enumeratedLosses _ _ [] _ = pure []
    enumeratedLosses lB vB (t :: rest) k = do
      l <- perStepLoss lB vB k entropyCoef valueCoef t
      ls <- enumeratedLosses lB vB rest (k + 1)
      pure (l :: ls)

-- Batched rollout's loss: per-env GAE off the env's own bootstrap, then
-- concat into one flat triples list and normalize across the whole batch
-- (matches PyTorch's SyncVectorEnv update where advantages are normalized
-- over T*N samples, not per-env).
buildLossBatched : {n : Nat} -> Actor -> Critic ->
                   Double -> Double -> Double -> Double ->
                   Vect n (List RollStep) -> Vect n Double ->
                   IO (Tensor [] Ex F WithGrad)
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

a2cEpoch : Optimizer Ex -> Config -> A2CState -> IO (A2CState, Double)
a2cEpoch opt cfg st = do
  startEnvs <- readIORef st.envRef
  -- Rollout-phase forward only extracts logits/values as Doubles (for
  -- sampling + bootstrap). The grad path is rebuilt fresh in
  -- buildLossBatched's batched forward, so withNoGrad skips tape append.
  rolled <- withNoGrad {ex=Ex} (rolloutBatched st.actor st.critic startEnvs RolloutLen)
  let stepLists = fst rolled
      finalEnvs = snd rolled
  writeIORef st.envRef finalEnvs
  bootstraps <- withNoGrad {ex=Ex} (computeBootstrapsBatched st.critic stepLists finalEnvs)
  loss <- buildLossBatched st.actor st.critic cfg.gamma cfg.lam
                              cfg.entropyCoef cfg.valueCoef stepLists bootstraps
  _ <- nativeTrainStep opt loss

  -- Per-env running returns: each env independently accumulates its own
  -- episodic return; on termination it adds to the reported list and
  -- resets to 0.
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
  let stateV = the (Tensor [1, ObsDim] Ex F WithGrad)
                 (MkTensor (bulkToTensor2d {ex=Ex} {dt=F} [obsTensor obs]) Nothing)
  logits <- forwardSeq {b=1} actor stateV
  let l0 = primItem2d {ex=Ex} logits.tensorPtr 0 0
      l1 = primItem2d {ex=Ex} logits.tensorPtr 0 1
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

%default partial

main : IO ()
main = do
  requireMachine {m = ChosenMachine}
  args <- getArgs
  let cfg = parseArgs defaultConfig specs (drop 1 args)
  srand cfg.seed
  tsetInitSeed {ex = Ex} cfg.seed

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
  resetSeedI <- randomInt32
  let initEnvs : VecEnv NumEnvs CPState
      initEnvs = fst (resetAll {state=CPState} {action=Nat} {obs=Vect 4 Double}
                              (cast resetSeedI))
  envRef <- newIORef initEnvs
  retRef <- newIORef (the (Vect NumEnvs Double) (replicate NumEnvs 0.0))
  let st0 = MkA2C actor critic envRef retRef
  -- Single Adam over both actor + critic (all params registered).
  opt <- adam cfg.lr ({ clip := NormClip 0.5 } defaultOpts)

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
  (trained, epochsDone, _) <- fit {batch = ()}
    (\s, _ => do
       (s', loss) <- a2cEpoch opt cfg s
       recordReturn metrics (negate loss)
       pure (s', loss))
    opt (generate (pure ()))
    trainCfg st0

  putStrLn ""
  let nEval = the Nat 30
  evalSum <- withNoGrad {ex=Ex} (evalN trained.actor nEval 0.0)
  let avgReturn = evalSum / cast (natToInteger nEval)
  putStrLn $ "Eval (" ++ show nEval ++ " episodes, greedy): avg_return=" ++ show avgReturn
  putStrLn ""
  putStrLn $ formatResult [("avg_return", show avgReturn),
                            ("epochs", show epochsDone),
                            ("seed", show cfg.seed)]
