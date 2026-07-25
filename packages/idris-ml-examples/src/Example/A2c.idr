module Example.A2c

import Control.Linear.LIO
import Data.IORef
import Data.Linear.Notation
import Data.List
import Data.Vect
import System

import Gym.ClassicControl.CartPole
import Gym.Env
import Gym.Vector
import Ml.Array
import Ml.Compat.Random
import Ml.Fit
import Ml.Hpo.LrFinder
import Ml.RL.Gae
import Ml.Rng
import Ml.Sampler
import Ml.Simple
import Ml.Train
import Random.Source

import BuildConfig

-- Actor + critic are linear `Seq`s; hide the IO `Nn.Seq` constructors.

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

mkActor : Init Actor
mkActor = scoped "actor" $ do
  l1 <- linear {i=ObsDim} {o=Hidden}
  l2 <- linear {i=Hidden} {o=Hidden}
  l3 <- linear {i=Hidden} {o=NumActions}
  pure (l1 ~~> tanhA ~~> l2 ~~> tanhA ~~> l3 ~~> Nil)

mkCritic : Init Critic
mkCritic = scoped "critic" $ do
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
sampleActionFromBatch : {n : Nat} -> Rng -> Tensor [n, NumActions] Ex F g ->
                        Vect n CPState -> IO (Vect n Nat, Vect n Double)
sampleActionFromBatch rng logProbsB envs = go 0 envs
  where
    go : Int -> Vect k CPState -> IO (Vect k Nat, Vect k Double)
    go _ []          = pure ([], [])
    go i (_ :: rest) = do
      let lp0 = primItem2d {ex=Ex} logProbsB.tensorPtr i 0
          lp1 = primItem2d {ex=Ex} logProbsB.tensorPtr i 1
      a <- rng.choice [Prelude.exp lp0, Prelude.exp lp1]
      (acts, lpVals) <- go (i + 1) rest
      pure (a :: acts, (if a == 0 then lp0 else lp1) :: lpVals)

-- Step every env with its action; auto-reset envs that terminate, drawing
-- the new state from CartPole's own U(-0.05, 0.05)^4 start distribution
-- (`Gym.Vector.stepAutoReset` threads the Source through those sub-resets).
stepAllAutoReset : {n : Nat} -> Source -> Vect n CPState -> Vect n Nat ->
                   (Vect n CPState, Vect n Double, Vect n Bool, Source)
stepAllAutoReset seed envs acts =
  case stepAutoReset {state=CPState} {action=Nat} {obs=Vect ObsDim Double}
                     seed (MkVecEnv envs) acts of
    (v', rewards, _, outcomes, seed') => (v'.envs, rewards, map done outcomes, seed')

mkRollSteps : Vect n (Vect ObsDim Double) -> Vect n Nat -> Vect n Double ->
              Vect n Double -> Vect n Bool -> Vect n RollStep
mkRollSteps [] [] [] [] []                                    = []
mkRollSteps (o :: os) (a :: as) (r :: rs) (v :: vs) (d :: ds) =
  MkRS o a r v d :: mkRollSteps os as rs vs ds

||| Batched per-env rollout. Each env steps RolloutLen times in lockstep;
||| one batched (actor, critic) forward per timestep amortises the
||| Idris-side per-op overhead across NumEnvs samples. Done envs
||| auto-reset so the [NumEnvs, ObsDim] shape stays constant.
rolloutBatchedL : {n : Nat} -> Rng -> (1 _ : Actor) -> (1 _ : Critic) -> VecEnv n CPState ->
                  Source -> Nat ->
                  L IO {use = 1} (LPair (!* (Vect n (List RollStep), VecEnv n CPState, Source))
                                        (LPair Actor Critic))
rolloutBatchedL rng actor critic v0 seed0 rolloutLen = do
  (MkBang (envs', stepLists, seedEnd) # (actor' # critic')) <-
    go rolloutLen actor critic v0.envs seed0 (replicate n [])
  pure1 (MkBang (map reverse stepLists, MkVecEnv envs', seedEnd) # (actor' # critic'))
  where
    mapIdx : (Nat -> a -> b) -> Vect k a -> Vect k b
    mapIdx _ []        = []
    mapIdx f (x :: xs) = f 0 x :: mapIdx (\i, v => f (S i) v) xs

    go : Nat -> (1 _ : Actor) -> (1 _ : Critic) -> Vect n CPState -> Source ->
         Vect n (List RollStep) ->
         L IO {use = 1} (LPair (!* (Vect n CPState, Vect n (List RollStep), Source))
                               (LPair Actor Critic))
    go Z actor critic envs seed accs     = pure1 (MkBang (envs, accs, seed) # (actor # critic))
    go (S k) actor critic envs seed accs = do
      stateV <- liftIO1 (ioRerun (\_ =>
        the (Tensor [n, ObsDim] Ex F WithGrad)
            (MkTensor (bulkToTensor2d {ex=Ex} {dt=F} (map (\s => obsTensor (observeVec s)) envs)) Nothing)))
      (MkBang logitsV # actor') <- forwardSeq {b=n} actor stateV
      let logProbsV = the (Tensor [n, NumActions] Ex F WithGrad)
                        (MkTensor (primLogSoftmax2d {ex=Ex} logitsV.tensorPtr) Nothing)
      (MkBang valuesV # critic') <- forwardSeq {b=n} critic stateV
      acts <- liftIO1 (fst <$> sampleActionFromBatch rng logProbsV envs)
      let valueRows : Vect n Double
          valueRows = mapIdx (\i, _ => primItem2d {ex=Ex} valuesV.tensorPtr (cast i) 0) envs
          obsVects : Vect n (Vect ObsDim Double)
          obsVects = map observeVec envs
      case stepAllAutoReset seed envs acts of
        (envs', rewards, dones, seed') =>
          let newSteps = mkRollSteps obsVects acts rewards valueRows dones
              accs' = zipWith (\acc, s => s :: acc) accs newSteps
          in go k actor' critic' envs' seed' accs'

----------------------------------------------------------------------
-- GAE helpers (pure Double — no autograd)
----------------------------------------------------------------------

bootstrapVL : (1 _ : Critic) -> Vect ObsDim Double -> L IO {use = 1} (LPair (!* Double) Critic)
bootstrapVL critic obs = do
  stateV <- liftIO1 (ioRerun (\_ =>
    the (Tensor [1, ObsDim] Ex F WithGrad)
        (MkTensor (bulkToTensor2d {ex=Ex} {dt=F} [obsTensor obs]) Nothing)))
  (MkBang valueV # critic') <- forwardSeq {b=1} critic stateV
  pure1 (MkBang (primItem2d {ex=Ex} valueV.tensorPtr 0 0) # critic')

computeBootstrapL : (1 _ : Critic) -> List RollStep -> CPState ->
                    L IO {use = 1} (LPair (!* Double) Critic)
computeBootstrapL critic [] _          = pure1 (MkBang 0.0 # critic)
computeBootstrapL critic steps finalSt =
  case last' steps of
    Nothing => pure1 (MkBang 0.0 # critic)
    Just ls => if ls.isDone then pure1 (MkBang 0.0 # critic)
                            else bootstrapVL critic (observeVec finalSt)

stepTriple : RollStep -> (Double, Double, Bool)
stepTriple s = (s.reward, s.value, s.isDone)

flattenTriple : (RollStep, (Double, Double)) -> (RollStep, Double, Double)
flattenTriple (sRec, (a, r)) = (sRec, a, r)

tripleAdv : (RollStep, Double, Double) -> Double
tripleAdv (_, a, _) = a

-- Denominator n-1, matching `Tensor.std()`'s PyTorch default on the reference
-- side, and the epsilon outside the root for the same reason.
normAdvs : List (RollStep, Double, Double) -> List (RollStep, Double, Double)
normAdvs triples =
  let advs   = map tripleAdv triples
      nN     = the Double (cast (natToInteger (length advs)))
      mu     = if nN > 0.0 then sum advs / nN else 0.0
      sqDevs = map (\a => (a - mu) * (a - mu)) advs
      vr     = if nN > 1.0 then sum sqDevs / (nN - 1.0) else 1.0
      sd     = sqrt vr + 1.0e-8
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
buildLossFromMergedL : (1 _ : Actor) -> (1 _ : Critic) -> Double -> Double ->
                       List (RollStep, Double, Double) ->
                       L IO {use = 1} (LPair (!* (Tensor [] Ex F WithGrad)) (LPair Actor Critic))
buildLossFromMergedL actor critic entropyCoef valueCoef merged = do
  let normalized = normAdvs merged
      normVec  = Data.Vect.fromList normalized
      n        = length normalized
      obsBatch = the (Vect (length normalized) (Vector ObsDim Double))
                     (map (\(s, _, _) => obsTensor s.obs) normVec)
  stackedV <- liftIO1 (ioRerun (\_ =>
    the (Tensor [n, ObsDim] Ex F WithGrad) (MkTensor (bulkToTensor2d {ex=Ex} {dt=F} obsBatch) Nothing)))
  (MkBang logitsB # actor') <- forwardSeq {b=n} actor stackedV
  (MkBang valuesB # critic') <- forwardSeq {b=n} critic stackedV
  loss <- liftIO1 $ do
            losses <- enumeratedLosses logitsB valuesB normVec 0
            aggregateLoss losses
  pure1 (MkBang loss # (actor' # critic'))
  where
    enumeratedLosses : {n : Nat} -> Tensor [n, NumActions] Ex F WithGrad ->
                       Tensor [n, 1] Ex F WithGrad ->
                       Vect k (RollStep, Double, Double) -> Int ->
                       IO (List (Tensor [] Ex F WithGrad))
    enumeratedLosses _ _ [] _            = pure []
    enumeratedLosses lB vB (t :: rest) k = do
      l <- perStepLoss lB vB k entropyCoef valueCoef t
      ls <- enumeratedLosses lB vB rest (k + 1)
      pure (l :: ls)

-- Batched rollout's loss: per-env GAE off the env's own bootstrap, then
-- concat into one flat triples list and normalize across the whole batch
-- (matches PyTorch's SyncVectorEnv update where advantages are normalized
-- over T*N samples, not per-env).
buildLossBatchedL : {n : Nat} -> (1 _ : Actor) -> (1 _ : Critic) ->
                    Double -> Double -> Double -> Double ->
                    Vect n (List RollStep) -> Vect n Double ->
                    L IO {use = 1} (LPair (!* (Tensor [] Ex F WithGrad)) (LPair Actor Critic))
buildLossBatchedL actor critic gamma lam entropyCoef valueCoef stepLists bootstraps =
  let mergedPerEnv : Vect n (List (RollStep, Double, Double))
      mergedPerEnv = zipWith
        (\steps, boot =>
          let triples = map stepTriple steps
              gaeOut = gae gamma lam boot triples
          in map flattenTriple (zip steps gaeOut))
        stepLists bootstraps
      flatMerged = concat (toList mergedPerEnv)
  in buildLossFromMergedL actor critic entropyCoef valueCoef flatMerged

----------------------------------------------------------------------
-- Config + epoch
----------------------------------------------------------------------

-- Actor + critic are **linear** fields (threaded single-owner through the
-- epoch); the env / running-return IORefs are ω.
record A2CState where
  constructor MkA2C
  1 actor  : Actor
  1 critic : Critic
  envRef  : IORef (VecEnv NumEnvs CPState)
  retRef  : IORef (Vect NumEnvs Double)
  seedRef : IORef Source
  rng     : Rng

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
  replay      : String

defaultConfig : Config
defaultConfig = MkConfig 7.0e-4 5000 0.99 0.95 0.01 0.5 42 False ""

specs : List (ArgSpec Config)
specs = [ Arg "--lr" (\v, c => { lr := cast v } c)
        , Arg "--epochs" (\v, c => { epochs := castNat v } c)
        , Arg "--gamma" (\v, c => { gamma := cast v } c)
        , Arg "--lambda" (\v, c => { lam := cast v } c)
        , Arg "--entropy" (\v, c => { entropyCoef := cast v } c)
        , Arg "--value-coef" (\v, c => { valueCoef := cast v } c)
        , Arg "--seed" (\v, c => { seed := castBits64 v } c)
        , Arg "--lr-find" (\v, c => { lrFind := (v == "1" || v == "true") } c)
        -- Replay recorded draws (`Ml.Rng.loadReplay` format) instead of
        -- sampling: actions come from the file's choice channel and env
        -- resets from its env channel, so the rollout reproduces exactly.
        , Arg "--replay" (\v, c => { replay := v } c)
        ]

-- Per-env bootstrap: critic forward on each env's final state. Skips
-- envs whose last step terminated (those contribute 0 to GAE).
computeBootstrapsBatchedL : (1 _ : Critic) -> Vect n (List RollStep) -> VecEnv n CPState ->
                            L IO {use = 1} (LPair (!* (Vect n Double)) Critic)
computeBootstrapsBatchedL critic stepLists v = batchOver critic stepLists v.envs
  where
    -- Thread the critic across envs, building the bootstrap Vect in env order.
    batchOver : (1 _ : Critic) -> Vect k (List RollStep) -> Vect k CPState ->
                L IO {use = 1} (LPair (!* (Vect k Double)) Critic)
    batchOver critic [] []                     = pure1 (MkBang [] # critic)
    batchOver critic (steps :: rest) (s :: ss) = do
      (MkBang b # critic') <- computeBootstrapL critic steps s
      (MkBang bs # critic'') <- batchOver critic' rest ss
      pure1 (MkBang (b :: bs) # critic'')

a2cEpochL : Optimizer Ex -> Config -> (1 _ : A2CState) -> L IO {use = 1} (LPair (!* Double) A2CState)
a2cEpochL opt cfg (MkA2C actor critic envRef retRef seedRef rng) = do
  startEnvs <- liftIO1 (readIORef envRef)
  startSeed <- liftIO1 (readIORef seedRef)
  -- Rollout-phase forward only extracts logits/values as Doubles (no grad);
  -- the grad path is rebuilt fresh in buildLossBatchedL's batched forward.
  (MkBang (stepLists, finalEnvs, finalSeed) # (actor' # critic')) <-
    withNoGradL {ex=Ex} (rolloutBatchedL rng actor critic startEnvs startSeed RolloutLen)
  liftIO1 (writeIORef envRef finalEnvs)
  liftIO1 (writeIORef seedRef finalSeed)
  (MkBang bootstraps # critic'') <-
    withNoGradL {ex=Ex} (computeBootstrapsBatchedL critic' stepLists finalEnvs)
  (MkBang loss # (actor'' # critic''')) <-
    buildLossBatchedL actor' critic'' cfg.gamma cfg.lam cfg.entropyCoef cfg.valueCoef stepLists bootstraps
  _ <- liftIO1 (trainStep opt loss)
  -- Per-env running returns (ω bookkeeping).
  reported <- liftIO1 $ do
    oldRunRets <- readIORef retRef
    case updateRetVect oldRunRets stepLists of
      (newRuns, epReturns) => do
        writeIORef retRef newRuns
        let nEp = length epReturns
            sumRew : Double
            sumRew   = sum (map (\steps => sum (map (\s => s.reward) steps)) stepLists)
        pure (if nEp > 0
              then sum epReturns / cast (natToInteger nEp)
              else sumRew / cast (natToInteger NumEnvs))
  pure1 (MkBang (negate reported) # MkA2C actor'' critic''' envRef retRef seedRef rng)
  where
    walkOne : Double -> List RollStep -> (Double, List Double)
    walkOne run []        = (run, [])
    walkOne run (s :: ss) =
      let r' = run + s.reward
      in case walkOne (if s.isDone then 0.0 else r') ss of
           (final, eps) =>
             if s.isDone then (final, r' :: eps) else (final, eps)

    updateRetVect : Vect k Double -> Vect k (List RollStep) ->
                    (Vect k Double, List Double)
    updateRetVect [] []                     = ([], [])
    updateRetVect (r :: rs) (steps :: rest) =
      case walkOne r steps of
        (r', eps) => case updateRetVect rs rest of
          (rs', restEps) => (r' :: rs', eps ++ restEps)

----------------------------------------------------------------------
-- Greedy evaluation
----------------------------------------------------------------------

greedyActL : (1 _ : Actor) -> Vect ObsDim Double -> L IO {use = 1} (LPair (!* Nat) Actor)
greedyActL actor obs = do
  stateV <- liftIO1 (ioRerun (\_ =>
    the (Tensor [1, ObsDim] Ex F WithGrad)
        (MkTensor (bulkToTensor2d {ex=Ex} {dt=F} [obsTensor obs]) Nothing)))
  (MkBang logits # actor') <- forwardSeq {b=1} actor stateV
  let l0 = primItem2d {ex=Ex} logits.tensorPtr 0 0
      l1 = primItem2d {ex=Ex} logits.tensorPtr 0 1
  pure1 (MkBang (if l0 >= l1 then the Nat 0 else 1) # actor')

evalEpL : (1 _ : Actor) -> CPState -> Nat -> Double -> L IO {use = 1} (LPair (!* Double) Actor)
evalEpL actor _ Z acc      = pure1 (MkBang acc # actor)
evalEpL actor st (S k) acc = do
  (MkBang a # actor') <- greedyActL actor (observeVec st)
  case cpStep st a of
    (r, st', outcome, _) =>
      if done outcome then pure1 (MkBang (acc + r) # actor')
      else evalEpL actor' st' k (acc + r)

-- Each episode starts from a fresh `reset` draw, as the reference's
-- `env.reset()` does. A fixed start would make every greedy episode the same
-- trajectory, so the mean over N of them would carry one sample's worth of
-- information.
evalNL : (1 _ : Actor) -> Source -> Nat -> Double -> L IO {use = 1} (LPair (!* Double) Actor)
evalNL actor _ Z acc        = pure1 (MkBang acc # actor)
evalNL actor seed (S k) acc = do
  let (st0, seed') = reset {state=CPState} {action=Nat} {obs=Vect ObsDim Double} seed
  (MkBang ep # actor') <- evalEpL actor st0 MaxSteps 0.0
  evalNL actor' seed' k (acc + ep)

----------------------------------------------------------------------
-- State construction / eval / discard (linear)
----------------------------------------------------------------------

buildStateL : (replayPath : String) -> L IO {use = 1} A2CState
buildStateL replayPath = do
  actor  <- runInitL mkActor
  critic <- runInitL mkCritic
  -- All stochastic input comes through one `Replay`: live draws normally,
  -- the recorded channels of `--replay <file>` otherwise.
  replay <- liftIO1 (case replayPath of
                       "" => liveReplay
                       p  => loadReplay p)
  let (initEnvs, seed0) = resetAll {state=CPState} {action=Nat} {obs=Vect ObsDim Double}
                                   replay.envSource
  envRef  <- liftIO1 (newIORef initEnvs)
  retRef  <- liftIO1 (newIORef (the (Vect NumEnvs Double) (replicate NumEnvs 0.0)))
  seedRef <- liftIO1 (newIORef seed0)
  pure1 (MkA2C actor critic envRef retRef seedRef replay.rng)

discardStateL : (1 _ : A2CState) -> L IO ()
discardStateL (MkA2C actor critic _ _ _ _) = do
  discard actor
  discard critic

finalReportL : Config -> Nat -> (1 _ : A2CState) -> L IO ()
finalReportL cfg epochsDone (MkA2C actor critic _ _ seedRef _) = do
  let nEval = the Nat 50
  evalSeed <- liftIO1 (readIORef seedRef)
  (MkBang evalSum # actor') <- withNoGradL {ex=Ex} (evalNL actor evalSeed nEval 0.0)
  discard actor'
  discard critic
  liftIO1 $ do
    let avgReturn = evalSum / cast (natToInteger nEval)
    putStrLn ""
    putStrLn $ "Eval (" ++ show nEval ++ " episodes, greedy): avg_return=" ++ show avgReturn
    putStrLn ""
    putStrLn $ formatResult [("avg_return", show avgReturn),
                              ("epochs", show epochsDone),
                              ("seed", show cfg.seed)]

----------------------------------------------------------------------
-- Main
----------------------------------------------------------------------

%default partial

lrFindCfg : LrFindConfig
lrFindCfg = { numIters := 100 } defaultLrFindConfig

-- Terminal consumer of the trained-but-unused lrFind state: discard it, then
-- print. A single `(1 _ : A2CState) -> L IO ()` so the bind continuation that
-- produces it is recognised as linear (mirrors `finalReportL`'s shape).
finishLrFind : (1 _ : LPair (!* LrFindResult) A2CState) -> L IO ()
finishLrFind (MkBang _ # st') = do
  discardStateL st'
  liftIO1 $ do
    putStrLn ""
    putStrLn "Done — re-run without --lr-find at the recommended LR."

runLrFind : Config -> IO ()
runLrFind cfg = Control.Linear.LIO.run $ do
  st0 <- buildStateL cfg.replay
  opt <- liftIO1 (adam cfg.lr ({ clip := NormClip 0.5 } defaultOpts))
  (LIO.(>>=))
    (lrFind {ex = Ex} {model = A2CState} {dp = ()} lrFindCfg
       (\s, _ => a2cEpochL opt cfg s) (pure ()) opt st0)
    finishLrFind

runTrain : Config -> IO ()
runTrain cfg = Control.Linear.LIO.run $ do
  st0 <- buildStateL cfg.replay
  -- Single Adam over both actor + critic (all params registered).
  opt <- liftIO1 (adam cfg.lr ({ clip := NormClip 0.5 } defaultOpts))
  metrics <- liftIO1 (newRLMetricsState 50)
  let trainCfg : TrainConfig A2CState
      trainCfg = { metricsL := readRLMetrics "recent_50" metrics }
                   (mkTrainConfig cfg.epochs 500 NoEarlyStop
                      (const (pure (the (List (String, String)) []))) (\_ => pure ()))
  (MkBang (epochsDone, _) # trained) <- fit {batch = ()}
    (\s, _ => do
       (MkBang loss # s') <- a2cEpochL opt cfg s
       dd <- liftIO1 (do recordReturn metrics (negate loss); pure loss)
       pure1 (MkBang dd # s'))
    opt (generate (pure ())) trainCfg st0
  finalReportL cfg epochsDone trained

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

  putStrLn ""

  if cfg.lrFind then runLrFind cfg else runTrain cfg
