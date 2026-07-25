module Example.Ppo

import Control.Linear.LIO
import Data.IORef
import Data.Linear.Notation
import Data.List
import Data.Vect
import System

import Gym.ClassicControl.Acrobot
import Gym.Env
import Gym.Vector
import Ml.Array
import Ml.Checkpoint
import Ml.Compat.Random
import Ml.Fit
import Ml.Floating
import Ml.Hpo.LrFinder
import Ml.Math
import Ml.RL.Gae
import Ml.Sampler
import Ml.Simple
import Ml.Train

import BuildConfig

-- Actor + critic are linear `Seq`s; hide the IO `Nn.Seq` constructors.

----------------------------------------------------------------------
-- Architecture: separate actor and critic MLPs with discrete-action
-- (categorical) policy on Acrobot. Both register into the C registry
-- under distinct scoped prefixes (actor./critic.), so a single Adam
-- steps both off the combined clipped-surrogate + value + entropy loss.
--
-- Actor:  Linear(6→64) → tanh → Linear(64→64) → tanh → Linear(64→3)
-- Critic: Linear(6→64) → tanh → Linear(64→64) → tanh → Linear(64→1)
----------------------------------------------------------------------

ObsDim     : Nat; ObsDim = 6
Hidden     : Nat; Hidden = 64
NumActions : Nat; NumActions = 3
EpisodeLen : Nat; EpisodeLen = 500   -- Acrobot defaultTimeLimit

||| Number of parallel envs run per ppoEpoch.
NumEnvs : Nat; NumEnvs = 4

||| Per-env rollout length. Total samples per ppoEpoch is
||| `NumEnvs * RolloutLen` (256 × 4 = 1024).
RolloutLen : Nat; RolloutLen = 256
BatchSize  : Nat; BatchSize = 64

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
-- Single-sample critic value (via [1, ObsDim] batched forward)
----------------------------------------------------------------------

criticValueL : (1 _ : Critic) -> Vect ObsDim Double -> L IO {use = 1} (LPair (!* Double) Critic)
criticValueL critic obs = do
  stateV <- liftIO1 (ioRerun (\_ =>
    the (Tensor [1, ObsDim] Ex F WithGrad)
        (MkTensor (bulkToTensor2d {ex=Ex} {dt=F} [obsTensor obs]) Nothing)))
  (MkBang outV # critic') <- forwardSeq {b=1} critic stateV
  pure1 (MkBang (primItem2d {ex=Ex} outV.tensorPtr 0 0) # critic')

----------------------------------------------------------------------
-- Batched rollout (NumEnvs parallel envs, one batched forward per step)
----------------------------------------------------------------------

-- Sample one action per env from a [n, NumActions] batched log-prob
-- tensor. Records the chosen action's log-prob per env (PPO needs it for
-- the importance ratio later).
sampleActionFromBatch : {n : Nat} -> Tensor [n, NumActions] Ex F g ->
                        Vect n AState -> IO (Vect n Nat, Vect n Double)
sampleActionFromBatch logProbsB envs = go 0 envs
  where
    go : Int -> Vect k AState -> IO (Vect k Nat, Vect k Double)
    go _ []          = pure ([], [])
    go i (_ :: rest) = do
      let lp0 = primItem2d {ex=Ex} logProbsB.tensorPtr i 0
          lp1 = primItem2d {ex=Ex} logProbsB.tensorPtr i 1
          lp2 = primItem2d {ex=Ex} logProbsB.tensorPtr i 2
      u <- randomRIO (the Double 0.0, 1.0)
      let a = categoricalSample [Prelude.exp lp0, Prelude.exp lp1, Prelude.exp lp2] u
          lp = case a of
                 0 => lp0
                 1 => lp1
                 _ => lp2
      (acts, lpVals) <- go (i + 1) rest
      pure (a :: acts, lp :: lpVals)

-- Step every env with its action; auto-reset envs that terminate OR hit
-- the EpisodeLen truncation cap.
stepAllAutoResetTrunc : Vect n AState -> Vect n Nat -> Vect n Nat ->
                        (Vect n AState, Vect n Double, Vect n Bool, Vect n Nat)
stepAllAutoResetTrunc [] [] []                        = ([], [], [], [])
stepAllAutoResetTrunc (s :: ss) (a :: as) (sl :: sls) =
  case aStep s a of
    (r, s', outcome, _) =>
      let natTerm = case outcome of
                      Terminated => True
                      _          => False
          truncate = sl == 1
          isDone   = natTerm || truncate
          nextS    = if isDone then MkA 0.0 0.0 0.0 0.0 else s'
          nextSl   = if isDone then EpisodeLen else sl `minus` 1
      in case stepAllAutoResetTrunc ss as sls of
           (rest, rs, ds, sls') => (nextS :: rest, r :: rs, isDone :: ds, nextSl :: sls')

mkRollSteps : Vect n (Vect ObsDim Double) -> Vect n Nat -> Vect n Double ->
              Vect n Double -> Vect n Double -> Vect n Bool ->
              Vect n RollStep
mkRollSteps [] [] [] [] [] []                                             = []
mkRollSteps (o :: os) (a :: as) (lp :: lps) (v :: vs) (r :: rs) (d :: ds) =
  MkRS o a lp v r d :: mkRollSteps os as lps vs rs ds

||| Batched per-env rollout. Each env steps RolloutLen times in lockstep;
||| one batched (actor, critic) forward per timestep. Threads per-env
||| stepsLeft for truncation. Done envs auto-reset.
rolloutBatchedL : {n : Nat} -> (1 _ : Actor) -> (1 _ : Critic) ->
                  VecEnv n AState -> Vect n Nat -> Nat ->
                  L IO {use = 1} (LPair (!* (Vect n (List RollStep), VecEnv n AState, Vect n Nat))
                                       (LPair Actor Critic))
rolloutBatchedL actor critic v0 sl0 rolloutLen = do
  (MkBang (envs', sls', stepLists) # (actor' # critic')) <- go rolloutLen actor critic v0.envs sl0 (replicate n [])
  pure1 (MkBang (map reverse stepLists, MkVecEnv envs', sls') # (actor' # critic'))
  where
    mapIdx : (Nat -> a -> b) -> Vect k a -> Vect k b
    mapIdx _ []        = []
    mapIdx f (x :: xs) = f 0 x :: mapIdx (\i, v => f (S i) v) xs

    go : Nat -> (1 _ : Actor) -> (1 _ : Critic) -> Vect n AState -> Vect n Nat -> Vect n (List RollStep) ->
         L IO {use = 1} (LPair (!* (Vect n AState, Vect n Nat, Vect n (List RollStep))) (LPair Actor Critic))
    go Z actor critic envs sls accs     = pure1 (MkBang (envs, sls, accs) # (actor # critic))
    go (S k) actor critic envs sls accs = withNoGradL {ex=Ex} $ do
      -- Per-step no-grad bracket: free this step's forward intermediates.
      stateV <- liftIO1 (ioRerun (\_ =>
        the (Tensor [n, ObsDim] Ex F WithGrad)
            (MkTensor (bulkToTensor2d {ex=Ex} {dt=F} (map (\s => obsTensor (observeVec s)) envs)) Nothing)))
      (MkBang logitsV # actor') <- forwardSeq {b=n} actor stateV
      let logProbsV = the (Tensor [n, NumActions] Ex F WithGrad)
                        (MkTensor (primLogSoftmax2d {ex=Ex} logitsV.tensorPtr) Nothing)
      (MkBang valuesV # critic') <- forwardSeq {b=n} critic stateV
      (acts, lps) <- liftIO1 (sampleActionFromBatch logProbsV envs)
      let valueRows : Vect n Double
          valueRows = mapIdx (\i, _ => primItem2d {ex=Ex} valuesV.tensorPtr (cast i) 0) envs
          obsVects : Vect n (Vect ObsDim Double)
          obsVects = map observeVec envs
      case stepAllAutoResetTrunc envs acts sls of
        (envs', rewards, dones, sls') =>
          let newSteps = mkRollSteps obsVects acts lps valueRows rewards dones
              accs' = zipWith (\acc, s => s :: acc) accs newSteps
          in go k actor' critic' envs' sls' accs'

----------------------------------------------------------------------
-- GAE + advantage normalization
----------------------------------------------------------------------

computeBootstrapL : (1 _ : Critic) -> List RollStep -> AState ->
                    L IO {use = 1} (LPair (!* Double) Critic)
computeBootstrapL critic [] _          = pure1 (MkBang 0.0 # critic)
computeBootstrapL critic steps finalSt =
  case last' steps of
    Nothing => pure1 (MkBang 0.0 # critic)
    Just ls => if ls.isDone then pure1 (MkBang 0.0 # critic)
                            else criticValueL critic (observeVec finalSt)

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
-- Per-step PPO loss (categorical policy)
----------------------------------------------------------------------

clipScalar : Double -> Double -> Double -> Double
clipScalar lo hi x = if x < lo then lo else if x > hi then hi else x

perStepLoss : {n : Nat} -> (logitsB : Tensor [n, NumActions] Ex F WithGrad) ->
              (valueB : Tensor [n, 1] Ex F WithGrad) -> (rowIdx : Int) ->
              Double -> Double -> Double ->
              (RollStep, Double, Double) -> IO (Tensor [] Ex F WithGrad)
perStepLoss logitsB valueB rowIdx clipEps entropyCoef valueCoef (step, adv, retT) = do
  logitsRow <- trowSelect logitsB rowIdx
  let logPT = the (Tensor [NumActions] Ex F WithGrad)
                  (MkTensor (primLogSoftmax {ex=Ex} logitsRow.tensorPtr 0) Nothing)
      aIdx : Int
      aIdx = cast {to=Int} (cast {to=Integer} step.action)
  lpNew <- telemSelect logPT aIdx
  let lpVal = primItem1d {ex=Ex} logPT.tensorPtr aIdx
  valueRow <- trowSelect valueB rowIdx
  valueV   <- telemSelect valueRow 0
  oldLPT   <- tconstScalar step.oldLogProb
  diffLP   <- tsub lpNew oldLPT
  let ratioVal = Prelude.exp (lpVal - step.oldLogProb)
  ratioV   <- texp diffLP
  surr1    <- tmulScalar ratioV adv
  let surr1Val = ratioVal * adv
      clipped  = clipScalar (1.0 - clipEps) (1.0 + clipEps) ratioVal
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
  let negEntV = the (Tensor [] Ex F WithGrad)
                    (MkTensor (primAdd {ex=Ex}
                              (primAdd {ex=Ex} (primMul {ex=Ex} p0V.tensorPtr lp0V.tensorPtr)
                                         (primMul {ex=Ex} p1V.tensorPtr lp1V.tensorPtr))
                              (primMul {ex=Ex} p2V.tensorPtr lp2V.tensorPtr))
                            Nothing)
  entTerm <- tmulScalar negEntV entropyCoef
  pure (MkTensor (primAdd {ex=Ex} (primAdd {ex=Ex} policyT.tensorPtr valueTerm.tensorPtr)
                       entTerm.tensorPtr) Nothing)

meanScalarLoss : (n : Nat) -> List (Tensor [] Ex F WithGrad) -> IO (Tensor [] Ex F WithGrad)
meanScalarLoss n losses = do
  zero <- tconstScalar 0.0
  let summed = foldl (\a, b => MkTensor (primAdd {ex=Ex} a.tensorPtr b.tensorPtr) Nothing) zero losses
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

-- Actor + critic are **linear** fields; the env / steps-left IORefs are ω.
record PPOState where
  constructor MkPPO
  1 actor  : Actor
  1 critic : Critic
  envRef   : IORef (VecEnv NumEnvs AState)
  stepsRef : IORef (Vect NumEnvs Nat)

-- Per-env bootstrap: critic value at each env's final state, threading the
-- (linear) critic across envs.
computeBootstrapsBatchedL : (1 _ : Critic) -> Vect n (List RollStep) -> VecEnv n AState ->
                            L IO {use = 1} (LPair (!* (Vect n Double)) Critic)
computeBootstrapsBatchedL critic stepLists v = batchOver critic stepLists v.envs
  where
    batchOver : (1 _ : Critic) -> Vect k (List RollStep) -> Vect k AState ->
                L IO {use = 1} (LPair (!* (Vect k Double)) Critic)
    batchOver critic [] []                     = pure1 (MkBang [] # critic)
    batchOver critic (steps :: rest) (s :: ss) = do
      (MkBang b # critic') <- computeBootstrapL critic steps s
      (MkBang bs # critic'') <- batchOver critic' rest ss
      pure1 (MkBang (b :: bs) # critic'')

-- Batched prepareRollout: per-env GAE → concat → normalize advantages
-- across the whole flat batch. Threads the critic (bootstrap forwards).
prepareRolloutBatchedL : (1 _ : Critic) -> Config -> Vect n (List RollStep) ->
                         VecEnv n AState ->
                         L IO {use = 1} (LPair (!* (List (RollStep, Double, Double))) Critic)
prepareRolloutBatchedL critic cfg stepLists finalEnvs = do
  (MkBang bootstraps # critic') <- computeBootstrapsBatchedL critic stepLists finalEnvs
  let mergedPerEnv : Vect n (List (RollStep, Double, Double))
      mergedPerEnv = zipWith
        (\steps, boot =>
          let triples = map stepTriple steps
              gaeOut = gae cfg.gamma cfg.lam boot triples
          in map flattenTriple (zip steps gaeOut))
        stepLists bootstraps
      flatMerged = concat (toList mergedPerEnv)
  pure1 (MkBang (normAdvs flatMerged) # critic')

-- Stack mini-batch obs into [B, ObsDim], one batched actor + critic
-- forward each, then per-sample loss expressions indexed into the
-- [B, NumActions] / [B, 1] tensors. One Adam steps both nets.
runBatchL : Optimizer Ex -> (1 _ : Actor) -> (1 _ : Critic) -> Config ->
            List (RollStep, Double, Double) -> L IO {use = 1} (LPair Actor Critic)
runBatchL opt actor critic cfg batch = withGenFreeL {ex=Ex} $ do
  -- Per-minibatch generation bracket: free this update's grad intermediates.
  let batchVec = Data.Vect.fromList batch
      n        = length batch
      obsBatch = the (Vect (length batch) (Vector ObsDim Double))
                     (map (\(s, _, _) => obsTensor s.obs) batchVec)
  stackedV <- liftIO1 (ioRerun (\_ =>
    the (Tensor [n, ObsDim] Ex F WithGrad) (MkTensor (bulkToTensor2d {ex=Ex} {dt=F} obsBatch) Nothing)))
  (MkBang logitsB # actor') <- forwardSeq {b=n} actor stackedV
  (MkBang valueB # critic') <- forwardSeq {b=n} critic stackedV
  loss <- liftIO1 $ do
            losses <- enumeratedLosses logitsB valueB batchVec 0
            meanScalarLoss n losses
  _ <- liftIO1 (trainStep opt loss)
  pure1 (actor' # critic')
  where
    enumeratedLosses : {n : Nat} -> Tensor [n, NumActions] Ex F WithGrad ->
                       Tensor [n, 1] Ex F WithGrad ->
                       Vect k (RollStep, Double, Double) -> Int ->
                       IO (List (Tensor [] Ex F WithGrad))
    enumeratedLosses _ _ [] _            = pure []
    enumeratedLosses lB vB (t :: rest) k = do
      l  <- perStepLoss lB vB k cfg.clipEps cfg.entropyCoef cfg.valueCoef t
      ls <- enumeratedLosses lB vB rest (k + 1)
      pure (l :: ls)

-- Iterate runBatchL over mini-batches, threading both nets.
runBatchesL : Optimizer Ex -> (1 _ : Actor) -> (1 _ : Critic) -> Config ->
              List (List (RollStep, Double, Double)) -> L IO {use = 1} (LPair Actor Critic)
runBatchesL _ actor critic _ []              = pure1 (actor # critic)
runBatchesL opt actor critic cfg (b :: rest) = do
  (actor' # critic') <- runBatchL opt actor critic cfg b
  runBatchesL opt actor' critic' cfg rest

kEpochUpdateL : Optimizer Ex -> (1 _ : Actor) -> (1 _ : Critic) -> Config ->
                List (RollStep, Double, Double) -> Nat -> L IO {use = 1} (LPair Actor Critic)
kEpochUpdateL _ actor critic _ _ Z               = pure1 (actor # critic)
kEpochUpdateL opt actor critic cfg prepped (S k) = do
  shuffled <- liftIO1 (shuffleIO prepped)
  let batches = chunksOf BatchSize shuffled
  (actor' # critic') <- runBatchesL opt actor critic cfg batches
  kEpochUpdateL opt actor' critic' cfg prepped k

ppoEpochL : Optimizer Ex -> Config -> (1 _ : PPOState) -> L IO {use = 1} (LPair (!* Double) PPOState)
ppoEpochL opt cfg (MkPPO actor critic envRef stepsRef) = do
  startEnvs <- liftIO1 (readIORef envRef)
  startSls  <- liftIO1 (readIORef stepsRef)
  (MkBang (stepLists, finalEnvs, finalSls) # (actor' # critic')) <-
    rolloutBatchedL actor critic startEnvs startSls RolloutLen
  liftIO1 (writeIORef envRef finalEnvs)
  liftIO1 (writeIORef stepsRef finalSls)
  (MkBang prepped # critic'') <-
    withNoGradL {ex=Ex} (prepareRolloutBatchedL critic' cfg stepLists finalEnvs)
  (actor'' # critic''') <- kEpochUpdateL opt actor' critic'' cfg prepped cfg.kEpochs
  let allReturns = concat (toList (map computeEpisodeReturns stepLists))
      nEp    = length allReturns
      sumEp  = sum allReturns
      sumRew = sum (map (\steps => sum (map (\s => s.reward) steps)) (toList stepLists))
      avgEp  = if nEp > 0
              then sumEp / cast (natToInteger nEp)
              else sumRew / cast (natToInteger NumEnvs)
  pure1 (MkBang (negate avgEp) # MkPPO actor'' critic''' envRef stepsRef)
  where
    computeEpisodeReturns : List RollStep -> List Double
    computeEpisodeReturns = go 0.0 []
      where
        go : Double -> List Double -> List RollStep -> List Double
        go _ acc []            = reverse acc
        go run acc (s :: rest) =
          if s.isDone
            then go 0.0 ((run + s.reward) :: acc) rest
            else go (run + s.reward) acc rest

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
      l2 = primItem2d {ex=Ex} logits.tensorPtr 0 2
  pure1 (MkBang (if l0 >= l1 && l0 >= l2 then the Nat 0
                 else if l1 >= l2 then 1
                 else 2) # actor')

evalEpL : (1 _ : Actor) -> AState -> Nat -> Double -> L IO {use = 1} (LPair (!* Double) Actor)
evalEpL actor _ Z acc      = pure1 (MkBang acc # actor)
evalEpL actor st (S k) acc = do
  (MkBang a # actor') <- greedyActL actor (observeVec st)
  case aStep st a of
    (r, st', outcome, _) =>
      case outcome of
        Terminated => pure1 (MkBang (acc + r) # actor')
        _          => evalEpL actor' st' k (acc + r)

evalNL : (1 _ : Actor) -> Nat -> Double -> L IO {use = 1} (LPair (!* Double) Actor)
evalNL actor Z acc     = pure1 (MkBang acc # actor)
evalNL actor (S k) acc = do
  (MkBang v # actor') <- evalEpL actor (MkA 0.0 0.0 0.0 0.0) EpisodeLen 0.0
  evalNL actor' k (acc + v)

----------------------------------------------------------------------
-- State construction / eval / discard (linear)
----------------------------------------------------------------------

buildStateL : L IO {use = 1} PPOState
buildStateL = do
  actor  <- runInitL mkActor
  critic <- runInitL mkCritic
  liftIO1 (maybeDumpInit {ex = ExampleExecutor})
  resetSeedI <- liftIO1 randomInt32
  let initEnvs : VecEnv NumEnvs AState
      initEnvs = fst (resetAll {state=AState} {action=Nat} {obs=Vect 6 Double}
                              (cast resetSeedI))
  envRef   <- liftIO1 (newIORef initEnvs)
  stepsRef <- liftIO1 (newIORef (the (Vect NumEnvs Nat) (replicate NumEnvs EpisodeLen)))
  pure1 (MkPPO actor critic envRef stepsRef)

discardStateL : (1 _ : PPOState) -> L IO ()
discardStateL (MkPPO actor critic _ _) = do
  discard actor
  discard critic

finalReportL : Config -> Nat -> (1 _ : PPOState) -> L IO ()
finalReportL cfg epochsDone (MkPPO actor critic _ _) = do
  let nEval = the Nat 20
  (MkBang evalSum # actor') <- withNoGradL {ex=Ex} (evalNL actor nEval 0.0)
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
lrFindCfg = { numIters := 30 } defaultLrFindConfig

-- Terminal linear consumer of the lrFind result. A named function with an
-- explicit `(1 _ : LPair ...)` signature so the bind continuation is linear
-- (the inline do-notation `<-` doesn't get recognised as linear for `lrFind`).
finishLrFind : (1 _ : LPair (!* LrFindResult) PPOState) -> L IO ()
finishLrFind (MkBang _ # st') = do
  discardStateL st'
  liftIO1 $ do
    putStrLn ""
    putStrLn "Done — re-run without --lr-find at the recommended LR."

runLrFind : Config -> IO ()
runLrFind cfg = Control.Linear.LIO.run $ do
  st0 <- buildStateL
  opt <- liftIO1 (adam cfg.lr ({ clip := NormClip 0.5 } defaultOpts))
  (LIO.(>>=))
    (lrFind {ex = Ex} {model = PPOState} {dp = ()} lrFindCfg
       (\s, _ => ppoEpochL opt cfg s) (pure ()) opt st0)
    finishLrFind

runTrain : Config -> IO ()
runTrain cfg = Control.Linear.LIO.run $ do
  st0 <- buildStateL
  -- Single Adam over both actor + critic (all params registered).
  opt <- liftIO1 (adam cfg.lr ({ clip := NormClip 0.5 } defaultOpts))
  metrics <- liftIO1 (newRLMetricsState 50)
  let trainCfg : TrainConfig PPOState
      trainCfg = { metricsL := readRLMetrics "recent_50" metrics }
                   (mkTrainConfig cfg.epochs 10 NoEarlyStop
                      (const (pure (the (List (String, String)) []))) (\_ => pure ()))
  (MkBang (epochsDone, _) # trained) <- fit {batch = ()}
    (\s, _ => do
       (MkBang loss # s') <- ppoEpochL opt cfg s
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

  putStrLn ""

  if cfg.lrFind then runLrFind cfg else runTrain cfg
