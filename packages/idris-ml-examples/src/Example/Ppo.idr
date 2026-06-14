module Example.Ppo

import Data.IORef
import Data.List
import Data.Vect
import System

import Array
import BuildConfig
import Compat.Random
import Floating
import Gym.ClassicControl.Acrobot
import Gym.Env
import Gym.Vector
import Hpo.LrFinder
import ML.Simple
import Math
import RL.Gae
import Sampler
import Train

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

criticValue : Critic -> Vect ObsDim Double -> IO Double
criticValue critic obs = do
  let stateV = the (Tensor [1, ObsDim] Ex F WithGrad)
                 (MkTensor (bulkToTensor2d {ex=Ex} {dt=F} [obsTensor obs]) Nothing)
  outV <- forwardSeq {b=1} critic stateV
  pure (primItem2d {ex=Ex} outV.tensorPtr 0 0)

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
    go _ [] = pure ([], [])
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

mkRollSteps : Vect n (Vect ObsDim Double) -> Vect n Nat -> Vect n Double ->
              Vect n Double -> Vect n Double -> Vect n Bool ->
              Vect n RollStep
mkRollSteps [] [] [] [] [] [] = []
mkRollSteps (o :: os) (a :: as) (lp :: lps) (v :: vs) (r :: rs) (d :: ds) =
  MkRS o a lp v r d :: mkRollSteps os as lps vs rs ds

||| Batched per-env rollout. Each env steps RolloutLen times in lockstep;
||| one batched (actor, critic) forward per timestep. Threads per-env
||| stepsLeft for truncation. Done envs auto-reset.
rolloutBatched : {n : Nat} -> Actor -> Critic ->
                 VecEnv n AState -> Vect n Nat -> Nat ->
                 IO (Vect n (List RollStep), VecEnv n AState, Vect n Nat)
rolloutBatched actor critic v0 sl0 rolloutLen = do
  (envs', sls', stepLists) <- go rolloutLen v0.envs sl0 (replicate n [])
  pure (map reverse stepLists, MkVecEnv envs', sls')
  where
    mapIdx : (Nat -> a -> b) -> Vect k a -> Vect k b
    mapIdx _ [] = []
    mapIdx f (x :: xs) = f 0 x :: mapIdx (\i, v => f (S i) v) xs

    go : Nat -> Vect n AState -> Vect n Nat -> Vect n (List RollStep) ->
         IO (Vect n AState, Vect n Nat, Vect n (List RollStep))
    go Z envs sls accs = pure (envs, sls, accs)
    go (S k) envs sls accs = withNoGrad {ex=Ex} $ do
      -- Per-step no-grad bracket: free this step's forward intermediates.
      let obsRows : Vect n (Vector ObsDim Double)
          obsRows = map (\s => obsTensor (observeVec s)) envs
          batchPtr = bulkToTensor2d {ex=Ex} {dt=F} obsRows
          stateV : Tensor [n, ObsDim] Ex F WithGrad
          stateV = MkTensor batchPtr Nothing
      logitsV <- forwardSeq {b=n} actor stateV
      let logProbsV = the (Tensor [n, NumActions] Ex F WithGrad)
                        (MkTensor (primLogSoftmax2d {ex=Ex} logitsV.tensorPtr) Nothing)
      valuesV <- forwardSeq {b=n} critic stateV
      (acts, lps) <- sampleActionFromBatch logProbsV envs
      let valueRows : Vect n Double
          valueRows = mapIdx (\i, _ => primItem2d {ex=Ex} valuesV.tensorPtr (cast i) 0) envs
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

record PPOState where
  constructor MkPPO
  actor    : Actor
  critic   : Critic
  envRef   : IORef (VecEnv NumEnvs AState)
  stepsRef : IORef (Vect NumEnvs Nat)

-- Per-env bootstrap: critic value at each env's final state, zeroed for
-- envs whose last step terminated.
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
-- across the whole flat batch.
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

-- Stack mini-batch obs into [B, ObsDim], one batched actor + critic
-- forward each, then per-sample loss expressions indexed into the
-- [B, NumActions] / [B, 1] tensors. One Adam steps both nets.
runBatch : Optimizer Ex -> Actor -> Critic -> Config ->
           List (RollStep, Double, Double) -> IO ()
runBatch opt actor critic cfg batch = withGenFree {ex=Ex} $ do
  -- Per-minibatch generation bracket: free this update's grad
  -- intermediates immediately (PPO runs K epochs × minibatches).
  let batchVec = Data.Vect.fromList batch
      n = length batch
      obsBatch = the (Vect (length batch) (Vector ObsDim Double))
                     (map (\(s, _, _) => obsTensor s.obs) batchVec)
      stackedT = bulkToTensor2d {ex=Ex} {dt=F} obsBatch
      stackedV = the (Tensor [n, ObsDim] Ex F WithGrad) (MkTensor stackedT Nothing)
  logitsB <- forwardSeq {b=n} actor stackedV
  valueB  <- forwardSeq {b=n} critic stackedV
  losses <- enumeratedLosses logitsB valueB batchVec 0
  loss <- meanScalarLoss n losses
  _ <- nativeTrainStep opt loss
  pure ()
  where
    enumeratedLosses : {n : Nat} -> Tensor [n, NumActions] Ex F WithGrad ->
                       Tensor [n, 1] Ex F WithGrad ->
                       Vect k (RollStep, Double, Double) -> Int ->
                       IO (List (Tensor [] Ex F WithGrad))
    enumeratedLosses _ _ [] _ = pure []
    enumeratedLosses lB vB (t :: rest) k = do
      l  <- perStepLoss lB vB k cfg.clipEps cfg.entropyCoef cfg.valueCoef t
      ls <- enumeratedLosses lB vB rest (k + 1)
      pure (l :: ls)

-- Iterate runBatch over mini-batches via do-block recursion (NOT
-- traverse_, whose `*>` desugaring crashes mlx after a tape reset).
runBatches : Optimizer Ex -> Actor -> Critic -> Config ->
             List (List (RollStep, Double, Double)) -> IO ()
runBatches _ _ _ _ [] = pure ()
runBatches opt actor critic cfg (b :: rest) = do
  runBatch opt actor critic cfg b
  runBatches opt actor critic cfg rest

kEpochUpdate : Optimizer Ex -> Actor -> Critic -> Config ->
               List (RollStep, Double, Double) -> Nat -> IO ()
kEpochUpdate _ _ _ _ _ Z = pure ()
kEpochUpdate opt actor critic cfg prepped (S k) = do
  shuffled <- shuffleIO prepped
  let batches = chunksOf BatchSize shuffled
  runBatches opt actor critic cfg batches
  kEpochUpdate opt actor critic cfg prepped k

ppoEpoch : Optimizer Ex -> Config -> PPOState -> IO (PPOState, Double)
ppoEpoch opt cfg st = do
  startEnvs <- readIORef st.envRef
  startSls  <- readIORef st.stepsRef
  rolled <- rolloutBatched st.actor st.critic startEnvs startSls RolloutLen
  let stepLists = fst rolled
      finalEnvs = fst (snd rolled)
      finalSls  = snd (snd rolled)
  writeIORef st.envRef finalEnvs
  writeIORef st.stepsRef finalSls

  prepped <- withNoGrad {ex=Ex} (prepareRolloutBatched st.critic cfg stepLists finalEnvs)
  kEpochUpdate opt st.actor st.critic cfg prepped cfg.kEpochs

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
  let stateV = the (Tensor [1, ObsDim] Ex F WithGrad)
                 (MkTensor (bulkToTensor2d {ex=Ex} {dt=F} [obsTensor obs]) Nothing)
  logits <- forwardSeq {b=1} actor stateV
  let l0 = primItem2d {ex=Ex} logits.tensorPtr 0 0
      l1 = primItem2d {ex=Ex} logits.tensorPtr 0 1
      l2 = primItem2d {ex=Ex} logits.tensorPtr 0 2
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

%default partial

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

  actor  <- mkActor
  critic <- mkCritic
  resetSeedI <- randomInt32
  let initEnvs : VecEnv NumEnvs AState
      initEnvs = fst (resetAll {state=AState} {action=Nat} {obs=Vect 6 Double}
                              (cast resetSeedI))
  envRef   <- newIORef initEnvs
  stepsRef <- newIORef (the (Vect NumEnvs Nat) (replicate NumEnvs EpisodeLen))
  let st0 = MkPPO actor critic envRef stepsRef
  -- Single Adam over both actor + critic (all params registered).
  opt <- adam cfg.lr ({ clip := NormClip 0.5 } defaultOpts)

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
  (trained, epochsDone, _) <- fit {batch = ()}
    (\s, _ => do
       (s', loss) <- ppoEpoch opt cfg s
       recordReturn metrics (negate loss)
       pure (s', loss))
    opt (generate (pure ()))
    trainCfg st0

  putStrLn ""
  let nEval = the Nat 20
  evalSum <- withNoGrad {ex=Ex} (evalN trained.actor nEval 0.0)
  let avgReturn = evalSum / cast (natToInteger nEval)
  putStrLn $ "Eval (" ++ show nEval ++ " episodes, greedy): avg_return=" ++ show avgReturn
  putStrLn ""
  putStrLn $ formatResult [("avg_return", show avgReturn),
                            ("epochs", show epochsDone),
                            ("seed", show cfg.seed)]
