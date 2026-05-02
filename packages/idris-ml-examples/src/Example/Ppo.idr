module Example.Ppo

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
import Math
import RL.Gae
import Sampler
import Tensor
import Train
import Util
import Device
import Variable


----------------------------------------------------------------------
-- Architecture: separate actor and critic MLPs + learnable state-
-- independent log_std (matching `torch_ref/models/ppo.py` exactly).
-- Actor:  Linear(3→64) → tanh → Linear(64→64) → tanh → Linear(64→1)  = mean
-- Critic: Linear(3→64) → tanh → Linear(64→64) → tanh → Linear(64→1)  = value
-- log_std is a standalone Variable parameter.
--
-- Actor and critic params are scoped via `autoNameScoped` (see A2c.idr
-- — same paramId-collision fix). Without scoping, both networks register
-- `ll0_weights` etc. and the second overwrites the first.
----------------------------------------------------------------------

-- --- Local autoName with scope prefix -------------------------------

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


-- --- Architecture ---------------------------------------------------

ObsDim : Nat; ObsDim = 3
Hidden : Nat; Hidden = 64
EpisodeLen : Nat; EpisodeLen = 200
RolloutLen : Nat; RolloutLen = 400
BatchSize : Nat; BatchSize = 64

Actor : Type
Actor = Network ObsDim [Hidden, Hidden, Hidden, Hidden] 1 (Variable CPU)

Critic : Type
Critic = Network ObsDim [Hidden, Hidden, Hidden, Hidden] 1 (Variable CPU)

mkActor : IO Actor
mkActor = do
  ll1 <- linearLayer {i=ObsDim} {o=Hidden}
  ll2 <- linearLayer {i=Hidden} {o=Hidden}
  ll3 <- linearLayer {i=Hidden} {o=1}
  pure (autoNameScoped "actor_"
    (ll1 ~> tanhLayer ~> ll2 ~> tanhLayer ~> OutputLayer ll3))

mkCritic : IO Critic
mkCritic = do
  ll1 <- linearLayer {i=ObsDim} {o=Hidden}
  ll2 <- linearLayer {i=Hidden} {o=Hidden}
  ll3 <- linearLayer {i=Hidden} {o=1}
  pure (autoNameScoped "critic_"
    (ll1 ~> tanhLayer ~> ll2 ~> tanhLayer ~> OutputLayer ll3))

mkLogStd : Variable CPU
mkLogStd = param "log_std" 0.0


----------------------------------------------------------------------
-- Observation helpers
----------------------------------------------------------------------

observeVec : PState -> Vect ObsDim Double
observeVec s = pObserve s

obsTensor : Vect ObsDim Double -> Vector ObsDim Double
obsTensor v = VTensor (map STensor v)


----------------------------------------------------------------------
-- Rollout record
----------------------------------------------------------------------

record RollStep where
  constructor MkRS
  obs        : Vect ObsDim Double
  action     : Double
  oldLogProb : Double
  value      : Double
  reward     : Double
  isDone     : Bool


----------------------------------------------------------------------
-- Gaussian policy helpers
----------------------------------------------------------------------

logTwoPiHalf : Double
logTwoPiHalf = 0.5 * Prelude.log (2.0 * 3.141592653589793)

gaussianLogProb : Double -> Double -> Double -> Double
gaussianLogProb mean logStd action =
  let std = Prelude.exp logStd
      z = (action - mean) / std
  in -0.5 * z * z - logStd - logTwoPiHalf

-- Scalar mean from actor.
actorMean : Actor -> Vect ObsDim Double -> Double
actorMean actor obs =
  let outT = snd (forwardVarTensor actor (bulkToTensor (obsTensor obs)))
  in prim__item1d outT 0

-- Scalar value from critic.
criticValue : Critic -> Vect ObsDim Double -> Double
criticValue critic obs =
  let outT = snd (forwardVarTensor critic (bulkToTensor (obsTensor obs)))
  in prim__item1d outT 0

sampleActionIO : Actor -> Critic -> Variable CPU -> Vect ObsDim Double ->
                 IO (Double, Double, Double)
sampleActionIO actor critic logStdV obs = do
  let mean   = actorMean actor obs
      v      = criticValue critic obs
      logStd = (refreshValue logStdV).value
  eps <- normalSample
  let std    = Prelude.exp logStd
      action = mean + std * eps
      lp     = gaussianLogProb mean logStd action
  pure (action, lp, v)


----------------------------------------------------------------------
-- Rollout: RolloutLen steps with auto-reset every EpisodeLen.
----------------------------------------------------------------------

rollout : Actor -> Critic -> Variable CPU -> PState -> Nat -> Nat ->
          IO (List RollStep, PState)
rollout _ _ _ st _ Z = pure ([], st)
rollout actor critic logStdV st stepsLeft (S k) = do
  let obs = observeVec st
  triple <- sampleActionIO actor critic logStdV obs
  let a  = fst triple
      lp = fst (snd triple)
      v  = snd (snd triple)
  case pStep st a of
    (r, st', _, _) => do
      let isBoundary = stepsLeft == 1
          stepRec = MkRS obs a lp v r isBoundary
          nextSt = if isBoundary then MkP 3.141592653589793 0.0 else st'
          nextStepsLeft = if isBoundary then EpisodeLen else stepsLeft `minus` 1
      recur <- rollout actor critic logStdV nextSt nextStepsLeft k
      pure (stepRec :: fst recur, snd recur)


----------------------------------------------------------------------
-- GAE + advantage normalization
----------------------------------------------------------------------

bootstrapV : Critic -> Vect ObsDim Double -> Double
bootstrapV critic obs = criticValue critic obs

computeBootstrap : Critic -> List RollStep -> PState -> Double
computeBootstrap _ [] _ = 0.0
computeBootstrap critic steps finalSt =
  case last' steps of
    Nothing => 0.0
    Just ls => if ls.isDone then 0.0 else bootstrapV critic (observeVec finalSt)

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
-- Per-step PPO loss
----------------------------------------------------------------------

clipScalar : Double -> Double -> Double -> Double
clipScalar lo hi x = if x < lo then lo else if x > hi then hi else x

-- Per-sample mean / value Variables are extracted from the batched
-- [B, 1] outputs via prim__select. The actor + critic forwards happen
-- once per mini-batch in `runBatch`, not once per transition.
perStepLoss : (meanB : AnyPtr) -> (valueB : AnyPtr) -> (rowIdx : Int) ->
              Variable CPU -> Double -> Double -> Double ->
              (RollStep, Double, Double) -> Variable CPU
perStepLoss meanB valueB rowIdx logStdV clipEps entropyCoef valueCoef (step, adv, retT) =
  let meanRow   = prim__select meanB 0 rowIdx     -- [1]
      meanPtr   = prim__select meanRow 0 0
      meanVal   = prim__item1d meanRow 0
      meanV     : Variable CPU
      meanV     = Var meanPtr Nothing meanVal

      valueRow  = prim__select valueB 0 rowIdx    -- [1]
      valuePtr  = prim__select valueRow 0 0
      valueVal  = prim__item1d valueRow 0
      valueV    : Variable CPU
      valueV    = Var valuePtr Nothing valueVal

      -- Gaussian log-prob with gradient flow through actor + logStdV:
      --   logπ = -0.5*((a-mean)/std)^2 - logStd - 0.5*log(2π)
      --        = -0.5*(a-mean)^2 * exp(-2*logStd) - logStd - c
      actC      : Variable CPU
      actC      = fromDouble step.action
      diffM     = actC - meanV
      halfC     : Variable CPU
      halfC     = fromDouble 0.5
      twoC      : Variable CPU
      twoC      = fromDouble 2.0
      zeroC     : Variable CPU
      zeroC     = fromDouble 0.0
      negTwoLs  = zeroC - twoC * logStdV
      varInv    = exp negTwoLs
      quadratic = halfC * diffM * diffM * varInv
      cC        : Variable CPU
      cC        = fromDouble logTwoPiHalf
      lpNew     = (zeroC - quadratic) - logStdV - cC

      lpOldC    : Variable CPU
      lpOldC    = fromDouble step.oldLogProb
      advC      : Variable CPU
      advC      = fromDouble adv

      diffLP    = lpNew - lpOldC
      ratioVal  = Prelude.exp diffLP.value
      ratioV    = exp diffLP
      surr1     = ratioV * advC

      clippedR  = clipScalar (1.0 - clipEps) (1.0 + clipEps) ratioVal
      surr2Val  = clippedR * adv
      surr1Val  = ratioVal * adv

      policyT   = if surr1Val <= surr2Val
                    then zeroC - surr1
                    else zeroC - fromDouble surr2Val

      retC      : Variable CPU
      retC      = fromDouble retT
      diffV     = valueV - retC
      valCoefC  : Variable CPU
      valCoefC  = fromDouble valueCoef
      valueTerm = valCoefC * halfC * diffV * diffV

      -- Entropy (Gaussian): H = 0.5*log(2πe) + logStd; subtract ent_coef*H.
      entCoefC  : Variable CPU
      entCoefC  = fromDouble entropyCoef
      entTerm   = zeroC - entCoefC * logStdV
  in policyT + valueTerm + entTerm


aggregateLoss : List (Variable CPU) -> Variable CPU
aggregateLoss losses =
  let zeroV  = the (Variable CPU) (fromDouble 0.0)
      sumV   = foldl (+) zeroV losses
      n      = the Double (cast (natToInteger (length losses)))
      nV     = the (Variable CPU) (fromDouble n)
  in sumV / nV


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
  epochs      : Nat       -- number of rollouts
  gamma       : Double
  lam         : Double
  clipEps     : Double
  kEpochs     : Nat
  entropyCoef : Double
  valueCoef   : Double
  seed        : Bits64

defaultConfig : Config
defaultConfig = MkConfig 3.0e-4 200 0.99 0.95 0.2 10 0.0 0.5 42

specs : List (ArgSpec Config)
specs = [ Arg "--lr" (\v, c => { lr := cast v } c)
        , Arg "--epochs" (\v, c => { epochs := castNat v } c)
        , Arg "--gamma" (\v, c => { gamma := cast v } c)
        , Arg "--lambda" (\v, c => { lam := cast v } c)
        , Arg "--clip" (\v, c => { clipEps := cast v } c)
        , Arg "--k-epochs" (\v, c => { kEpochs := castNat v } c)
        , Arg "--entropy" (\v, c => { entropyCoef := cast v } c)
        , Arg "--value-coef" (\v, c => { valueCoef := cast v } c)
        , Arg "--seed" (\v, c => { seed := castBits64 v } c)
        ]

record PPOState where
  constructor MkPPO
  actor   : Actor
  critic  : Critic
  logStdV : Variable CPU
  envRef  : IORef PState


prepareRollout : Critic -> Config -> List RollStep -> PState ->
                 List (RollStep, Double, Double)
prepareRollout critic cfg steps finalSt =
  let bootstrap = computeBootstrap critic steps finalSt
      triples   = map stepTriple steps
      gaeOut    = gae cfg.gamma cfg.lam bootstrap triples
      merged    = map flattenTriple (zip steps gaeOut)
  in normAdvs merged


-- Stack mini-batch obs into [B, ObsDim], do one batched actor + critic
-- forward each, then build per-sample loss expressions by indexing into
-- the [B, 1] mean / value tensors. Replaces O(B) per-sample
-- `forwardVarTensor` calls with two batched calls per mini-batch.
runBatch : NativeOptimizer -> Actor -> Critic -> Variable CPU -> Config ->
           List (RollStep, Double, Double) -> IO ()
runBatch opt actor critic logStdV cfg batch = do
  let batchVec  = Data.Vect.fromList batch
      n         = length batch
      obsBatch : Vect (length batch) (Vector ObsDim Double)
      obsBatch  = map (\(s, _, _) => obsTensor s.obs) batchVec
      stackedT  = bulkToTensor2d obsBatch
      meanB     = snd (forwardVarTensorBatch actor n stackedT)
      valueB    = snd (forwardVarTensorBatch critic n stackedT)
      losses    = enumeratedLosses meanB valueB batchVec 0
      loss      = aggregateLoss losses
  _ <- pure (nativeTrainStep opt loss)
  pure ()
  where
    enumeratedLosses : (meanB : AnyPtr) -> (valueB : AnyPtr) ->
                       Vect k (RollStep, Double, Double) -> Int ->
                       List (Variable CPU)
    enumeratedLosses _ _ [] _ = []
    enumeratedLosses mB vB (t :: rest) k =
      perStepLoss mB vB k logStdV cfg.clipEps cfg.entropyCoef cfg.valueCoef t
        :: enumeratedLosses mB vB rest (k + 1)


kEpochUpdate : NativeOptimizer -> Actor -> Critic -> Variable CPU -> Config ->
               List (RollStep, Double, Double) -> Nat -> IO ()
kEpochUpdate _ _ _ _ _ _ Z = pure ()
kEpochUpdate opt actor critic logStdV cfg prepped (S k) = do
  shuffled <- shuffleIO prepped
  let batches = chunksOf BatchSize shuffled
  traverse_ (runBatch opt actor critic logStdV cfg) batches
  kEpochUpdate opt actor critic logStdV cfg prepped k


ppoEpoch : NativeOptimizer -> Config -> PPOState -> IO (PPOState, Double)
ppoEpoch opt cfg st = do
  startSt <- readIORef st.envRef
  rolled  <- rollout st.actor st.critic st.logStdV startSt EpisodeLen RolloutLen
  let steps   = fst rolled
      finalSt = snd rolled
  writeIORef st.envRef finalSt

  let prepped = prepareRollout st.critic cfg steps finalSt
  kEpochUpdate opt st.actor st.critic st.logStdV cfg prepped cfg.kEpochs

  let sumRew = sum (map (\s => s.reward) steps)
      nEp    = length (filter (\s => s.isDone) steps)
      avgEp  = if nEp > 0 then sumRew / cast (natToInteger nEp) else sumRew
  pure (st, negate avgEp)


----------------------------------------------------------------------
-- Greedy evaluation
----------------------------------------------------------------------

greedyAct : Actor -> Vect ObsDim Double -> Double
greedyAct actor obs = actorMean actor obs

evalEp : Actor -> PState -> Nat -> Double -> Double
evalEp _ _ Z acc = acc
evalEp actor st (S k) acc =
  let a = greedyAct actor (observeVec st)
  in case pStep st a of
       (r, st', _, _) => evalEp actor st' k (acc + r)

evalN : Actor -> Nat -> Double -> Double
evalN _ Z acc = acc
evalN actor (S k) acc =
  evalN actor k (acc + evalEp actor (MkP 3.141592653589793 0.0) EpisodeLen 0.0)


----------------------------------------------------------------------
-- Main
----------------------------------------------------------------------

main : IO ()
main = do
  args <- getArgs
  let cfg = parseArgs defaultConfig specs (drop 1 args)
  srand cfg.seed

  putStrLn "=== PPO on Pendulum (separate actor + critic + log_std) ==="
  putStrLn $ "Config: lr=" ++ show cfg.lr
           ++ " epochs=" ++ show cfg.epochs
           ++ " rollout=" ++ show RolloutLen
           ++ " batch=" ++ show BatchSize
           ++ " gamma=" ++ show cfg.gamma
           ++ " lambda=" ++ show cfg.lam
           ++ " clip=" ++ show cfg.clipEps
           ++ " K=" ++ show cfg.kEpochs
           ++ " seed=" ++ show cfg.seed

  actor  <- mkActor
  critic <- mkCritic
  let logStdV = mkLogStd
  envRef <- newIORef (the PState (MkP 3.141592653589793 0.0))
  let st0 = MkPPO actor critic logStdV envRef
      opt = nativeAdamGlobalClip cfg.lr 0.9 0.999 1.0e-8 0.5

  putStrLn ""

  let trainCfg : TrainConfig PPOState
      trainCfg = MkTrainConfig cfg.epochs 10 NoEarlyStop (const (pure []))
  (trained, epochsDone, _) <- runTrainingIO
    (\s, _ => ppoEpoch opt cfg s)
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
