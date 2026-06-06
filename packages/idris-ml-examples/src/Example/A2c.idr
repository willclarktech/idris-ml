module Example.A2c

import Data.List
import Data.Vect
import Data.IORef
import System
import Compat.Random

import Floating
import Gym.ClassicControl.CartPole
import Gym.Env
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
  let stateT  = bulkToTensor {d=ExampleExecutor} {dt=ExampleDType} (obsTensor obs)
      stateV  = the (TVec ObsDim ExampleExecutor ExampleDType WithGrad) (MkTensor stateT Nothing)
  (_, logitsV) <- forwardVar actor stateV
  let logPT   = primLogSoftmax {d=ExampleExecutor} logitsV.tensorPtr 0
      lp0     = primItem1d {d=ExampleExecutor} logPT 0
      lp1     = primItem1d {d=ExampleExecutor} logPT 1
  (_, valueV) <- forwardVar critic stateV
  let v       = primItem1d {d=ExampleExecutor} valueV.tensorPtr 0
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
-- GAE helpers (pure Double — no autograd)
----------------------------------------------------------------------

bootstrapV : Critic -> Vect ObsDim Double -> IO Double
bootstrapV critic obs = do
  let stateV = the (TVec ObsDim ExampleExecutor ExampleDType WithGrad) (MkTensor (bulkToTensor {d=ExampleExecutor} {dt=ExampleDType} (obsTensor obs)) Nothing)
  (_, valueV) <- forwardVar critic stateV
  pure (primItem1d {d=ExampleExecutor} valueV.tensorPtr 0)

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
                 (MkTensor (primLogSoftmax {d=ExampleExecutor} logitsRow.tensorPtr 0) Nothing)
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
                  (primAdd {d=ExampleExecutor} (primMul {d=ExampleExecutor} p0V.tensorPtr lp0V.tensorPtr)
                             (primMul {d=ExampleExecutor} p1V.tensorPtr lp1V.tensorPtr))
                  Nothing)
  entTerm <- tmulScalar negEntV entropyCoef
  pure (MkTensor (primAdd {d=ExampleExecutor} (primAdd {d=ExampleExecutor} policyT.tensorPtr valueTerm.tensorPtr)
                          entTerm.tensorPtr) Nothing)


aggregateLoss : List (Tensor [] ExampleExecutor ExampleDType WithGrad) -> IO (Tensor [] ExampleExecutor ExampleDType WithGrad)
aggregateLoss losses = do
  zero <- tconstScalar 0.0
  let summed = foldl (\a, b => MkTensor (primAdd {d=ExampleExecutor} a.tensorPtr b.tensorPtr) Nothing) zero losses
      n = the Double (cast (natToInteger (length losses)))
  tmulScalar summed (1.0 / n)


-- Pair each rollout step (after GAE + advantage normalization) with its
-- batch row index, then build one batched actor + critic forward and
-- index into the resulting [B, NumActions] / [B, 1] tensors per-sample.
-- Caller supplies `bootstrap` precomputed (intended to be done in
-- withNoGrad — the bootstrap forward through critic doesn't need grad
-- tracking, the value is just a Double consumed by GAE).
buildLoss : Actor -> Critic -> Double -> Double -> Double -> Double ->
            Double -> List RollStep -> IO (Tensor [] ExampleExecutor ExampleDType WithGrad)
buildLoss actor critic gamma lam entropyCoef valueCoef bootstrap steps = do
  let triples = map stepTriple steps
      gaeOut = gae gamma lam bootstrap triples
      merged = map flattenTriple (zip steps gaeOut)
      normalized = normAdvs merged
      normVec = Data.Vect.fromList normalized
      n = length normalized
      obsBatch = the (Vect (length normalized) (Vector ObsDim Double))
                     (map (\(s, _, _) => obsTensor s.obs) normVec)
      stackedT = bulkToTensor2d {d=ExampleExecutor} {dt=ExampleDType} obsBatch
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


----------------------------------------------------------------------
-- Config + epoch
----------------------------------------------------------------------

record A2CState where
  constructor MkA2C
  actor  : Actor
  critic : Critic
  envRef : IORef CPState
  retRef : IORef Double

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

a2cEpoch : NativeOptimizer ExampleExecutor -> Config -> A2CState -> IO (A2CState, Double)
a2cEpoch opt cfg st = do
  startSt <- readIORef st.envRef
  -- Rollout-phase forward only extracts logits/values as Doubles
  -- (for sampling + bootstrap). The grad path is rebuilt fresh in
  -- buildLoss's batched forward, so the rollout's per-step forward
  -- doesn't need autograd tracking. withNoGrad skips tape append
  -- (tape/mlx) and disables libtorch's autograd graph (torch).
  rolled <- withNoGrad {d=ExampleExecutor} (rollout st.actor st.critic startSt RolloutLen)
  let steps = fst rolled
      finalSt = snd rolled
  writeIORef st.envRef finalSt
  -- Bootstrap forward (one critic forward on finalSt) doesn't need
  -- grad either — GAE consumes the value as a Double. Pull it out
  -- of buildLoss and run inside withNoGrad like the rollout.
  bootstrap <- withNoGrad {d=ExampleExecutor} (computeBootstrap st.critic steps finalSt)
  loss <- buildLoss st.actor st.critic cfg.gamma cfg.lam
                       cfg.entropyCoef cfg.valueCoef bootstrap steps
  _ <- nativeTrainStep opt loss

  let sumRew = sum (map (\s => s.reward) steps)
      wasDone = lastTerminated steps
  runRet <- readIORef st.retRef
  let newRet = runRet + sumRew
      reported = if wasDone then newRet else sumRew
  writeIORef st.retRef (if wasDone then 0.0 else newRet)
  pure (st, negate reported)


----------------------------------------------------------------------
-- Greedy evaluation
----------------------------------------------------------------------

greedyAct : Actor -> Vect ObsDim Double -> IO Nat
greedyAct actor obs = do
  let stateV = the (TVec ObsDim ExampleExecutor ExampleDType WithGrad) (MkTensor (bulkToTensor {d=ExampleExecutor} {dt=ExampleDType} (obsTensor obs)) Nothing)
  (_, logits) <- forwardVar actor stateV
  let l0 = primItem1d {d=ExampleExecutor} logits.tensorPtr 0
      l1 = primItem1d {d=ExampleExecutor} logits.tensorPtr 1
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
  envRef <- newIORef (the CPState (MkCP 0 0 0 0))
  retRef <- newIORef (the Double 0.0)
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
  (trained, epochsDone, _) <- runTrainingIO {d=ExampleExecutor}
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
  evalSum <- withNoGrad {d=ExampleExecutor} (evalN trained.actor nEval 0.0)
  let avgReturn = evalSum / cast (natToInteger nEval)
  putStrLn $ "Eval (" ++ show nEval ++ " episodes, greedy): avg_return=" ++ show avgReturn
  putStrLn ""
  putStrLn $ formatResult [("avg_return", show avgReturn),
                            ("epochs", show epochsDone),
                            ("seed", show cfg.seed)]
