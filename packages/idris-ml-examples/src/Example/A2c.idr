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
import Tensor
import Train
import Util
import Device
import Variable


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
Actor = Network ObsDim [Hidden, Hidden, Hidden, Hidden] NumActions CPU

Critic : Type
Critic = Network ObsDim [Hidden, Hidden, Hidden, Hidden] 1 CPU

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
obsTensor v = VTensor (map STensor v)


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
  let stateT  = bulkToTensor (obsTensor obs)
      stateV  = the (TVec ObsDim CPU) (MkVar stateT Nothing)
      logitsV = snd (forwardVar actor stateV)
      logPT   = prim__logSoftmax logitsV.tensorPtr 0
      lp0     = prim__item1d logPT 0
      lp1     = prim__item1d logPT 1
      valueV  = snd (forwardVar critic stateV)
      v       = prim__item1d valueV.tensorPtr 0
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

bootstrapV : Critic -> Vect ObsDim Double -> Double
bootstrapV critic obs =
  let stateV = the (TVec ObsDim CPU) (MkVar (bulkToTensor (obsTensor obs)) Nothing)
      valueV = snd (forwardVar critic stateV)
  in prim__item1d valueV.tensorPtr 0

computeBootstrap : Critic -> List RollStep -> CPState -> Double
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
-- Per-step A2C loss ( typed-surface, autograd-tracked)
----------------------------------------------------------------------

perStepLoss : {n : Nat} -> (logitsB : Variable [n, NumActions] CPU) ->
              (valuesB : Variable [n, 1] CPU) -> (rowIdx : Int) ->
              Double -> Double ->
              (RollStep, Double, Double) -> Variable [] CPU
perStepLoss logitsB valuesB rowIdx entropyCoef valueCoef (step, adv, retT) =
  let logitsRow = the (TVec NumActions CPU) (trowSelect logitsB rowIdx)
      logPT = the (Variable [NumActions] CPU)
                 (MkVar (prim__logSoftmax logitsRow.tensorPtr 0) Nothing)
      aIdx : Int
      aIdx = cast {to=Int} (cast {to=Integer} step.action)
      logProbV = the (Variable [] CPU) (telemSelect logPT aIdx)

      valueRow = the (TVec 1 CPU) (trowSelect valuesB rowIdx)
      valueV = the (Variable [] CPU) (telemSelect valueRow 0)

      retC = the (Variable [] CPU) (tconstScalar retT)

      -- Policy gradient: -logπ(a|s) * advantage. `adv` is a fixed Double
      -- (no grad path back to the value head); just scale logProbV by -adv.
      policyT = tmulScalar logProbV (negate adv)

      -- Value loss: valueCoef * (V(s) - return)^2
      diff = tsub valueV retC
      valueTerm = tmulScalar (tmul diff diff) valueCoef

      -- Entropy bonus: -entropyCoef * H(π) where H(π) = -Σ p_i log p_i.
      -- Build (-H(π)) as Σ p_i log p_i using grad-tracked Variable arithmetic.
      lp0V = the (Variable [] CPU) (telemSelect logPT 0)
      lp1V = the (Variable [] CPU) (telemSelect logPT 1)
      p0V = texp lp0V
      p1V = texp lp1V
      negEntV = the (Variable [] CPU) (MkVar
                  (prim__add (prim__mul p0V.tensorPtr lp0V.tensorPtr)
                             (prim__mul p1V.tensorPtr lp1V.tensorPtr))
                  Nothing)
      entTerm = tmulScalar negEntV entropyCoef
  in MkVar (prim__add (prim__add policyT.tensorPtr valueTerm.tensorPtr)
                       entTerm.tensorPtr) Nothing


aggregateLoss : List (Variable [] CPU) -> Variable [] CPU
aggregateLoss losses =
  let zero = tconstScalar 0.0
      summed = foldl (\a, b => MkVar (prim__add a.tensorPtr b.tensorPtr) Nothing) zero losses
      n = the Double (cast (natToInteger (length losses)))
  in tmulScalar summed (1.0 / n)


-- Pair each rollout step (after GAE + advantage normalization) with its
-- batch row index, then build one batched actor + critic forward and
-- index into the resulting [B, NumActions] / [B, 1] tensors per-sample.
buildLoss : Actor -> Critic -> Double -> Double -> Double -> Double ->
            List RollStep -> CPState -> Variable [] CPU
buildLoss actor critic gamma lam entropyCoef valueCoef steps finalSt =
  let bootstrap = computeBootstrap critic steps finalSt
      triples = map stepTriple steps
      gaeOut = gae gamma lam bootstrap triples
      merged = map flattenTriple (zip steps gaeOut)
      normalized = normAdvs merged
      normVec = Data.Vect.fromList normalized
      n = length normalized
      obsBatch = the (Vect (length normalized) (Vector ObsDim Double))
                     (map (\(s, _, _) => obsTensor s.obs) normVec)
      stackedT = bulkToTensor2d obsBatch
      stackedV = the (Variable [n, ObsDim] CPU) (MkVar stackedT Nothing)
      logitsB = snd (forwardVarBatch actor stackedV)
      valuesB = snd (forwardVarBatch critic stackedV)
      losses = the (List (Variable [] CPU)) (enumeratedLosses logitsB valuesB normVec 0)
  in aggregateLoss losses
  where
    enumeratedLosses : {n : Nat} -> Variable [n, NumActions] CPU ->
                       Variable [n, 1] CPU ->
                       Vect k (RollStep, Double, Double) -> Int ->
                       List (Variable [] CPU)
    enumeratedLosses _ _ [] _ = []
    enumeratedLosses lB vB (t :: rest) k =
      perStepLoss lB vB k entropyCoef valueCoef t :: enumeratedLosses lB vB rest (k + 1)


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

a2cEpoch : NativeOptimizer -> Config -> A2CState -> IO (A2CState, Double)
a2cEpoch opt cfg st = do
  startSt <- readIORef st.envRef
  rolled <- rollout st.actor st.critic startSt RolloutLen
  let steps = fst rolled
      finalSt = snd rolled
  writeIORef st.envRef finalSt
  let loss = buildLoss st.actor st.critic cfg.gamma cfg.lam
                       cfg.entropyCoef cfg.valueCoef steps finalSt
  _ <- pure (nativeTrainStep opt loss)

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

greedyAct : Actor -> Vect ObsDim Double -> Nat
greedyAct actor obs =
  let stateV = the (TVec ObsDim CPU) (MkVar (bulkToTensor (obsTensor obs)) Nothing)
      logits = snd (forwardVar actor stateV)
      l0 = prim__item1d logits.tensorPtr 0
      l1 = prim__item1d logits.tensorPtr 1
  in if l0 >= l1 then 0 else 1

evalEp : Actor -> CPState -> Nat -> Double -> Double
evalEp _ _ Z acc = acc
evalEp actor st (S k) acc =
  let a = greedyAct actor (observeVec st)
  in case cpStep st a of
       (r, st', outcome, _) =>
         if done outcome then acc + r
         else evalEp actor st' k (acc + r)

evalN : Actor -> Nat -> Double -> Double
evalN _ Z acc = acc
evalN actor (S k) acc =
  evalN actor k (acc + evalEp actor (MkCP 0 0 0 0) MaxSteps 0.0)


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

  let trainCfg : TrainConfig A2CState
      trainCfg = MkTrainConfig cfg.epochs 500 NoEarlyStop (const (pure [])) (\_ => pure ())
  (trained, epochsDone, _) <- runTrainingIO
    (\s, _ => a2cEpoch opt cfg s)
    (pure ())
    trainCfg st0

  putStrLn ""
  let nEval = the Nat 30
      avgReturn = evalN trained.actor nEval 0.0 / cast (natToInteger nEval)
  putStrLn $ "Eval (" ++ show nEval ++ " episodes, greedy): avg_return=" ++ show avgReturn
  putStrLn ""
  putStrLn $ formatResult [("avg_return", show avgReturn),
                            ("epochs", show epochsDone),
                            ("seed", show cfg.seed)]
