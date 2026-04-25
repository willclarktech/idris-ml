module Example.A2c

import Data.List
import Data.Vect
import Data.IORef
import System
import Compat.Random

import Endofunctor
import Floating
import Gym.ClassicControl.CartPole
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
-- Architecture: single combined actor-critic network. Output is a
-- 3-vector: [logit_0, logit_1, value]. Shared parameters end-to-end;
-- sidesteps the paramId-prefix problem that would arise from two
-- separately-autoNamed networks. A more orthodox A2C with a shared
-- trunk and separate heads would need a branching network type, which
-- isn't currently supported by the linear `Network` chain.
----------------------------------------------------------------------

ObsDim : Nat; ObsDim = 4
Hidden : Nat; Hidden = 64
NumActions : Nat; NumActions = 2
OutDim : Nat; OutDim = 3   -- NumActions logits + 1 value
MaxSteps : Nat; MaxSteps = cartPoleMaxSteps
RolloutLen : Nat; RolloutLen = 10

ACNet : Type
ACNet = Network ObsDim [Hidden, Hidden, Hidden, Hidden] OutDim (Variable CPU)


mkACNet : IO ACNet
mkACNet = do
  ll1 <- linearLayer {i=ObsDim} {o=Hidden}
  ll2 <- linearLayer {i=Hidden} {o=Hidden}
  ll3 <- linearLayer {i=Hidden} {o=OutDim}
  pure (autoName (ll1 ~> tanhLayer ~> ll2 ~> tanhLayer ~> OutputLayer ll3))


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
-- Forward: read out logits (indices 0,1) and value (index 2) via the
-- tensor forward. Scalar extraction via prim__item1d.
----------------------------------------------------------------------

-- Returns (logit0, logit1, value) for a given observation.
forwardAC : ACNet -> Vect ObsDim Double -> (Double, Double, Double)
forwardAC net obs =
  let outT = snd (forwardVarTensor net (bulkToTensor (obsTensor obs)))
      l0 = prim__item1d outT 0
      l1 = prim__item1d outT 1
      v  = prim__item1d outT 2
  in (l0, l1, v)


sampleActionIO : ACNet -> Vect ObsDim Double -> IO (Nat, Double)
sampleActionIO net obs = do
  let (l0, l1, v) = forwardAC net obs
      -- log_softmax over logits
      maxL = if l0 >= l1 then l0 else l1
      el0  = Prelude.exp (l0 - maxL)
      el1  = Prelude.exp (l1 - maxL)
      z    = el0 + el1
      p0   = el0 / z
      p1   = el1 / z
  u <- randomRIO (the Double 0.0, 1.0)
  let a = categoricalSample [p0, p1] u
  pure (a, v)


rollout : ACNet -> CPState -> Nat -> IO (List RollStep, CPState)
rollout _ st Z = pure ([], st)
rollout net st (S k) = do
  let obs = observeVec st
  pair <- sampleActionIO net obs
  let a = fst pair
      v = snd pair
  case cpStep st a of
    (r, st', outcome, _) => do
      let isDone = done outcome
          stepRec = MkRS obs a r v isDone
          nextSt = if isDone then MkCP 0 0 0 0 else st'
      recur <- rollout net nextSt k
      pure (stepRec :: fst recur, snd recur)


----------------------------------------------------------------------
-- GAE pipeline helpers (top-level to avoid do-block let quirks).
----------------------------------------------------------------------

bootstrapV : ACNet -> Vect ObsDim Double -> Double
bootstrapV net obs =
  let (_, _, v) = forwardAC net obs
  in v

computeBootstrap : ACNet -> List RollStep -> CPState -> Double
computeBootstrap _ [] _ = 0.0
computeBootstrap net steps finalSt =
  case last' steps of
    Nothing => 0.0
    Just ls => if ls.isDone then 0.0 else bootstrapV net (observeVec finalSt)

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
-- Per-step loss via Variable arithmetic.
--   policy: -logπ(a|s) * advantage
--   value : value_coef * (V(s) - return)^2
--   ent   : -entropy_coef * H(π)
----------------------------------------------------------------------

perStepLoss : ACNet -> Double -> Double ->
              (RollStep, Double, Double) -> Variable CPU
perStepLoss net entropyCoef valueCoef (step, adv, retT) =
  let stateT  = bulkToTensor (obsTensor step.obs)
      outT    = snd (forwardVarTensor net stateT)

      -- Slice out the logits (first 2 elements) and run log_softmax through
      -- the C op so gradients flow correctly back to both logits.
      logitsT = prim__narrow outT 0 0 2
      logPT   = prim__logSoftmax logitsT 0
      lp0Val  = prim__item1d logPT 0
      lp1Val  = prim__item1d logPT 1
      vVal    = prim__item1d outT 2

      aIdx : Int
      aIdx     = cast {to=Int} (cast {to=Integer} step.action)
      logProbPtr = prim__select logPT 0 aIdx
      selLPVal   = if step.action == 0 then lp0Val else lp1Val
      logProbV   = Var logProbPtr Nothing selLPVal

      valuePtr   = prim__select outT 0 2
      valueV     = Var valuePtr Nothing vVal

      advC     : Variable CPU
      advC     = fromDouble adv
      retC     : Variable CPU
      retC     = fromDouble retT
      zeroC    : Variable CPU
      zeroC    = fromDouble 0.0
      valCoefC : Variable CPU
      valCoefC = fromDouble valueCoef

      policyT  = zeroC - (logProbV * advC)

      diff     = valueV - retC
      valueTerm : Variable CPU
      valueTerm = valCoefC * diff * diff

      p0v      = Prelude.exp lp0Val
      p1v      = Prelude.exp lp1Val
      entH     = negate (p0v * lp0Val + p1v * lp1Val)
      entTerm  : Variable CPU
      entTerm  = zeroC - fromDouble (entropyCoef * entH)
  in policyT + valueTerm + entTerm


aggregateLoss : List (Variable CPU) -> Variable CPU
aggregateLoss losses =
  let zeroV  = the (Variable CPU) (fromDouble 0.0)
      sumV   = foldl (+) zeroV losses
      n      = the Double (cast (natToInteger (length losses)))
      nV     = the (Variable CPU) (fromDouble n)
  in sumV / nV


buildLoss : ACNet -> Double -> Double -> Double -> Double ->
            List RollStep -> CPState -> Variable CPU
buildLoss net gamma lam entropyCoef valueCoef steps finalSt =
  let bootstrap  = computeBootstrap net steps finalSt
      triples    = map stepTriple steps
      gaeOut     = gae gamma lam bootstrap triples
      merged     = map flattenTriple (zip steps gaeOut)
      normalized = normAdvs merged
      lossFn : (RollStep, Double, Double) -> Variable CPU
      lossFn     = perStepLoss net entropyCoef valueCoef
      losses     = map lossFn normalized
  in aggregateLoss losses


----------------------------------------------------------------------
-- Config + epoch
----------------------------------------------------------------------

record A2CState where
  constructor MkA2C
  net    : ACNet
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

defaultConfig : Config
defaultConfig = MkConfig 3.0e-3 5000 0.99 0.95 0.05 0.5 42

specs : List (ArgSpec Config)
specs = [ Arg "--lr" (\v, c => { lr := cast v } c)
        , Arg "--epochs" (\v, c => { epochs := castNat v } c)
        , Arg "--gamma" (\v, c => { gamma := cast v } c)
        , Arg "--lambda" (\v, c => { lam := cast v } c)
        , Arg "--entropy" (\v, c => { entropyCoef := cast v } c)
        , Arg "--value-coef" (\v, c => { valueCoef := cast v } c)
        , Arg "--seed" (\v, c => { seed := castBits64 v } c)
        ]

lastTerminated : List RollStep -> Bool
lastTerminated steps = case last' steps of
  Just ls => ls.isDone
  Nothing => False

a2cEpoch : NativeOptimizer -> Config -> A2CState -> IO (A2CState, Double)
a2cEpoch opt cfg st = do
  startSt <- readIORef st.envRef
  rolled  <- rollout st.net startSt RolloutLen
  let steps   = fst rolled
      finalSt = snd rolled
  writeIORef st.envRef finalSt
  let loss = buildLoss st.net cfg.gamma cfg.lam
                       cfg.entropyCoef cfg.valueCoef steps finalSt
  _ <- pure (nativeTrainStep opt loss)

  let sumRew  = sum (map (\s => s.reward) steps)
      wasDone = lastTerminated steps
  runRet <- readIORef st.retRef
  let newRet   = runRet + sumRew
      reported = if wasDone then newRet else sumRew
  writeIORef st.retRef (if wasDone then 0.0 else newRet)
  pure (st, negate reported)


----------------------------------------------------------------------
-- Greedy evaluation (argmax on logits)
----------------------------------------------------------------------

greedyAct : ACNet -> Vect ObsDim Double -> Nat
greedyAct net obs =
  let (l0, l1, _) = forwardAC net obs
  in if l0 >= l1 then 0 else 1

evalEp : ACNet -> CPState -> Nat -> Double -> Double
evalEp _ _ Z acc = acc
evalEp net st (S k) acc =
  let a = greedyAct net (observeVec st)
  in case cpStep st a of
       (r, st', outcome, _) =>
         if done outcome then acc + r
         else evalEp net st' k (acc + r)

evalN : ACNet -> Nat -> Double -> Double
evalN _ Z acc = acc
evalN net (S k) acc =
  evalN net k (acc + evalEp net (MkCP 0 0 0 0) MaxSteps 0.0)


----------------------------------------------------------------------
-- Main
----------------------------------------------------------------------

main : IO ()
main = do
  args <- getArgs
  let cfg = parseArgs defaultConfig specs (drop 1 args)
  srand cfg.seed

  putStrLn "=== A2C on CartPole ==="
  putStrLn $ "Config: lr=" ++ show cfg.lr
           ++ " epochs=" ++ show cfg.epochs
           ++ " gamma=" ++ show cfg.gamma
           ++ " lambda=" ++ show cfg.lam
           ++ " entropy=" ++ show cfg.entropyCoef
           ++ " seed=" ++ show cfg.seed

  net    <- mkACNet
  envRef <- newIORef (the CPState (MkCP 0 0 0 0))
  retRef <- newIORef (the Double 0.0)
  let st0 = MkA2C net envRef retRef
      opt = nativeAdamGlobalClip cfg.lr 0.9 0.999 1.0e-8 0.5

  putStrLn ""

  let trainCfg : TrainConfig A2CState
      trainCfg = MkTrainConfig cfg.epochs 100 NoEarlyStop (const (pure []))
  (trained, epochsDone, _) <- runTrainingIO
    (\s, _ => a2cEpoch opt cfg s)
    (pure ())
    trainCfg st0

  putStrLn ""
  let nEval = the Nat 30
      avgReturn = evalN trained.net nEval 0.0 / cast (natToInteger nEval)
  putStrLn $ "Eval (" ++ show nEval ++ " episodes, greedy): avg_return=" ++ show avgReturn
  putStrLn ""
  putStrLn $ formatResult [("avg_return", show avgReturn),
                            ("epochs", show epochsDone),
                            ("seed", show cfg.seed)]
