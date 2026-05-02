module Example.A2c

import Data.List
import Data.SortedMap
import Data.Vect
import Data.IORef
import System
import Compat.Random

import Endofunctor
import Floating
import Gym.ClassicControl.CartPole
import Gym.Env
import Hpo.LrFinder
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
-- Architecture: separate actor and critic MLPs (aligned with PyTorch
-- reference `a2c.py`). We use `autoName` to generate paramIds like
-- "ll0_weight0", then `emap (prefixParamIdLocal "actor_")` /
-- `emap (prefixParamIdLocal "critic_")` to rescope them, so the two
-- networks register disjoint keys with the single optimizer.
--   Actor  : 4 -> 64 -> 64 -> 2 (action logits)
--   Critic : 4 -> 64 -> 64 -> 1 (state value)
----------------------------------------------------------------------

-- Local `autoNameNetwork`/`autoNameAny` because the `-o <file>` Idris
-- invocation used by Makefile example targets fails to pick up symbols
-- newly exported from `idris-ml` (even after a clean install). The
-- `--build <pkg>.ipkg` path sees them fine — this is a single-file
-- Idris resolution quirk. Inlining sidesteps it.
--
-- This is the critical path: it re-registers each layer's consolidated
-- weight/bias tensors (the ones used by the `applyVarTensor` fast path)
-- under a scoped paramId, which is what makes the optimizer see the
-- actor and critic as distinct parameter groups. A simpler `emap` +
-- `setParamId` approach only renames the scalar *view* Variables and
-- leaves the consolidated weight tensor registered under the colliding
-- unprefixed name — the second network's `autoName` then overwrites the
-- first's registry entry, silently zeroing the first's gradient flow.
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

ObsDim : Nat; ObsDim = 4
Hidden : Nat; Hidden = 64
NumActions : Nat; NumActions = 2
MaxSteps : Nat; MaxSteps = cartPoleMaxSteps
RolloutLen : Nat; RolloutLen = 20

Actor : Type
Actor = Network ObsDim [Hidden, Hidden, Hidden, Hidden] NumActions (Variable CPU)

Critic : Type
Critic = Network ObsDim [Hidden, Hidden, Hidden, Hidden] 1 (Variable CPU)


mkActor : IO Actor
mkActor = do
  ll1 <- linearLayer {i=ObsDim} {o=Hidden}
  ll2 <- linearLayer {i=Hidden} {o=Hidden}
  ll3 <- linearLayer {i=Hidden} {o=NumActions}
  pure (autoNameScoped "actor_"
    (ll1 ~> tanhLayer ~> ll2 ~> tanhLayer ~> OutputLayer ll3))

mkCritic : IO Critic
mkCritic = do
  ll1 <- linearLayer {i=ObsDim} {o=Hidden}
  ll2 <- linearLayer {i=Hidden} {o=Hidden}
  ll3 <- linearLayer {i=Hidden} {o=1}
  pure (autoNameScoped "critic_"
    (ll1 ~> tanhLayer ~> ll2 ~> tanhLayer ~> OutputLayer ll3))


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
-- Top-level helpers (keep out of do-blocks to dodge Idris 2 parser
-- quirks with multi-binding let + case).
----------------------------------------------------------------------

sampleActionIO : Actor -> Critic -> Vect ObsDim Double -> IO (Nat, Double)
sampleActionIO actor critic obs = do
  let stateT  = bulkToTensor (obsTensor obs)
      logitsT = snd (forwardVarTensor actor stateT)
      logPT   = prim__logSoftmax logitsT 0
      lp0     = prim__item1d logPT 0
      lp1     = prim__item1d logPT 1
      valueT  = snd (forwardVarTensor critic stateT)
      v       = prim__item1d valueT 0
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
-- GAE helpers
----------------------------------------------------------------------

bootstrapV : Critic -> Vect ObsDim Double -> Double
bootstrapV critic obs =
  let valueT = snd (forwardVarTensor critic (bulkToTensor (obsTensor obs)))
  in prim__item1d valueT 0

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
-- Per-step A2C loss
--
-- The per-sample logits/value Variables are built from a row of the
-- batched [B, NumActions] / [B, 1] tensors via `prim__select`. Forward
-- through actor + critic happens once per epoch (batched) — see
-- `buildLoss` below.
----------------------------------------------------------------------

perStepLoss : (logitsB : AnyPtr) -> (valuesB : AnyPtr) -> (rowIdx : Int) ->
              Double -> Double ->
              (RollStep, Double, Double) -> Variable CPU
perStepLoss logitsB valuesB rowIdx entropyCoef valueCoef (step, adv, retT) =
  let logitsRow = prim__select logitsB 0 rowIdx        -- [NumActions]
      logPT     = prim__logSoftmax logitsRow 0
      aIdx : Int
      aIdx      = cast {to=Int} (cast {to=Integer} step.action)
      selLP     = prim__select logPT 0 aIdx
      lpVal     = if step.action == 0 then prim__item1d logPT 0 else prim__item1d logPT 1
      logProbV  = Var selLP Nothing lpVal

      valueRow  = prim__select valuesB 0 rowIdx        -- [1]
      vVal      = prim__item1d valueRow 0
      valueV    = Var (prim__select valueRow 0 0) Nothing vVal

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

      -- Entropy H(π) = -Σ p_i log p_i, built with grad-tracked Variables
      -- so the entropy bonus actually pulls the policy back from collapse.
      lp0Val   = prim__item1d logPT 0
      lp1Val   = prim__item1d logPT 1
      lp0V     : Variable CPU
      lp0V     = Var (prim__select logPT 0 0) Nothing lp0Val
      lp1V     : Variable CPU
      lp1V     = Var (prim__select logPT 0 1) Nothing lp1Val
      p0V      : Variable CPU
      p0V      = exp lp0V
      p1V      : Variable CPU
      p1V      = exp lp1V
      negEntV  = p0V * lp0V + p1V * lp1V      -- = -H(π)
      entCoefC : Variable CPU
      entCoefC = fromDouble entropyCoef
      entTerm  = entCoefC * negEntV           -- loss += ent_coef * (-H)
  in policyT + valueTerm + entTerm


aggregateLoss : List (Variable CPU) -> Variable CPU
aggregateLoss losses =
  let zeroV  = the (Variable CPU) (fromDouble 0.0)
      sumV   = foldl (+) zeroV losses
      n      = the Double (cast (natToInteger (length losses)))
      nV     = the (Variable CPU) (fromDouble n)
  in sumV / nV


-- Pair each rollout step (after GAE + advantage normalization) with its
-- batch row index, then build one batched actor + critic forward and
-- index into the resulting [B, NumActions] / [B, 1] tensors per-sample.
-- Replaces O(B) per-step `forwardVarTensor` calls with two batched ones.
buildLoss : Actor -> Critic -> Double -> Double -> Double -> Double ->
            List RollStep -> CPState -> Variable CPU
buildLoss actor critic gamma lam entropyCoef valueCoef steps finalSt =
  let bootstrap  = computeBootstrap critic steps finalSt
      triples    = map stepTriple steps
      gaeOut     = gae gamma lam bootstrap triples
      merged     = map flattenTriple (zip steps gaeOut)
      normalized = normAdvs merged
      normVec    = Data.Vect.fromList normalized
      n          = length normalized
      obsBatch : Vect (length normalized) (Vector ObsDim Double)
      obsBatch   = map (\(s, _, _) => obsTensor s.obs) normVec
      stackedT   = bulkToTensor2d obsBatch
      logitsB    = snd (forwardVarTensorBatch actor n stackedT)
      valuesB    = snd (forwardVarTensorBatch critic n stackedT)
      losses     = enumeratedLosses logitsB valuesB normVec 0
  in aggregateLoss losses
  where
    enumeratedLosses : (logitsB : AnyPtr) -> (valuesB : AnyPtr) ->
                       Vect k (RollStep, Double, Double) -> Int ->
                       List (Variable CPU)
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

||| Defaults aligned with PyTorch `a2c.py`: lr=7e-4, entropy=0.01,
||| rollout=20, gamma=0.99, lam=0.95.
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
  rolled  <- rollout st.actor st.critic startSt RolloutLen
  let steps   = fst rolled
      finalSt = snd rolled
  writeIORef st.envRef finalSt
  let loss = buildLoss st.actor st.critic cfg.gamma cfg.lam
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
-- Greedy evaluation
----------------------------------------------------------------------

greedyAct : Actor -> Vect ObsDim Double -> Nat
greedyAct actor obs =
  let logits = snd (forwardVarTensor actor (bulkToTensor (obsTensor obs)))
      l0 = prim__item1d logits 0
      l1 = prim__item1d logits 1
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

  actor  <- mkActor
  critic <- mkCritic
  envRef <- newIORef (the CPState (MkCP 0 0 0 0))
  retRef <- newIORef (the Double 0.0)
  let st0 = MkA2C actor critic envRef retRef
      opt = nativeAdamGlobalClip cfg.lr 0.9 0.999 1.0e-8 0.5

  putStrLn ""

  -- HPO branch: --lr-find runs lr_find using episode-return-as-loss.
  -- See hyperparameter-tuning-2026.md — same negative-loss caveat as
  -- Reinforce/Dqn; result is informational only.
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
