module Example.Ppo

import Data.List
import Data.SortedMap
import Data.Vect
import Data.IORef
import System
import Compat.Random

import Endofunctor
import Floating
import Gym.ClassicControl.Acrobot
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
-- Architecture: separate actor and critic MLPs with discrete-action
-- (categorical) policy on Acrobot. Pendulum (continuous-action Gaussian)
-- was the original env but didn't converge at CPU-feasible rollout
-- sizes; Acrobot is the canonical "PPO clipped-surrogate demonstrates"
-- benchmark — discrete actions, sparse reward, longer horizon.
--
-- Actor:  Linear(6→64) → tanh → Linear(64→64) → tanh → Linear(64→3)  = action logits
-- Critic: Linear(6→64) → tanh → Linear(64→64) → tanh → Linear(64→1)  = value
--
-- Actor and critic params are scoped via `autoNameScoped` (paramId-
-- collision fix; see CLAUDE.md "ParamId scoping for multi-network
-- examples"). Without scoping, both networks register `ll0_weights`
-- etc. and the second overwrites the first.
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

ObsDim : Nat; ObsDim = 6
Hidden : Nat; Hidden = 64
NumActions : Nat; NumActions = 3
EpisodeLen : Nat; EpisodeLen = 500   -- Acrobot defaultTimeLimit
RolloutLen : Nat; RolloutLen = 1024
BatchSize : Nat; BatchSize = 64

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

observeVec : AState -> Vect ObsDim Double
observeVec s = aObserve s

obsTensor : Vect ObsDim Double -> Vector ObsDim Double
obsTensor v = VTensor (map STensor v)


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
-- Categorical policy helpers
----------------------------------------------------------------------

-- Scalar value from critic.
criticValue : Critic -> Vect ObsDim Double -> Double
criticValue critic obs =
  let outT = snd (forwardVarTensor critic (bulkToTensor (obsTensor obs)))
  in prim__item1d outT 0

-- Sample a discrete action from the actor's categorical distribution.
-- Returns (action, log π_old(a|s), value(s)).
sampleActionIO : Actor -> Critic -> Vect ObsDim Double -> IO (Nat, Double, Double)
sampleActionIO actor critic obs = do
  let stateT  = bulkToTensor (obsTensor obs)
      logitsT = snd (forwardVarTensor actor stateT)
      logPT   = prim__logSoftmax logitsT 0
      lp0     = prim__item1d logPT 0
      lp1     = prim__item1d logPT 1
      lp2     = prim__item1d logPT 2
      v       = criticValue critic obs
  u <- randomRIO (the Double 0.0, 1.0)
  let a = categoricalSample [Prelude.exp lp0, Prelude.exp lp1, Prelude.exp lp2] u
      lp = case a of
             0 => lp0
             1 => lp1
             _ => lp2
  pure (a, lp, v)


----------------------------------------------------------------------
-- Rollout: RolloutLen steps with auto-reset on natural termination
-- and at EpisodeLen truncation.
----------------------------------------------------------------------

rollout : Actor -> Critic -> AState -> Nat -> Nat ->
          IO (List RollStep, AState, List Double)
rollout _ _ st _ Z = pure ([], st, [])
rollout actor critic st stepsLeft (S k) = do
  let obs = observeVec st
  triple <- sampleActionIO actor critic obs
  let a  = fst triple
      lp = fst (snd triple)
      v  = snd (snd triple)
  case aStep st a of
    (r, st', outcome, _) => do
      let natTerm = case outcome of
                      Terminated => True
                      _          => False
          truncate = stepsLeft == 1
          isDone = natTerm || truncate
          stepRec = MkRS obs a lp v r isDone
          nextSt = if isDone then MkA 0.0 0.0 0.0 0.0 else st'
          nextStepsLeft = if isDone then EpisodeLen else stepsLeft `minus` 1
      recur <- rollout actor critic nextSt nextStepsLeft k
      let (stepsRest, finalSt, retsRest) = recur
          retsCarry = if isDone then 0.0 :: retsRest else retsRest
      pure (stepRec :: stepsRest, finalSt, retsCarry)


----------------------------------------------------------------------
-- GAE + advantage normalization
----------------------------------------------------------------------

bootstrapV : Critic -> Vect ObsDim Double -> Double
bootstrapV critic obs = criticValue critic obs

computeBootstrap : Critic -> List RollStep -> AState -> Double
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
-- Per-step PPO loss (categorical policy)
----------------------------------------------------------------------

clipScalar : Double -> Double -> Double -> Double
clipScalar lo hi x = if x < lo then lo else if x > hi then hi else x

-- Per-sample logits/value Variables are extracted from the batched
-- [B, NumActions] / [B, 1] outputs via prim__select. The actor + critic
-- forwards happen once per mini-batch in `runBatch`.
perStepLoss : (logitsB : AnyPtr) -> (valueB : AnyPtr) -> (rowIdx : Int) ->
              Double -> Double -> Double ->
              (RollStep, Double, Double) -> Variable CPU
perStepLoss logitsB valueB rowIdx clipEps entropyCoef valueCoef (step, adv, retT) =
  let logitsRow = prim__select logitsB 0 rowIdx        -- [NumActions]
      logPT     = prim__logSoftmax logitsRow 0
      aIdx : Int
      aIdx      = cast {to=Int} (cast {to=Integer} step.action)
      selLP     = prim__select logPT 0 aIdx
      lpVal     = case step.action of
                    0 => prim__item1d logPT 0
                    1 => prim__item1d logPT 1
                    _ => prim__item1d logPT 2
      lpNew     : Variable CPU
      lpNew     = Var selLP Nothing lpVal

      valueRow  = prim__select valueB 0 rowIdx          -- [1]
      valuePtr  = prim__select valueRow 0 0
      valueVal  = prim__item1d valueRow 0
      valueV    : Variable CPU
      valueV    = Var valuePtr Nothing valueVal

      lpOldC    : Variable CPU
      lpOldC    = fromDouble step.oldLogProb
      advC      : Variable CPU
      advC      = fromDouble adv

      diffLP    = lpNew - lpOldC
      ratioVal  = Prelude.exp diffLP.value
      ratioV    = exp diffLP
      surr1     = ratioV * advC

      zeroC     : Variable CPU
      zeroC     = fromDouble 0.0
      halfC     : Variable CPU
      halfC     = fromDouble 0.5

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

      -- Entropy H(π) = -Σ p_i log p_i, built with grad-tracked Variables
      -- so the bonus actually pulls the policy back from collapse.
      lp0Val   = prim__item1d logPT 0
      lp1Val   = prim__item1d logPT 1
      lp2Val   = prim__item1d logPT 2
      lp0V     : Variable CPU
      lp0V     = Var (prim__select logPT 0 0) Nothing lp0Val
      lp1V     : Variable CPU
      lp1V     = Var (prim__select logPT 0 1) Nothing lp1Val
      lp2V     : Variable CPU
      lp2V     = Var (prim__select logPT 0 2) Nothing lp2Val
      p0V      : Variable CPU
      p0V      = exp lp0V
      p1V      : Variable CPU
      p1V      = exp lp1V
      p2V      : Variable CPU
      p2V      = exp lp2V
      negEntV  = p0V * lp0V + p1V * lp1V + p2V * lp2V    -- = -H(π)
      entCoefC : Variable CPU
      entCoefC = fromDouble entropyCoef
      entTerm  = entCoefC * negEntV
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
  lrFind      : Bool

defaultConfig : Config
defaultConfig = MkConfig 3.0e-4 100 0.99 0.95 0.2 10 0.01 0.5 42 False

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
        , Arg "--lr-find" (\v, c => { lrFind := (v == "1" || v == "true") } c)
        ]

record PPOState where
  constructor MkPPO
  actor   : Actor
  critic  : Critic
  envRef  : IORef AState


prepareRollout : Critic -> Config -> List RollStep -> AState ->
                 List (RollStep, Double, Double)
prepareRollout critic cfg steps finalSt =
  let bootstrap = computeBootstrap critic steps finalSt
      triples   = map stepTriple steps
      gaeOut    = gae cfg.gamma cfg.lam bootstrap triples
      merged    = map flattenTriple (zip steps gaeOut)
  in normAdvs merged


-- Stack mini-batch obs into [B, ObsDim], do one batched actor + critic
-- forward each, then build per-sample loss expressions by indexing into
-- the [B, NumActions] / [B, 1] tensors. Replaces O(B) per-sample
-- `forwardVarTensor` calls with two batched calls per mini-batch.
runBatch : NativeOptimizer -> Actor -> Critic -> Config ->
           List (RollStep, Double, Double) -> IO ()
runBatch opt actor critic cfg batch = do
  let batchVec  = Data.Vect.fromList batch
      n         = length batch
      obsBatch : Vect (length batch) (Vector ObsDim Double)
      obsBatch  = map (\(s, _, _) => obsTensor s.obs) batchVec
      stackedT  = bulkToTensor2d obsBatch
      logitsB   = snd (forwardVarTensorBatch actor n stackedT)
      valueB    = snd (forwardVarTensorBatch critic n stackedT)
      losses    = enumeratedLosses logitsB valueB batchVec 0
      loss      = aggregateLoss losses
  _ <- pure (nativeTrainStep opt loss)
  pure ()
  where
    enumeratedLosses : (logitsB : AnyPtr) -> (valueB : AnyPtr) ->
                       Vect k (RollStep, Double, Double) -> Int ->
                       List (Variable CPU)
    enumeratedLosses _ _ [] _ = []
    enumeratedLosses lB vB (t :: rest) k =
      perStepLoss lB vB k cfg.clipEps cfg.entropyCoef cfg.valueCoef t
        :: enumeratedLosses lB vB rest (k + 1)


kEpochUpdate : NativeOptimizer -> Actor -> Critic -> Config ->
               List (RollStep, Double, Double) -> Nat -> IO ()
kEpochUpdate _ _ _ _ _ Z = pure ()
kEpochUpdate opt actor critic cfg prepped (S k) = do
  shuffled <- shuffleIO prepped
  let batches = chunksOf BatchSize shuffled
  traverse_ (runBatch opt actor critic cfg) batches
  kEpochUpdate opt actor critic cfg prepped k


ppoEpoch : NativeOptimizer -> Config -> PPOState -> IO (PPOState, Double)
ppoEpoch opt cfg st = do
  startSt <- readIORef st.envRef
  rolled  <- rollout st.actor st.critic startSt EpisodeLen RolloutLen
  let steps   = fst rolled
      finalSt = fst (snd rolled)
  writeIORef st.envRef finalSt

  let prepped = prepareRollout st.critic cfg steps finalSt
  kEpochUpdate opt st.actor st.critic cfg prepped cfg.kEpochs

  -- Average return over completed episodes in this rollout.
  let episodeReturns : List Double
      episodeReturns = computeEpisodeReturns steps
      nEp = length episodeReturns
      sumEp = sum episodeReturns
      avgEp = if nEp > 0 then sumEp / cast (natToInteger nEp) else sum (map (\s => s.reward) steps)
  pure (st, negate avgEp)
  where
    -- Walk the rollout and split into episode segments by isDone, summing
    -- rewards within each. Trailing partial episode (no isDone) is dropped.
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

greedyAct : Actor -> Vect ObsDim Double -> Nat
greedyAct actor obs =
  let logits = snd (forwardVarTensor actor (bulkToTensor (obsTensor obs)))
      l0 = prim__item1d logits 0
      l1 = prim__item1d logits 1
      l2 = prim__item1d logits 2
  in if l0 >= l1 && l0 >= l2 then 0
     else if l1 >= l2 then 1
     else 2

evalEp : Actor -> AState -> Nat -> Double -> Double
evalEp _ _ Z acc = acc
evalEp actor st (S k) acc =
  let a = greedyAct actor (observeVec st)
  in case aStep st a of
       (r, st', outcome, _) =>
         case outcome of
           Terminated => acc + r
           _          => evalEp actor st' k (acc + r)

evalN : Actor -> Nat -> Double -> Double
evalN _ Z acc = acc
evalN actor (S k) acc =
  evalN actor k (acc + evalEp actor (MkA 0.0 0.0 0.0 0.0) EpisodeLen 0.0)


----------------------------------------------------------------------
-- Main
----------------------------------------------------------------------

main : IO ()
main = do
  args <- getArgs
  let cfg = parseArgs defaultConfig specs (drop 1 args)
  srand cfg.seed

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
  envRef <- newIORef (the AState (MkA 0.0 0.0 0.0 0.0))
  let st0 = MkPPO actor critic envRef
      opt = nativeAdamGlobalClip cfg.lr 0.9 0.999 1.0e-8 0.5

  putStrLn ""

  -- HPO branch: --lr-find runs lr_find using one full PPO rollout per
  -- iter (`ppoEpoch` does the rollout + K mini-batch updates). Acrobot
  -- episode returns are negative (avg_return uses negate avgEp), so the
  -- "loss" stays positive — the negative-loss heuristic bug doesn't trip.
  -- Each iter is heavy (1024 env steps + K=10 mini-batches), so use 30 iters.
  when cfg.lrFind $ do
    let lrCfg : LrFindConfig
        lrCfg = { numIters := 30 } defaultLrFindConfig
    _ <- lrFind lrCfg
      (\s, _ => ppoEpoch opt cfg s)
      (pure ()) opt st0
    putStrLn ""
    putStrLn "Done — re-run without --lr-find at the recommended LR."
    exitSuccess

  let trainCfg : TrainConfig PPOState
      trainCfg = MkTrainConfig cfg.epochs 10 NoEarlyStop (const (pure [])) (\_ => pure ())
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
