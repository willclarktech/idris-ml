module Example.Reinforce

import Data.List
import Data.Vect
import System
import Compat.Random

import Floating
import Gym.ClassicControl.CartPole
import Gym.Env
import Hpo.LrFinder
import Layer.Activation
import Layer.Core
import Layer.Linear
import Sampler
import Array
import Train
import Util
import Device
import Tensor


MaxSteps : Nat; MaxSteps = cartPoleMaxSteps

export
observe : CPState -> Vector 4 Double
observe s = VArray (map SArray (cpObserve s))


----------------------------------------------------------------------
-- Episode Rollout (tensor-level, autograd-tracked)
----------------------------------------------------------------------

-- (logProbTensorPtr, logProbDoubleVal, reward)
public export
StepRec : Type
StepRec = (AnyPtr, Double, Double)

export
rolloutEp : {hs : List Nat} ->
            Network 4 hs 2 CPU WithGrad -> CPState -> List Double -> Nat ->
            List StepRec -> List StepRec
rolloutEp _ _ _ Z acc = reverse acc
rolloutEp _ _ [] _ acc = reverse acc
rolloutEp model st (r :: rs) (S k) acc =
  let stateT = bulkToTensor (observe st)
      stateV = the (TVec 4 CPU WithGrad) (MkTensor stateT Nothing)
      (_, predV) = forwardVar model stateV
      logProbsT = prim__logSoftmax predV.tensorPtr 0
      lp0 = prim__item1d logProbsT 0
      lp1 = prim__item1d logProbsT 1
      action = categoricalSample [exp lp0, exp lp1] r
      selLP = prim__select logProbsT 0 (cast {to=Int} action)
      selLPVal = if action == 0 then lp0 else lp1
  in case cpStep st action of
       (reward, st', outcome, _) =>
         let acc' = (selLP, selLPVal, reward) :: acc
         in if done outcome then reverse acc'
            else rolloutEp model st' rs k acc'


||| Run N parallel episodes with one batched policy forward per
||| timestep. Each env has its own RNG sequence and starting state.
||| Returns N independent StepRec lists (same shape as N sequential
||| rolloutEp calls).
|||
||| Done envs are "frozen" — their state passes through the batched
||| forward (so the [N, 4] shape stays constant timestep-to-timestep,
||| keeping the tape graph stable for `mx::compile`-style fusers) but
||| no further StepRec is appended for that env. Loop exits early if
||| all envs terminate.
export
rolloutEpBatched : {n : Nat} -> {hs : List Nat} ->
                   Network 4 hs 2 CPU WithGrad ->
                   Vect n CPState ->
                   Vect n (List Double) ->
                   Nat ->
                   Vect n (List StepRec)
rolloutEpBatched model states0 rss0 maxSteps =
  map reverse (go maxSteps states0 rss0 (replicate n False) (replicate n []))
  where
    -- Per-env action selection + env step, given the batched log-prob
    -- tensor and this env's integer index. Frozen (done) envs and
    -- RNG-exhausted envs pass through unchanged.
    perEnv : Tensor [n, 2] CPU WithGrad -> Int ->
             CPState -> List Double -> Bool -> List StepRec ->
             (CPState, List Double, Bool, List StepRec)
    perEnv _         _ st rs  True  acc = (st, rs, True, acc)
    perEnv _         _ st []  _     acc = (st, [], True, acc)
    perEnv logProbsV i st (r :: rs) False acc =
      let logProbsT = logProbsV.tensorPtr
          lp0       = prim__item2d logProbsT i 0
          lp1       = prim__item2d logProbsT i 1
          action    = categoricalSample [exp lp0, exp lp1] r
          rowPtr    = prim__select logProbsT 0 i
          selLP     = prim__select rowPtr 0 (cast {to=Int} action)
          selLPVal  = if action == 0 then lp0 else lp1
      in case cpStep st action of
           (reward, st', outcome, _) =>
             (st', rs, done outcome, (selLP, selLPVal, reward) :: acc)

    -- Walk the four parallel Vects together, threading the row index.
    -- Each pattern strips one element from each Vect simultaneously.
    stepAllEnvs : Tensor [n, 2] CPU WithGrad -> Int ->
                  Vect k CPState -> Vect k (List Double) -> Vect k Bool ->
                  Vect k (List StepRec) ->
                  (Vect k CPState, Vect k (List Double), Vect k Bool, Vect k (List StepRec))
    stepAllEnvs _         _ []         []         []         []         = ([], [], [], [])
    stepAllEnvs logProbsV i (st :: sts) (rs :: rss) (d :: ds) (acc :: accs) =
      let (st', rs', d', acc') = perEnv logProbsV i st rs d acc
          (sts', rss', ds', accs') = stepAllEnvs logProbsV (i + 1) sts rss ds accs
      in (st' :: sts', rs' :: rss', d' :: ds', acc' :: accs')

    -- Recursive loop. Returns accumulators in REVERSED order.
    go : Nat ->
         Vect n CPState -> Vect n (List Double) -> Vect n Bool ->
         Vect n (List StepRec) ->
         Vect n (List StepRec)
    go Z _ _ _ accs = accs
    go (S k) sts rss dones accs =
      if all id (toList dones) then accs
      else
        let obsRows : Vect n (Vector 4 Double)
            obsRows = map observe sts
            batchPtr = bulkToTensor2d obsRows
            stateV : Tensor [n, 4] CPU WithGrad
            stateV = MkTensor batchPtr Nothing
            predV : Tensor [n, 2] CPU WithGrad
            predV = snd (forwardVarBatch model stateV)
            logProbsV : Tensor [n, 2] CPU WithGrad
            logProbsV = MkTensor (prim__logSoftmax2d predV.tensorPtr) Nothing
        in case stepAllEnvs logProbsV 0 sts rss dones accs of
             (sts', rss', dones', accs') => go k sts' rss' dones' accs'


----------------------------------------------------------------------
-- REINFORCE Loss
----------------------------------------------------------------------

discReturns : Double -> List Double -> List Double
discReturns gamma rewards = reverse (go 0.0 (reverse rewards))
  where
    go : Double -> List Double -> List Double
    go _ [] = []
    go g (r :: rs) = let g' = r + gamma * g in g' :: go g' rs

-- Compute per-episode step losses with advantage. Each loss is a
-- scalar `Tensor [] CPU` carrying the autograd graph back to the policy.
epStepLosses : Double -> Double -> List StepRec -> List (Tensor [] CPU WithGrad)
epStepLosses gamma baseline steps =
  let rewards = map (\(_, _, r) => r) steps
      rets = discReturns gamma rewards
  in zipWith (\(lp, _, _), gt =>
       the (Tensor [] CPU WithGrad) (MkTensor (prim__mulScalar lp (baseline - gt)) Nothing))
     steps rets

export
sumRewards : List StepRec -> Double
sumRewards steps = foldl (\a, (_, _, r) => a + r) 0.0 steps

-- Mean-reduce a non-empty list of scalar TVars. Empty case returns a
-- fresh zero scalar (degenerate; runs only if the rollout produced no
-- steps).
averageLoss : List (Tensor [] CPU WithGrad) -> Tensor [] CPU WithGrad
averageLoss [] = MkTensor (prim__createScalar 0.0 0) Nothing
averageLoss (x :: xs) =
  let n = cast {to=Double} (1 + length xs)
      addT : Tensor [] CPU WithGrad -> Tensor [] CPU WithGrad -> Tensor [] CPU WithGrad
      addT a b = MkTensor (prim__add a.tensorPtr b.tensorPtr) Nothing
      s = foldl addT x xs
  in MkTensor (prim__mulScalar s.tensorPtr (1.0 / n)) Nothing

computeLoss : {hs : List Nat} -> Double ->
              Network 4 hs 2 CPU WithGrad -> List (List Double) ->
              (Tensor [] CPU WithGrad, Double)
computeLoss gamma model randomBatch =
  let episodes = map (\rs => rolloutEp model (MkCP 0 0 0 0) rs MaxSteps []) randomBatch
      epReturns = map sumRewards episodes
      nEp = cast {to=Double} (natToInteger (length epReturns))
      baseline = foldl (+) 0.0 epReturns / nEp
      stepLosses = concatMap (epStepLosses gamma baseline) episodes
  in (averageLoss stepLosses, baseline)

||| Batched-rollout variant. Same loss math; the only difference is
||| how the per-step logProbs are obtained — one batched forward over
||| `Tensor [n, 4]` per timestep instead of N sequential single-env
||| forwards. Gradients flow back through the same shape of graph
||| (one batched forward op per timestep instead of N separate ops).
computeLossBatched : {n : Nat} -> {hs : List Nat} -> Double ->
                     Network 4 hs 2 CPU WithGrad -> Vect n (List Double) ->
                     (Tensor [] CPU WithGrad, Double)
computeLossBatched gamma model randomBatchV =
  let initStates : Vect n CPState = replicate n (MkCP 0 0 0 0)
      epsV  = rolloutEpBatched model initStates randomBatchV MaxSteps
      eps   = toList epsV
      epReturns = map sumRewards eps
      nEp = cast {to=Double} (natToInteger (length epReturns))
      baseline = foldl (+) 0.0 epReturns / nEp
      stepLosses = concatMap (epStepLosses gamma baseline) eps
  in (averageLoss stepLosses, baseline)


----------------------------------------------------------------------
-- Training
----------------------------------------------------------------------

epochRL : {hs : List Nat} -> NativeOptimizer -> Double ->
          Network 4 hs 2 CPU WithGrad -> List (List Double) ->
          (Network 4 hs 2 CPU WithGrad, Double, Double)
epochRL opt gamma model batch =
  let (loss, avgRet) = computeLoss gamma model batch
      lossVal = nativeTrainStep opt loss
  in (model, lossVal, avgRet)

epochRLBatched : {n : Nat} -> {hs : List Nat} ->
                 NativeOptimizer -> Double ->
                 Network 4 hs 2 CPU WithGrad -> Vect n (List Double) ->
                 (Network 4 hs 2 CPU WithGrad, Double, Double)
epochRLBatched opt gamma model batchV =
  let (loss, avgRet) = computeLossBatched gamma model batchV
      lossVal = nativeTrainStep opt loss
  in (model, lossVal, avgRet)

genBatch : Nat -> IO (List (List Double))
genBatch batchSz = go batchSz
  where
    genN : Nat -> IO (List Double)
    genN Z = pure []
    genN (S k) = do
      r <- randomRIO (the Double 0.0, 1.0)
      rs <- genN k
      pure (r :: rs)

    go : Nat -> IO (List (List Double))
    go Z = pure []
    go (S k) = do
      ep <- genN MaxSteps
      rest <- go k
      pure (ep :: rest)

genBatchV : (n : Nat) -> IO (Vect n (List Double))
genBatchV Z = pure []
genBatchV (S k) = do
  ep <- go MaxSteps
  rest <- genBatchV k
  pure (ep :: rest)
  where
    go : Nat -> IO (List Double)
    go Z = pure []
    go (S k') = do
      r <- randomRIO (the Double 0.0, 1.0)
      rs <- go k'
      pure (r :: rs)


----------------------------------------------------------------------
-- Evaluation (greedy argmax)
----------------------------------------------------------------------

evalEp : {hs : List Nat} ->
         Network 4 hs 2 CPU WithGrad -> CPState -> Nat -> Double -> Double
evalEp _ _ Z acc = acc
evalEp model st (S k) acc =
  let stateT = bulkToTensor (observe st)
      stateV = the (TVec 4 CPU WithGrad) (MkTensor stateT Nothing)
      (_, predV) = forwardVar model stateV
      logitsT = predV.tensorPtr
      action = if prim__item1d logitsT 0 >= prim__item1d logitsT 1 then the Nat 0 else 1
  in case cpStep st action of
       (reward, st', outcome, _) =>
         if done outcome then acc + reward
         else evalEp model st' k (acc + reward)

evalN : {hs : List Nat} -> Network 4 hs 2 CPU WithGrad -> Nat -> Double -> Double
evalN _ Z acc = acc
evalN model (S k) acc =
  evalN model k (acc + evalEp model (MkCP 0 0 0 0) MaxSteps 0.0)


----------------------------------------------------------------------
-- Config & Main
----------------------------------------------------------------------

record Config where
  constructor MkConfig
  lr : Double
  epochs : Nat
  seed : Bits64
  gamma : Double
  batchSz : Nat
  lrFind : Bool
  batched : Bool  -- Job 4 Phase B: use batched policy forward per timestep

defaultConfig : Config
defaultConfig = MkConfig 0.001 2000 42 0.99 10 False False

specs : List (ArgSpec Config)
specs = [ Arg "--lr" (\v, c => { lr := cast v } c)
        , Arg "--epochs" (\v, c => { epochs := castNat v } c)
        , Arg "--seed" (\v, c => { seed := castBits64 v } c)
        , Arg "--gamma" (\v, c => { gamma := cast v } c)
        , Arg "--batch" (\v, c => { batchSz := castNat v } c)
        , Arg "--lr-find" (\v, c => { lrFind := (v == "1" || v == "true") } c)
        , Arg "--batched" (\v, c => { batched := (v == "1" || v == "true") } c) ]

main : IO ()
main = do
  args <- getArgs
  let cfg = parseArgs defaultConfig specs (drop 1 args)

  srand cfg.seed

  let opt = nativeAdamGlobalClip cfg.lr 0.9 0.999 1.0e-8 1.0

  putStrLn "=== REINFORCE on CartPole ==="
  putStrLn $ "Config: lr=" ++ show cfg.lr ++ " epochs=" ++ show cfg.epochs
           ++ " gamma=" ++ show cfg.gamma ++ " batch=" ++ show cfg.batchSz
           ++ " seed=" ++ show cfg.seed

  ll1Any <- linearLayerAny {i=4} {o=128} "ll1"
  ll2Any <- linearLayerAny {i=128} {o=2} "ll2"
  let model : Network 4 [128, 128] 2 CPU WithGrad
      model = ll1Any ~~> tanhLayerAny ~~> OutputLayer ll2Any
  putStrLn ""

  when cfg.lrFind $ do
    let lrCfg : LrFindConfig
        lrCfg = { numIters := 100 } defaultLrFindConfig
    _ <- lrFind lrCfg
      (\m, d => let (m', loss, _) = epochRL opt cfg.gamma m d
                in pure (m', loss))
      (genBatch cfg.batchSz) opt model
    putStrLn ""
    putStrLn "Done — re-run without --lr-find at the recommended LR."
    exitSuccess

  metrics <- newRLMetricsState 100
  let n : Nat = cfg.batchSz
      modelType : Type = Network 4 [128, 128] 2 CPU WithGrad
  (trained, epochsDone, _) <- (
    if cfg.batched
      then runTrainingIO {dp = Vect n (List Double)}
             (\m, d => do
                let (m', loss, avgRet) = epochRLBatched opt cfg.gamma m d
                recordReturn metrics avgRet
                pure (m', loss))
             (genBatchV n)
             ({ metrics := \_ => readRLMetrics "recent_100" metrics }
                (simpleConfig {model = modelType} cfg.epochs))
             model
      else runTrainingIO {dp = List (List Double)}
             (\m, d => do
                let (m', loss, avgRet) = epochRL opt cfg.gamma m d
                recordReturn metrics avgRet
                pure (m', loss))
             (genBatch cfg.batchSz)
             ({ metrics := \_ => readRLMetrics "recent_100" metrics }
                (simpleConfig {model = modelType} cfg.epochs))
             model)

  putStrLn ""
  putStrLn "Eval (100 episodes, greedy):"
  let nEval = the Nat 100
      totalReturn = evalN trained nEval 0.0
      avgReturn = totalReturn / cast (natToInteger nEval)
  putStrLn $ "  avg_return=" ++ show avgReturn

  putStrLn ""
  putStrLn $ formatResult [("avg_return", show avgReturn),
                            ("epochs", show epochsDone),
                            ("seed", show cfg.seed)]
