module Example.Reinforce

import Data.List
import Data.Vect
import System
import Compat.Random

import ML.Simple
import Array            -- Vector / VArray / SArray (bulkToTensor2d input)
import Gym.ClassicControl.CartPole
import Gym.Env
import Gym.Vector
import Hpo.LrFinder
import Sampler          -- categoricalSample
import Train            -- simpleConfig / RL metrics
import BuildConfig      -- ChosenMachine / requireMachine

MaxSteps : Nat; MaxSteps = cartPoleMaxSteps

-- Array-typed observation row, the shape `bulkToTensor2d` consumes
-- (distinct from the Env method's plain `Vect 4 Double`).
export
observe : CPState -> Vector 4 Double
observe s = VArray (map SArray (cpObserve s))

-- Policy network: MLP 4 -> 128 -> tanh -> 2 (action logits).
Policy : Type
Policy = Seq 4 2 Ex F WithGrad

----------------------------------------------------------------------
-- Episode Rollout (tensor-level, autograd-tracked)
----------------------------------------------------------------------

-- (logProbTensorPtr, logProbDoubleVal, reward)
public export
StepRec : Type
StepRec = (AnyPtr, Double, Double)

export
rolloutEp : Policy -> CPState -> List Double -> Nat ->
            List StepRec -> IO (List StepRec)
rolloutEp _ _ _ Z acc = pure (reverse acc)
rolloutEp _ _ [] _ acc = pure (reverse acc)
rolloutEp model st (r :: rs) (S k) acc = do
  let stateT = bulkToTensor2d {ex=Ex} {dt=F} [observe st]
      stateV = the (Tensor [1, 4] Ex F WithGrad) (MkTensor stateT Nothing)
  predV <- forwardSeq {b=1} model stateV
  let logProbsT = primLogSoftmax2d {ex=Ex} predV.tensorPtr
      lp0 = primItem2d {ex=Ex} logProbsT 0 0
      lp1 = primItem2d {ex=Ex} logProbsT 0 1
      action = categoricalSample [exp lp0, exp lp1] r
      rowPtr = primSelect {ex=Ex} logProbsT 0 0
      selLP = primSelect {ex=Ex} rowPtr 0 (cast {to=Int} action)
      selLPVal = if action == 0 then lp0 else lp1
  case cpStep st action of
    (reward, st', outcome, _) =>
      let acc' = (selLP, selLPVal, reward) :: acc
      in if done outcome then pure (reverse acc')
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
rolloutEpBatched : {n : Nat} ->
                   Policy ->
                   VecEnv n CPState ->
                   Vect n (List Double) ->
                   Nat ->
                   IO (Vect n (List StepRec))
rolloutEpBatched model (MkVecEnv states0) rss0 maxSteps = do
  result <- go maxSteps states0 rss0 (replicate n False) (replicate n [])
  pure (map reverse result)
  where
    -- Per-env action selection + env step, given the batched log-prob
    -- tensor and this env's integer index. Frozen (done) envs and
    -- RNG-exhausted envs pass through unchanged.
    perEnv : Tensor [n, 2] Ex F WithGrad -> Int ->
             CPState -> List Double -> Bool -> List StepRec ->
             (CPState, List Double, Bool, List StepRec)
    perEnv _         _ st rs  True  acc = (st, rs, True, acc)
    perEnv _         _ st []  _     acc = (st, [], True, acc)
    perEnv logProbsV i st (r :: rs) False acc =
      let logProbsT = logProbsV.tensorPtr
          lp0       = primItem2d {ex=Ex} logProbsT i 0
          lp1       = primItem2d {ex=Ex} logProbsT i 1
          action    = categoricalSample [exp lp0, exp lp1] r
          rowPtr    = primSelect {ex=Ex} logProbsT 0 i
          selLP     = primSelect {ex=Ex} rowPtr 0 (cast {to=Int} action)
          selLPVal  = if action == 0 then lp0 else lp1
      in case cpStep st action of
           (reward, st', outcome, _) =>
             (st', rs, done outcome, (selLP, selLPVal, reward) :: acc)

    -- Walk the four parallel Vects together, threading the row index.
    stepAllEnvs : Tensor [n, 2] Ex F WithGrad -> Int ->
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
         IO (Vect n (List StepRec))
    go Z _ _ _ accs = pure accs
    go (S k) sts rss dones accs =
      if all id (toList dones) then pure accs
      else do
        let obsRows : Vect n (Vector 4 Double)
            obsRows = map observe sts
            batchPtr = bulkToTensor2d {ex=Ex} {dt=F} obsRows
            stateV : Tensor [n, 4] Ex F WithGrad
            stateV = MkTensor batchPtr Nothing
        predV <- forwardSeq {b=n} model stateV
        let logProbsV : Tensor [n, 2] Ex F WithGrad
            logProbsV = MkTensor (primLogSoftmax2d {ex=Ex} predV.tensorPtr) Nothing
        case stepAllEnvs logProbsV 0 sts rss dones accs of
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
-- scalar `Tensor []` carrying the autograd graph back to the policy.
epStepLosses : Double -> Double -> List StepRec -> List (Tensor [] Ex F WithGrad)
epStepLosses gamma baseline steps =
  let rewards = map (\(_, _, r) => r) steps
      rets = discReturns gamma rewards
  in zipWith (\(lp, _, _), gt =>
       the (Tensor [] Ex F WithGrad) (MkTensor (primMulScalar {ex=Ex} lp (baseline - gt)) Nothing))
     steps rets

export
sumRewards : List StepRec -> Double
sumRewards steps = foldl (\a, (_, _, r) => a + r) 0.0 steps

-- Mean-reduce a non-empty list of scalar tensors. Empty case returns a
-- fresh zero scalar (degenerate; runs only if the rollout produced no
-- steps).
averageLoss : List (Tensor [] Ex F WithGrad) -> Tensor [] Ex F WithGrad
averageLoss [] = MkTensor (dtCreateScalar {ex=Ex} {t=F} 0.0 0 (deviceStreamTag {ex=Ex})) Nothing
averageLoss (x :: xs) =
  let n = cast {to=Double} (1 + length xs)
      addT : Tensor [] Ex F WithGrad -> Tensor [] Ex F WithGrad -> Tensor [] Ex F WithGrad
      addT a b = MkTensor (primAdd {ex=Ex} a.tensorPtr b.tensorPtr) Nothing
      s = foldl addT x xs
  in MkTensor (primMulScalar {ex=Ex} s.tensorPtr (1.0 / n)) Nothing

computeLoss : Double -> Policy -> List (List Double) ->
              IO (Tensor [] Ex F WithGrad, Double)
computeLoss gamma model randomBatch = do
  episodes <- traverse (\rs => rolloutEp model (MkCP 0 0 0 0) rs MaxSteps []) randomBatch
  let epReturns = map sumRewards episodes
      nEp = cast {to=Double} (natToInteger (List.length epReturns))
      baseline = foldl (+) 0.0 epReturns / nEp
      stepLosses = concatMap (epStepLosses gamma baseline) episodes
  pure (averageLoss stepLosses, baseline)

computeLossBatched : {n : Nat} -> Double -> Policy -> Vect n (List Double) ->
                     IO (Tensor [] Ex F WithGrad, Double)
computeLossBatched gamma model randomBatchV = do
  resetSeedI <- randomInt32
  let initEnvs : VecEnv n CPState
      initEnvs = fst (resetAll {state=CPState} {action=Nat} {obs=Vect 4 Double}
                              (cast resetSeedI))
  epsV  <- rolloutEpBatched model initEnvs randomBatchV MaxSteps
  let eps   = toList epsV
      epReturns = map sumRewards eps
      nEp = cast {to=Double} (natToInteger (List.length epReturns))
      baseline = foldl (+) 0.0 epReturns / nEp
      stepLosses = concatMap (epStepLosses gamma baseline) eps
  pure (averageLoss stepLosses, baseline)

----------------------------------------------------------------------
-- Training
----------------------------------------------------------------------

epochRL : Optimizer Ex -> Double -> Policy -> List (List Double) ->
          IO (Policy, Double, Double)
epochRL opt gamma model batch = do
  (loss, avgRet) <- computeLoss gamma model batch
  lossVal <- nativeTrainStep opt loss
  pure (model, lossVal, avgRet)

epochRLBatched : {n : Nat} -> Optimizer Ex -> Double -> Policy -> Vect n (List Double) ->
                 IO (Policy, Double, Double)
epochRLBatched opt gamma model batchV = do
  (loss, avgRet) <- computeLossBatched gamma model batchV
  lossVal <- nativeTrainStep opt loss
  pure (model, lossVal, avgRet)

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

evalEp : Policy -> CPState -> Nat -> Double -> IO Double
evalEp _ _ Z acc = pure acc
evalEp model st (S k) acc = do
  let stateT = bulkToTensor2d {ex=Ex} {dt=F} [observe st]
      stateV = the (Tensor [1, 4] Ex F WithGrad) (MkTensor stateT Nothing)
  predV <- forwardSeq {b=1} model stateV
  let logitsT = predV.tensorPtr
      action = if primItem2d {ex=Ex} logitsT 0 0 >= primItem2d {ex=Ex} logitsT 0 1 then the Nat 0 else 1
  case cpStep st action of
    (reward, st', outcome, _) =>
      if done outcome then pure (acc + reward)
      else evalEp model st' k (acc + reward)

evalN : Policy -> Nat -> Double -> IO Double
evalN _ Z acc = pure acc
evalN model (S k) acc = do
  v <- evalEp model (MkCP 0 0 0 0) MaxSteps 0.0
  evalN model k (acc + v)

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
  batched : Bool  -- use batched policy forward per timestep

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

%default partial

main : IO ()
main = do
  requireMachine {m = ChosenMachine}
  args <- getArgs
  let cfg = parseArgs defaultConfig specs (drop 1 args)

  srand cfg.seed
  tsetInitSeed {ex = Ex} cfg.seed

  opt <- adam cfg.lr ({ clip := NormClip 1.0 } defaultOpts)

  putStrLn "=== REINFORCE on CartPole ==="
  putStrLn $ "Config: lr=" ++ show cfg.lr ++ " epochs=" ++ show cfg.epochs
           ++ " gamma=" ++ show cfg.gamma ++ " batch=" ++ show cfg.batchSz
           ++ " seed=" ++ show cfg.seed

  model <- runInit $ do
    l1 <- linear {i=4} {o=128}
    l2 <- linear {i=128} {o=2}
    pure (l1 ~~> tanhA ~~> l2 ~~> Nil)
  putStrLn ""

  when cfg.lrFind $ do
    let lrCfg : LrFindConfig
        lrCfg = { numIters := 100 } defaultLrFindConfig
    _ <- lrFind lrCfg
      (\m, d => do
         (m', loss, _) <- epochRL opt cfg.gamma m d
         pure (m', loss))
      (genBatch cfg.batchSz) opt model
    putStrLn ""
    putStrLn "Done — re-run without --lr-find at the recommended LR."
    exitSuccess

  metrics <- newRLMetricsState 100
  let n : Nat = cfg.batchSz
  (trained, epochsDone, _) <- (
    if cfg.batched
      then fit {batch = Vect n (List Double)}
             (\m, d => do
                (m', loss, avgRet) <- epochRLBatched opt cfg.gamma m d
                recordReturn metrics avgRet
                pure (m', loss))
             opt (generate (genBatchV n))
             ({ metrics := \_ => readRLMetrics "recent_100" metrics }
                (simpleConfig {model = Policy} cfg.epochs))
             model
      else fit {batch = List (List Double)}
             (\m, d => do
                (m', loss, avgRet) <- epochRL opt cfg.gamma m d
                recordReturn metrics avgRet
                pure (m', loss))
             opt (generate (genBatch cfg.batchSz))
             ({ metrics := \_ => readRLMetrics "recent_100" metrics }
                (simpleConfig {model = Policy} cfg.epochs))
             model)

  putStrLn ""
  putStrLn "Eval (100 episodes, greedy):"
  let nEval = the Nat 100
  totalReturn <- withNoGrad {ex=Ex} (evalN trained nEval 0.0)
  let avgReturn = totalReturn / cast (natToInteger nEval)
  putStrLn $ "  avg_return=" ++ show avgReturn

  putStrLn ""
  putStrLn $ formatResult [("avg_return", show avgReturn),
                            ("epochs", show epochsDone),
                            ("seed", show cfg.seed)]
