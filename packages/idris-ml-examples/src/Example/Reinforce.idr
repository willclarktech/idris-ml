module Example.Reinforce

import Control.Linear.LIO
import Data.Linear.Notation
import Data.List
import Data.Vect
import System

import Array
import BuildConfig
import Compat.Random
import FitL
import Gym.ClassicControl.CartPole
import Gym.Env
import Gym.Vector
import Hpo.LrFinder
import ML.Simple
import Sampler
import Train

-- The policy is a linear `SeqL`; hide the IO `Nn.Seq` constructors so the chain
-- builder + threading resolve to SeqL.
%hide Nn.Seq.Nil
%hide Nn.Seq.(::)
%hide Nn.Seq.(~~>)

MaxSteps : Nat; MaxSteps = cartPoleMaxSteps

-- Array-typed observation row, the shape `bulkToTensor2d` consumes
-- (distinct from the Env method's plain `Vect 4 Double`).
export
observe : CPState -> Vector 4 Double
observe s = VArray (map SArray (cpObserve s))

-- Policy network: MLP 4 -> 128 -> tanh -> 2 (action logits). A linear `SeqL`
-- threaded single-owner through every forward (rollout + eval).
public export
Policy : Type
Policy = SeqL 4 2 Ex F WithGrad

----------------------------------------------------------------------
-- Episode Rollout (tensor-level, autograd-tracked)
----------------------------------------------------------------------

-- (logProbTensorPtr, logProbDoubleVal, reward)
public export
StepRec : Type
StepRec = (AnyPtr, Double, Double)

export
rolloutEpL : (1 _ : Policy) -> CPState -> List Double -> Nat ->
             List StepRec -> L IO {use = 1} (LPair (!* (List StepRec)) Policy)
rolloutEpL pol _ _ Z acc              = pure1 (MkBang (reverse acc) # pol)
rolloutEpL pol _ [] _ acc             = pure1 (MkBang (reverse acc) # pol)
rolloutEpL pol st (r :: rs) (S k) acc = do
  stateV <- liftIO1 (ioRerun (\_ =>
    the (Tensor [1, 4] Ex F WithGrad) (MkTensor (bulkToTensor2d {ex=Ex} {dt=F} [observe st]) Nothing)))
  (MkBang predV # pol') <- forwardSeqL {b=1} pol stateV
  let logProbsT = primLogSoftmax2d {ex=Ex} predV.tensorPtr
      lp0      = primItem2d {ex=Ex} logProbsT 0 0
      lp1      = primItem2d {ex=Ex} logProbsT 0 1
      action   = categoricalSample [exp lp0, exp lp1] r
      rowPtr   = primSelect {ex=Ex} logProbsT 0 0
      selLP    = primSelect {ex=Ex} rowPtr 0 (cast {to=Int} action)
      selLPVal = if action == 0 then lp0 else lp1
  case cpStep st action of
    (reward, st', outcome, _) =>
      let acc' = (selLP, selLPVal, reward) :: acc
      in if done outcome then pure1 (MkBang (reverse acc') # pol')
         else rolloutEpL pol' st' rs k acc'

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
rolloutEpBatchedL : {n : Nat} ->
                    (1 _ : Policy) ->
                    VecEnv n CPState ->
                    Vect n (List Double) ->
                    Nat ->
                    L IO {use = 1} (LPair (!* (Vect n (List StepRec))) Policy)
rolloutEpBatchedL pol (MkVecEnv states0) rss0 maxSteps = do
  (MkBang result # pol') <- go pol maxSteps states0 rss0 (replicate n False) (replicate n [])
  pure1 (MkBang (map reverse result) # pol')
  where
    -- Per-env action selection + env step, given the batched log-prob
    -- tensor and this env's integer index. Frozen (done) envs and
    -- RNG-exhausted envs pass through unchanged.
    perEnv : Tensor [n, 2] Ex F WithGrad -> Int ->
             CPState -> List Double -> Bool -> List StepRec ->
             (CPState, List Double, Bool, List StepRec)
    perEnv _         _ st rs  True  acc       = (st, rs, True, acc)
    perEnv _         _ st []  _     acc       = (st, [], True, acc)
    perEnv logProbsV i st (r :: rs) False acc =
      let logProbsT = logProbsV.tensorPtr
          lp0      = primItem2d {ex=Ex} logProbsT i 0
          lp1      = primItem2d {ex=Ex} logProbsT i 1
          action   = categoricalSample [exp lp0, exp lp1] r
          rowPtr   = primSelect {ex=Ex} logProbsT 0 i
          selLP    = primSelect {ex=Ex} rowPtr 0 (cast {to=Int} action)
          selLPVal = if action == 0 then lp0 else lp1
      in case cpStep st action of
           (reward, st', outcome, _) =>
             (st', rs, done outcome, (selLP, selLPVal, reward) :: acc)

    -- Walk the four parallel Vects together, threading the row index.
    stepAllEnvs : Tensor [n, 2] Ex F WithGrad -> Int ->
                  Vect k CPState -> Vect k (List Double) -> Vect k Bool ->
                  Vect k (List StepRec) ->
                  (Vect k CPState, Vect k (List Double), Vect k Bool, Vect k (List StepRec))
    stepAllEnvs _         _ []         []         []         []             = ([], [], [], [])
    stepAllEnvs logProbsV i (st :: sts) (rs :: rss) (d :: ds) (acc :: accs) =
      let (st', rs', d', acc') = perEnv logProbsV i st rs d acc
          (sts', rss', ds', accs') = stepAllEnvs logProbsV (i + 1) sts rss ds accs
      in (st' :: sts', rs' :: rss', d' :: ds', acc' :: accs')

    -- Recursive loop, threading the linear policy. Accumulators REVERSED.
    go : (1 _ : Policy) -> Nat ->
         Vect n CPState -> Vect n (List Double) -> Vect n Bool ->
         Vect n (List StepRec) ->
         L IO {use = 1} (LPair (!* (Vect n (List StepRec))) Policy)
    go pol Z _ _ _ accs             = pure1 (MkBang accs # pol)
    go pol (S k) sts rss dones accs =
      if all id (toList dones) then pure1 (MkBang accs # pol)
      else do
        let obsRows : Vect n (Vector 4 Double)
            obsRows  = map observe sts
        stateV <- liftIO1 (ioRerun (\_ =>
          the (Tensor [n, 4] Ex F WithGrad) (MkTensor (bulkToTensor2d {ex=Ex} {dt=F} obsRows) Nothing)))
        (MkBang predV # pol') <- forwardSeqL {b=n} pol stateV
        let logProbsV : Tensor [n, 2] Ex F WithGrad
            logProbsV = MkTensor (primLogSoftmax2d {ex=Ex} predV.tensorPtr) Nothing
        case stepAllEnvs logProbsV 0 sts rss dones accs of
          (sts', rss', dones', accs') => go pol' k sts' rss' dones' accs'

----------------------------------------------------------------------
-- REINFORCE Loss
----------------------------------------------------------------------

discReturns : Double -> List Double -> List Double
discReturns gamma rewards = reverse (go 0.0 (reverse rewards))
  where
    go : Double -> List Double -> List Double
    go _ []        = []
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
averageLoss []        = MkTensor (dtCreateScalar {ex=Ex} {t=F} 0.0 0 (deviceStreamTag {ex=Ex})) Nothing
averageLoss (x :: xs) =
  let n = cast {to=Double} (1 + length xs)
      addT : Tensor [] Ex F WithGrad -> Tensor [] Ex F WithGrad -> Tensor [] Ex F WithGrad
      addT a b = MkTensor (primAdd {ex=Ex} a.tensorPtr b.tensorPtr) Nothing
      s        = foldl addT x xs
  in MkTensor (primMulScalar {ex=Ex} s.tensorPtr (1.0 / n)) Nothing

-- Thread the linear policy across the batch's episodes (one rolloutEpL each),
-- then build the REINFORCE loss + baseline from the collected StepRecs.
computeLossL : Double -> (1 _ : Policy) -> List (List Double) ->
               L IO {use = 1} (LPair (!* (Tensor [] Ex F WithGrad, Double)) Policy)
computeLossL gamma pol randomBatch = do
  (MkBang episodes # pol') <- foldEps pol randomBatch []
  let epReturns  = map sumRewards episodes
      nEp        = cast {to=Double} (natToInteger (List.length epReturns))
      baseline   = foldl (+) 0.0 epReturns / nEp
      stepLosses = concatMap (epStepLosses gamma baseline) episodes
  pure1 (MkBang (averageLoss stepLosses, baseline) # pol')
  where
    foldEps : (1 _ : Policy) -> List (List Double) -> List (List StepRec) ->
              L IO {use = 1} (LPair (!* (List (List StepRec))) Policy)
    foldEps pol []          acc  = pure1 (MkBang (reverse acc) # pol)
    foldEps pol (rs :: rest) acc = do
      (MkBang ep # pol') <- rolloutEpL pol (MkCP 0 0 0 0) rs MaxSteps []
      foldEps pol' rest (ep :: acc)

computeLossBatchedL : {n : Nat} -> Double -> (1 _ : Policy) -> Vect n (List Double) ->
                      L IO {use = 1} (LPair (!* (Tensor [] Ex F WithGrad, Double)) Policy)
computeLossBatchedL gamma pol randomBatchV = do
  resetSeedI <- liftIO1 randomInt32
  let initEnvs : VecEnv n CPState
      initEnvs = fst (resetAll {state=CPState} {action=Nat} {obs=Vect 4 Double}
                              (cast resetSeedI))
  (MkBang epsV # pol') <- rolloutEpBatchedL pol initEnvs randomBatchV MaxSteps
  let eps   = toList epsV
      epReturns  = map sumRewards eps
      nEp        = cast {to=Double} (natToInteger (List.length epReturns))
      baseline   = foldl (+) 0.0 epReturns / nEp
      stepLosses = concatMap (epStepLosses gamma baseline) eps
  pure1 (MkBang (averageLoss stepLosses, baseline) # pol')

----------------------------------------------------------------------
-- Training
----------------------------------------------------------------------

genBatch : Nat -> IO (List (List Double))
genBatch batchSz = go batchSz
  where
    genN : Nat -> IO (List Double)
    genN Z     = pure []
    genN (S k) = do
      r <- randomRIO (the Double 0.0, 1.0)
      rs <- genN k
      pure (r :: rs)

    go : Nat -> IO (List (List Double))
    go Z     = pure []
    go (S k) = do
      ep <- genN MaxSteps
      rest <- go k
      pure (ep :: rest)

genBatchV : (n : Nat) -> IO (Vect n (List Double))
genBatchV Z     = pure []
genBatchV (S k) = do
  ep <- go MaxSteps
  rest <- genBatchV k
  pure (ep :: rest)
  where
    go : Nat -> IO (List Double)
    go Z      = pure []
    go (S k') = do
      r <- randomRIO (the Double 0.0, 1.0)
      rs <- go k'
      pure (r :: rs)

----------------------------------------------------------------------
-- Evaluation (greedy argmax)
----------------------------------------------------------------------

-- Greedy eval, threading the linear policy through each step.
evalEpL : (1 _ : Policy) -> CPState -> Nat -> Double -> L IO {use = 1} (LPair (!* Double) Policy)
evalEpL pol _ Z acc      = pure1 (MkBang acc # pol)
evalEpL pol st (S k) acc = do
  stateV <- liftIO1 (ioRerun (\_ =>
    the (Tensor [1, 4] Ex F WithGrad) (MkTensor (bulkToTensor2d {ex=Ex} {dt=F} [observe st]) Nothing)))
  (MkBang predV # pol') <- forwardSeqL {b=1} pol stateV
  let logitsT = predV.tensorPtr
      action = if primItem2d {ex=Ex} logitsT 0 0 >= primItem2d {ex=Ex} logitsT 0 1 then the Nat 0 else 1
  case cpStep st action of
    (reward, st', outcome, _) =>
      if done outcome then pure1 (MkBang (acc + reward) # pol')
      else evalEpL pol' st' k (acc + reward)

evalNL : (1 _ : Policy) -> Nat -> Double -> L IO {use = 1} (LPair (!* Double) Policy)
evalNL pol Z acc     = pure1 (MkBang acc # pol)
evalNL pol (S k) acc = do
  (MkBang v # pol') <- evalEpL pol (MkCP 0 0 0 0) MaxSteps 0.0
  evalNL pol' k (acc + v)

----------------------------------------------------------------------
-- Config & Main
----------------------------------------------------------------------

record Config where
  constructor MkConfig
  lr      : Double
  epochs  : Nat
  seed    : Bits64
  gamma   : Double
  batchSz : Nat
  lrFind  : Bool
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

-- Top-level `Init Policy` (the linear SeqL MLP).
mkPolicy : Init Policy
mkPolicy = do
  l1 <- linear {i=4} {o=128}
  l2 <- linear {i=128} {o=2}
  pure (l1 ~~> tanhA ~~> l2 ~~> Nil)

-- Eval + RESULT report; consumes the trained (linear) policy under withNoGradL.
evalReportL : Config -> Nat -> (1 _ : Policy) -> L IO ()
evalReportL cfg epochsDone trained = do
  liftIO1 (putStrLn "" >> putStrLn "Eval (100 episodes, greedy):")
  let nEval = the Nat 100
  (MkBang totalReturn # trained') <- withNoGradL {ex=Ex} (evalNL trained nEval 0.0)
  discardL trained'
  liftIO1 $ do
    let avgReturn = totalReturn / cast (natToInteger nEval)
    putStrLn $ "  avg_return=" ++ show avgReturn
    putStrLn ""
    putStrLn $ formatResult [("avg_return", show avgReturn),
                              ("epochs", show epochsDone),
                              ("seed", show cfg.seed)]

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

  metrics <- newRLMetricsState 100
  let n : Nat = cfg.batchSz

  -- Three self-contained run blocks (avoids binding a linear result across an
  -- `if`): lrFind, batched training, non-batched training. The policy is born
  -- linear (runInitL), threaded through the rollout + fitL, eval'd, discarded.
  -- RL metrics are model-free → they ride metricsL.
  if cfg.lrFind
    then Control.Linear.LIO.run $ do
      model <- runInitL mkPolicy
      liftIO1 (putStrLn "")
      let lrCfg : LrFindConfig
          lrCfg = { numIters := 100 } defaultLrFindConfig
      (MkBang _ # m') <- lrFindL lrCfg
        (\m, d => do
           (MkBang (loss, _) # m') <- computeLossL cfg.gamma m d
           dd <- liftIO1 (nativeTrainStep opt loss)
           pure1 (MkBang dd # m'))
        (genBatch cfg.batchSz) opt model
      discardL m'
      liftIO1 $ do
        putStrLn ""
        putStrLn "Done — re-run without --lr-find at the recommended LR."
    else if cfg.batched
      then Control.Linear.LIO.run $ do
        model <- runInitL mkPolicy
        liftIO1 (putStrLn "")
        (MkBang (epochsDone, _) # trained) <-
          fitL {batch = Vect n (List Double)}
               (\m, d => do
                  (MkBang (loss, avgRet) # m') <- computeLossBatchedL cfg.gamma m d
                  dd <- liftIO1 (do x <- nativeTrainStep opt loss; recordReturn metrics avgRet; pure x)
                  pure1 (MkBang dd # m'))
               opt (generate (genBatchV n))
               ({ metricsL := readRLMetrics "recent_100" metrics }
                  (simpleConfig {model = Policy} cfg.epochs))
               model
        evalReportL cfg epochsDone trained
      else Control.Linear.LIO.run $ do
        model <- runInitL mkPolicy
        liftIO1 (putStrLn "")
        (MkBang (epochsDone, _) # trained) <-
          fitL {batch = List (List Double)}
               (\m, d => do
                  (MkBang (loss, avgRet) # m') <- computeLossL cfg.gamma m d
                  dd <- liftIO1 (do x <- nativeTrainStep opt loss; recordReturn metrics avgRet; pure x)
                  pure1 (MkBang dd # m'))
               opt (generate (genBatch cfg.batchSz))
               ({ metricsL := readRLMetrics "recent_100" metrics }
                  (simpleConfig {model = Policy} cfg.epochs))
               model
        evalReportL cfg epochsDone trained
