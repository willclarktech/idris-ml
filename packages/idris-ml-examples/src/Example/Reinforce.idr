module Example.Reinforce

import Control.Linear.LIO
import Data.IORef
import Data.Linear.Notation
import Data.List
import Data.Vect
import System

import Gym.ClassicControl.CartPole
import Gym.Env
import Gym.Vector
import Ml.Array
import Ml.Checkpoint
import Ml.Compat.Random
import Ml.Fit
import Ml.Hpo.LrFinder
import Ml.Rng
import Ml.Sampler
import Ml.Simple
import Ml.Train

import BuildConfig

-- The policy is a linear `Seq`; hide the IO `Nn.Seq` constructors so the chain
-- builder + threading resolve to Seq.

MaxSteps : Nat; MaxSteps = cartPoleMaxSteps

-- Array-typed observation row, the shape `bulkToTensor2d` consumes
-- (distinct from the Env method's plain `Vect 4 Double`).
export
observe : CPState -> Vector 4 Double
observe s = VArray (map SArray (cpObserve s))

-- Policy network: MLP 4 -> 128 -> tanh -> 2 (action logits). A linear `Seq`
-- threaded single-owner through every forward (rollout + eval).
public export
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
rolloutEpL : Rng -> (1 _ : Policy) -> CPState -> Nat ->
             List StepRec -> L IO {use = 1} (LPair (!* (List StepRec)) Policy)
rolloutEpL _   pol _ Z acc      = pure1 (MkBang (reverse acc) # pol)
rolloutEpL rng pol st (S k) acc = do
  stateV <- liftIO1 (ioRerun (\_ =>
    the (Tensor [1, 4] Ex F WithGrad) (MkTensor (bulkToTensor2d {ex=Ex} {dt=F} [observe st]) Nothing)))
  (MkBang predV # pol') <- forwardSeq {b=1} pol stateV
  let logProbsT = primLogSoftmax2d {ex=Ex} predV.tensorPtr
      lp0 = primItem2d {ex=Ex} logProbsT 0 0
      lp1 = primItem2d {ex=Ex} logProbsT 0 1
  action <- liftIO1 (rng.choice [exp lp0, exp lp1])
  let rowPtr   = primSelect {ex=Ex} logProbsT 0 0
      selLP    = primSelect {ex=Ex} rowPtr 0 (cast {to=Int} action)
      selLPVal = if action == 0 then lp0 else lp1
  case cpStep st action of
    (reward, st', outcome, _) =>
      let acc' = (selLP, selLPVal, reward) :: acc
      in if done outcome then pure1 (MkBang (reverse acc') # pol')
         else rolloutEpL rng pol' st' k acc'

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
                    Rng ->
                    (1 _ : Policy) ->
                    VecEnv n CPState ->
                    Nat ->
                    L IO {use = 1} (LPair (!* (Vect n (List StepRec))) Policy)
rolloutEpBatchedL rng pol (MkVecEnv states0) maxSteps = do
  (MkBang result # pol') <- go pol maxSteps states0 (replicate n False) (replicate n [])
  pure1 (MkBang (map reverse result) # pol')
  where
    -- Per-env action selection + env step, given the batched log-prob
    -- tensor and this env's integer index. Frozen (done) envs pass
    -- through without drawing.
    perEnv : Tensor [n, 2] Ex F WithGrad -> Int ->
             CPState -> Bool -> List StepRec ->
             IO (CPState, Bool, List StepRec)
    perEnv _         _ st True  acc = pure (st, True, acc)
    perEnv logProbsV i st False acc = do
      let logProbsT = logProbsV.tensorPtr
          lp0 = primItem2d {ex=Ex} logProbsT i 0
          lp1 = primItem2d {ex=Ex} logProbsT i 1
      action <- rng.choice [exp lp0, exp lp1]
      let rowPtr   = primSelect {ex=Ex} logProbsT 0 i
          selLP    = primSelect {ex=Ex} rowPtr 0 (cast {to=Int} action)
          selLPVal = if action == 0 then lp0 else lp1
      case cpStep st action of
        (reward, st', outcome, _) =>
          pure (st', done outcome, (selLP, selLPVal, reward) :: acc)

    -- Walk the parallel Vects together, threading the row index.
    stepAllEnvs : Tensor [n, 2] Ex F WithGrad -> Int ->
                  Vect k CPState -> Vect k Bool -> Vect k (List StepRec) ->
                  IO (Vect k CPState, Vect k Bool, Vect k (List StepRec))
    stepAllEnvs _         _ []          []        []            = pure ([], [], [])
    stepAllEnvs logProbsV i (st :: sts) (d :: ds) (acc :: accs) = do
      (st', d', acc') <- perEnv logProbsV i st d acc
      (sts', ds', accs') <- stepAllEnvs logProbsV (i + 1) sts ds accs
      pure (st' :: sts', d' :: ds', acc' :: accs')

    -- Recursive loop, threading the linear policy. Accumulators REVERSED.
    go : (1 _ : Policy) -> Nat ->
         Vect n CPState -> Vect n Bool -> Vect n (List StepRec) ->
         L IO {use = 1} (LPair (!* (Vect n (List StepRec))) Policy)
    go pol Z _ _ accs           = pure1 (MkBang accs # pol)
    go pol (S k) sts dones accs =
      if all id (toList dones) then pure1 (MkBang accs # pol)
      else do
        let obsRows : Vect n (Vector 4 Double)
            obsRows  = map observe sts
        stateV <- liftIO1 (ioRerun (\_ =>
          the (Tensor [n, 4] Ex F WithGrad) (MkTensor (bulkToTensor2d {ex=Ex} {dt=F} obsRows) Nothing)))
        (MkBang predV # pol') <- forwardSeq {b=n} pol stateV
        let logProbsV : Tensor [n, 2] Ex F WithGrad
            logProbsV = MkTensor (primLogSoftmax2d {ex=Ex} predV.tensorPtr) Nothing
        (sts', dones', accs') <- liftIO1 (stepAllEnvs logProbsV 0 sts dones accs)
        go pol' k sts' dones' accs'

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
computeLossL : Rng -> (nextSource : IO Source) -> (putSource : Source -> IO ()) ->
               Double -> (batchSz : Nat) -> (1 _ : Policy) ->
               L IO {use = 1} (LPair (!* (Tensor [] Ex F WithGrad, Double)) Policy)
computeLossL rng nextSource putSource gamma batchSz pol = do
  (MkBang episodes # pol') <- foldEps pol batchSz []
  let epReturns  = map sumRewards episodes
      nEp        = cast {to=Double} (natToInteger (List.length epReturns))
      baseline   = foldl (+) 0.0 epReturns / nEp
      stepLosses = concatMap (epStepLosses gamma baseline) episodes
  pure1 (MkBang (averageLoss stepLosses, baseline) # pol')
  where
    foldEps : (1 _ : Policy) -> Nat -> List (List StepRec) ->
              L IO {use = 1} (LPair (!* (List (List StepRec))) Policy)
    foldEps pol Z acc     = pure1 (MkBang (reverse acc) # pol)
    foldEps pol (S k) acc = do
      -- Fresh `reset` draw per episode, as the reference's `env.reset()`
      -- does. `nextSource` is a fresh per-episode Seeded source normally
      -- and the recording's env channel under --replay; `putSource` hands
      -- the advanced source back so successive resets keep consuming it.
      src <- liftIO1 nextSource
      let (st0, src') = reset {state=CPState} {action=Nat} {obs=Vect 4 Double} src
      liftIO1 (putSource src')
      (MkBang ep # pol') <- rolloutEpL rng pol st0 MaxSteps []
      foldEps pol' k (ep :: acc)

computeLossBatchedL : {n : Nat} -> Rng -> (nextSource : IO Source) -> (putSource : Source -> IO ()) ->
                      Double -> (1 _ : Policy) ->
                      L IO {use = 1} (LPair (!* (Tensor [] Ex F WithGrad, Double)) Policy)
computeLossBatchedL rng nextSource putSource gamma pol = do
  src <- liftIO1 nextSource
  let (initEnvs, src') = the (VecEnv n CPState, Source)
                             (resetAll {state=CPState} {action=Nat} {obs=Vect 4 Double} src)
  liftIO1 (putSource src')
  (MkBang epsV # pol') <- rolloutEpBatchedL rng pol initEnvs MaxSteps
  let eps   = toList epsV
      epReturns  = map sumRewards eps
      nEp        = cast {to=Double} (natToInteger (List.length epReturns))
      baseline   = foldl (+) 0.0 epReturns / nEp
      stepLosses = concatMap (epStepLosses gamma baseline) eps
  pure1 (MkBang (averageLoss stepLosses, baseline) # pol')

----------------------------------------------------------------------
-- Training
----------------------------------------------------------------------

----------------------------------------------------------------------
-- Evaluation (greedy argmax)
----------------------------------------------------------------------

-- Greedy eval, threading the linear policy through each step.
evalEpL : (1 _ : Policy) -> CPState -> Nat -> Double -> L IO {use = 1} (LPair (!* Double) Policy)
evalEpL pol _ Z acc      = pure1 (MkBang acc # pol)
evalEpL pol st (S k) acc = do
  stateV <- liftIO1 (ioRerun (\_ =>
    the (Tensor [1, 4] Ex F WithGrad) (MkTensor (bulkToTensor2d {ex=Ex} {dt=F} [observe st]) Nothing)))
  (MkBang predV # pol') <- forwardSeq {b=1} pol stateV
  let logitsT = predV.tensorPtr
      action = if primItem2d {ex=Ex} logitsT 0 0 >= primItem2d {ex=Ex} logitsT 0 1 then the Nat 0 else 1
  case cpStep st action of
    (reward, st', outcome, _) =>
      if done outcome then pure1 (MkBang (acc + reward) # pol')
      else evalEpL pol' st' k (acc + reward)

-- Each episode starts from a fresh `reset` draw, as the reference's
-- `env.reset()` does. A fixed start would make every greedy episode the same
-- trajectory, so the mean over N of them would carry one sample's worth of
-- information.
evalNL : (1 _ : Policy) -> Nat -> Double -> L IO {use = 1} (LPair (!* Double) Policy)
evalNL pol Z acc     = pure1 (MkBang acc # pol)
evalNL pol (S k) acc = do
  resetSeedI <- liftIO1 randomInt32
  let (st0, _) = reset {state=CPState} {action=Nat} {obs=Vect 4 Double}
                       (Seeded (cast resetSeedI))
  (MkBang v # pol') <- evalEpL pol st0 MaxSteps 0.0
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
  replay  : String

defaultConfig : Config
defaultConfig = MkConfig 0.001 2000 42 0.99 10 False False ""

specs : List (ArgSpec Config)
specs = [ Arg "--lr" (\v, c => { lr := cast v } c)
        , Arg "--epochs" (\v, c => { epochs := castNat v } c)
        , Arg "--seed" (\v, c => { seed := castBits64 v } c)
        , Arg "--gamma" (\v, c => { gamma := cast v } c)
        , Arg "--batch" (\v, c => { batchSz := castNat v } c)
        , Arg "--lr-find" (\v, c => { lrFind := (v == "1" || v == "true") } c)
        , Arg "--batched" (\v, c => { batched := (v == "1" || v == "true") } c)
        -- Replay recorded draws (`Ml.Rng.loadReplay` format) instead of
        -- sampling: actions come from the file's decision channel and
        -- episode resets from its env channel, so the rollout reproduces a
        -- recorded run exactly.
        , Arg "--replay" (\v, c => { replay := v } c) ]

-- Top-level `Init Policy` (the linear Seq MLP).
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
  discard trained'
  liftIO1 $ do
    let avgReturn = totalReturn / cast (natToInteger nEval)
    putStrLn $ "  avg_return=" ++ show avgReturn
    putStrLn ""
    putStrLn $ formatResult [("avg_return", show avgReturn),
                              ("epochs", show epochsDone),
                              ("seed", show cfg.seed)]

%default partial

lrFindCfg : LrFindConfig
lrFindCfg = { numIters := 100 } defaultLrFindConfig

-- Terminal linear consumer of the lrFind result. A named function with an
-- explicit `(1 _ : LPair ...)` signature so the bind continuation is linear
-- (the inline do-notation `<-` doesn't get recognised as linear for `lrFind`).
finishLrFind : (1 _ : LPair (!* LrFindResult) Policy) -> L IO ()
finishLrFind (MkBang _ # m') = do
  discard m'
  liftIO1 $ do
    putStrLn ""
    putStrLn "Done — re-run without --lr-find at the recommended LR."

runLrFind : Rng -> IO Source -> (Source -> IO ()) -> Config -> Optimizer Ex -> IO ()
runLrFind rng nextSource putSource cfg opt = Control.Linear.LIO.run $ do
  model <- runInitL mkPolicy
  liftIO1 (putStrLn "")
  (LIO.(>>=))
    (lrFind {ex = Ex} {model = Policy} {dp = ()} lrFindCfg
       (\m, _ => do
          (MkBang (loss, _) # m') <- computeLossL rng nextSource putSource cfg.gamma cfg.batchSz m
          dd <- liftIO1 (trainStep opt loss)
          pure1 (MkBang dd # m'))
       (pure ()) opt model)
    finishLrFind

runTrainBatched : Rng -> IO Source -> (Source -> IO ()) -> Config -> Optimizer Ex -> RLMetricsState -> (n : Nat) -> IO ()
runTrainBatched rng nextSource putSource cfg opt metrics n = Control.Linear.LIO.run $ do
  model <- runInitL mkPolicy
  liftIO1 (putStrLn "")
  (MkBang (epochsDone, _) # trained) <-
    fit {batch = ()}
         (\m, _ => do
            (MkBang (loss, avgRet) # m') <- computeLossBatchedL {n} rng nextSource putSource cfg.gamma m
            dd <- liftIO1 (do x <- trainStep opt loss; recordReturn metrics avgRet; pure x)
            pure1 (MkBang dd # m'))
         opt (generate (pure ()))
         ({ metricsL := readRLMetrics "recent_100" metrics }
            (simpleConfig {model = Policy} cfg.epochs))
         model
  evalReportL cfg epochsDone trained

runTrainSeq : Rng -> IO Source -> (Source -> IO ()) -> Config -> Optimizer Ex -> RLMetricsState -> IO ()
runTrainSeq rng nextSource putSource cfg opt metrics = Control.Linear.LIO.run $ do
  model <- runInitL mkPolicy
  liftIO1 (putStrLn "")
  (MkBang (epochsDone, _) # trained) <-
    fit {batch = ()}
         (\m, _ => do
            (MkBang (loss, avgRet) # m') <- computeLossL rng nextSource putSource cfg.gamma cfg.batchSz m
            dd <- liftIO1 (do x <- trainStep opt loss; recordReturn metrics avgRet; pure x)
            pure1 (MkBang dd # m'))
         opt (generate (pure ()))
         ({ metricsL := readRLMetrics "recent_100" metrics }
            (simpleConfig {model = Policy} cfg.epochs))
         model
  evalReportL cfg epochsDone trained

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
  -- linear (runInitL), threaded through the rollout + fit, eval'd, discarded.
  -- RL metrics are model-free → they ride metricsL.
  -- Stochastic inputs: live draws normally, the recorded channels of
  -- `--replay <file>` otherwise. Episode resets draw a fresh Seeded source
  -- per episode when live (matching the reference's per-episode
  -- env.reset()); under replay the recording's env channel is threaded
  -- across resets via an IORef.
  let mkLive : IO (Rng, IO Source, Source -> IO ())
      mkLive = do rng <- liveRng
                  pure (rng, (\i => Seeded (cast i)) <$> randomInt32, \_ => pure ())
      mkReplay : String -> IO (Rng, IO Source, Source -> IO ())
      mkReplay pth = do replay <- loadReplay pth
                        srcRef <- newIORef replay.envSource
                        pure (replay.rng, readIORef srcRef, writeIORef srcRef)
  (rng, nextSource, putSource) <- case cfg.replay of
                                    ""  => mkLive
                                    pth => mkReplay pth
  if cfg.lrFind
    then runLrFind rng nextSource putSource cfg opt
    else if cfg.batched
      then runTrainBatched rng nextSource putSource cfg opt metrics n
      else runTrainSeq rng nextSource putSource cfg opt metrics
