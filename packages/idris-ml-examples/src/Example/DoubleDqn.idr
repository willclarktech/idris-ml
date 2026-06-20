module Example.DoubleDqn

import Control.Linear.LIO
import Data.Fin
import Data.IORef
import Data.Linear.Notation
import Data.List
import Data.String
import Data.Vect
import System

import Array
import BuildConfig
import Compat.Random
import Fit
import Gym.ClassicControl.CartPole
import Gym.Env
import Gym.Vector
import Hpo.LrFinder
import ML.Simple
import RL.ReplayBuffer
import Train
import Train.Freeze

-- The Q-nets are linear `Seq`s; hide the IO `Nn.Seq` constructors.

----------------------------------------------------------------------
-- Architecture: MLP 4 -> 64 -> relu -> 64 -> relu -> 2
----------------------------------------------------------------------

ObsDim     : Nat; ObsDim = 4
Hidden     : Nat; Hidden = 64
NumActions : Nat; NumActions = 2
MaxSteps   : Nat; MaxSteps = cartPoleMaxSteps

||| Parallel envs collecting transitions in lockstep. Mirrors PyTorch's
||| `gym.vector.SyncVectorEnv`. Env-0 is the "primary" (its episode
||| boundary marks the end of an Idris epoch).
NumEnvs : Nat; NumEnvs = 4

QNet : Type
QNet = Seq ObsDim NumActions Ex F WithGrad

||| Build a Q-network with all params registered under `<scope>.*`. Reuse
||| the same architecture for online and target nets; their exact param names
||| (from `reflectNames`) are paired positionally by `polyakUpdatePaired` for
||| the target sync — the scope string is just a registry namespace, not a
||| matching key. An `Init` (run with `runInitL` to be born linear).
mkQNet : (scope : String) -> Init QNet
mkQNet scope = scoped scope $ do
  l1 <- linear {i=ObsDim} {o=Hidden}
  l2 <- linear {i=Hidden} {o=Hidden}
  l3 <- linear {i=Hidden} {o=NumActions}
  pure (l1 ~~> reluA ~~> l2 ~~> reluA ~~> l3 ~~> Nil)

----------------------------------------------------------------------
-- Observation helper
----------------------------------------------------------------------

observeVec : CPState -> Vect ObsDim Double
observeVec s = cpObserve s

obsTensor : Vect ObsDim Double -> Vector ObsDim Double
obsTensor v = VArray (map SArray v)

----------------------------------------------------------------------
-- Epsilon-greedy action selection (uses online net's current weights)
----------------------------------------------------------------------

epsilonAt : Nat -> Double -> Double -> Nat -> Double
epsilonAt step start end decaySteps =
  let frac = min (cast (natToInteger step) / cast (natToInteger decaySteps)) 1.0
  in start + frac * (end - start)

-- Argmax over Q(s, *) from the online net (single [1, ObsDim] forward).
-- Argmax over Q(s, *) from the online net (threads the linear net).
greedyActionL : (1 _ : QNet) -> Vect ObsDim Double -> L IO {use = 1} (LPair (!* Nat) QNet)
greedyActionL online obs = do
  stateV <- liftIO1 (ioRerun (\_ =>
    the (Tensor [1, ObsDim] Ex F WithGrad) (MkTensor (bulkToTensor2d {ex=Ex} {dt=F} [obsTensor obs]) Nothing)))
  (MkBang qV # online') <- forwardSeq {b=1} online stateV
  let q0 = primItem2d {ex=Ex} qV.tensorPtr 0 0
      q1 = primItem2d {ex=Ex} qV.tensorPtr 0 1
  pure1 (MkBang (if q0 >= q1 then the Nat 0 else 1) # online')

-- Batched epsilon-greedy: given a [N, NumActions] Q-tensor and the
-- current N envs, sample one action per env (one randomRIO per env).
epsGreedyBatched : {n : Nat} -> Tensor [n, NumActions] Ex F g ->
                   Vect n CPState -> Double -> IO (Vect n Nat)
epsGreedyBatched qB envs eps = go 0 envs
  where
    go : Int -> Vect k CPState -> IO (Vect k Nat)
    go _ []          = pure []
    go i (_ :: rest) = do
      u <- randomRIO (the Double 0.0, 1.0)
      a <- if u < eps
             then do
               u2 <- randomRIO (the Double 0.0, 1.0)
               pure (if u2 < 0.5 then 0 else 1)
             else do
               let q0 = primItem2d {ex=Ex} qB.tensorPtr i 0
                   q1 = primItem2d {ex=Ex} qB.tensorPtr i 1
               pure (if q0 >= q1 then 0 else 1)
      as <- go (i + 1) rest
      pure (a :: as)

----------------------------------------------------------------------
-- Double DQN loss (batched online Q; bootstrap target forwarded
-- per-sample and read back as a Double — no autograd chain into either
-- net, the optimizer is scoped to "online" only).
----------------------------------------------------------------------

-- Argmax over row 0 of a [1, NumActions] tensor pointer (the online net's
-- action selection in the Double DQN target).
rowArgmax2d : AnyPtr -> Nat
rowArgmax2d t =
  let v0 = primItem2d {ex=Ex} t 0 0
      v1 = primItem2d {ex=Ex} t 0 1
  in if v0 >= v1 then 0 else 1

-- Double DQN bootstrap target: the ONLINE net selects a* = argmax_a
-- Q_online(s', a) (under no-grad), the TARGET net evaluates Q_target(s', a*).
-- Decoupling selection from evaluation cuts vanilla DQN's maximization bias.
-- Both linear nets are threaded; the result is a plain Double (the readbacks
-- produce values, not tensor ops, so no graph connects to the loss).
computeTargetValL : (1 _ : QNet) -> (1 _ : QNet) -> Double -> Transition ObsDim 1 ->
                    L IO {use = 1} (LPair (!* Double) (LPair QNet QNet))
computeTargetValL online target gamma t = do
  selV <- liftIO1 (ioRerun (\_ =>
    the (Tensor [1, ObsDim] Ex F WithGrad) (MkTensor (bulkToTensor2d {ex=Ex} {dt=F} [obsTensor t.nextObs]) Nothing)))
  (MkBang aStar # online') <- withNoGradL {ex=Ex} $ do
    (MkBang qOn # online') <- forwardSeq {b=1} online selV
    pure1 (MkBang (rowArgmax2d qOn.tensorPtr) # online')
  evalV <- liftIO1 (ioRerun (\_ =>
    the (Tensor [1, ObsDim] Ex F WithGrad) (MkTensor (bulkToTensor2d {ex=Ex} {dt=F} [obsTensor t.nextObs]) Nothing)))
  (MkBang qTg # target') <- forwardSeq {b=1} target evalV
  let q0 = primItem2d {ex=Ex} qTg.tensorPtr 0 0
      q1        = primItem2d {ex=Ex} qTg.tensorPtr 0 1
      nextQ     = if aStar == 0 then q0 else q1
      bootstrap = if t.done then 0.0 else gamma * nextQ
  pure1 (MkBang (t.reward + bootstrap) # (online' # target'))

actionIdx : Vect 1 Double -> Int
actionIdx [a] = cast {to=Int} (cast {to=Integer} a)

perSampleLoss : {n : Nat} -> (qOutB : Tensor [n, NumActions] Ex F WithGrad) ->
                Transition ObsDim 1 -> Double -> Int -> IO (Tensor [] Ex F WithGrad)
perSampleLoss qOutB t tv k = do
  let aIdx = actionIdx t.action
  qRow    <- trowSelect qOutB k
  qScalar <- telemSelect qRow aIdx
  targetT <- tconstScalar tv
  diff    <- tsub qScalar targetT
  tmul diff diff

meanScalarLoss : (n : Nat) -> List (Tensor [] Ex F WithGrad) -> IO (Tensor [] Ex F WithGrad)
meanScalarLoss n losses = do
  zero <- tconstScalar 0.0
  let summed = foldl (\a, b => MkTensor (primAdd {ex=Ex} a.tensorPtr b.tensorPtr) Nothing) zero losses
  tmulScalar summed (1.0 / cast n)

-- Thread BOTH (linear) nets across the batch, collecting Double DQN
-- bootstrap target values in order.
foldTargetsL : (1 _ : QNet) -> (1 _ : QNet) -> Double -> List (Transition ObsDim 1) -> List Double ->
               L IO {use = 1} (LPair (!* (List Double)) (LPair QNet QNet))
foldTargetsL online target _     []          acc = pure1 (MkBang (reverse acc) # (online # target))
foldTargetsL online target gamma (t :: rest) acc = do
  (MkBang tv # (online' # target')) <- computeTargetValL online target gamma t
  foldTargetsL online' target' gamma rest (tv :: acc)

-- Double DQN loss: thread both nets through the per-sample bootstrap fold
-- (online selects, target evaluates), then the batched Q_online(s, a) forward,
-- then the per-sample squared TD error (pure on the ω forward output) → mean.
-- Returns the loss beside BOTH nets (nested LPair).
batchLossBatchedL : (n : Nat) -> (1 _ : QNet) -> (1 _ : QNet) -> Double ->
                    Vect n (Transition ObsDim 1) ->
                    L IO {use = 1} (LPair (!* (Tensor [] Ex F WithGrad)) (LPair QNet QNet))
batchLossBatchedL n online target gamma batch = do
  (MkBang targetVals # (online' # target')) <- foldTargetsL online target gamma (toList batch) []
  obsBV <- liftIO1 (ioRerun (\_ =>
    the (Tensor [n, ObsDim] Ex F WithGrad)
        (MkTensor (bulkToTensor2d {ex=Ex} {dt=F} (map (\t => obsTensor t.obs) batch)) Nothing)))
  (MkBang qOutB # online'') <- forwardSeq {b=n} online' obsBV
  loss <- liftIO1 $ do
            losses <- go qOutB (toList batch) targetVals 0
            meanScalarLoss n losses
  pure1 (MkBang loss # (online'' # target'))
  where
    go : {n : Nat} -> Tensor [n, NumActions] Ex F WithGrad ->
         List (Transition ObsDim 1) ->
         List Double -> Int -> IO (List (Tensor [] Ex F WithGrad))
    go _ [] _ _                            = pure []
    go _ _ [] _                            = pure []
    go qOutB (t :: tRest) (tv :: tvRest) k = do
      l <- perSampleLoss qOutB t tv k
      ls <- go qOutB tRest tvRest (k + 1)
      pure (l :: ls)

----------------------------------------------------------------------
-- Double DQN state threaded through training
----------------------------------------------------------------------

-- The two Q-nets are **linear** fields (threaded single-owner through the
-- rollout); the buffer / IORefs / config scalars are ω.
record DqnState where
  constructor MkDqnState
  1 qNet       : QNet
  1 target     : QNet
  buffer       : ReplayBuffer ObsDim 1
  envsRef      : IORef (VecEnv NumEnvs CPState)
  stepRef      : IORef Nat
  cfgEpsStart  : Double
  cfgEpsEnd    : Double
  cfgEpsDecay  : Nat
  cfgSyncEvery : Nat
  cfgBatch     : Nat
  cfgGamma     : Double
  -- exact registry names of the online / target nets (from reflectNames),
  -- paired positionally by polyakUpdatePaired — no string-prefix scoping.
  onNames  : List String
  tgtNames : List String

----------------------------------------------------------------------
-- Episode rollout with Double DQN updates
----------------------------------------------------------------------

actionToVec : Nat -> Vect 1 Double
actionToVec a = [cast (natToInteger a)]

-- Train-if-ready, threading BOTH nets (online + target) through the batch loss
-- + optimizer step (trainStep updates online's params by registry scope;
-- the net values pass through). Returns both nets in a nested LPair.
trainIfReadyL : Optimizer Ex -> ReplayBuffer ObsDim 1 -> Nat -> Double ->
                (1 _ : QNet) -> (1 _ : QNet) -> L IO {use = 1} (LPair QNet QNet)
trainIfReadyL opt buffer cfgBatch gamma online target = do
  bufSz <- liftIO1 (bufferSize buffer)
  if bufSz < cfgBatch
    then pure1 (online # target)
    else do
      mBatch <- liftIO1 (sampleN cfgBatch buffer)
      case mBatch of
        Just batchVec => do
          (MkBang loss # (online' # target')) <- batchLossBatchedL cfgBatch online target gamma batchVec
          _ <- liftIO1 (trainStep opt loss)
          pure1 (online' # target')
        Nothing => pure1 (online # target)

----------------------------------------------------------------------
-- Batched episode rollout: NumEnvs parallel envs collect transitions
-- in lockstep; one batched action-selection forward per outer step.
-- env-0 is the "primary" env — its episode boundary marks the end of
-- a training epoch.
----------------------------------------------------------------------

stepAllAutoResetDqn : Vect n CPState -> Vect n Nat ->
                     (Vect n CPState, Vect n Double, Vect n Bool)
stepAllAutoResetDqn [] []               = ([], [], [])
stepAllAutoResetDqn (s :: ss) (a :: as) =
  case cpStep s a of
    (r, s', outcome, _) =>
      let isDone = done outcome
          nextS  = if isDone then MkCP 0 0 0 0 else s'
      in case stepAllAutoResetDqn ss as of
           (rest, rs, ds) => (nextS :: rest, r :: rs, isDone :: ds)

pushAllTransitions : ReplayBuffer ObsDim 1 ->
                     Vect n CPState -> Vect n Nat -> Vect n Double ->
                     Vect n CPState -> Vect n Bool -> IO ()
pushAllTransitions _ [] [] [] [] []                                        = pure ()
pushAllTransitions buf (s :: ss) (a :: as) (r :: rs) (s' :: ss') (d :: ds) = do
  push buf (MkTransition (observeVec s) (actionToVec a) r (observeVec s') d)
  pushAllTransitions buf ss as rs ss' ds

runEpisodeBatchedL : Optimizer Ex -> (1 _ : DqnState) -> L IO {use = 1} (LPair (!* Double) DqnState)
runEpisodeBatchedL opt (MkDqnState qNet target buffer envsRef stepRef epsStart epsEnd epsDecay syncEvery batch gamma onNames tgtNames) = do
  startEnvs <- liftIO1 (readIORef envsRef)
  (MkBang ret # (qNet' # target')) <- go qNet target startEnvs.envs MaxSteps 0.0
  pure1 (MkBang ret # MkDqnState qNet' target' buffer envsRef stepRef epsStart epsEnd epsDecay syncEvery batch gamma onNames tgtNames)
  where
    -- Thread BOTH nets (nested LPair) through the lockstep rollout; the ω
    -- buffer / IORefs / config are captured from the outer match.
    go : (1 _ : QNet) -> (1 _ : QNet) -> Vect NumEnvs CPState -> Nat -> Double ->
         L IO {use = 1} (LPair (!* Double) (LPair QNet QNet))
    go qNet target _ Z ret = do
      liftIO1 (writeIORef envsRef (MkVecEnv (replicate NumEnvs (MkCP 0 0 0 0))))
      pure1 (MkBang ret # (qNet # target))
    go qNet target envs (S steps) ret = do
      stepCount <- liftIO1 (readIORef stepRef)
      let eps = epsilonAt stepCount epsStart epsEnd epsDecay
      stateV <- liftIO1 (ioRerun (\_ =>
        the (Tensor [NumEnvs, ObsDim] Ex F WithGrad)
            (MkTensor (bulkToTensor2d {ex=Ex} {dt=F} (map (\s => obsTensor (observeVec s)) envs)) Nothing)))
      -- Batched action-selection forward under no-grad: read qB INSIDE the
      -- bracket (before the exit drain frees it), thread qNet out.
      (MkBang actions # qNet') <- withNoGradL {ex=Ex} $ do
        (MkBang qB # qNet') <- forwardSeq {b=NumEnvs} qNet stateV
        acts <- liftIO1 (epsGreedyBatched qB envs eps)
        pure1 (MkBang acts # qNet')
      case stepAllAutoResetDqn envs actions of
        (envs', rewards, dones) => do
          liftIO1 (pushAllTransitions buffer envs actions rewards envs' dones)
          liftIO1 (writeIORef stepRef (stepCount + 1))
          let ret0  = head rewards
              done0 = head dones
              ret'  = ret + ret0
          (qNet'' # target') <- trainIfReadyL opt buffer batch gamma qNet' target
          liftIO1 (when ((stepCount + 1) `mod` syncEvery == 0) $ do
                     _ <- polyakUpdatePaired {ex=Ex} onNames tgtNames 1.0
                     pure ())
          if done0
            then do
              liftIO1 (writeIORef envsRef (MkVecEnv envs'))
              pure1 (MkBang ret' # (qNet'' # target'))
            else go qNet'' target' envs' steps ret'

----------------------------------------------------------------------
-- Config & epoch
----------------------------------------------------------------------

record Config where
  constructor MkConfig
  lr         : Double
  epochs     : Nat
  gamma      : Double
  batchSize  : Nat
  bufferCap  : Nat
  targetSync : Nat
  epsStart   : Double
  epsEnd     : Double
  epsDecay   : Nat
  seed       : Bits64
  lrFind     : Bool

defaultConfig : Config
defaultConfig = MkConfig 5.0e-4 300 0.99 64 10000 100 1.0 0.05 10000 42 False

specs : List (ArgSpec Config)
specs = [ Arg "--lr" (\v, c => { lr := cast v } c)
        , Arg "--epochs" (\v, c => { epochs := castNat v } c)
        , Arg "--gamma" (\v, c => { gamma := cast v } c)
        , Arg "--batch" (\v, c => { batchSize := castNat v } c)
        , Arg "--buffer-cap" (\v, c => { bufferCap := castNat v } c)
        , Arg "--target-sync" (\v, c => { targetSync := castNat v } c)
        , Arg "--eps-start" (\v, c => { epsStart := cast v } c)
        , Arg "--eps-end" (\v, c => { epsEnd := cast v } c)
        , Arg "--eps-decay" (\v, c => { epsDecay := castNat v } c)
        , Arg "--seed" (\v, c => { seed := castBits64 v } c)
        , Arg "--lr-find" (\v, c => { lrFind := (v == "1" || v == "true") } c)
        ]

----------------------------------------------------------------------
-- Greedy evaluation
----------------------------------------------------------------------

evalEpL : (1 _ : QNet) -> CPState -> Nat -> Double -> L IO {use = 1} (LPair (!* Double) QNet)
evalEpL q _ Z acc      = pure1 (MkBang acc # q)
evalEpL q st (S k) acc = do
  (MkBang action # q') <- greedyActionL q (observeVec st)
  case cpStep st action of
    (reward, st', outcome, _) =>
      if done outcome then pure1 (MkBang (acc + reward) # q')
      else evalEpL q' st' k (acc + reward)

evalNL : (1 _ : QNet) -> Nat -> Double -> L IO {use = 1} (LPair (!* Double) QNet)
evalNL q Z acc     = pure1 (MkBang acc # q)
evalNL q (S k) acc = do
  (MkBang ep # q') <- evalEpL q (MkCP 0 0 0 0) MaxSteps 0.0
  evalNL q' k (acc + ep)

----------------------------------------------------------------------
-- Main
----------------------------------------------------------------------

-- Build the born-linear DQN state inside the linear block: nets via runInitL
-- (so they're linear), then the initial hard target sync + buffer + env/step
-- refs (ω). Optimizer is constructed by the caller *after* this (params must be
-- registered first).
buildStateL : Config -> L IO {use = 1} DqnState
buildStateL cfg = do
  qNet0   <- runInitL (mkQNet "online")
  target0 <- runInitL (mkQNet "target")
  -- Capture the nets' exact param names once (static for the run); pair them
  -- positionally for every target sync, so no naming convention is load-bearing.
  let (MkBang onNames # qNet0)    = reflectNames qNet0
  let (MkBang tgtNames # target0) = reflectNames target0
  liftIO1 (do _ <- polyakUpdatePaired {ex=Ex} onNames tgtNames 1.0; pure ())
  buffer  <- liftIO1 (mkBuffer {obsDim = ObsDim, actDim = 1} cfg.bufferCap)
  resetSeedI <- liftIO1 randomInt32
  let initEnvs : VecEnv NumEnvs CPState
      initEnvs = fst (resetAll {state=CPState} {action=Nat} {obs=Vect 4 Double}
                              (cast resetSeedI))
  envsRef <- liftIO1 (newIORef initEnvs)
  stepRef <- liftIO1 (newIORef (the Nat 0))
  pure1 (MkDqnState qNet0 target0 buffer envsRef stepRef
                    cfg.epsStart cfg.epsEnd cfg.epsDecay
                    cfg.targetSync cfg.batchSize cfg.gamma onNames tgtNames)

-- Discard the (linear) state: both nets are linear → discard; ω fields drop.
discardStateL : (1 _ : DqnState) -> L IO ()
discardStateL (MkDqnState qNet target _ _ _ _ _ _ _ _ _ _ _) = do
  discard qNet
  discard target

-- Final greedy eval (consumes the trained linear state): eval the online net
-- under withNoGradL, discard both nets, report.
finalReportL : Config -> Nat -> (1 _ : DqnState) -> L IO ()
finalReportL cfg epochsDone (MkDqnState qNet target _ _ _ _ _ _ _ _ _ _ _) = do
  let nEval = the Nat 30
  (MkBang totalReturn # qNet') <- withNoGradL {ex=Ex} (evalNL qNet nEval 0.0)
  discard qNet'
  discard target
  liftIO1 $ do
    let avgReturn = totalReturn / cast (natToInteger nEval)
    putStrLn ""
    putStrLn $ "Eval (" ++ show nEval ++ " episodes, greedy): avg_return=" ++ show avgReturn
    putStrLn ""
    putStrLn $ formatResult [("avg_return", show avgReturn),
                              ("epochs", show epochsDone),
                              ("seed", show cfg.seed)]

%default partial

lrFindCfg : LrFindConfig
lrFindCfg = { numIters := 30 } defaultLrFindConfig

-- Terminal linear consumer of the lrFind result. A named function with an
-- explicit `(1 _ : LPair ...)` signature so the bind continuation is linear
-- (the inline do-notation `<-` doesn't get recognised as linear for `lrFind`).
finishLrFind : (1 _ : LPair (!* LrFindResult) DqnState) -> L IO ()
finishLrFind (MkBang _ # st') = do
  discardStateL st'
  liftIO1 $ do
    putStrLn ""
    putStrLn "Done — re-run without --lr-find at the recommended LR."

runLrFind : Config -> IO ()
runLrFind cfg = Control.Linear.LIO.run $ do
  st0 <- buildStateL cfg
  opt <- liftIO1 (adam cfg.lr ({ clip := NormClip 10.0 } defaultOpts))
  online <- liftIO1 (namesMatching {ex=Ex} (isPrefixOf "online"))
  liftIO1 (restrictTo opt online)
  (LIO.(>>=))
    (lrFind {ex = Ex} {model = DqnState} {dp = ()} lrFindCfg
       (\st, _ => do
          (MkBang ret # st') <- runEpisodeBatchedL opt st
          pure1 (MkBang (negate ret) # st'))
       (pure ()) opt st0)
    finishLrFind

runTrain : Config -> IO ()
runTrain cfg = Control.Linear.LIO.run $ do
  st0 <- buildStateL cfg
  -- Adam owns the "online" params only — the target net syncs via polyakUpdatePaired.
  -- Typed ownership: zero LR on everything outside the online net's registry
  -- names (restrictTo), so the optimizer can't leak updates into the target.
  opt <- liftIO1 (adam cfg.lr ({ clip := NormClip 10.0 } defaultOpts))
  online <- liftIO1 (namesMatching {ex=Ex} (isPrefixOf "online"))
  liftIO1 (restrictTo opt online)
  metrics <- liftIO1 (newRLMetricsState 50)
  let trainCfg : TrainConfig DqnState
      trainCfg = { metricsL := readRLMetrics "recent_50" metrics }
                   (mkTrainConfig cfg.epochs 25 NoEarlyStop
                      (const (pure (the (List (String, String)) []))) (\_ => pure ()))
  (MkBang (epochsDone, _) # trained) <- fit {batch = ()}
    (\st, _ => do
       (MkBang ret # st') <- runEpisodeBatchedL opt st
       dd <- liftIO1 (do recordReturn metrics ret; pure (negate ret))
       pure1 (MkBang dd # st'))
    opt (generate (pure ())) trainCfg st0
  finalReportL cfg epochsDone trained

main : IO ()
main = do
  requireMachine {m = ChosenMachine}
  args <- getArgs
  let cfg = parseArgs defaultConfig specs (drop 1 args)
  srand cfg.seed
  tsetInitSeed {ex = Ex} cfg.seed

  putStrLn "=== Double DQN on CartPole ==="
  putStrLn $ "Config: lr=" ++ show cfg.lr
           ++ " epochs=" ++ show cfg.epochs
           ++ " gamma=" ++ show cfg.gamma
           ++ " batch=" ++ show cfg.batchSize
           ++ " buffer=" ++ show cfg.bufferCap
           ++ " target_sync=" ++ show cfg.targetSync
           ++ " eps=" ++ show cfg.epsStart ++ "→" ++ show cfg.epsEnd
           ++ " seed=" ++ show cfg.seed
  putStrLn ""

  if cfg.lrFind then runLrFind cfg else runTrain cfg
