module Example.MountainCar

import Control.Linear.LIO
import Data.Fin
import Data.IORef
import Data.Linear.Notation
import Data.List
import Data.Vect
import System

import Array
import BuildConfig
import Compat.Random
import FitL
import Gym.ClassicControl.MountainCar
import Gym.Env
import Gym.Vector
import Hpo.LrFinder
import ML.Simple
import RL.ReplayBuffer
import Train

-- The Q-nets are linear `SeqL`s; hide the IO `Nn.Seq` constructors.
%hide Nn.Seq.Nil
%hide Nn.Seq.(::)
%hide Nn.Seq.(~~>)

----------------------------------------------------------------------
-- DQN on MountainCar-v0 with reward shaping.
--
-- MountainCar has a sparse reward (-1 per step until goal at pos >= 0.5).
-- Random exploration almost never reaches the goal in 200 steps, so DQN
-- can't learn from raw reward alone. We add velocity-magnitude shaping
-- (|v| * shapingScale) as a dense intermediate signal.
--
-- Architecture: MLP 2 -> 64 -> relu -> 64 -> relu -> 3.
----------------------------------------------------------------------

ObsDim     : Nat; ObsDim = 2
Hidden     : Nat; Hidden = 64
NumActions : Nat; NumActions = 3
MaxSteps   : Nat; MaxSteps = 200

||| Parallel envs collecting transitions in lockstep. Mirrors
||| `Example.Dqn.NumEnvs`. env-0 is the primary.
NumEnvs : Nat; NumEnvs = 4

QNet : Type
QNet = SeqL ObsDim NumActions Ex F WithGrad

mkQNet : (scope : String) -> Init QNet
mkQNet scope = scoped scope $ do
  l1 <- linear {i=ObsDim} {o=Hidden}
  l2 <- linear {i=Hidden} {o=Hidden}
  l3 <- linear {i=Hidden} {o=NumActions}
  pure (l1 ~~> reluA ~~> l2 ~~> reluA ~~> l3 ~~> Nil)

----------------------------------------------------------------------
-- Observation helper
----------------------------------------------------------------------

obsTensor : Vect ObsDim Double -> Vector ObsDim Double
obsTensor v = VArray (map SArray v)

----------------------------------------------------------------------
-- Epsilon-greedy
----------------------------------------------------------------------

epsilonAt : Nat -> Double -> Double -> Nat -> Double
epsilonAt step start end decaySteps =
  let frac = min (cast (natToInteger step) / cast (natToInteger decaySteps)) 1.0
  in start + frac * (end - start)

-- Argmax over Q(s, *) from the online net (threads the linear net).
greedyActionL : (1 _ : QNet) -> Vect ObsDim Double -> L IO {use = 1} (LPair (!* Nat) QNet)
greedyActionL online obs = do
  stateV <- liftIO1 (ioRerun (\_ =>
    the (Tensor [1, ObsDim] Ex F WithGrad) (MkTensor (bulkToTensor2d {ex=Ex} {dt=F} [obsTensor obs]) Nothing)))
  (MkBang qV # online') <- forwardSeqL {b=1} online stateV
  let q0 = primItem2d {ex=Ex} qV.tensorPtr 0 0
      q1 = primItem2d {ex=Ex} qV.tensorPtr 0 1
      q2 = primItem2d {ex=Ex} qV.tensorPtr 0 2
  pure1 (MkBang (if q0 >= q1 && q0 >= q2 then the Nat 0
                 else if q1 >= q2 then 1
                 else 2) # online')

-- Batched epsilon-greedy across NumEnvs envs: one batched forward,
-- then per-env eps-vs-greedy with independent random draws.
epsGreedyBatched : {n : Nat} -> Tensor [n, NumActions] Ex F g ->
                   Vect n MCState -> Double -> IO (Vect n Nat)
epsGreedyBatched qB envs eps = go 0 envs
  where
    go : Int -> Vect k MCState -> IO (Vect k Nat)
    go _ []          = pure []
    go i (_ :: rest) = do
      u <- randomRIO (the Double 0.0, 1.0)
      a <- if u < eps
             then do
               u2 <- randomRIO (the Double 0.0, 1.0)
               pure (if u2 < (1.0 / 3.0) then 0
                     else if u2 < (2.0 / 3.0) then 1
                     else 2)
             else do
               let q0 = primItem2d {ex=Ex} qB.tensorPtr i 0
                   q1 = primItem2d {ex=Ex} qB.tensorPtr i 1
                   q2 = primItem2d {ex=Ex} qB.tensorPtr i 2
               pure (if q0 >= q1 && q0 >= q2 then 0
                     else if q1 >= q2 then 1
                     else 2)
      as <- go (i + 1) rest
      pure (a :: as)

----------------------------------------------------------------------
-- Batched DQN loss (mirrors Example.Dqn).
----------------------------------------------------------------------

rowMax2d : AnyPtr -> Double
rowMax2d t =
  let v0 = primItem2d {ex=Ex} t 0 0
      v1 = primItem2d {ex=Ex} t 0 1
      v2 = primItem2d {ex=Ex} t 0 2
  in if v0 >= v1 && v0 >= v2 then v0
     else if v1 >= v2 then v1
     else v2

computeTargetValL : (1 _ : QNet) -> Double -> Transition ObsDim 1 ->
                    L IO {use = 1} (LPair (!* Double) QNet)
computeTargetValL target gamma t = do
  stateV <- liftIO1 (ioRerun (\_ =>
    the (Tensor [1, ObsDim] Ex F WithGrad) (MkTensor (bulkToTensor2d {ex=Ex} {dt=F} [obsTensor t.nextObs]) Nothing)))
  (MkBang qV # target') <- forwardSeqL {b=1} target stateV
  let nextMax = rowMax2d qV.tensorPtr
      bootstrap = if t.done then 0.0 else gamma * nextMax
  pure1 (MkBang (t.reward + bootstrap) # target')

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

-- Thread the (linear) target net across the batch, collecting bootstrap values.
foldTargetsL : (1 _ : QNet) -> Double -> List (Transition ObsDim 1) -> List Double ->
               L IO {use = 1} (LPair (!* (List Double)) QNet)
foldTargetsL target _     []          acc = pure1 (MkBang (reverse acc) # target)
foldTargetsL target gamma (t :: rest) acc = do
  (MkBang tv # target') <- computeTargetValL target gamma t
  foldTargetsL target' gamma rest (tv :: acc)

batchLossBatchedL : (n : Nat) -> (1 _ : QNet) -> (1 _ : QNet) -> Double ->
                    Vect n (Transition ObsDim 1) ->
                    L IO {use = 1} (LPair (!* (Tensor [] Ex F WithGrad)) (LPair QNet QNet))
batchLossBatchedL n online target gamma batch = do
  (MkBang targetVals # target') <- foldTargetsL target gamma (toList batch) []
  obsBV <- liftIO1 (ioRerun (\_ =>
    the (Tensor [n, ObsDim] Ex F WithGrad)
        (MkTensor (bulkToTensor2d {ex=Ex} {dt=F} (map (\t => obsTensor t.obs) batch)) Nothing)))
  (MkBang qOutB # online') <- forwardSeqL {b=n} online obsBV
  loss <- liftIO1 $ do
            losses <- go qOutB (toList batch) targetVals 0
            meanScalarLoss n losses
  pure1 (MkBang loss # (online' # target'))
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
-- DQN state
----------------------------------------------------------------------

-- The two Q-nets are **linear** fields; the buffer / IORefs / config are ω.
record DqnState where
  constructor MkDqnState
  1 qNet       : QNet
  1 target     : QNet
  buffer       : ReplayBuffer ObsDim 1
  envsRef      : IORef (VecEnv NumEnvs MCState)
  stepRef      : IORef Nat
  cfgEpsStart  : Double
  cfgEpsEnd    : Double
  cfgEpsDecay  : Nat
  cfgSyncEvery : Nat
  cfgBatch     : Nat
  cfgGamma     : Double
  cfgShaping   : Double  -- multiplier on |vel| reward bonus

----------------------------------------------------------------------
-- Episode rollout
----------------------------------------------------------------------

actionToVec : Nat -> Vect 1 Double
actionToVec a = [cast (natToInteger a)]

-- Train-if-ready, threading both nets through the batch loss + step inside a
-- linear generation bracket (withGenFreeL: frees the replay step's grad
-- intermediates so the within-epoch live handle count stays small — one epoch
-- is ~106k ops; registry params rc>1 are spared).
trainIfReadyL : Optimizer Ex -> ReplayBuffer ObsDim 1 -> Nat -> Double ->
                (1 _ : QNet) -> (1 _ : QNet) -> L IO {use = 1} (LPair QNet QNet)
trainIfReadyL opt buffer cfgBatch gamma online target = do
  bufSz <- liftIO1 (bufferSize buffer)
  if bufSz < cfgBatch
    then pure1 (online # target)
    else do
      mBatch <- liftIO1 (sampleN cfgBatch buffer)
      case mBatch of
        Just batchVec => withGenFreeL {ex=Ex} $ do
          (MkBang loss # (online' # target')) <- batchLossBatchedL cfgBatch online target gamma batchVec
          _ <- liftIO1 (nativeTrainStep opt loss)
          pure1 (online' # target')
        Nothing => pure1 (online # target)

----------------------------------------------------------------------
-- Batched episode rollout: NumEnvs parallel envs collect transitions
-- in lockstep; one batched action-selection forward per outer step.
-- Reward shaping is applied per-env using each env's old / new state.
----------------------------------------------------------------------

stepAllAutoResetMC : Vect n MCState -> Vect n Nat ->
                     (Vect n MCState, Vect n Double, Vect n Bool)
stepAllAutoResetMC [] []               = ([], [], [])
stepAllAutoResetMC (s :: ss) (a :: as) =
  case mcStep s a of
    (r, s', outcome, _) =>
      let isDone = done outcome
          nextS  = if isDone then MkMC (-0.5) 0.0 else s'
      in case stepAllAutoResetMC ss as of
           (rest, rs, ds) => (nextS :: rest, r :: rs, isDone :: ds)

pushAllTransitionsMC : ReplayBuffer ObsDim 1 -> Double ->
                       Vect n MCState -> Vect n Nat -> Vect n Double ->
                       Vect n MCState -> Vect n Bool -> IO ()
pushAllTransitionsMC _ _ [] [] [] [] []                                              = pure ()
pushAllTransitionsMC buf shaping (s :: ss) (a :: as) (r :: rs) (s' :: ss') (d :: ds) = do
  let shapedR = r + shaping * abs s'.mcVel
  push buf (MkTransition (mcObserve s) (actionToVec a) shapedR (mcObserve s') d)
  pushAllTransitionsMC buf shaping ss as rs ss' ds

runEpisodeBatchedL : Optimizer Ex -> (1 _ : DqnState) -> L IO {use = 1} (LPair (!* Double) DqnState)
runEpisodeBatchedL opt (MkDqnState qNet target buffer envsRef stepRef epsStart epsEnd epsDecay syncEvery batch gamma shaping) = do
  startEnvs <- liftIO1 (readIORef envsRef)
  (MkBang ret # (qNet' # target')) <- go qNet target startEnvs.envs MaxSteps 0.0
  pure1 (MkBang ret # MkDqnState qNet' target' buffer envsRef stepRef epsStart epsEnd epsDecay syncEvery batch gamma shaping)
  where
    go : (1 _ : QNet) -> (1 _ : QNet) -> Vect NumEnvs MCState -> Nat -> Double ->
         L IO {use = 1} (LPair (!* Double) (LPair QNet QNet))
    go qNet target _ Z ret = do
      liftIO1 (writeIORef envsRef (MkVecEnv (replicate NumEnvs (MkMC (-0.5) 0.0))))
      pure1 (MkBang ret # (qNet # target))
    go qNet target envs (S steps) ret = do
      stepCount <- liftIO1 (readIORef stepRef)
      let eps = epsilonAt stepCount epsStart epsEnd epsDecay
      stateV <- liftIO1 (ioRerun (\_ =>
        the (Tensor [NumEnvs, ObsDim] Ex F WithGrad)
            (MkTensor (bulkToTensor2d {ex=Ex} {dt=F} (map (\s => obsTensor (mcObserve s)) envs)) Nothing)))
      (MkBang actions # qNet') <- withNoGradL {ex=Ex} $ do
        (MkBang qB # qNet') <- forwardSeqL {b=NumEnvs} qNet stateV
        acts <- liftIO1 (epsGreedyBatched qB envs eps)
        pure1 (MkBang acts # qNet')
      case stepAllAutoResetMC envs actions of
        (envs', rewards, dones) => do
          liftIO1 (pushAllTransitionsMC buffer shaping envs actions rewards envs' dones)
          liftIO1 (writeIORef stepRef (stepCount + 1))
          let ret0  = head rewards
              done0 = head dones
              ret'  = ret + ret0
          (qNet'' # target') <- trainIfReadyL opt buffer batch gamma qNet' target
          liftIO1 (when ((stepCount + 1) `mod` syncEvery == 0) $ do
                     _ <- polyakUpdate {ex=Ex} 1.0 "online" "target"
                     pure ())
          if done0
            then do
              liftIO1 (writeIORef envsRef (MkVecEnv envs'))
              pure1 (MkBang ret' # (qNet'' # target'))
            else go qNet'' target' envs' steps ret'

----------------------------------------------------------------------
-- Config & main
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
  shaping    : Double
  seed       : Bits64
  lrFind     : Bool

defaultConfig : Config
defaultConfig =
  MkConfig 1.0e-3 1000 0.99 64 50000 200 1.0 0.05 50000 10.0 42 False

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
        , Arg "--shaping" (\v, c => { shaping := cast v } c)
        , Arg "--seed" (\v, c => { seed := castBits64 v } c)
        , Arg "--lr-find" (\v, c => { lrFind := (v == "1" || v == "true") } c)
        ]

----------------------------------------------------------------------
-- Greedy evaluation (raw reward, no shaping).
----------------------------------------------------------------------

evalEpL : (1 _ : QNet) -> MCState -> Nat -> Double -> L IO {use = 1} (LPair (!* Double) QNet)
evalEpL q _ Z acc      = pure1 (MkBang acc # q)
evalEpL q st (S k) acc = do
  (MkBang action # q') <- greedyActionL q (mcObserve st)
  case mcStep st action of
    (reward, st', outcome, _) =>
      if done outcome then pure1 (MkBang (acc + reward) # q')
      else evalEpL q' st' k (acc + reward)

evalNL : (1 _ : QNet) -> Nat -> Double -> L IO {use = 1} (LPair (!* Double) QNet)
evalNL q Z acc     = pure1 (MkBang acc # q)
evalNL q (S k) acc = do
  (MkBang ep # q') <- evalEpL q (MkMC (-0.5) 0.0) MaxSteps 0.0
  evalNL q' k (acc + ep)

----------------------------------------------------------------------
-- State construction / eval / discard (linear)
----------------------------------------------------------------------

buildStateL : Config -> L IO {use = 1} DqnState
buildStateL cfg = do
  qNet0   <- runInitL (mkQNet "online")
  target0 <- runInitL (mkQNet "target")
  liftIO1 (do _ <- polyakUpdate {ex=Ex} 1.0 "online" "target"; pure ())
  buffer  <- liftIO1 (mkBuffer {obsDim = ObsDim, actDim = 1} cfg.bufferCap)
  resetSeedI <- liftIO1 randomInt32
  let initEnvs : VecEnv NumEnvs MCState
      initEnvs = fst (resetAll {state=MCState} {action=Nat} {obs=Vect 2 Double}
                              (cast resetSeedI))
  envsRef <- liftIO1 (newIORef initEnvs)
  stepRef <- liftIO1 (newIORef (the Nat 0))
  pure1 (MkDqnState qNet0 target0 buffer envsRef stepRef
                    cfg.epsStart cfg.epsEnd cfg.epsDecay
                    cfg.targetSync cfg.batchSize cfg.gamma cfg.shaping)

discardStateL : (1 _ : DqnState) -> L IO ()
discardStateL (MkDqnState qNet target _ _ _ _ _ _ _ _ _ _) = do
  discardL qNet
  discardL target

finalReportL : Config -> Nat -> (1 _ : DqnState) -> L IO ()
finalReportL cfg epochsDone (MkDqnState qNet target _ _ _ _ _ _ _ _ _ _) = do
  let nEval = the Nat 30
  (MkBang totalReturn # qNet') <- withNoGradL {ex=Ex} (evalNL qNet nEval 0.0)
  discardL qNet'
  discardL target
  liftIO1 $ do
    let avgReturn = totalReturn / cast (natToInteger nEval)
    putStrLn ""
    putStrLn $ "Eval (" ++ show nEval ++ " episodes, greedy): avg_return=" ++ show avgReturn
    putStrLn ""
    putStrLn $ formatResult [("avg_return", show avgReturn),
                              ("epochs", show epochsDone),
                              ("seed", show cfg.seed)]

----------------------------------------------------------------------
-- Main
----------------------------------------------------------------------

%default partial

main : IO ()
main = do
  requireMachine {m = ChosenMachine}
  args <- getArgs
  let cfg = parseArgs defaultConfig specs (drop 1 args)
  srand cfg.seed
  tsetInitSeed {ex = Ex} cfg.seed

  putStrLn "=== DQN on MountainCar ==="
  putStrLn $ "Config: lr=" ++ show cfg.lr
           ++ " epochs=" ++ show cfg.epochs
           ++ " gamma=" ++ show cfg.gamma
           ++ " batch=" ++ show cfg.batchSize
           ++ " buffer=" ++ show cfg.bufferCap
           ++ " target_sync=" ++ show cfg.targetSync
           ++ " eps=" ++ show cfg.epsStart ++ "→" ++ show cfg.epsEnd
           ++ " shaping=" ++ show cfg.shaping
           ++ " seed=" ++ show cfg.seed
  putStrLn ""

  if cfg.lrFind
    then Control.Linear.LIO.run $ do
      st0 <- buildStateL cfg
      opt <- liftIO1 (adam {scope="online"} cfg.lr ({ clip := NormClip 10.0 } defaultOpts))
      let lrCfg : LrFindConfig
          lrCfg = { numIters := 30 } defaultLrFindConfig
      (MkBang _ # st') <- lrFindL lrCfg
        (\st, _ => do
           (MkBang ret # st') <- runEpisodeBatchedL opt st
           pure1 (MkBang (negate ret) # st'))
        (pure ()) opt st0
      discardStateL st'
      liftIO1 $ do
        putStrLn ""
        putStrLn "Done — re-run without --lr-find at the recommended LR."
    else Control.Linear.LIO.run $ do
      st0 <- buildStateL cfg
      opt <- liftIO1 (adam {scope="online"} cfg.lr ({ clip := NormClip 10.0 } defaultOpts))
      metrics <- liftIO1 (newRLMetricsState 50)
      let trainCfg : TrainConfig DqnState
          trainCfg = { metricsL := readRLMetrics "recent_50" metrics }
                       (mkTrainConfig cfg.epochs 50 NoEarlyStop
                          (const (pure (the (List (String, String)) []))) (\_ => pure ()))
      (MkBang (epochsDone, _) # trained) <- fitL {batch = ()}
        (\st, _ => do
           (MkBang ret # st') <- runEpisodeBatchedL opt st
           dd <- liftIO1 (do recordReturn metrics ret; pure (negate ret))
           pure1 (MkBang dd # st'))
        opt (generate (pure ())) trainCfg st0
      finalReportL cfg epochsDone trained
