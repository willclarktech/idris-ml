module Example.MountainCarCont

import Data.List
import Data.SortedMap
import Data.Vect
import Data.IORef
import System
import Compat.Random

import Endofunctor
import Floating
import Gym.ClassicControl.MountainCarCont
import Gym.Env
import Layer
import Layer.Core
import Math
import RL.ReplayBuffer
import Sampler
import Tensor
import Train
import Util
import Device
import Variable


----------------------------------------------------------------------
-- SAC on MountainCarContinuous-v0 with velocity-magnitude reward shaping.
--
-- MountainCarCont's reward is sparse (terminal +100 for reaching goal,
-- per-step -0.1*action²). Random Gaussian exploration almost never finds
-- the goal in 999 steps — same problem as the discrete MountainCar +
-- DQN. Solution mirrors Example.MountainCar: shape the *training*
-- reward with `r_shaped = r_raw + shaping * |v_next|` to densify the
-- signal toward kinetic energy. Eval reports the *raw* return so the
-- metric stays comparable to standard MountainCarCont reporting.
--
-- Architecture mirrors Example.Sac (separate actor + Q1 + Q2 + 2 target
-- Q nets, scoped paramIds, three Adam optimizers, polyak τ-soft target
-- update). Aligned with `torch_ref/models/mountain_car_cont.py`.
----------------------------------------------------------------------

-- --- Inline autoName-with-scope (single-file invocation quirk — see Sac.idr).

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

-- Inline FFI wrappers (single-file invocation quirk — see Sac.idr).

%foreign "C:optimizer_create_adam_group,libidrisml"
prim__mkAdamGroupLocal : Double -> Double -> Double -> Double -> String -> AnyPtr

mkAdamGroup : String -> Double -> Double -> NativeOptimizer
mkAdamGroup scope lr clip =
  MkNativeOptimizer (prim__mkAdamGroupLocal lr 0.9 0.999 1.0e-8 scope) (NormClip clip)

%foreign "C:polyak_blend,libidrisml"
prim__polyakLocal : Double -> String -> String -> Int

polyakBlend : Double -> String -> String -> IO Int
polyakBlend tau on tg = pure (prim__polyakLocal tau on tg)


-- --- Constants ------------------------------------------------------

ObsDim : Nat; ObsDim = 2
ActDim : Nat; ActDim = 1
QInputDim : Nat; QInputDim = 3          -- ObsDim + ActDim
Hidden : Nat; Hidden = 64
EpisodeLen : Nat; EpisodeLen = 999
MaxAct : Double; MaxAct = 1.0


-- --- Architectures --------------------------------------------------

ActorNet : Type
ActorNet = Network ObsDim [Hidden, Hidden, Hidden, Hidden] 1 (Variable CPU)

QNet : Type
QNet = Network QInputDim [Hidden, Hidden, Hidden, Hidden] 1 (Variable CPU)


mkActor : IO ActorNet
mkActor = do
  ll1 <- linearLayer {i=ObsDim} {o=Hidden}
  ll2 <- linearLayer {i=Hidden} {o=Hidden}
  ll3 <- linearLayer {i=Hidden} {o=1}
  pure (autoNameScoped "actor_"
    (ll1 ~> reluLayer ~> ll2 ~> reluLayer ~> OutputLayer ll3))

mkQ : String -> IO QNet
mkQ scope = do
  ll1 <- linearLayer {i=QInputDim} {o=Hidden}
  ll2 <- linearLayer {i=Hidden} {o=Hidden}
  ll3 <- linearLayer {i=Hidden} {o=1}
  pure (autoNameScoped scope
    (ll1 ~> reluLayer ~> ll2 ~> reluLayer ~> OutputLayer ll3))

mkLogStd : Variable CPU
mkLogStd = param "actor_log_std" 0.0


-- --- Observation helpers --------------------------------------------

observeVec : MCCState -> Vect ObsDim Double
observeVec s = mccObserve s

obsTensor : Vect ObsDim Double -> Vector ObsDim Double
obsTensor v = VTensor (map STensor v)

qInput : Vect ObsDim Double -> Double -> Vect QInputDim Double
qInput obs a = obs ++ [a]

qInputTensor : Vect QInputDim Double -> Vector QInputDim Double
qInputTensor v = VTensor (map STensor v)


-- --- Gaussian / squash helpers --------------------------------------

logTwoPiHalf : Double
logTwoPiHalf = 0.5 * Prelude.log (2.0 * 3.141592653589793)

squashCorrection : Double -> Double
squashCorrection u =
  let tu = Math.tanh u
  in Prelude.log (1.0 - tu * tu + 1.0e-6) + Prelude.log MaxAct


-- --- Forward helpers ------------------------------------------------

actorMean : ActorNet -> Vect ObsDim Double -> Double
actorMean actor obs =
  let outT = snd (forwardVarTensor actor (bulkToTensor (obsTensor obs)))
  in prim__item1d outT 0

qValue : QNet -> Vect ObsDim Double -> Double -> Double
qValue q obs action =
  let outT = snd (forwardVarTensor q (bulkToTensor (qInputTensor (qInput obs action))))
  in prim__item1d outT 0


sampleActionIO : ActorNet -> Variable CPU -> Vect ObsDim Double ->
                 IO (Double, Double)
sampleActionIO actor logStdV obs = do
  let mean   = actorMean actor obs
      logStd = (refreshValue logStdV).value
      std    = Prelude.exp logStd
  eps <- normalSample
  let u        = mean + std * eps
      action   = Math.tanh u * MaxAct
      lp_u     = -0.5 * ((u - mean) / std) * ((u - mean) / std) - logStd - logTwoPiHalf
      lp       = lp_u - squashCorrection u
  pure (action, lp)


-- --- SAC state -------------------------------------------------------

record SACState where
  constructor MkSAC
  actor   : ActorNet
  q1      : QNet
  q2      : QNet
  q1Tgt   : QNet
  q2Tgt   : QNet
  logStdV : Variable CPU
  buffer  : ReplayBuffer ObsDim ActDim
  stepRef : IORef Nat
  envRef  : IORef MCCState
  epLenRef : IORef Nat
  retRef  : IORef Double
  lastEpRef : IORef Double


record Config where
  constructor MkConfig
  lr           : Double
  epochs       : Nat
  gamma        : Double
  alpha        : Double
  bufferCap    : Nat
  batchSize    : Nat
  warmupSteps  : Nat
  tau          : Double
  shaping      : Double
  clipNorm     : Double
  seed         : Bits64
  esThreshold  : Double
  esWindow     : Nat
  esPatience   : Nat
  lrFind       : Bool

defaultConfig : Config
defaultConfig = MkConfig 3.0e-4 30000 0.99 0.2 100000 64 1000 0.005 10.0 1.0 42
                         (-85.0) 500 5 False

specs : List (ArgSpec Config)
specs = [ Arg "--lr" (\v, c => { lr := cast v } c)
        , Arg "--epochs" (\v, c => { epochs := castNat v } c)
        , Arg "--gamma" (\v, c => { gamma := cast v } c)
        , Arg "--alpha" (\v, c => { alpha := cast v } c)
        , Arg "--buffer-cap" (\v, c => { bufferCap := castNat v } c)
        , Arg "--batch" (\v, c => { batchSize := castNat v } c)
        , Arg "--warmup" (\v, c => { warmupSteps := castNat v } c)
        , Arg "--tau" (\v, c => { tau := cast v } c)
        , Arg "--shaping" (\v, c => { shaping := cast v } c)
        , Arg "--clip" (\v, c => { clipNorm := cast v } c)
        , Arg "--seed" (\v, c => { seed := castBits64 v } c)
        , Arg "--es-threshold" (\v, c => { esThreshold := cast v } c)
        , Arg "--es-window" (\v, c => { esWindow := castNat v } c)
        , Arg "--es-patience" (\v, c => { esPatience := castNat v } c)
        , Arg "--lr-find" (\v, c => { lrFind := (v == "1" || v == "true") } c)
        ]


-- --- Q-network loss (batched) ---------------------------------------

computeTargetVal : QNet -> QNet -> ActorNet -> Variable CPU ->
                   Double -> Double -> Transition ObsDim ActDim -> IO Double
computeTargetVal q1Tgt q2Tgt actor logStdV gamma alpha t = do
  nextPair <- sampleActionIO actor logStdV t.nextObs
  let nextAction = fst nextPair
      nextLogP   = snd nextPair
      q1NextD    = qValue q1Tgt t.nextObs nextAction
      q2NextD    = qValue q2Tgt t.nextObs nextAction
      minQNextD  = if q1NextD <= q2NextD then q1NextD else q2NextD
      doneMask   = if t.done then 0.0 else 1.0
  pure (t.reward + gamma * doneMask * (minQNextD - alpha * nextLogP))

qLossBatch : (n : Nat) -> QNet -> QNet -> QNet -> ActorNet -> Variable CPU ->
             Double -> Double -> Vect n (Transition ObsDim ActDim) ->
             IO (Variable CPU)
qLossBatch n qOnline q1Tgt q2Tgt actor logStdV gamma alpha batch = do
  targetVals <- traverse (computeTargetVal q1Tgt q2Tgt actor logStdV gamma alpha) batch
  let qInputs : Vect n (Vector QInputDim Double)
      qInputs   = map (\t => qInputTensor (qInput t.obs (oneAct t.action))) batch
      qInputBT  = bulkToTensor2d qInputs
      qOutB     = snd (forwardVarTensorBatch qOnline n qInputBT)
      losses    = perSampleLosses qOutB targetVals 0
      n_d       = the Double (cast (natToInteger n))
      sumV      = foldl (+) (the (Variable CPU) (fromDouble 0.0)) losses
      nV        = the (Variable CPU) (fromDouble n_d)
  pure (sumV / nV)
  where
    oneAct : Vect ActDim Double -> Double
    oneAct [a] = a
    perSampleLosses : (qOutB : AnyPtr) -> Vect k Double -> Int -> List (Variable CPU)
    perSampleLosses _ [] _ = []
    perSampleLosses qOutB (tv :: rest) k =
      let qRow = prim__select qOutB 0 k
          qScalar = prim__item1d qRow 0
          qV : Variable CPU
          qV = Var (prim__select qRow 0 0) Nothing qScalar
          targetC : Variable CPU
          targetC = fromDouble tv
          diff = qV - targetC
      in (diff * diff) :: perSampleLosses qOutB rest (k + 1)


-- --- Actor loss with reparameterization -----------------------------

buildScalarColumnT : {n : Nat} -> Vect n Double -> AnyPtr
buildScalarColumnT {n} xs =
  let rows : Vect n (Vector 1 Double)
      rows = map (\x => VTensor [STensor x]) xs
  in bulkToTensor2d rows

actorPerStepLoss : (meanB : AnyPtr) -> (uBT : AnyPtr) ->
                   (q1B : AnyPtr) -> (q2B : AnyPtr) ->
                   Variable CPU -> Double -> Double ->
                   Int -> Double -> Variable CPU
actorPerStepLoss meanB uBT q1B q2B logStdV alpha stdVal rowIdx eps =
  let q1Row   = prim__select q1B 0 rowIdx
      q1Val   = prim__item1d q1Row 0
      q1V     : Variable CPU
      q1V     = Var (prim__select q1Row 0 0) Nothing q1Val
      q2Row   = prim__select q2B 0 rowIdx
      q2Val   = prim__item1d q2Row 0
      q2V     : Variable CPU
      q2V     = Var (prim__select q2Row 0 0) Nothing q2Val
      minQV   = if q1Val <= q2Val then q1V else q2V

      meanRow = prim__select meanB 0 rowIdx
      meanVal = prim__item1d meanRow 0
      meanV   : Variable CPU
      meanV   = Var (prim__select meanRow 0 0) Nothing meanVal

      uRow    = prim__select uBT 0 rowIdx
      uVal    = prim__item1d uRow 0
      uV      : Variable CPU
      uV      = Var (prim__select uRow 0 0) Nothing uVal

      halfC   : Variable CPU
      halfC   = fromDouble 0.5
      twoC    : Variable CPU
      twoC    = fromDouble 2.0
      zeroC   : Variable CPU
      zeroC   = fromDouble 0.0

      diffM    = uV - meanV
      negTwoLs = zeroC - twoC * logStdV
      varInv   = exp negTwoLs
      quad     = halfC * diffM * diffM * varInv
      cC       : Variable CPU
      cC       = fromDouble logTwoPiHalf
      lpU      = (zeroC - quad) - logStdV - cC
      corrC    : Variable CPU
      corrC    = fromDouble (squashCorrection uVal)
      lpV      = lpU - corrC

      alphaC   : Variable CPU
      alphaC   = fromDouble alpha
  in alphaC * lpV - minQV

actorLossBatch : (n : Nat) -> ActorNet -> QNet -> QNet -> Variable CPU ->
                 Double -> Vect n (Vect ObsDim Double) -> IO (Variable CPU)
actorLossBatch n actor q1 q2 logStdV alpha obsBatch = do
  let logStd = (refreshValue logStdV).value
      stdVal = Prelude.exp logStd
  epses <- traverse (\_ => normalSample) obsBatch
  let obsTensors : Vect n (Vector ObsDim Double)
      obsTensors = map obsTensor obsBatch
      obsBT      = bulkToTensor2d obsTensors
      meanB      = snd (forwardVarTensorBatch actor n obsBT)
      epsScales  = map (\e => stdVal * e) epses
      epsBT      = buildScalarColumnT epsScales
      uBT        = tensorAdd meanB epsBT
      aSquashedBT = prim__tanh uBT
      aReparamBT = prim__mulScalar aSquashedBT MaxAct
      qInputBT   = prim__concat2dAxis1 obsBT aReparamBT
      q1B        = snd (forwardVarTensorBatch q1 n qInputBT)
      q2B        = snd (forwardVarTensorBatch q2 n qInputBT)
      losses     = enumeratedLosses meanB uBT q1B q2B epses 0
      n_d        = the Double (cast (natToInteger n))
      sumV       = foldl (+) (the (Variable CPU) (fromDouble 0.0)) losses
      nV         = the (Variable CPU) (fromDouble n_d)
  pure (sumV / nV)
  where
    enumeratedLosses : (meanB : AnyPtr) -> (uBT : AnyPtr) ->
                       (q1B : AnyPtr) -> (q2B : AnyPtr) ->
                       Vect k Double -> Int -> List (Variable CPU)
    enumeratedLosses _ _ _ _ [] _ = []
    enumeratedLosses meanB uBT q1B q2B (eps :: rest) k =
      let stdVal = Prelude.exp (refreshValue logStdV).value
      in actorPerStepLoss meanB uBT q1B q2B logStdV alpha stdVal k eps
           :: enumeratedLosses meanB uBT q1B q2B rest (k + 1)


-- --- Batch update -----------------------------------------------------

runBatchUpdate : NativeOptimizer -> NativeOptimizer -> NativeOptimizer ->
                 SACState -> Config -> {n : Nat} ->
                 Vect n (Transition ObsDim ActDim) -> IO ()
runBatchUpdate q1Opt q2Opt actorOpt st cfg {n} batch = do
  q1LossV <- qLossBatch n st.q1 st.q1Tgt st.q2Tgt st.actor st.logStdV
                        cfg.gamma cfg.alpha batch
  _ <- pure (nativeTrainStep q1Opt q1LossV)
  q2LossV <- qLossBatch n st.q2 st.q1Tgt st.q2Tgt st.actor st.logStdV
                        cfg.gamma cfg.alpha batch
  _ <- pure (nativeTrainStep q2Opt q2LossV)
  let obsVec : Vect n (Vect ObsDim Double)
      obsVec = map (\t => t.obs) batch
  aLossV <- actorLossBatch n st.actor st.q1 st.q2 st.logStdV cfg.alpha obsVec
  _ <- pure (nativeTrainStep actorOpt aLossV)
  pure ()


-- --- Main loop -------------------------------------------------------

sacStep : NativeOptimizer -> NativeOptimizer -> NativeOptimizer ->
          Config -> SACState -> IO (SACState, Double)
sacStep q1Opt q2Opt actorOpt cfg st = do
  stepCount <- readIORef st.stepRef
  envState  <- readIORef st.envRef
  epLen     <- readIORef st.epLenRef
  let obs = observeVec envState

  action <- if stepCount < cfg.warmupSteps
              then randomRIO (the Double (negate MaxAct), MaxAct)
              else do
                pair <- sampleActionIO st.actor st.logStdV obs
                pure (fst pair)

  case mccStep envState action of
    (rawR, envState', outcome, _) => do
      let nextObs = observeVec envState'
          terminated = case outcome of
                         Terminated => True
                         _          => False
          truncated  = (epLen + 1) >= EpisodeLen
          isDone     = terminated || truncated
          -- Buffer's done flag should reflect TRUE termination only
          -- (so Q-target bootstrap continues at truncation boundaries).
          bufferDone = terminated
          shapedR    = rawR + cfg.shaping * abs envState'.mccVel
          nextSt     = if isDone then MkMCC (-0.5) 0.0 else envState'
          trans      = MkTransition obs [action] shapedR nextObs bufferDone
      push st.buffer trans
      writeIORef st.envRef nextSt
      writeIORef st.stepRef (stepCount + 1)
      writeIORef st.epLenRef (if isDone then 0 else epLen + 1)

      runRet <- readIORef st.retRef
      let newRet = runRet + rawR
      if isDone
        then do writeIORef st.lastEpRef newRet
                writeIORef st.retRef 0.0
        else writeIORef st.retRef newRet

      bufSz <- bufferSize st.buffer
      _ <- if bufSz >= cfg.batchSize && stepCount >= cfg.warmupSteps
             then do
               mBatch <- sampleN cfg.batchSize st.buffer
               case mBatch of
                 Nothing    => pure ()
                 Just batch => do
                   runBatchUpdate q1Opt q2Opt actorOpt st cfg batch
                   _ <- polyakBlend cfg.tau "q1_" "q1tgt_"
                   _ <- polyakBlend cfg.tau "q2_" "q2tgt_"
                   pure ()
             else pure ()

      lastEp <- readIORef st.lastEpRef
      pure (st, negate lastEp)


-- --- Greedy evaluation ----------------------------------------------

greedyAct : ActorNet -> Vect ObsDim Double -> Double
greedyAct actor obs =
  let mean = actorMean actor obs
  in Math.tanh mean * MaxAct

evalEp : ActorNet -> MCCState -> Nat -> Double -> Double
evalEp _ _ Z acc = acc
evalEp actor st (S k) acc =
  let a = greedyAct actor (mccObserve st)
  in case mccStep st a of
       (r, st', outcome, _) =>
         case outcome of
           Terminated => acc + r
           _          => evalEp actor st' k (acc + r)

evalN : ActorNet -> Nat -> Double -> Double
evalN _ Z acc = acc
evalN actor (S k) acc =
  evalN actor k (acc + evalEp actor (MkMCC (-0.5) 0.0) EpisodeLen 0.0)


-- --- Main -----------------------------------------------------------

main : IO ()
main = do
  args <- getArgs
  let cfg = parseArgs defaultConfig specs (drop 1 args)
  srand cfg.seed

  putStrLn "=== SAC on MountainCarContinuous ==="
  putStrLn $ "Config: lr=" ++ show cfg.lr
           ++ " steps=" ++ show cfg.epochs
           ++ " gamma=" ++ show cfg.gamma
           ++ " alpha=" ++ show cfg.alpha
           ++ " batch=" ++ show cfg.batchSize
           ++ " warmup=" ++ show cfg.warmupSteps
           ++ " tau=" ++ show cfg.tau
           ++ " shaping=" ++ show cfg.shaping
           ++ " seed=" ++ show cfg.seed

  actor <- mkActor
  q1    <- mkQ "q1_"
  q2    <- mkQ "q2_"
  q1Tgt <- mkQ "q1tgt_"
  q2Tgt <- mkQ "q2tgt_"
  let logStdV = mkLogStd

  _ <- polyakBlend 1.0 "q1_" "q1tgt_"
  _ <- polyakBlend 1.0 "q2_" "q2tgt_"

  buffer    <- mkBuffer {obsDim=ObsDim, actDim=ActDim} cfg.bufferCap
  stepRef   <- newIORef (the Nat 0)
  envRef    <- newIORef (the MCCState (MkMCC (-0.5) 0.0))
  epLenRef  <- newIORef (the Nat 0)
  retRef    <- newIORef (the Double 0.0)
  lastEpRef <- newIORef (the Double 0.0)

  let st0 = MkSAC actor q1 q2 q1Tgt q2Tgt logStdV buffer stepRef envRef epLenRef retRef lastEpRef
      actorOpt = mkAdamGroup "actor_" cfg.lr cfg.clipNorm
      q1Opt    = mkAdamGroup "q1_"    cfg.lr cfg.clipNorm
      q2Opt    = mkAdamGroup "q2_"    cfg.lr cfg.clipNorm

  putStrLn ""

  when cfg.lrFind $ do
    putStrLn "lr_find skipped for SAC: per-step epochs + warmup don't fit"
    putStrLn "the LR-range-test pattern. See docs/develop/hyperparameter-tuning-2026.md."
    exitSuccess

  let trainCfg : TrainConfig SACState
      trainCfg = MkTrainConfig cfg.epochs 2000
                            (WindowedAvg cfg.esThreshold cfg.esWindow cfg.esPatience)
                            (const (pure [])) (\_ => pure ())
  (trained, epochsDone, _) <- runTrainingIO
    (\s, _ => sacStep q1Opt q2Opt actorOpt cfg s)
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
