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

observe : CPState -> Vector 4 Double
observe s = VArray (map SArray (cpObserve s))


----------------------------------------------------------------------
-- Episode Rollout (tensor-level, autograd-tracked)
----------------------------------------------------------------------

-- (logProbTensorPtr, logProbDoubleVal, reward)
StepRec : Type
StepRec = (AnyPtr, Double, Double)

rolloutEp : {hs : List Nat} ->
            Network 4 hs 2 CPU -> CPState -> List Double -> Nat ->
            List StepRec -> List StepRec
rolloutEp _ _ _ Z acc = reverse acc
rolloutEp _ _ [] _ acc = reverse acc
rolloutEp model st (r :: rs) (S k) acc =
  let stateT = bulkToTensor (observe st)
      stateV = the (TVec 4 CPU) (MkTensor stateT Nothing)
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
epStepLosses : Double -> Double -> List StepRec -> List (Tensor [] CPU)
epStepLosses gamma baseline steps =
  let rewards = map (\(_, _, r) => r) steps
      rets = discReturns gamma rewards
  in zipWith (\(lp, _, _), gt =>
       the (Tensor [] CPU) (MkTensor (prim__mulScalar lp (baseline - gt)) Nothing))
     steps rets

sumRewards : List StepRec -> Double
sumRewards steps = foldl (\a, (_, _, r) => a + r) 0.0 steps

-- Mean-reduce a non-empty list of scalar TVars. Empty case returns a
-- fresh zero scalar (degenerate; runs only if the rollout produced no
-- steps).
averageLoss : List (Tensor [] CPU) -> Tensor [] CPU
averageLoss [] = MkTensor (prim__createScalar 0.0 0) Nothing
averageLoss (x :: xs) =
  let n = cast {to=Double} (1 + length xs)
      addT : Tensor [] CPU -> Tensor [] CPU -> Tensor [] CPU
      addT a b = MkTensor (prim__add a.tensorPtr b.tensorPtr) Nothing
      s = foldl addT x xs
  in MkTensor (prim__mulScalar s.tensorPtr (1.0 / n)) Nothing

computeLoss : {hs : List Nat} -> Double ->
              Network 4 hs 2 CPU -> List (List Double) ->
              Tensor [] CPU
computeLoss gamma model randomBatch =
  let episodes = map (\rs => rolloutEp model (MkCP 0 0 0 0) rs MaxSteps []) randomBatch
      epReturns = map sumRewards episodes
      nEp = cast {to=Double} (natToInteger (length epReturns))
      baseline = foldl (+) 0.0 epReturns / nEp
      stepLosses = concatMap (epStepLosses gamma baseline) episodes
  in averageLoss stepLosses


----------------------------------------------------------------------
-- Training
----------------------------------------------------------------------

epochRL : {hs : List Nat} -> NativeOptimizer -> Double ->
          Network 4 hs 2 CPU -> List (List Double) ->
          (Network 4 hs 2 CPU, Double)
epochRL opt gamma model batch =
  let loss = computeLoss gamma model batch
      lossVal = nativeTrainStep opt loss
  in (model, lossVal)

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


----------------------------------------------------------------------
-- Evaluation (greedy argmax)
----------------------------------------------------------------------

evalEp : {hs : List Nat} ->
         Network 4 hs 2 CPU -> CPState -> Nat -> Double -> Double
evalEp _ _ Z acc = acc
evalEp model st (S k) acc =
  let stateT = bulkToTensor (observe st)
      stateV = the (TVec 4 CPU) (MkTensor stateT Nothing)
      (_, predV) = forwardVar model stateV
      logitsT = predV.tensorPtr
      action = if prim__item1d logitsT 0 >= prim__item1d logitsT 1 then the Nat 0 else 1
  in case cpStep st action of
       (reward, st', outcome, _) =>
         if done outcome then acc + reward
         else evalEp model st' k (acc + reward)

evalN : {hs : List Nat} -> Network 4 hs 2 CPU -> Nat -> Double -> Double
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

defaultConfig : Config
defaultConfig = MkConfig 0.001 2000 42 0.99 10 False

specs : List (ArgSpec Config)
specs = [ Arg "--lr" (\v, c => { lr := cast v } c)
        , Arg "--epochs" (\v, c => { epochs := castNat v } c)
        , Arg "--seed" (\v, c => { seed := castBits64 v } c)
        , Arg "--gamma" (\v, c => { gamma := cast v } c)
        , Arg "--batch" (\v, c => { batchSz := castNat v } c)
        , Arg "--lr-find" (\v, c => { lrFind := (v == "1" || v == "true") } c) ]

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
  let model : Network 4 [128, 128] 2 CPU
      model = ll1Any ~~> tanhLayerAny ~~> OutputLayer ll2Any
  putStrLn ""

  when cfg.lrFind $ do
    let lrCfg : LrFindConfig
        lrCfg = { numIters := 100 } defaultLrFindConfig
    _ <- lrFind lrCfg
      (\m, d => let (m', loss) = epochRL opt cfg.gamma m d
                in pure (m', loss))
      (genBatch cfg.batchSz) opt model
    putStrLn ""
    putStrLn "Done — re-run without --lr-find at the recommended LR."
    exitSuccess

  (trained, epochsDone, _) <- runTraining
    (\m, d => epochRL opt cfg.gamma m d) (genBatch cfg.batchSz) (simpleConfig cfg.epochs) model

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
