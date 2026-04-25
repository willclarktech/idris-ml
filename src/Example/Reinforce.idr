module Example.Reinforce

import Data.List
import Data.Vect
import System
import Compat.Random

import Floating
import Layer
import Sampler
import Tensor
import Train
import Util
import Device
import Variable


----------------------------------------------------------------------
-- CartPole Environment (Gymnasium-compatible constants)
----------------------------------------------------------------------

Gravity : Double;       Gravity = 9.8
MassCart : Double;      MassCart = 1.0
MassPole : Double;      MassPole = 0.1
TotalMass : Double;     TotalMass = MassCart + MassPole
HalfPoleLen : Double;   HalfPoleLen = 0.5
PoleMassLen : Double;   PoleMassLen = MassPole * HalfPoleLen
ForceMag : Double;      ForceMag = 10.0
Tau : Double;           Tau = 0.02
ThetaThresh : Double;   ThetaThresh = 12.0 * 2.0 * 3.141592653589793 / 360.0
XThresh : Double;       XThresh = 2.4
MaxSteps : Nat;         MaxSteps = 200

record CPState where
  constructor MkCP
  cpX, cpXDot, cpTheta, cpThetaDot : Double

cpStep : CPState -> Nat -> (Double, CPState, Bool)
cpStep s action =
  let force = if action == 1 then ForceMag else negate ForceMag
      cosT = prim__doubleCos s.cpTheta
      sinT = prim__doubleSin s.cpTheta
      temp = (force + PoleMassLen * s.cpThetaDot * s.cpThetaDot * sinT) / TotalMass
      tAcc = (Gravity * sinT - cosT * temp) /
             (HalfPoleLen * (4.0 / 3.0 - MassPole * cosT * cosT / TotalMass))
      xAcc = temp - PoleMassLen * tAcc * cosT / TotalMass
      s' = MkCP (s.cpX + Tau * s.cpXDot) (s.cpXDot + Tau * xAcc)
                (s.cpTheta + Tau * s.cpThetaDot) (s.cpThetaDot + Tau * tAcc)
  in (1.0, s', abs s'.cpX > XThresh || abs s'.cpTheta > ThetaThresh)

observe : CPState -> Vector 4 Double
observe s = VTensor [STensor s.cpX, STensor s.cpXDot,
                     STensor s.cpTheta, STensor s.cpThetaDot]


----------------------------------------------------------------------
-- Episode Rollout (tensor-level, autograd-tracked)
----------------------------------------------------------------------

-- (logProbTensorPtr, logProbDoubleVal, reward)
StepRec : Type
StepRec = (AnyPtr, Double, Double)

rolloutEp : {hs : List Nat} ->
            Network 4 hs 2 (Variable CPU) -> CPState -> List Double -> Nat ->
            List StepRec -> List StepRec
rolloutEp _ _ _ Z acc = reverse acc
rolloutEp _ _ [] _ acc = reverse acc
rolloutEp model st (r :: rs) (S k) acc =
  let stateT = bulkToTensor (observe st)
      pair = forwardVarTensor model stateT
      logProbsT = prim__logSoftmax (snd pair) 0
      lp0 = prim__item1d logProbsT 0
      lp1 = prim__item1d logProbsT 1
      action = categoricalSample [exp lp0, exp lp1] r
      selLP = prim__select logProbsT 0 (cast {to=Int} action)
      selLPVal = if action == 0 then lp0 else lp1
  in case cpStep st action of
       (reward, st', done) =>
         let acc' = (selLP, selLPVal, reward) :: acc
         in if done then reverse acc'
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

-- Compute per-episode step losses with advantage
epStepLosses : Double -> Double -> List StepRec -> List (Variable CPU)
epStepLosses gamma baseline steps =
  let rewards = map (\(_, _, r) => r) steps
      rets = discReturns gamma rewards
  in zipWith (\(lp, lpVal, _), gt =>
       Var (prim__mulScalar lp (baseline - gt)) Nothing (lpVal * (baseline - gt)))
     steps rets

sumRewards : List StepRec -> Double
sumRewards steps = foldl (\a, (_, _, r) => a + r) 0.0 steps

averageLoss : List (Variable CPU) -> Variable CPU
averageLoss losses =
  let n = fromDouble (cast (natToInteger (length losses)))
      s = foldl (+) (fromDouble 0.0) losses
  in s / n

computeLoss : {hs : List Nat} -> Double ->
              Network 4 hs 2 (Variable CPU) -> List (List Double) ->
              Variable CPU
computeLoss gamma model randomBatch =
  let episodes = map (\rs => rolloutEp model (MkCP 0 0 0 0) rs MaxSteps []) randomBatch
      epReturns = map sumRewards episodes
      nEp = cast {to=Double} (natToInteger (length epReturns))
      baseline = foldl (+) 0.0 epReturns / nEp
      stepLosses = concatMap (epStepLosses gamma baseline) episodes
      avg = averageLoss stepLosses
  in Var avg.tensorPtr Nothing (negate baseline)


----------------------------------------------------------------------
-- Training
----------------------------------------------------------------------

epochRL : {hs : List Nat} -> NativeOptimizer -> Double ->
          Network 4 hs 2 (Variable CPU) -> List (List Double) ->
          (Network 4 hs 2 (Variable CPU), Double)
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
         Network 4 hs 2 (Variable CPU) -> CPState -> Nat -> Double -> Double
evalEp _ _ Z acc = acc
evalEp model st (S k) acc =
  let stateT = bulkToTensor (observe st)
      pair = forwardVarTensor model stateT
      logitsT = snd pair
      action = if prim__item1d logitsT 0 >= prim__item1d logitsT 1 then the Nat 0 else 1
  in case cpStep st action of
       (reward, st', done) =>
         if done then acc + reward
         else evalEp model st' k (acc + reward)

evalN : {hs : List Nat} -> Network 4 hs 2 (Variable CPU) -> Nat -> Double -> Double
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

defaultConfig : Config
defaultConfig = MkConfig 0.001 2000 42 0.99 10

specs : List (ArgSpec Config)
specs = [ Arg "--lr" (\v, c => { lr := cast v } c)
        , Arg "--epochs" (\v, c => { epochs := castNat v } c)
        , Arg "--seed" (\v, c => { seed := castBits64 v } c)
        , Arg "--gamma" (\v, c => { gamma := cast v } c)
        , Arg "--batch" (\v, c => { batchSz := castNat v } c) ]

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

  ll1 <- linearLayer {i=4} {o=128}
  ll2 <- linearLayer {i=128} {o=2}
  let model = autoName $ ll1 ~> tanhLayer ~> OutputLayer ll2
  putStrLn $ "Architecture: " ++ show model
  putStrLn ""

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
