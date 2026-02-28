-- | NTM Associative Recall Task
-- |
-- | Exercises CONTENT-BASED addressing: key-value pairs are stored in
-- | memory, then queried in shuffled order. The model must use cosine
-- | similarity (content addressing) to find the matching key and
-- | retrieve the associated value. Sequential shifting alone cannot
-- | solve this because queries arrive in random order.
-- |
-- | See NtmCopy.idr for the location-based addressing counterpart.

module Example.NtmAssociativeRecall

import Data.List
import Data.String
import Data.Vect
import System
import System.Random

import Backprop
import Curriculum
import DataPoint
import Debug
import Floating
import Generate
import Layer
import Math
import Optimizer
import Schedule
import Tensor
import Variable


----------------------------------------------------------------------
-- Configuration
----------------------------------------------------------------------

||| Input/output size = number of symbols (0 = <BLANK>, 1-7 = data)
W : Nat
W = 8

||| Number of memory slots
N : Nat
N = 16

||| Controller hidden layer size
H : Nat
H = 40

||| Training batch size (data points per chunk)
BatchSize : Nat
BatchSize = 16

||| Evaluation batch size
TestSize : Nat
TestSize = 20


----------------------------------------------------------------------
-- Decode/Display Helpers
----------------------------------------------------------------------

decodeOutput : Vect n (List (Vector W Variable)) -> Vect n (List (Fin W))
decodeOutput = map (map argmax)

showSequences : Vect n (List (Fin W)) -> String
showSequences seqs = show $ map (map finToNat) seqs

matchCount : List (Fin W) -> List (Fin W) -> Nat
matchCount [] [] = 0
matchCount (x :: xs) (y :: ys) = (if x == y then 1 else 0) + matchCount xs ys
matchCount _ _ = 0

totalLen : Vect n (List a) -> Nat
totalLen [] = 0
totalLen (x :: xs) = length x + totalLen xs

accuracy : Vect n (List (Fin W)) -> Vect n (List (Fin W)) -> Double
accuracy preds targets =
  let len = totalLen targets
      correct = sum $ zipWith matchCount (toList preds) (toList targets)
  in if len == 0 then 0.0 else cast correct / cast len


----------------------------------------------------------------------
-- Curriculum Stages
----------------------------------------------------------------------

genData : Nat -> Nat -> IO (Vect BatchSize (RecurrentDataPoint W W Variable))
genData mnK mxK = map (map (map fromDouble)) (randomBatchVect (associativeRecallTask {w=W}) BatchSize mnK mxK)

stages : List (Stage W W BatchSize)
stages =
  [ MkStage "Stage 1 (K=2 pairs)" 0.12 (genData 2 2)
  , MkStage "Stage 2 (K=3 pairs)" 0.10 (genData 3 3)
  , MkStage "Stage 3 (K=3-4 pairs)" 0.08 (genData 3 4)
  , MkStage "Stage 4 (K=4-5 pairs)" 0.0  (genData 4 5)
  ]


----------------------------------------------------------------------
-- CLI Argument Parsing
----------------------------------------------------------------------

record Config where
  constructor MkConfig
  lr : Double
  maxNorm : Double
  beta1 : Double
  beta2 : Double
  eps : Double
  divFinal : Double
  epochs : Nat
  patience : Nat
  seed : Bits64
  diagnose : Bool
  diagnoseVerbose : Bool

defaultConfig : Config
defaultConfig = MkConfig 0.001 10.0 0.9 0.999 (pow 10 (-8)) 10.0 10000 800 123456 False False

parseConfig : List String -> Config
parseConfig args = go args defaultConfig
  where
    go : List String -> Config -> Config
    go [] c = c
    go ("--lr" :: v :: rest) c = go rest ({ lr := cast v } c)
    go ("--max-norm" :: v :: rest) c = go rest ({ maxNorm := cast v } c)
    go ("--beta1" :: v :: rest) c = go rest ({ beta1 := cast v } c)
    go ("--beta2" :: v :: rest) c = go rest ({ beta2 := cast v } c)
    go ("--eps" :: v :: rest) c = go rest ({ eps := cast v } c)
    go ("--div-final" :: v :: rest) c = go rest ({ divFinal := cast v } c)
    go ("--epochs" :: v :: rest) c = go rest ({ epochs := cast (cast {to=Integer} v) } c)
    go ("--patience" :: v :: rest) c = go rest ({ patience := cast (cast {to=Integer} v) } c)
    go ("--seed" :: v :: rest) c = go rest ({ seed := cast (cast {to=Integer} v) } c)
    go ("--diagnose" :: rest) c = go rest ({ diagnose := True } c)
    go ("--diagnose-verbose" :: rest) c = go rest ({ diagnose := True, diagnoseVerbose := True } c)
    go (_ :: rest) c = go rest c


----------------------------------------------------------------------
-- Main
----------------------------------------------------------------------

main : IO ()
main = do
  args <- getArgs
  let cfg = parseConfig (drop 1 args)

  srand cfg.seed

  putStrLn "=== NTM Associative Recall (Curriculum) ==="
  putStrLn $ "Config: lr=" ++ show cfg.lr
           ++ " maxNorm=" ++ show cfg.maxNorm
           ++ " beta1=" ++ show cfg.beta1
           ++ " beta2=" ++ show cfg.beta2
           ++ " epochs=" ++ show cfg.epochs
           ++ " patience=" ++ show cfg.patience
           ++ " seed=" ++ show cfg.seed
           ++ " H=" ++ show H
  putStrLn ""

  -- Build NTM with logSoftmax output
  controllerHidden <- linearLayer {i = NtmInputWidth W, o = H}
  controllerOut <- linearLayer {i = H, o = NtmOutputWidth N W}
  let controller = controllerHidden ~> tanhLayer ~> OutputLayer controllerOut
  ntm <- ntmLayer {n = N, w = W} controller
  let model = nameNetworkParams "ntm" $ ntm ~> OutputLayer logSoftmaxLayer

  putStr "Model:\t\t"
  printLn model
  putStrLn ""

  -- Curriculum training
  let makeOpt = \lr => adamGlobalClip lr cfg.beta1 cfg.beta2 cfg.eps cfg.maxNorm
  let schedule = oneCycle cfg.lr 25.0 cfg.divFinal 0.25 cfg.epochs
  putStrLn "Training (curriculum + one-cycle)..."
  (trained, finalSt, epochsDone) <- runCurriculum makeOpt schedule model
    nllLoss stages cfg.epochs cfg.patience 100 initState

  putStrLn ""

  -- Final evaluation on fresh random data
  k2Batch <- randomBatchVect (associativeRecallTask {w=W}) TestSize 2 2
  k3Batch <- randomBatchVect (associativeRecallTask {w=W}) TestSize 3 3
  k5Batch <- randomBatchVect (associativeRecallTask {w=W}) TestSize 5 5
  let k2Pts = map (map fromDouble) k2Batch
  let k3Pts = map (map fromDouble) k3Batch
  let k5Pts = map (map fromDouble) k5Batch
  let k2Targets = map (\dp => map argmax (ys dp)) k2Pts
  let k3Targets = map (\dp => map argmax (ys dp)) k3Pts
  let k5Targets = map (\dp => map argmax (ys dp)) k5Pts
  let k2Preds = decodeOutput $ evaluateRecurrent trained k2Pts
  let k3Preds = decodeOutput $ evaluateRecurrent trained k3Pts
  let k5Preds = decodeOutput $ evaluateRecurrent trained k5Pts
  let k2Acc = accuracy k2Preds k2Targets
  let k3Acc = accuracy k3Preds k3Targets
  let k5Acc = accuracy k5Preds k5Targets

  putStrLn "Eval (random sequences):"
  putStr "  K=2 pairs:\t"
  putStrLn $ show k2Acc
  putStr "  K=3 pairs:\t"
  putStrLn $ show k3Acc
  putStr "  K=5 pairs:\t"
  putStrLn $ show k5Acc

  -- Diagnostics
  when cfg.diagnose $ do
    let dblModel = toDoubleNetwork trained
    putStrLn ""
    putStrLn "=== NTM Diagnostic Analysis ==="

    let diagnoseOne : RecurrentDataPoint W W Double -> String -> IO (Maybe NtmSummary)
        diagnoseOne dp label = do
          let inputs = xs dp
          -- seqLen = 2*K + 1 (store phase + delimiter)
          let sl = length inputs `div` 2
          let (_, _, snaps) = debugForwardRecurrent dblModel inputs
          case computeSummary sl snaps of
            Nothing => do
              putStrLn $ label ++ ": no NTM entry found"
              pure Nothing
            Just s => do
              printSummary label s
              printAddrGrid s
              putStrLn ""
              pure (Just s)

    -- K=2 pairs
    putStrLn "--- K=2 Pairs ---"
    d2a <- (associativeRecallTask {w=W}).generatePoint 2
    d2b <- (associativeRecallTask {w=W}).generatePoint 2
    s0 <- diagnoseOne d2a "Diag (K=2, a)"
    s1 <- diagnoseOne d2b "Diag (K=2, b)"

    -- K=3 pairs
    putStrLn "--- K=3 Pairs ---"
    d3a <- (associativeRecallTask {w=W}).generatePoint 3
    d3b <- (associativeRecallTask {w=W}).generatePoint 3
    d3c <- (associativeRecallTask {w=W}).generatePoint 3
    t0 <- diagnoseOne d3a "Diag (K=3, a)"
    t1 <- diagnoseOne d3b "Diag (K=3, b)"
    t2 <- diagnoseOne d3c "Diag (K=3, c)"

    -- K=5 pairs
    putStrLn "--- K=5 Pairs ---"
    d5a <- (associativeRecallTask {w=W}).generatePoint 5
    d5b <- (associativeRecallTask {w=W}).generatePoint 5
    d5c <- (associativeRecallTask {w=W}).generatePoint 5
    u0 <- diagnoseOne d5a "Diag (K=5, a)"
    u1 <- diagnoseOne d5b "Diag (K=5, b)"
    u2 <- diagnoseOne d5c "Diag (K=5, c)"

    -- Verbose raw dumps
    when cfg.diagnoseVerbose $ do
      putStrLn "--- Verbose: K=2 ---"
      let (_, _, snaps0) = debugForwardRecurrent dblModel (xs d2a)
      printDiagnostics "K=2" snaps0
      putStrLn ""
      putStrLn "--- Verbose: K=3 ---"
      let (_, _, snapsL) = debugForwardRecurrent dblModel (xs d3a)
      printDiagnostics "K=3" snapsL
      putStrLn ""
      putStrLn "--- Verbose: K=5 ---"
      let (_, _, snaps5) = debugForwardRecurrent dblModel (xs d5a)
      printDiagnostics "K=5" snaps5

    -- Aggregate comparison
    let shortSums = mapMaybe id [s0, s1]
    let longSums = mapMaybe id [u0, u1, u2]
    case (avgSummaries shortSums, avgSummaries longSums) of
      (Just avgShort, Just avgLong) => do
        putStrLn ""
        printComparison avgShort avgLong
      _ => putStrLn "\n  Insufficient data for comparison"

  -- Machine-readable result line for sweep script
  putStrLn $ "RESULT\t"
           ++ show cfg.lr ++ "\t"
           ++ show cfg.maxNorm ++ "\t"
           ++ show cfg.beta1 ++ "\t"
           ++ show cfg.beta2 ++ "\t"
           ++ show cfg.epochs ++ "\t"
           ++ show cfg.patience ++ "\t"
           ++ show epochsDone ++ "\t"
           ++ show cfg.seed ++ "\t"
           ++ show H ++ "\t"
           ++ show k2Acc ++ "\t"
           ++ show k3Acc ++ "\t"
           ++ show k5Acc
