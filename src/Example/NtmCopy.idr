-- | NTM Copy Task
-- |
-- | Exercises LOCATION-BASED addressing: the model writes symbols
-- | sequentially into memory (shift right), then reads them back in
-- | the same order (shift right again). Content addressing is not
-- | required — sequential shifting through memory slots suffices.
-- |
-- | See NtmAssociativeRecall.idr for the content-based addressing
-- | counterpart.

module Example.NtmCopy

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

||| Input/output size = number of symbols (0 = <BLANK>)
W : Nat
W = 3

||| Number of memory slots
N : Nat
N = 10

||| Controller hidden layer size
H : Nat
H = 20

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
genData mnL mxL = map (map (map fromDouble)) (randomBatchVect (copyTask {w=W}) BatchSize mnL mxL)

stages : List (Stage W W BatchSize)
stages =
  [ MkStage "Stage 1 (len 1-3)" 0.15 (genData 1 3)
  , MkStage "Stage 2 (len 1-5)" 0.10 (genData 1 5)
  , MkStage "Stage 3 (len 1-8)" 0.0  (genData 1 8)
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
defaultConfig = MkConfig 0.001 50.0 0.9 0.999 (pow 10 (-8)) 10.0 6000 200 123456 False False

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

  putStrLn "=== NTM Copy Task (Curriculum) ==="
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
  let model = autoName $ ntm ~> OutputLayer logSoftmaxLayer

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
  shortBatch <- randomBatchVect (copyTask {w=W}) TestSize 1 3
  fullBatch <- randomBatchVect (copyTask {w=W}) TestSize 1 8
  let shortPts = map (map fromDouble) shortBatch
  let fullPts = map (map fromDouble) fullBatch
  let shortTargets = map (\dp => map argmax (ys dp)) shortPts
  let fullTargets = map (\dp => map argmax (ys dp)) fullPts
  let shortPreds = decodeOutput $ evaluateRecurrent trained shortPts
  let fullPreds = decodeOutput $ evaluateRecurrent trained fullPts
  let shortAcc = accuracy shortPreds shortTargets
  let fullAcc = accuracy fullPreds fullTargets

  putStrLn "Eval (random sequences):"
  putStr "  Short (len 1-3):\t"
  putStrLn $ show shortAcc
  putStr "  Full (len 1-8):\t"
  putStrLn $ show fullAcc

  -- Diagnostics
  when cfg.diagnose $ do
    let dblModel = toDoubleNetwork trained
    putStrLn ""
    putStrLn "=== NTM Diagnostic Analysis ==="

    let diagnoseOne : RecurrentDataPoint W W Double -> String -> IO (Maybe NtmSummary)
        diagnoseOne dp label = do
          let inputs = xs dp
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

    -- Short sequences
    putStrLn "--- Short Sequences ---"
    d3 <- (copyTask {w=W}).generatePoint 3
    d5 <- (copyTask {w=W}).generatePoint 5
    s0 <- diagnoseOne d3 "Diag (len=3)"
    s1 <- diagnoseOne d5 "Diag (len=5)"

    -- Long sequences
    putStrLn "--- Long Sequences ---"
    d8a <- (copyTask {w=W}).generatePoint 8
    d8b <- (copyTask {w=W}).generatePoint 8
    d8c <- (copyTask {w=W}).generatePoint 8
    t0 <- diagnoseOne d8a "Diag (len=8, a)"
    t1 <- diagnoseOne d8b "Diag (len=8, b)"
    t2 <- diagnoseOne d8c "Diag (len=8, c)"

    -- Verbose raw dumps
    when cfg.diagnoseVerbose $ do
      putStrLn "--- Verbose: Short ---"
      let (_, _, snaps0) = debugForwardRecurrent dblModel (xs d3)
      printDiagnostics "Short" snaps0
      putStrLn ""
      putStrLn "--- Verbose: Long ---"
      let (_, _, snapsL) = debugForwardRecurrent dblModel (xs d8a)
      printDiagnostics "Long" snapsL

    -- Aggregate comparison
    let shortSums = mapMaybe id [s0, s1]
    let longSums = mapMaybe id [t0, t1, t2]
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
           ++ show shortAcc ++ "\t"
           ++ show fullAcc
