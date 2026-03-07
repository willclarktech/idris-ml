module Example.Profile

import Data.List
import Data.Vect
import System
import System.Clock
import System.Random

import Backprop
import DataPoint
import Endofunctor
import Floating
import Generate
import Layer
import Math
import Optimizer
import Tensor
import Util
import Variable


----------------------------------------------------------------------
-- Timing
----------------------------------------------------------------------

elapsedMs : Clock Monotonic -> Clock Monotonic -> Double
elapsedMs t0 t1 =
  let s = cast {to=Double} (seconds t1 - seconds t0)
      ns = cast {to=Double} (nanoseconds t1 - nanoseconds t0)
  in s * 1000.0 + ns / 1000000.0

padL : Nat -> String -> String
padL n s = pack (replicate (minus n (length s)) ' ') ++ s

-- Truncate a Double to 1 decimal place string
showMs : Double -> String
showMs d =
  let whole = cast {to=Integer} d
      frac = cast {to=Integer} (abs ((d - cast whole) * 10))
  in show whole ++ "." ++ show frac

fmtMs : Double -> String
fmtMs d = padL 10 (showMs d)


----------------------------------------------------------------------
-- NTM Copy Task Setup (matches NtmCopy.idr)
----------------------------------------------------------------------

||| Binary vector width (data channels)
W : Nat
W = 8

||| Input width = data + delimiter channel
InputW : Nat
InputW = S W

||| Output width = data channels only
OutputW : Nat
OutputW = W

||| Number of memory slots
N : Nat
N = 128

||| Memory width
M : Nat
M = 20

||| Controller hidden size
H : Nat
H = 100

||| Batch size (data points per epoch)
BatchSize : Nat
BatchSize = 16


----------------------------------------------------------------------
-- Tape Histogram
----------------------------------------------------------------------

showHist : Nat -> String
showHist dummy =
  let constOps  = tapeCountTag 0 dummy
      shadowOps = tapeCountTag 25 dummy
  in let scalarOps = tapeCountRange 1 10 dummy + tapeCountRange 19 20 dummy
         tensorOps = tapeCountRange 11 18 dummy + tapeCountRange 21 24 dummy
  in let totalOps = tapeSize dummy
  in "  Tape histogram:\n"
  ++ "    ConstOps:  " ++ padL 8 (show constOps) ++ "\n"
  ++ "    ScalarOps: " ++ padL 8 (show scalarOps) ++ "\n"
  ++ "    TensorOps: " ++ padL 8 (show tensorOps) ++ "\n"
  ++ "    ShadowOps: " ++ padL 8 (show shadowOps) ++ "\n"
  ++ "    Total:     " ++ padL 8 (show totalOps) ++ "\n"
  ++ "    Tensor detail: MatVec=" ++ show (tapeCountTag 11 dummy)
  ++ " Dot=" ++ show (tapeCountTag 12 dummy)
  ++ " Softmax=" ++ show (tapeCountTag 13 dummy)
  ++ " LogSoftmax=" ++ show (tapeCountTag 14 dummy)
  ++ " BatchCosSim=" ++ show (tapeCountTag 15 dummy) ++ "\n"
  ++ "                   ReadOp=" ++ show (tapeCountTag 16 dummy)
  ++ " WriteOp=" ++ show (tapeCountTag 17 dummy)
  ++ " InterpWrite=" ++ show (tapeCountTag 18 dummy)
  ++ " Interpolate=" ++ show (tapeCountTag 21 dummy) ++ "\n"
  ++ "                   Shift=" ++ show (tapeCountTag 22 dummy)
  ++ " Focus=" ++ show (tapeCountTag 23 dummy)
  ++ " LstmCell=" ++ show (tapeCountTag 24 dummy)


----------------------------------------------------------------------
-- Profiled Epoch (inlined two-phase forward with sub-phase timing)
----------------------------------------------------------------------

-- Process a single data point's encoding phase: feed encoding inputs,
-- return updated model. Discard outputs (encoding phase).
encodeOne : Network InputW [] OutputW Variable ->
            TwoPhaseDataPoint InputW OutputW Variable ->
            Network InputW [] OutputW Variable
encodeOne m dp = fst (forwardRecurrentVar m (encodingInputs dp))

-- Process a single data point's output phase: feed zero inputs,
-- return updated model and predictions.
outputOne : Network InputW [] OutputW Variable ->
            TwoPhaseDataPoint InputW OutputW Variable ->
            (Network InputW [] OutputW Variable, List (Vector OutputW Variable))
outputOne m dp =
  let zeroInput : Vector InputW Variable
      zeroInput = map (const (fromDouble 0.0)) zeros
      outputInputs = Data.List.replicate (length (targets dp)) zeroInput
  in forwardRecurrentVar m outputInputs

-- Compute per-sequence loss from predictions and targets.
lossOne : LossFunction Variable ->
          TwoPhaseDataPoint InputW OutputW Variable ->
          List (Vector OutputW Variable) -> Variable
lossOne lossFn dp preds =
  Util.mean (zipWith lossFn preds (targets dp))

profileEpoch :
  DenseOptimizer ->
  Vect BatchSize (TwoPhaseDataPoint InputW OutputW Variable) ->
  LossFunction Variable ->
  Network InputW [] OutputW Variable ->
  DenseOptimizerState ->
  Nat ->
  IO (Network InputW [] OutputW Variable, DenseOptimizerState)
profileEpoch opt dataPoints lossFn model st epochNum = do
  let dpList = toList dataPoints

  -- Sub-phase 1: Encoding forward (feed encoding inputs, discard outputs)
  t0 <- clockTime Monotonic
  let encodedModels = foldl (\m, dp => encodeOne m dp) model dpList
  let ts1 = tapeSize 0  -- forces evaluation of encoding
  t1 <- clockTime Monotonic

  -- Sub-phase 2: Output forward (feed zero inputs, collect predictions)
  let (outModel, allPreds) = foldl
        (\(m, ps), dp =>
          let (m', preds) = outputOne m dp
          in (m', ps ++ [preds]))
        (encodedModels, the (List (List (Vector OutputW Variable))) [])
        dpList
  let ts2 = tapeSize 1  -- forces evaluation of output forward
  t2 <- clockTime Monotonic

  -- Sub-phase 3: Loss computation (BCE on predictions vs targets)
  let perSeqLosses = zipWith (lossOne lossFn) dpList allPreds
      loss = Util.mean perSeqLosses
  let lossVal = loss.value  -- forces evaluation
  let ts3 = tapeSize 2
  t3 <- clockTime Monotonic

  -- Capture tape histogram before backward resets it
  let hist = showHist 3

  -- Phase 2: Backward pass
  let denseBuf = collectGradsDense 1.0 loss st.buf
  t4 <- clockTime Monotonic

  -- Phase 3: Optimizer step
  let st' = opt.step denseBuf st
  t5 <- clockTime Monotonic

  -- Phase 4: Apply deltas + sync buffers
  let model' = applyDeltasAndSyncNetwork denseBuf model
  t6 <- clockTime Monotonic

  let line = padL 5 (show epochNum)
          ++ fmtMs (elapsedMs t0 t1)
          ++ fmtMs (elapsedMs t1 t2)
          ++ fmtMs (elapsedMs t2 t3)
          ++ fmtMs (elapsedMs t3 t4)
          ++ fmtMs (elapsedMs t4 t5)
          ++ fmtMs (elapsedMs t5 t6)
          ++ padL 10 (show ts3)
          ++ "    " ++ show lossVal
  putStrLn line

  -- Print histogram on first epoch only
  when (epochNum == 1) $ do
    putStrLn ""
    putStrLn hist
    putStrLn ""

  pure (model', st')


----------------------------------------------------------------------
-- Profile Loop
----------------------------------------------------------------------

profileLoop :
  DenseOptimizer ->
  Vect BatchSize (TwoPhaseDataPoint InputW OutputW Variable) ->
  LossFunction Variable ->
  Network InputW [] OutputW Variable ->
  DenseOptimizerState ->
  Nat -> Nat ->
  IO (Network InputW [] OutputW Variable, DenseOptimizerState)
profileLoop opt dataPoints lossFn model st cur count =
  if cur >= count
    then pure (model, st)
    else do
      (model', st') <- profileEpoch opt dataPoints lossFn model st (cur + 1)
      profileLoop opt dataPoints lossFn model' st' (cur + 1) count


----------------------------------------------------------------------
-- Main
----------------------------------------------------------------------

main : IO ()
main = do
  srand 123456

  putStrLn "=== NTM Copy Forward-Pass Profile ==="
  putStrLn $ "Architecture: N=" ++ show N ++ " M=" ++ show M ++ " H=" ++ show H
  putStrLn $ "Batch=" ++ show BatchSize ++ " seqLen=1-20"
  putStrLn ""

  -- Build NTM model (same as NtmCopy.idr: no output activation, BCE loss)
  ntm <- ntmLayer {inputSize = InputW, outputSize = OutputW, n = N, m = M, h = H}
  let model = autoName $ OutputLayer ntm

  let numPids = getNumPids 0
  let opt = rmspropValueClipMomentumDense 0.0001 0.95 1.0e-8 10.0 0.9
  let st0 = initDenseState numPids

  -- Generate a fixed batch for consistent profiling
  tGen0 <- clockTime Monotonic
  batch <- copyTaskBinaryBatchVect {w = W} BatchSize 1 20
  let dataPoints = map (map fromDouble) batch
  tGen1 <- clockTime Monotonic
  putStrLn $ "Data generation: " ++ showMs (elapsedMs tGen0 tGen1) ++ " ms"
  putStrLn ""

  -- Warmup: 5 epochs (untimed)
  putStrLn "Warmup (5 epochs)..."
  (warmModel, warmSt) <- go 0 model st0
  putStrLn ""

  let header = padL 5 "Epoch"
            ++ padL 10 "Enc(ms)"
            ++ padL 10 "Out(ms)"
            ++ padL 10 "Loss(ms)"
            ++ padL 10 "Bwd(ms)"
            ++ padL 10 "Opt(ms)"
            ++ padL 10 "Sync(ms)"
            ++ padL 10 "TapeSize"
            ++ "    Loss"
  putStrLn header

  -- Profile: 10 epochs with sub-phase timing
  (finalModel, _) <- profileLoop opt dataPoints binaryCrossEntropyWithLogits warmModel warmSt 0 10

  putStrLn "\nDone."

  where
    -- Warmup loop using epochTwoPhaseDense
    go : Nat -> Network InputW [] OutputW Variable ->
         DenseOptimizerState ->
         IO (Network InputW [] OutputW Variable, DenseOptimizerState)
    go 5 m s = pure (m, s)
    go k m s = do
      batch <- copyTaskBinaryBatchVect {w = W} BatchSize 1 20
      let dps = map (map fromDouble) batch
          opt = rmspropValueClipMomentumDense 0.0001 0.95 1.0e-8 10.0 0.9
          (m', s', loss) = epochTwoPhaseDense opt dps binaryCrossEntropyWithLogits m s
      putStrLn $ "  warmup " ++ show (k + 1) ++ ": loss=" ++ show loss
      go (k + 1) m' s'
