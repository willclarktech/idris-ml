-- | NTM Copy Task
-- |
-- | Binary vector copy with an LSTM-controller NTM, on the v1 Nn/fit
-- | surface. Two-phase sequence: encode the input rows (writing to
-- | memory), then decode by feeding zero inputs and reading the copied
-- | rows back. Sigmoid output + BCE-with-logits loss, RMSprop.

module Example.NtmCopy

import Control.Linear.LIO
import Data.Linear.Notation
import Data.List
import Data.Vect
import System

import BuildConfig
import Compat.Random
import Fit
import ML.Simple
import Train

----------------------------------------------------------------------
-- Configuration (dims)
----------------------------------------------------------------------

W : Nat
W = 8

InputW : Nat
InputW = W + 1  -- data channels + delimiter

OutputW : Nat
OutputW = W

N : Nat
N = 128

M : Nat
M = 20

H : Nat
H = 100

Model : Type
Model = Ntm N M H InputW OutputW Ex F WithGrad

----------------------------------------------------------------------
-- Copy-task data
----------------------------------------------------------------------

Seq : Type
Seq = (List (Vect InputW Double), List (Vect OutputW Double))

randomInt : (lo, hi : Nat) -> IO Nat
randomInt lo hi = do
  n <- randomRIO (cast {to=Int32} (natToInteger lo), cast {to=Int32} (natToInteger hi))
  pure (fromInteger (cast {to=Integer} n))

randomBitVec : (w : Nat) -> IO (Vect w Double)
randomBitVec w = traverse (\_ => do b <- randomRIO (the Int32 0, 1)
                                    pure (if b == 1 then 1.0 else 0.0))
                          (Vect.replicate w ())

-- Input row = data ++ [0] (delimiter channel off); delimiter = 0s ++ [1].
-- Target rows = the data rows (copy them back during decode).
genCopySeq : (seqLen : Nat) -> IO Seq
genCopySeq seqLen = do
  dataRows <- sequence (List.replicate seqLen (randomBitVec W))
  let inputRows = map (\r => r ++ [0.0]) dataRows ++ [Vect.replicate W 0.0 ++ [1.0]]
  pure (inputRows, dataRows)

genBatch : (n, minLen, maxLen : Nat) -> IO (List Seq)
genBatch Z _ _               = pure []
genBatch (S k) minLen maxLen = do
  len <- randomInt minLen maxLen
  dp <- genCopySeq len
  rest <- genBatch k minLen maxLen
  pure (dp :: rest)

----------------------------------------------------------------------
-- Two-phase loss
----------------------------------------------------------------------

zeroIn : IO (Tensor [InputW] Ex F WithGrad)
zeroIn = retypeGrad <$> tensor {dims = [InputW]} (Const 0.0)

sumLosses : List (Tensor [] Ex F WithGrad) -> IO (Tensor [] Ex F WithGrad)
sumLosses []        = assert_total $ idris_crash "NtmCopy.sumLosses: empty"
sumLosses (x :: xs) = go x xs
  where
    go : Tensor [] Ex F WithGrad -> List (Tensor [] Ex F WithGrad) -> IO (Tensor [] Ex F WithGrad)
    go acc []        = pure acc
    go acc (y :: ys) = do s <- tadd acc y; go s ys

-- The model is a bare `Ntm` recurrent layer (no wrapper record), so it is
-- threaded single-owner directly through `recurStep` at every timestep — a
-- stale memory/controller-state reuse is a compile-time linearity error.

-- Encode: feed input rows, write to memory, discard outputs, thread the cell.
encodeAllL : (1 _ : Model) -> List (Vect InputW Double) -> L IO {use = 1} Model
encodeAllL cell []            = pure1 cell
encodeAllL cell (row :: rest) = do
  x <- liftIO1 (retypeGrad <$> tensor {dims = [InputW]} (FromVect row))
  (MkBang _ # cell') <- recurStep cell x
  encodeAllL cell' rest

-- Decode: feed zeros, read rows back, BCE per step vs target, threading the
-- cell and collecting the (ω) per-step losses in forward order.
decodeLossesL : (1 _ : Model) -> List (Vect OutputW Double) -> List (Tensor [] Ex F WithGrad) ->
                L IO {use = 1} (LPair (!* (List (Tensor [] Ex F WithGrad))) Model)
decodeLossesL cell []            acc  = pure1 (MkBang (reverse acc) # cell)
decodeLossesL cell (trow :: rest) acc = do
  z <- liftIO1 zeroIn
  (MkBang out # cell') <- recurStep cell z
  l <- liftIO1 $ do
         y <- retypeGrad <$> tensor {dims = [OutputW]} (FromVect trow)
         tbceLoss out y
  decodeLossesL cell' rest (l :: acc)

twoPhaseLossL : (1 _ : Model) -> Seq -> L IO {use = 1} (LPair (!* (Tensor [] Ex F WithGrad)) Model)
twoPhaseLossL cell (encIns, targs) = do
  enc <- encodeAllL (recurReset cell) encIns
  (MkBang ls # enc') <- decodeLossesL enc targs []
  mean <- liftIO1 $ do s <- sumLosses ls; (1.0 / cast (length targs)) *: s
  pure1 (MkBang mean # enc')

-- Linear-resource epoch step, fine-grained: thread the cell across the batch's
-- two-phase sequences, accumulate the (ω) losses, one optimizer step.
recurEpochL : Optimizer Ex -> (1 _ : Model) -> List Seq ->
              L IO {use = 1} (LPair (!* Double) Model)
recurEpochL opt cell0 batch = do
  (MkBang ls # cellFinal) <- foldBatch cell0 batch []
  d <- liftIO1 $ do
         s    <- sumLosses ls
         mean <- (1.0 / cast (length batch)) *: s
         nativeTrainStep opt mean
  pure1 (MkBang d # cellFinal)
  where
    foldBatch : (1 _ : Model) -> List Seq -> List (Tensor [] Ex F WithGrad) ->
                L IO {use = 1} (LPair (!* (List (Tensor [] Ex F WithGrad))) Model)
    foldBatch cell []          acc = pure1 (MkBang (reverse acc) # cell)
    foldBatch cell (s :: rest) acc = do
      (MkBang l # cell') <- twoPhaseLossL cell s
      foldBatch cell' rest (l :: acc)

----------------------------------------------------------------------
-- Eval: bit accuracy over a fresh test batch (no grad)
----------------------------------------------------------------------

-- Decode under the linear no-grad bracket, threading the cell and counting
-- (matching bits, total bits).
scoreSeqL : (1 _ : Model) -> Seq -> L IO {use = 1} (LPair (!* (Nat, Nat)) Model)
scoreSeqL cell0 (encIns, targs) = withNoGradL {ex = Ex} $ do
  enc <- encodeAllL (recurReset cell0) encIns
  go enc targs 0 0
  where
    go : (1 _ : Model) -> List (Vect OutputW Double) -> Nat -> Nat ->
         L IO {use = 1} (LPair (!* (Nat, Nat)) Model)
    go cell []            correct tot  = pure1 (MkBang (correct, tot) # cell)
    go cell (trow :: rest) correct tot = do
      z <- liftIO1 zeroIn
      (MkBang out # cell') <- recurStep cell z
      let logits  = [ primItem1d {ex = Ex} out.tensorPtr (cast j) | j <- [the Nat 0 .. OutputW `minus` 1] ]
          matches = length [ () | (lg, tv) <- zip logits (toList trow), (lg >= 0.0) == (tv >= 0.5) ]
      go cell' rest (correct + matches) (tot + OutputW)

bitAccuracyL : (1 _ : Model) -> List Seq -> L IO {use = 1} (LPair (!* Double) Model)
bitAccuracyL cell0 batch = do
  (MkBang scores # cellFinal) <- foldScore cell0 batch []
  let (corrects, totals) = unzip scores
      correct = sum corrects
      tot     = sum totals
  pure1 (MkBang (if tot == 0 then 0.0 else cast correct / cast tot) # cellFinal)
  where
    foldScore : (1 _ : Model) -> List Seq -> List (Nat, Nat) ->
                L IO {use = 1} (LPair (!* (List (Nat, Nat))) Model)
    foldScore cell []          acc = pure1 (MkBang (reverse acc) # cell)
    foldScore cell (s :: rest) acc = do
      (MkBang sc # cell') <- scoreSeqL cell s
      foldScore cell' rest (sc :: acc)

----------------------------------------------------------------------
-- Config & Main
----------------------------------------------------------------------

record Config where
  constructor MkConfig
  lr          : Double
  clipVal     : Double
  alpha       : Double
  momentum    : Double
  epochs      : Nat
  esThreshold : Double
  esWindow    : Nat
  esPatience  : Nat
  seed        : Bits64
  minLen      : Nat
  maxLen      : Nat
  batch       : Nat

defaultConfig : Config
defaultConfig = MkConfig 0.0001 10.0 0.95 0.9 10000 0.01 1000 3 42 1 20 1

specs : List (ArgSpec Config)
specs = [ Arg "--lr" (\v, c => { lr := cast v } c)
        , Arg "--clip" (\v, c => { clipVal := cast v } c)
        , Arg "--alpha" (\v, c => { alpha := cast v } c)
        , Arg "--momentum" (\v, c => { momentum := cast v } c)
        , Arg "--epochs" (\v, c => { epochs := castNat v } c)
        , Arg "--es-threshold" (\v, c => { esThreshold := cast v } c)
        , Arg "--es-window" (\v, c => { esWindow := castNat v } c)
        , Arg "--es-patience" (\v, c => { esPatience := castNat v } c)
        , Arg "--seed" (\v, c => { seed := castBits64 v } c)
        , Arg "--min-len" (\v, c => { minLen := castNat v } c)
        , Arg "--max-len" (\v, c => { maxLen := castNat v } c)
        , Arg "--batch" (\v, c => { batch := castNat v } c) ]

%default partial

main : IO ()
main = do
  requireMachine {m = ChosenMachine}
  args <- getArgs
  let cfg = parseArgs defaultConfig specs (drop 1 args)

  srand cfg.seed
  tsetInitSeed {ex = Ex} cfg.seed

  putStrLn "=== NTM Copy ==="
  putStrLn $ "Config: lr=" ++ show cfg.lr ++ " clip=" ++ show cfg.clipVal
           ++ " epochs=" ++ show cfg.epochs ++ " seed=" ++ show cfg.seed
           ++ " batch=" ++ show cfg.batch
           ++ " seqLen=" ++ show cfg.minLen ++ "-" ++ show cfg.maxLen
  putStrLn $ "Architecture: N=" ++ show N ++ " M=" ++ show M ++ " H=" ++ show H

  opt <- rmsprop cfg.lr {alpha = cfg.alpha} {momentum = cfg.momentum}
                 ({ clip := NormClip cfg.clipVal } defaultOpts)
  let dataStream = generate (genBatch cfg.batch cfg.minLen cfg.maxLen)

  -- Linear surface end to end: model born linear (runInitL), threaded through
  -- fit (recurEpochL borrows-and-returns it each epoch), eval via withModelL,
  -- final handle discarded. Final loss is discarded: with windowed-percentile
  -- early stop the engine's returned loss isn't meaningful; bit accuracy is
  -- the headline.
  Control.Linear.LIO.run $ do
    model <- runInitL (ntm {n = N} {m = M} {h = H} {i = InputW} {o = OutputW})
    liftIO1 (putStrLn "")
    (MkBang (epochsDone, _) # trained) <-
      fit (recurEpochL opt) opt dataStream
           (windowedPercentileConfig cfg.epochs 0.10 cfg.esThreshold cfg.esWindow cfg.esPatience)
           model
    liftIO1 (putStrLn "" >> putStrLn "Eval:")
    testBatch <- liftIO1 (genBatch 100 1 20)
    (MkBang acc # trained') <- bitAccuracyL trained testBatch
    discard trained'
    liftIO1 $ do
      putStrLn $ "  Bit accuracy (len 1-20): " ++ show (acc * 100.0) ++ "%"
      putStrLn ""
      putStrLn $ formatResult [("epochs", show epochsDone),
                               ("acc", show acc), ("seed", show cfg.seed)]
