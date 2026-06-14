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
import FitL
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

-- Encode: feed input rows, write to memory, discard outputs. Thread cell.
encodeAll : Model -> List (Vect InputW Double) -> IO Model
encodeAll cell []            = pure cell
encodeAll cell (row :: rest) = do
  x <- retypeGrad <$> tensor {dims = [InputW]} (FromVect row)
  (cell', _) <- recurStep cell x
  encodeAll cell' rest

-- Decode: feed zeros, read rows back, BCE per step vs target.
decodeLosses : Model -> List (Vect OutputW Double) -> IO (List (Tensor [] Ex F WithGrad))
decodeLosses _ []                = pure []
decodeLosses cell (trow :: rest) = do
  z <- zeroIn
  (cell', out) <- recurStep cell z
  y <- retypeGrad <$> tensor {dims = [OutputW]} (FromVect trow)
  l <- tbceLoss out y
  ls <- decodeLosses cell' rest
  pure (l :: ls)

twoPhaseLoss : Model -> Seq -> IO (Tensor [] Ex F WithGrad)
twoPhaseLoss model (encIns, targs) = do
  enc <- encodeAll (recurReset model) encIns
  ls  <- decodeLosses enc targs
  s   <- sumLosses ls
  (1.0 / cast (length targs)) *: s

-- Borrow a linear NTM for an IO action that needs it (consume-match-rebuild-
-- delegate): the model is a bare `Ntm` layer rather than a wrapper record, so
-- match its `MkNtm` constructor (binding all fields at ω), build a reusable ω
-- model, run the IO action, and return the model beside the banged result.
-- One match here; every linear read site goes through this helper.
withModelL : {0 a : Type} -> (1 _ : Model) -> (Model -> IO a) ->
             L IO {use = 1} (LPair (!* a) Model)
withModelL (MkNtm ctrl rfc wfc ofc memInit iro memS raS waS roS) act = do
  let m : Model := MkNtm ctrl rfc wfc ofc memInit iro memS raS waS roS
  r <- liftIO1 (act m)
  pure1 (MkBang r # m)

-- Linear-resource epoch step: borrow the model, run the two-phase batch loss +
-- optimizer step, thread the model back.
recurEpochL : Optimizer Ex -> (1 _ : Model) -> List Seq ->
              L IO {use = 1} (LPair (!* Double) Model)
recurEpochL opt model batch =
  withModelL model (\m => do
    ls   <- traverse (twoPhaseLoss m) batch
    s    <- sumLosses ls
    mean <- (1.0 / cast (length batch)) *: s
    nativeTrainStep opt mean)

----------------------------------------------------------------------
-- Eval: bit accuracy over a fresh test batch (no grad)
----------------------------------------------------------------------

-- Decode under no-grad, counting (matching bits, total bits).
scoreSeq : Model -> Seq -> IO (Nat, Nat)
scoreSeq model (encIns, targs) = withNoGrad {ex = Ex} $ do
  enc <- encodeAll (recurReset model) encIns
  go enc targs 0 0
  where
    go : Model -> List (Vect OutputW Double) -> Nat -> Nat -> IO (Nat, Nat)
    go _ [] correct tot                = pure (correct, tot)
    go cell (trow :: rest) correct tot = do
      z <- zeroIn
      (cell', out) <- recurStep cell z
      let logits  = [ primItem1d {ex = Ex} out.tensorPtr (cast j) | j <- [the Nat 0 .. OutputW `minus` 1] ]
          matches = length [ () | (lg, tv) <- zip logits (toList trow), (lg >= 0.0) == (tv >= 0.5) ]
      go cell' rest (correct + matches) (tot + OutputW)

bitAccuracy : Model -> List Seq -> IO Double
bitAccuracy model batch = do
  scores <- traverse (scoreSeq model) batch
  let (corrects, totals) = unzip scores
      correct = sum corrects
      tot     = sum totals
  pure (if tot == 0 then 0.0 else cast correct / cast tot)

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
  -- fitL (recurEpochL borrows-and-returns it each epoch), eval via withModelL,
  -- final handle discarded. Final loss is discarded: with windowed-percentile
  -- early stop the engine's returned loss isn't meaningful; bit accuracy is
  -- the headline.
  Control.Linear.LIO.run $ do
    model <- runInitL (ntm {n = N} {m = M} {h = H} {i = InputW} {o = OutputW})
    liftIO1 (putStrLn "")
    (MkBang (epochsDone, _) # trained) <-
      fitL (recurEpochL opt) opt dataStream
           (windowedPercentileConfig cfg.epochs 0.10 cfg.esThreshold cfg.esWindow cfg.esPatience)
           model
    liftIO1 (putStrLn "" >> putStrLn "Eval:")
    (MkBang acc # trained') <- withModelL trained (\m => do
      testBatch <- genBatch 100 1 20
      bitAccuracy m testBatch)
    discardL trained'
    liftIO1 $ do
      putStrLn $ "  Bit accuracy (len 1-20): " ++ show (acc * 100.0) ++ "%"
      putStrLn ""
      putStrLn $ formatResult [("epochs", show epochsDone),
                               ("acc", show acc), ("seed", show cfg.seed)]
