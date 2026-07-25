-- | NTM Associative Recall Task
-- |
-- | Store a sequence of items, then (given a query item) recall the item
-- | that followed it. LSTM-controller NTM, on the v1 Nn/fit surface. Same
-- | two-phase encode/decode shape as NtmCopy; the data is the
-- | associative-recall encoding (items with delimiters, then a query).

module Example.NtmAssociativeRecall

import Control.Linear.LIO
import Data.Linear.Notation
import Data.List
import Data.Maybe
import Data.Vect
import System

import Ml.Compat.Random
import Ml.Fit
import Ml.Simple
import Ml.Train

import BuildConfig

----------------------------------------------------------------------
-- Configuration (dims)
----------------------------------------------------------------------

W : Nat
W = 6

InputW : Nat
InputW = W + 2  -- data channels + item-delim + query-delim

OutputW : Nat
OutputW = W

SeqLen : Nat
SeqLen = 3   -- rows per item

N : Nat
N = 128

M : Nat
M = 20

H : Nat
H = 100

Model : Type
Model = Ntm N M H InputW OutputW Ex F WithGrad

----------------------------------------------------------------------
-- Associative-recall data (was Generate.recallTaskBinary)
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

nth : Nat -> List a -> Maybe a
nth _ []            = Nothing
nth Z (x :: _)      = Just x
nth (S k) (_ :: xs) = nth k xs

-- Structure (InputW = W+2): each item is SeqLen data rows padded with two
-- zero channels, prefixed by an item-delimiter (channel W=1). After all
-- items: a query-delimiter (channel W+1=1), the queried item's rows, and a
-- closing query-delimiter. Target = the item that FOLLOWS the queried one.
genRecallSeq : (numItems : Nat) -> IO Seq
genRecallSeq numItems = do
  items <- sequence (List.replicate numItems
             (sequence (List.replicate SeqLen (randomBitVec W))))
  queryIdx <- randomInt 0 (numItems `minus` 2)
  let itemDelim  = Vect.replicate W 0.0 ++ [1.0, 0.0]
      queryDelim = Vect.replicate W 0.0 ++ [0.0, 1.0]
      padRow : Vect W Double -> Vect InputW Double
      padRow r   = r ++ [0.0, 0.0]
      encItems   = concatMap (\item => itemDelim :: map padRow item) items
      queryItem  = fromMaybe [] (nth queryIdx items)
      targetItem = fromMaybe [] (nth (S queryIdx) items)
      encQuery   = queryDelim :: (map padRow queryItem ++ [queryDelim])
  pure (encItems ++ encQuery, targetItem)

genBatch : (n, minItems, maxItems : Nat) -> IO (List Seq)
genBatch Z _ _                   = pure []
genBatch (S k) minItems maxItems = do
  ni <- randomInt (max 2 minItems) (max 2 maxItems)
  dp <- genRecallSeq ni
  rest <- genBatch k minItems maxItems
  pure (dp :: rest)

----------------------------------------------------------------------
-- Two-phase loss (identical shape to NtmCopy)
----------------------------------------------------------------------

zeroIn : IO (Tensor [InputW] Ex F WithGrad)
zeroIn = retypeGrad <$> tensor {dims = [InputW]} (Const 0.0)

sumLosses : List (Tensor [] Ex F WithGrad) -> IO (Tensor [] Ex F WithGrad)
sumLosses []        = assert_total $ idris_crash "NtmAssociativeRecall.sumLosses: empty"
sumLosses (x :: xs) = go x xs
  where
    go : Tensor [] Ex F WithGrad -> List (Tensor [] Ex F WithGrad) -> IO (Tensor [] Ex F WithGrad)
    go acc []        = pure acc
    go acc (y :: ys) = do s <- tadd acc y; go s ys

-- The model is a bare `Ntm` recurrent layer threaded single-owner through
-- `recurStep` at every timestep (see Example.NtmCopy for the shared shape).

encodeAllL : (1 _ : Model) -> List (Vect InputW Double) -> L IO {use = 1} Model
encodeAllL cell []            = pure1 cell
encodeAllL cell (row :: rest) = do
  x <- liftIO1 (retypeGrad <$> tensor {dims = [InputW]} (FromVect row))
  (MkBang _ # cell') <- recurStep cell x
  encodeAllL cell' rest

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

recurEpochL : Optimizer Ex -> (1 _ : Model) -> List Seq ->
              L IO {use = 1} (LPair (!* Double) Model)
recurEpochL opt cell0 batch = do
  (MkBang ls # cellFinal) <- foldBatch cell0 batch []
  d <- liftIO1 $ do
         s    <- sumLosses ls
         mean <- (1.0 / cast (length batch)) *: s
         trainStep opt mean
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

-- Score one sequence: (matching bits, total bits, whole-sequence-correct).
scoreSeqL : (1 _ : Model) -> Seq -> L IO {use = 1} (LPair (!* (Nat, Nat, Bool)) Model)
scoreSeqL cell0 (encIns, targs) = withNoGradL {ex = Ex} $ do
  enc <- encodeAllL (recurReset cell0) encIns
  go enc targs 0 0 True
  where
    go : (1 _ : Model) -> List (Vect OutputW Double) -> Nat -> Nat -> Bool ->
         L IO {use = 1} (LPair (!* (Nat, Nat, Bool)) Model)
    go cell []            correct tot allOk  = pure1 (MkBang (correct, tot, allOk) # cell)
    go cell (trow :: rest) correct tot allOk = do
      z <- liftIO1 zeroIn
      (MkBang out # cell') <- recurStep cell z
      let logits  = [ primItem1d {ex = Ex} out.tensorPtr (cast j) | j <- [the Nat 0 .. OutputW `minus` 1] ]
          matches = length [ () | (lg, tv) <- zip logits (toList trow), (lg >= 0.0) == (tv >= 0.5) ]
      go cell' rest (correct + matches) (tot + OutputW) (allOk && matches == OutputW)

-- (per-bit accuracy, per-sequence accuracy). Per-bit is the headline; per-sequence
-- (a sequence counts only if every bit matches) is the stricter signal.
bitAccuracyL : (1 _ : Model) -> List Seq -> L IO {use = 1} (LPair (!* (Double, Double)) Model)
bitAccuracyL cell0 batch = do
  (MkBang scores # cellFinal) <- foldScore cell0 batch []
  let correct = sum [ c | (c, _, _) <- scores ]
      tot    = sum [ t | (_, t, _) <- scores ]
      seqOk  = length (filter (\(_, _, ok) => ok) scores)
      nSeqs  = length scores
      bitAcc = if tot == 0 then 0.0 else cast correct / cast tot
      seqAcc = if nSeqs == 0 then 0.0 else cast seqOk / cast nSeqs
  pure1 (MkBang (bitAcc, seqAcc) # cellFinal)
  where
    foldScore : (1 _ : Model) -> List Seq -> List (Nat, Nat, Bool) ->
                L IO {use = 1} (LPair (!* (List (Nat, Nat, Bool))) Model)
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
  minItems    : Nat
  maxItems    : Nat
  batch       : Nat

defaultConfig : Config
defaultConfig = MkConfig 0.0001 10.0 0.95 0.9 30000 0.01 1000 3 42 2 6 1

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
        , Arg "--min-items" (\v, c => { minItems := castNat v } c)
        , Arg "--max-items" (\v, c => { maxItems := castNat v } c)
        , Arg "--batch" (\v, c => { batch := castNat v } c) ]

%default partial

main : IO ()
main = do
  requireMachine {m = ChosenMachine}
  args <- getArgs
  let cfg = parseArgs defaultConfig specs (drop 1 args)

  srand cfg.seed
  tsetInitSeed {ex = Ex} cfg.seed

  putStrLn "=== NTM Associative Recall ==="
  putStrLn $ "Config: lr=" ++ show cfg.lr ++ " clip=" ++ show cfg.clipVal
           ++ " epochs=" ++ show cfg.epochs ++ " seed=" ++ show cfg.seed
           ++ " batch=" ++ show cfg.batch
           ++ " items=" ++ show cfg.minItems ++ "-" ++ show cfg.maxItems
  putStrLn $ "Architecture: N=" ++ show N ++ " M=" ++ show M ++ " H=" ++ show H

  opt <- rmsprop cfg.lr {alpha = cfg.alpha} {momentum = cfg.momentum}
                 ({ clip := NormClip cfg.clipVal } defaultOpts)
  let dataStream = generate (genBatch cfg.batch cfg.minItems cfg.maxItems)

  -- Linear surface end to end (see Example.NtmCopy).
  Control.Linear.LIO.run $ do
    model <- runInitL (ntm {n = N} {m = M} {h = H} {i = InputW} {o = OutputW})
    liftIO1 (putStrLn "")
    (MkBang (epochsDone, _) # trained) <-
      fit (recurEpochL opt) opt dataStream
           (windowedPercentileConfig cfg.epochs 0.10 cfg.esThreshold cfg.esWindow cfg.esPatience)
           model
    liftIO1 (putStrLn "" >> putStrLn "Eval:")
    -- Mixed 2-6 (the gated metric) plus the per-K split the reference
    -- reports, which shows where along the difficulty curve a run breaks.
    mixedBatch <- liftIO1 (genBatch 100 2 6)
    (MkBang (acc, seqAcc) # trained1) <- bitAccuracyL trained mixedBatch
    k2Batch <- liftIO1 (genBatch 100 2 2)
    (MkBang (accK2, seqK2) # trained2) <- bitAccuracyL trained1 k2Batch
    k4Batch <- liftIO1 (genBatch 100 4 4)
    (MkBang (accK4, seqK4) # trained3) <- bitAccuracyL trained2 k4Batch
    k6Batch <- liftIO1 (genBatch 100 6 6)
    (MkBang (accK6, seqK6) # trained4) <- bitAccuracyL trained3 k6Batch
    discard trained4
    liftIO1 $ do
      putStrLn $ "  mixed 2-6: " ++ show (acc * 100.0) ++ "% bit, "
               ++ show (seqAcc * 100.0) ++ "% seq"
      putStrLn $ "  K=2 items: " ++ show (accK2 * 100.0) ++ "% bit, "
               ++ show (seqK2 * 100.0) ++ "% seq"
      putStrLn $ "  K=4 items: " ++ show (accK4 * 100.0) ++ "% bit, "
               ++ show (seqK4 * 100.0) ++ "% seq"
      putStrLn $ "  K=6 items: " ++ show (accK6 * 100.0) ++ "% bit, "
               ++ show (seqK6 * 100.0) ++ "% seq"
      putStrLn ""
      putStrLn $ formatResult [("epochs", show epochsDone),
                               ("acc", show acc), ("seq_acc", show seqAcc),
                               ("acc_k2", show accK2), ("acc_k4", show accK4),
                               ("acc_k6", show accK6),
                               ("seq_acc_k2", show seqK2), ("seq_acc_k4", show seqK4),
                               ("seq_acc_k6", show seqK6),
                               ("seed", show cfg.seed)]
