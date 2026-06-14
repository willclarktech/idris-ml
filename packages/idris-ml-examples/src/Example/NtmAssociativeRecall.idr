-- | NTM Associative Recall Task
-- |
-- | Store a sequence of items, then (given a query item) recall the item
-- | that followed it. LSTM-controller NTM, on the v1 Nn/fit surface. Same
-- | two-phase encode/decode shape as NtmCopy; the data is the
-- | associative-recall encoding (items with delimiters, then a query).

module Example.NtmAssociativeRecall

import Data.List
import Data.Maybe
import Data.Vect
import System

import BuildConfig
import Compat.Random
import ML.Simple
import Train

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
nth _ []        = Nothing
nth Z (x :: _)  = Just x
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
      padRow r = r ++ [0.0, 0.0]
      encItems   = concatMap (\item => itemDelim :: map padRow item) items
      queryItem  = fromMaybe [] (nth queryIdx items)
      targetItem = fromMaybe [] (nth (S queryIdx) items)
      encQuery   = queryDelim :: (map padRow queryItem ++ [queryDelim])
  pure (encItems ++ encQuery, targetItem)

genBatch : (n, minItems, maxItems : Nat) -> IO (List Seq)
genBatch Z _ _ = pure []
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
sumLosses [] = assert_total $ idris_crash "NtmAssociativeRecall.sumLosses: empty"
sumLosses (x :: xs) = go x xs
  where
    go : Tensor [] Ex F WithGrad -> List (Tensor [] Ex F WithGrad) -> IO (Tensor [] Ex F WithGrad)
    go acc []        = pure acc
    go acc (y :: ys) = do s <- tadd acc y; go s ys

encodeAll : Model -> List (Vect InputW Double) -> IO Model
encodeAll cell [] = pure cell
encodeAll cell (row :: rest) = do
  x <- retypeGrad <$> tensor {dims = [InputW]} (FromVect row)
  (cell', _) <- recurStep cell x
  encodeAll cell' rest

decodeLosses : Model -> List (Vect OutputW Double) -> IO (List (Tensor [] Ex F WithGrad))
decodeLosses _ [] = pure []
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

recurEpoch : Optimizer Ex -> Model -> List Seq -> IO (Model, Double)
recurEpoch opt model batch = do
  ls   <- traverse (twoPhaseLoss model) batch
  s    <- sumLosses ls
  mean <- (1.0 / cast (length batch)) *: s
  d    <- nativeTrainStep opt mean
  pure (model, d)

----------------------------------------------------------------------
-- Eval: bit accuracy over a fresh test batch (no grad)
----------------------------------------------------------------------

scoreSeq : Model -> Seq -> IO (Nat, Nat)
scoreSeq model (encIns, targs) = withNoGrad {ex = Ex} $ do
  enc <- encodeAll (recurReset model) encIns
  go enc targs 0 0
  where
    go : Model -> List (Vect OutputW Double) -> Nat -> Nat -> IO (Nat, Nat)
    go _ [] correct tot = pure (correct, tot)
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
  lr : Double
  clipVal : Double
  alpha : Double
  momentum : Double
  epochs : Nat
  esThreshold : Double
  esWindow : Nat
  esPatience : Nat
  seed : Bits64
  minItems : Nat
  maxItems : Nat
  batch : Nat

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
  model <- runInit (ntm {n = N} {m = M} {h = H} {i = InputW} {o = OutputW})
  let dataStream = generate (genBatch cfg.batch cfg.minItems cfg.maxItems)
  putStrLn ""

  (trained, epochsDone, _) <-
    fit (recurEpoch opt) opt dataStream
        (windowedPercentileConfig cfg.epochs 0.10 cfg.esThreshold cfg.esWindow cfg.esPatience)
        model

  putStrLn ""
  putStrLn "Eval:"
  testBatch <- genBatch 100 2 6
  acc <- bitAccuracy trained testBatch
  putStrLn $ "  Bit accuracy (2-6 items): " ++ show (acc * 100.0) ++ "%"
  putStrLn ""
  putStrLn $ formatResult [("epochs", show epochsDone),
                           ("acc", show acc), ("seed", show cfg.seed)]
