module Generate

import Data.Fin
import Data.List
import Data.Nat
import Data.Stream
import Data.Vect
import Compat.Random

import DataPoint
import Math
import Array
import Executor
import Tensor
import BuildConfig


----------------------------------------------------------------------
-- Port: SequenceTask
----------------------------------------------------------------------

||| A sequence task knows how to produce a RecurrentDataPoint
||| given a sequence length. Different tasks (copy, repeat copy,
||| sorting, associative recall) provide different implementations.
public export
record SequenceTask (i : Nat) (o : Nat) where
  constructor MkSequenceTask
  name : String
  generatePoint : (len : Nat) -> IO (RecurrentDataPoint i o Double)


----------------------------------------------------------------------
-- Generic infrastructure
----------------------------------------------------------------------

||| Random integer in [lo, hi] inclusive
export
randomInt : (lo, hi : Nat) -> IO Nat
randomInt lo hi = do
  n <- randomRIO (cast {to=Int32} (natToInteger lo), cast {to=Int32} (natToInteger hi))
  pure (fromInteger (cast {to=Integer} n))

||| Generate n data points with lengths uniformly sampled from [minLen, maxLen]
export
randomBatch : SequenceTask i o -> (n : Nat) -> (minLen, maxLen : Nat)
           -> IO (List (RecurrentDataPoint i o Double))
randomBatch task Z _ _ = pure []
randomBatch task (S k) minLen maxLen = do
  len <- randomInt minLen maxLen
  dp <- task.generatePoint len
  rest <- randomBatch task k minLen maxLen
  pure (dp :: rest)

||| Generate exactly n data points as a Vect
export
randomBatchVect : SequenceTask i o -> (n : Nat) -> (minLen, maxLen : Nat)
               -> IO (Vect n (RecurrentDataPoint i o Double))
randomBatchVect task Z _ _ = pure []
randomBatchVect task (S k) minLen maxLen = do
  len <- randomInt minLen maxLen
  dp <- task.generatePoint len
  rest <- randomBatchVect task k minLen maxLen
  pure (dp :: rest)


----------------------------------------------------------------------
-- Primitives for building tasks
----------------------------------------------------------------------

||| Generate a list of n random non-blank symbols (values 1..w-1)
export
randomSymbols : {w : Nat} -> (len : Nat) -> IO (List (Fin w))
randomSymbols {w = Z} _ = pure []
randomSymbols {w = S Z} _ = pure []
randomSymbols {w = S (S k)} Z = pure []
randomSymbols {w = S (S k)} (S n) = do
  val <- randomInt 1 (S k)
  let sym = restrict (S k) (cast val)
  rest <- randomSymbols {w = S (S k)} n
  pure (sym :: rest)


----------------------------------------------------------------------
-- Adapter: Copy task
----------------------------------------------------------------------

||| Encode a symbol sequence as a copy-task data point.
||| Input:  sequence ++ blanks  (write phase)
||| Output: blanks ++ sequence  (read phase)
||| Symbol 0 is the blank token.
export
copyTaskPoint : {w : Nat} -> List (Fin w) -> RecurrentDataPoint w w Double
copyTaskPoint {w = Z} _ = MkRecurrentDataPoint [] []
copyTaskPoint {w = S k} sequence =
  let len = length sequence
      pad = Data.List.replicate len FZ
      inp = sequence ++ pad
      outp = pad ++ sequence
      xs = map (oneHotEncode {n = S k}) inp
      ys = map (oneHotEncode {n = S k}) outp
      toDouble : Vector (S k) Nat -> Vector (S k) Double
      toDouble = map (fromInteger . natToInteger)
  in MkRecurrentDataPoint (map toDouble xs) (map toDouble ys)

||| Copy task adapter: generates random copy sequences
export
copyTask : {w : Nat} -> SequenceTask w w
copyTask = MkSequenceTask "copy" $ \len => do
  symbols <- randomSymbols {w} len
  pure $ copyTaskPoint symbols


----------------------------------------------------------------------
-- Adapter: Associative Recall task
----------------------------------------------------------------------

||| Remove element at index from a list, returning the element and remaining list.
removeAt : Nat -> List a -> Maybe (a, List a)
removeAt _ [] = Nothing
removeAt Z (x :: xs) = Just (x, xs)
removeAt (S k) (x :: xs) = map (\(y, ys) => (y, x :: ys)) (removeAt k xs)

||| Fisher-Yates shuffle using selection sort (O(n^2) but n is small).
shuffleList : List a -> IO (List a)
shuffleList [] = pure []
shuffleList xs = do
  let n = length xs
  idx <- randomInt 0 (minus n 1)
  case removeAt idx xs of
    Nothing => pure xs
    Just (picked, rest) => do
      shuffled <- shuffleList rest
      pure (picked :: shuffled)

||| Non-blank symbols: [1, 2, ..., w-1] as Fin w values.
nonBlankSymbols : {w : Nat} -> List (Fin w)
nonBlankSymbols {w = Z} = []
nonBlankSymbols {w = S k} = go k
  where
    go : (m : Nat) -> List (Fin (S k))
    go Z = []
    go (S j) = restrict k (cast (S j)) :: go j

||| Encode an associative recall data point from key-value pairs and query order.
|||
||| Sequence structure for K pairs (4K+1 timesteps):
|||   Store:  k1 v1 k2 v2 ... kK vK   (2K steps)
|||   Delim:  blank                     (1 step)
|||   Query:  q1 blank q2 blank ... qK blank  (2K steps)
|||
||| Output is blank everywhere except on blank-input timesteps during
||| the query phase, where the correct value for the preceding query
||| key appears.
export
associativeRecallPoint : {w : Nat} -> List (Fin w, Fin w) -> List (Fin w)
                       -> RecurrentDataPoint w w Double
associativeRecallPoint {w = Z} _ _ = MkRecurrentDataPoint [] []
associativeRecallPoint {w = S k} pairs queryOrder =
  let blank = the (Fin (S k)) FZ
      -- Store phase: k1 v1 k2 v2 ...
      storeIn  = concatMap (\(key, val) => [key, val]) pairs
      storeOut = Data.List.replicate (length storeIn) blank
      -- Delimiter
      delimIn  = [blank]
      delimOut = [blank]
      -- Build lookup from keys to values
      lookup : Fin (S k) -> Fin (S k)
      lookup q = case find (\(key, _) => key == q) pairs of
                   Just (_, val) => val
                   Nothing => blank
      -- Query phase: q1 blank q2 blank ... qK blank
      queryIn  = concatMap (\q => the (List (Fin (S k))) [q, blank]) queryOrder
      queryOut = concatMap (\q => the (List (Fin (S k))) [blank, lookup q]) queryOrder
      -- Full sequences
      inp  = storeIn ++ delimIn ++ queryIn
      outp = storeOut ++ delimOut ++ queryOut
      xs = map (oneHotEncode {n = S k}) inp
      ys = map (oneHotEncode {n = S k}) outp
      toDouble : Vector (S k) Nat -> Vector (S k) Double
      toDouble = map (fromInteger . natToInteger)
  in MkRecurrentDataPoint (map toDouble xs) (map toDouble ys)

||| Associative recall task adapter: generates random key-value pairs
||| and queries them in shuffled order. The len parameter is the
||| number of pairs K (clamped to w-1 non-blank symbols).
export
associativeRecallTask : {w : Nat} -> SequenceTask w w
associativeRecallTask = MkSequenceTask "associative-recall" $ \len => do
  let symbols = nonBlankSymbols {w}
  let maxK = length symbols
  let k = min len maxK
  shuffledSyms <- shuffleList symbols
  let keys = take k shuffledSyms
  values <- randomSymbols {w} k
  let pairs = zip keys values
  queryKeys <- shuffleList keys
  pure $ associativeRecallPoint pairs queryKeys


----------------------------------------------------------------------
-- Binary vector generation helpers
----------------------------------------------------------------------

||| Generate a random binary vector (each element 0.0 or 1.0).
randomBinaryVector : {w : Nat} -> IO (Vector w Double)
randomBinaryVector {w = Z} = pure (VArray [])
randomBinaryVector {w = S k} = do
  bits <- traverse (\_ => do
    b <- randomRIO (the Int32 0, 1)
    pure (if b == 1 then 1.0 else 0.0)) (replicate (S k) ())
  pure (VArray (map SArray bits))

||| Make a vector with a 1.0 at the specified position and 0.0 elsewhere.
||| Position is from the end: 0 = last channel, 1 = second-to-last, etc.
makeDelimiter : {w : Nat} -> (channelFromEnd : Nat) -> Vector w Double
makeDelimiter {w} pos =
  let go : (n : Nat) -> Vect n (Array [] Double)
      go Z = []
      go (S k) = (if k == pos then SArray 1.0 else SArray 0.0) :: go k
  in VArray (go w)


----------------------------------------------------------------------
-- Binary copy task (PyTorch-aligned)
----------------------------------------------------------------------

||| Append one element to a vector: Vector w -> Vector (S w).
||| Channel ordering: original elements first, new element last.
appendElem : {w : Nat} -> Array [] Double -> Vector w Double -> Vector (S w) Double
appendElem {w = Z} e (VArray []) = VArray [e]
appendElem {w = S k} e (VArray (x :: xs)) =
  let (VArray rest) = appendElem e (VArray xs)
  in VArray (x :: rest)

||| Generate n random binary vectors of width w.
genBinaryRows : {w : Nat} -> (n : Nat) -> IO (List (Vector w Double))
genBinaryRows Z = pure []
genBinaryRows (S k) = do
  row <- randomBinaryVector {w}
  rest <- genBinaryRows k
  pure (row :: rest)

||| Generate a binary copy task data point.
||| Input: seq_len rows of [binary_data(w), 0] + 1 delimiter [0..0, 1]
||| Target: seq_len rows of binary_data(w)
||| Input width: w+1, output width: w
export
copyTaskBinary : {w : Nat} -> (seqLen : Nat) -> IO (TwoPhaseDataPoint (S w) w Double)
copyTaskBinary {w} seqLen = do
  dataRows <- genBinaryRows {w} seqLen
  let inputRows = map (appendElem (SArray 0.0)) dataRows
      delimiter = makeDelimiter {w = S w} 0
  pure $ MkTwoPhaseDataPoint (inputRows ++ [delimiter]) dataRows

||| Binary copy task as a two-phase sequence task.
export
copyTaskBinaryBatch : {w : Nat} -> (batchSize : Nat) -> (minLen, maxLen : Nat)
                   -> IO (List (TwoPhaseDataPoint (S w) w Double))
copyTaskBinaryBatch Z _ _ = pure []
copyTaskBinaryBatch (S k) minLen maxLen = do
  len <- randomInt minLen maxLen
  dp <- copyTaskBinary {w} len
  rest <- copyTaskBinaryBatch k minLen maxLen
  pure (dp :: rest)

||| Binary copy task batch as a Vect.
export
copyTaskBinaryBatchVect : {w : Nat} -> (n : Nat) -> (minLen, maxLen : Nat)
                       -> IO (Vect n (TwoPhaseDataPoint (S w) w Double))
copyTaskBinaryBatchVect Z _ _ = pure []
copyTaskBinaryBatchVect (S k) minLen maxLen = do
  len <- randomInt minLen maxLen
  dp <- copyTaskBinary {w} len
  rest <- copyTaskBinaryBatchVect k minLen maxLen
  pure (dp :: rest)


----------------------------------------------------------------------
-- Binary associative recall task (PyTorch-aligned)
----------------------------------------------------------------------

||| Pad a data vector with two zero channels: [data, 0, 0] -> Vector (S (S w)).
padData2 : {w : Nat} -> Vector w Double -> Vector (S (S w)) Double
padData2 = appendElem (SArray 0.0) . appendElem (SArray 0.0)

||| Generate a list of items, each item is seqLen binary vectors.
genItems : {w : Nat} -> (numItems, seqLen : Nat) -> IO (List (List (Vector w Double)))
genItems Z _ = pure []
genItems (S k) seqLen = do
  item <- genBinaryRows {w} seqLen
  rest <- genItems k seqLen
  pure (item :: rest)

||| Generate a binary recall task data point.
||| Each item is seqLen binary vectors of width w.
||| Input width: w+2 (data + item_delim + query_delim)
||| Output width: w
||| Structure: [item_delim item₁] ... [item_delim itemₙ] [query_delim query_item query_delim]
||| Target: item following the queried item (seqLen vectors of width w)
export
recallTaskBinary : {w : Nat} -> (numItems : Nat) -> (seqLen : Nat)
                -> IO (TwoPhaseDataPoint (S (S w)) w Double)
recallTaskBinary {w} numItems seqLen = do
  items <- genItems {w} numItems seqLen
  queryIdx <- randomInt 0 (minus numItems 2)
  let -- Item delimiter: [0...0, 1, 0] (channel w = 1)
      itemDelim : Vector (S (S w)) Double
      itemDelim = makeDelimiter {w = S (S w)} 1
      -- Query delimiter: [0...0, 0, 1] (channel w+1 = 1)
      queryDelim : Vector (S (S w)) Double
      queryDelim = makeDelimiter {w = S (S w)} 0
      -- Build encoding: [item_delim item₁_rows] ... [item_delim itemₙ_rows]
      encItems = concatMap (\item => itemDelim :: map (padData2 {w}) item) items
      -- Query item and target (list indexing)
      queryItem = fromMaybe [] (getAt queryIdx items)
      targetItem = fromMaybe [] (getAt (S queryIdx) items)
      -- Query phase: [query_delim] [query_item_rows] [query_delim]
      encQuery = queryDelim :: map (padData2 {w}) queryItem ++ [queryDelim]
  pure $ MkTwoPhaseDataPoint (encItems ++ encQuery) targetItem

||| Binary recall task batch as a Vect.
export
recallTaskBinaryBatchVect : {w : Nat} -> (n : Nat) -> (minItems, maxItems : Nat) ->
                           (seqLen : Nat) ->
                           IO (Vect n (TwoPhaseDataPoint (S (S w)) w Double))
recallTaskBinaryBatchVect Z _ _ _ = pure []
recallTaskBinaryBatchVect (S k) minItems maxItems seqLen = do
  numItems <- randomInt (max 2 minItems) (max 2 maxItems)
  dp <- recallTaskBinary {w} numItems seqLen
  rest <- recallTaskBinaryBatchVect k minItems maxItems seqLen
  pure (dp :: rest)


----------------------------------------------------------------------
-- Reversal task (for Transformer example)
----------------------------------------------------------------------

||| Generate one sequence reversal data point.
||| Full sequence: [t0..t_{n-1}, SEP, t_{n-1}..t0, EOS]
||| Input (teacher-forced): first seqLen tokens, one-hot encoded.
||| Target: tokens 1..seqLen, one-hot encoded.
export
reversalPoint : {inputDim, outputDim : Nat} ->
                (vocabSize : Nat) -> (inputLen : Nat) -> (seqLen : Nat) ->
                (sepToken : Nat) -> (eosToken : Nat) ->
                IO (DataPoint inputDim outputDim Double)
reversalPoint vocabSize inputLen seqLen sepToken eosToken = do
  tokens <- sequence (replicate inputLen (randomInt 0 (minus vocabSize 3)))
  let revToks = reverse tokens
      fullSeq = tokens ++ [sepToken] ++ revToks ++ [eosToken]
      inputToks = Data.List.take seqLen fullSeq
      targetToks = Data.List.take seqLen (drop 1 fullSeq)
      oneHot : Nat -> List Double
      oneHot tok = map (\i => if finToNat i == tok then 1.0 else 0.0)
                       (toList (Data.Vect.Fin.range {len=vocabSize}))
      inputFlat = concatMap oneHot inputToks
      targetFlat = concatMap oneHot targetToks
      toVect : (n : Nat) -> List Double -> Vect n (Scalar Double)
      toVect Z _ = []
      toVect (S k) [] = SArray 0.0 :: toVect k []
      toVect (S k) (x :: xs) = SArray x :: toVect k xs
  pure $ MkDataPoint (VArray (toVect inputDim inputFlat))
                     (VArray (toVect outputDim targetFlat))

||| Generate a batch of reversal data points as a Vect.
export
reversalBatchVect : {inputDim, outputDim : Nat} ->
                    (vocabSize : Nat) -> (inputLen : Nat) -> (seqLen : Nat) ->
                    (sepToken : Nat) -> (eosToken : Nat) ->
                    (n : Nat) -> IO (Vect n (DataPoint inputDim outputDim Double))
reversalBatchVect _ _ _ _ _ Z = pure []
reversalBatchVect vocabSize inputLen seqLen sepToken eosToken (S k) = do
  dp <- reversalPoint vocabSize inputLen seqLen sepToken eosToken
  rest <- reversalBatchVect vocabSize inputLen seqLen sepToken eosToken k
  pure (dp :: rest)


-- Pack a list of Nats into a C int buffer (for passing token indices to C)
packTokens : AnyPtr -> Int -> List Nat -> AnyPtr
packTokens buf _ [] = buf
packTokens buf off (tok :: rest) =
  let buf' = prim__setInt buf off (cast tok)
  in packTokens buf' (off + 1) rest

||| Pack token indices as doubles (for embedding-based transformer input)
packTokensDouble : AnyPtr -> Int -> List Nat -> AnyPtr
packTokensDouble buf _ [] = buf
packTokensDouble buf off (tok :: rest) =
  packTokensDouble (prim__setDouble buf off (cast {to=Double} (natToInteger tok))) (off + 1) rest

||| Generate one reversal data point as pre-allocated C tensors.
||| One-hot encoding done in C (single FFI call per sequence).
export
reversalTensorPoint : (i : Nat) -> (o : Nat) ->
                      (vocabSize : Nat) -> (inputLen : Nat) -> (seqLen : Nat) ->
                      (sepToken : Nat) -> (eosToken : Nat) ->
                      IO (TensorDataPoint i o)
reversalTensorPoint i o vocabSize inputLen seqLen sepToken eosToken = do
  tokens <- sequence (replicate inputLen (randomInt 0 (minus vocabSize 3)))
  let revToks = reverse tokens
      fullSeq = tokens ++ [sepToken] ++ revToks ++ [eosToken]
      inputToks = Data.List.take seqLen fullSeq
      targetToks = Data.List.take seqLen (drop 1 fullSeq)
      sI = cast {to=Int} seqLen
      vI = cast {to=Int} vocabSize
      -- Input: token indices as doubles [seqLen]. Routed through the
      -- dtype-aware creator so mlx-gpu (F32-only) doesn't get an F64
      -- buffer pushed onto a Metal stream (which mlx aborts with
      -- "float64 is not supported on the GPU"). dtypeTag {t=ExampleDType}
      -- pins the on-device storage to the build's dtype.
      inT = dtCreate1d {ex=ExampleExecutor} {t=ExampleDType} sI (packTokensDouble (prim__allocDoubles sI) 0 inputToks) 0 (deviceStreamTag {ex=ExampleExecutor})
      -- Target: one-hot [seqLen * vocabSize] for cross-entropy
      tgtIdxBuf = packTokens (prim__allocInts sI) 0 targetToks
  pure $ MkTensorDataPoint inT (primOneHot {ex=ExampleExecutor} tgtIdxBuf sI vI (dtypeTag {t=ExampleDType}))

||| Generate a batch of reversal tensor data points.
export
reversalTensorBatchVect : (i : Nat) -> (o : Nat) ->
                          (vocabSize : Nat) -> (inputLen : Nat) -> (seqLen : Nat) ->
                          (sepToken : Nat) -> (eosToken : Nat) ->
                          (n : Nat) -> IO (Vect n (TensorDataPoint i o))
reversalTensorBatchVect _ _ _ _ _ _ _ Z = pure []
reversalTensorBatchVect i o vocabSize inputLen seqLen sepToken eosToken (S k) = do
  dp <- reversalTensorPoint i o vocabSize inputLen seqLen sepToken eosToken
  rest <- reversalTensorBatchVect i o vocabSize inputLen seqLen sepToken eosToken k
  pure (dp :: rest)


||| Generate one sorting data point as pre-allocated C tensors.
||| Input: [t0..t_{n-1}, SEP, sorted_0..sorted_{n-1}, EOS]
export
sortingTensorPoint : (i : Nat) -> (o : Nat) ->
                     (vocabSize : Nat) -> (inputLen : Nat) -> (seqLen : Nat) ->
                     (sepToken : Nat) -> (eosToken : Nat) ->
                     IO (TensorDataPoint i o)
sortingTensorPoint i o vocabSize inputLen seqLen sepToken eosToken = do
  tokens <- sequence (replicate inputLen (randomInt 0 (minus vocabSize 3)))
  let sorted = Data.List.sort tokens
      fullSeq = tokens ++ [sepToken] ++ sorted ++ [eosToken]
      inputToks = Data.List.take seqLen fullSeq
      targetToks = Data.List.take seqLen (drop 1 fullSeq)
      sI = cast {to=Int} seqLen
      vI = cast {to=Int} vocabSize
      -- See reversalTensorPoint for the dtype rationale.
      inT = dtCreate1d {ex=ExampleExecutor} {t=ExampleDType} sI (packTokensDouble (prim__allocDoubles sI) 0 inputToks) 0 (deviceStreamTag {ex=ExampleExecutor})
      tgtIdxBuf = packTokens (prim__allocInts sI) 0 targetToks
  pure $ MkTensorDataPoint inT (primOneHot {ex=ExampleExecutor} tgtIdxBuf sI vI (dtypeTag {t=ExampleDType}))

||| Generate a batch of sorting tensor data points.
export
sortingTensorBatchVect : (i : Nat) -> (o : Nat) ->
                         (vocabSize : Nat) -> (inputLen : Nat) -> (seqLen : Nat) ->
                         (sepToken : Nat) -> (eosToken : Nat) ->
                         (n : Nat) -> IO (Vect n (TensorDataPoint i o))
sortingTensorBatchVect _ _ _ _ _ _ _ Z = pure []
sortingTensorBatchVect i o vocabSize inputLen seqLen sepToken eosToken (S k) = do
  dp <- sortingTensorPoint i o vocabSize inputLen seqLen sepToken eosToken
  rest <- sortingTensorBatchVect i o vocabSize inputLen seqLen sepToken eosToken k
  pure (dp :: rest)


----------------------------------------------------------------------
-- Pattern Sequence Data (for RNN/LSTM examples)
----------------------------------------------------------------------

generatePatternSeq : Nat -> (List Double, List Double)
generatePatternSeq len =
  let infinitePattern = cycle [0, 1, 0]
  in (take len infinitePattern, take len (drop 1 infinitePattern))

prepScalars : List Double -> List (Vector 1 Double)
prepScalars ns = map (flatten . SArray) ns

||| Generate repeating pattern data: input = pattern, output = next element.
||| Pattern is [0,1,0,0,1,0,...]. Used by RNN and LSTM examples.
export
patternData : (n : Nat) -> Vect n (RecurrentDataPoint 1 1 Double)
patternData n =
  let pairs = map (generatePatternSeq . (+3) . finToNat) (Data.Vect.Fin.range {len=n})
  in map (\(is, os) => MkRecurrentDataPoint (prepScalars is) (prepScalars os)) pairs
