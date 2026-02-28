module Generate

import Data.Fin
import Data.List
import Data.Vect
import System.Random

import DataPoint
import Math
import Tensor


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
