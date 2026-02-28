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
