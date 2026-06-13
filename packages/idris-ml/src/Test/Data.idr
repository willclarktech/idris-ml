module Test.Data

import Data.Vect
import Data.Fin
import Data.List
import Data.Maybe

import Test.Harness
import Dataset
import DataStream
import Executor
import Tensor
import Test.Config


-- Access a dataset at a raw Nat index via natToFin (no compile-time
-- bound needed; the Fin is built at the dataset's runtime size).
getAt : Dataset a -> a -> Nat -> IO a
getAt d dflt k = case natToFin k d.size of
                   Just f  => d.item f
                   Nothing => pure dflt

fromVectRoundTrip : IO Bool
fromVectRoundTrip = do
  let d = fromVect [10.0, 20.0, 30.0]
  a <- getAt d (-1.0) 0
  b <- getAt d (-1.0) 1
  c <- getAt d (-1.0) 2
  check ("fromVect round-trip (size=" ++ show d.size
         ++ ", [" ++ show a ++ "," ++ show b ++ "," ++ show c ++ "])")
        (d.size == 3 && a == 10.0 && b == 20.0 && c == 30.0)

fromIndexedSquares : IO Bool
fromIndexedSquares = do
  let d = fromIndexed 5 (\n => pure (n * n))
  v0 <- getAt d 999 0
  v3 <- getAt d 999 3
  v4 <- getAt d 999 4
  check ("fromIndexed squares (size=" ++ show d.size
         ++ ", [0]=" ++ show v0 ++ " [3]=" ++ show v3 ++ " [4]=" ++ show v4 ++ ")")
        (d.size == 5 && v0 == 0 && v3 == 9 && v4 == 16)

-- Pull n elements from a stream into a list.
pullList : Nat -> DataStream a -> IO (List a)
pullList Z     _ = pure []
pullList (S k) s = do
  x <- s.next
  rest <- pullList k s
  pure (x :: rest)

streamNoShuffleInOrder : IO Bool
streamNoShuffleInOrder = do
  s <- stream NoShuffle (fromVect (the (Vect 5 Nat) [0, 1, 2, 3, 4]))
  xs <- pullList 5 s   -- first pass
  ys <- pullList 5 s   -- second pass: cursor wraps, no reshuffle → in order
  check ("stream NoShuffle in-order + epoch wrap (" ++ show xs ++ ", " ++ show ys ++ ")")
        (xs == [0, 1, 2, 3, 4] && ys == [0, 1, 2, 3, 4])

streamShufflePermutes : IO Bool
streamShufflePermutes = do
  s <- stream Shuffle (fromVect (the (Vect 5 Nat) [0, 1, 2, 3, 4]))
  xs <- pullList 5 s
  check ("stream Shuffle is a permutation (sorted " ++ show (sort xs) ++ ")")
        (sort xs == [0, 1, 2, 3, 4])

generateRepeats : IO Bool
generateRepeats = do
  let s = generate (pure (the Nat 7))
  xs <- pullList 3 s
  check ("generate repeats + epochLen Nothing (" ++ show xs ++ ")")
        (xs == [7, 7, 7] && isNothing s.epochLen)

-- batched collation value-check: 3 (input[2], target[1]) pairs with
-- known values, NoShuffle so order is preserved, b=3 → one ([3,2],[3,1])
-- batch. Read back every cell via primItem2d and compare to the source.
vec2 : Double -> Double -> IO (Tensor [2] TestExecutor TestDType NoGrad)
vec2 a b = tensor {ex=TestExecutor} {dt=TestDType} {dims=[2]} (FromVect [a, b])

vec1 : Double -> IO (Tensor [1] TestExecutor TestDType NoGrad)
vec1 a = tensor {ex=TestExecutor} {dt=TestDType} {dims=[1]} (FromVect [a])

batchedCollates : IO Bool
batchedCollates = do
  i0 <- vec2 1.0 2.0; o0 <- vec1 10.0
  i1 <- vec2 3.0 4.0; o1 <- vec1 20.0
  i2 <- vec2 5.0 6.0; o2 <- vec1 30.0
  let ds = fromVect [(i0, o0), (i1, o1), (i2, o2)]
  s <- stream NoShuffle ds
  let bs = batched {b = 3} {i = 2} {o = 1} s
  (inB, tgtB) <- bs.next
  let inOk = all (\(r, c, v) => primItem2d {ex=TestExecutor} inB.tensorPtr r c == v)
                 (the (List (Int, Int, Double))
                   [ (0,0,1.0),(0,1,2.0),(1,0,3.0),(1,1,4.0),(2,0,5.0),(2,1,6.0) ])
      tgtOk = all (\(r, v) => primItem2d {ex=TestExecutor} tgtB.tensorPtr r 0 == v)
                  (the (List (Int, Double)) [ (0,10.0),(1,20.0),(2,30.0) ])
  check "batched collates ([3,2] inputs + [3,1] targets, value-checked)" (inOk && tgtOk)

batched1Collates : IO Bool
batched1Collates = do
  i0 <- vec2 7.0 8.0
  i1 <- vec2 9.0 1.0
  let ds = fromVect [i0, i1]
  s <- stream NoShuffle ds
  let bs = batched1 {b = 2} {i = 2} s
  inB <- bs.next
  let ok = all (\(r, c, v) => primItem2d {ex=TestExecutor} inB.tensorPtr r c == v)
               (the (List (Int, Int, Double)) [ (0,0,7.0),(0,1,8.0),(1,0,9.0),(1,1,1.0) ])
  check "batched1 collates single [2,2] (value-checked)" ok

export
tests : List (IO Bool)
tests = [ fromVectRoundTrip, fromIndexedSquares
        , streamNoShuffleInOrder, streamShufflePermutes, generateRepeats
        , batchedCollates, batched1Collates ]
