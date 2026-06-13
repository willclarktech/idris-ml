module Test.Data

import Data.Vect
import Data.Fin
import Data.List
import Data.Maybe

import Test.Harness
import Dataset
import DataStream


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

export
tests : List (IO Bool)
tests = [ fromVectRoundTrip, fromIndexedSquares
        , streamNoShuffleInOrder, streamShufflePermutes, generateRepeats ]
