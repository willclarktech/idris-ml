-- | DataLoader: Reusable batched data pipeline with shuffle/batch/repeat.
-- |
-- | Two loaders:
-- |   mkGeneratorLoader — for synthetic data (fresh random batch each call)
-- |   mkIndexedLoader   — for file-backed datasets (shuffled epoch iteration)

module DataLoader

import Data.IORef
import Data.Vect
import Compat.Random


----------------------------------------------------------------------
-- C FFI: index array with Fisher-Yates shuffle
----------------------------------------------------------------------

%foreign "C:create_index_array,libidrisml"
prim__createIndexArray : Int -> AnyPtr

%foreign "C:shuffle_index_array,libidrisml"
prim__shuffleIndexArray : AnyPtr -> Int -> AnyPtr

%foreign "C:index_array_get,libidrisml"
prim__indexArrayGet : AnyPtr -> Int -> Int


----------------------------------------------------------------------
-- Generator loader (synthetic data)
----------------------------------------------------------------------

||| Build a Vect of n data points by calling gen n times.
||| Replaces the ad-hoc recursive IO pattern used by most examples.
export
mkGeneratorLoader : {n : Nat} -> IO dp -> IO (Vect n dp)
mkGeneratorLoader {n = Z} _ = pure []
mkGeneratorLoader {n = S k} gen = do
  x <- gen
  rest <- mkGeneratorLoader gen
  pure (x :: rest)


----------------------------------------------------------------------
-- Indexed loader (file-backed datasets)
----------------------------------------------------------------------

||| Fetch n items from a C index array, calling getItem for each.
fetchN : AnyPtr -> Int -> (n : Nat) -> (Nat -> IO dp) -> IO (Vect n dp)
fetchN _ _ Z _ = pure []
fetchN arr pos (S k) get = do
  let idx = prim__indexArrayGet arr pos
  item <- get (cast {to=Nat} (cast {to=Integer} idx))
  rest <- fetchN arr (pos + 1) k get
  pure (item :: rest)

||| Create a shuffled-index loader for file-backed datasets.
|||
||| Allocates an index array [0..datasetSize-1], shuffles it, and returns
||| an IO action that yields the next batch of batchSize items on each call.
||| When the dataset is exhausted, reshuffles and starts a new pass.
|||
||| @datasetSize Number of items in the dataset
||| @getItem     Callback: index -> IO dataPoint
export
mkIndexedLoader :
  {batchSize : Nat} ->
  (datasetSize : Nat) ->
  (Nat -> IO dp) ->
  IO (IO (Vect batchSize dp))
mkIndexedLoader {batchSize} dsSize getItem = do
  let sizeI = cast {to=Int} (natToInteger dsSize)
      bsI = cast {to=Int} (natToInteger batchSize)
  -- Allocate and shuffle index array
  let arr = prim__createIndexArray sizeI
      arr' = prim__shuffleIndexArray arr sizeI
  posRef <- newIORef (the Int 0)
  -- Return the batch-fetching IO action
  pure $ do
    pos <- readIORef posRef
    -- Reshuffle if not enough items remain
    pos' <- if pos + bsI > sizeI
      then do
        let reshuffled = prim__shuffleIndexArray arr' sizeI
        -- Force evaluation by reading element 0 (FFI side-effect threading)
        let _ = prim__indexArrayGet reshuffled 0
        writeIORef posRef 0
        pure 0
      else pure pos
    batch <- fetchN arr' pos' batchSize getItem
    writeIORef posRef (pos' + bsI)
    pure batch
