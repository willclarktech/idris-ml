||| Indexed, random-access data source — the v1 data-loading primitive.
||| `Dataset` is one of PyTorch's three orthogonal joints (indexed
||| access); ordering lives in `Stream`'s `ShuffleSpec`, batching +
||| collation in `Stream`'s `batched`. A `Dataset` knows its size
||| and how to materialise the sample at each in-bounds index.
module Dataset

import Data.Vect
import Data.Fin

||| Indexed data source. `item` takes a `Fin size`, so out-of-bounds
||| access is unrepresentable — no runtime bounds check, no partiality.
public export
record Dataset (sample : Type) where
  constructor MkDataset
  size : Nat
  item : Fin size -> IO sample

||| Dataset backed by an in-memory `Vect`. `index` is total.
export
fromVect : {n : Nat} -> Vect n sample -> Dataset sample
fromVect {n} xs = MkDataset n (\i => pure (index i xs))

||| Dataset backed by a file/IO callback. The callback receives the raw
||| `Nat` index (already bounds-guaranteed by the `Fin`); use this to
||| wrap an idx reader, a memory-mapped file, etc.
export
fromIndexed : (size : Nat) -> (Nat -> IO sample) -> Dataset sample
fromIndexed size get = MkDataset size (\i => get (finToNat i))
