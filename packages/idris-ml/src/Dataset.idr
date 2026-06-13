||| Indexed, random-access data source — the v1 data-loading primitive.
||| `Dataset` is one of PyTorch's three orthogonal joints (indexed
||| access); ordering lives in `Stream`'s `ShuffleSpec`, batching +
||| collation in `Stream`'s `batched`. A `Dataset` knows its size
||| and how to materialise the sample at each in-bounds index.
module Dataset

import Data.Vect
import Data.Fin

import Executor
import Tensor

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

||| Provided IDX (MNIST-family) dataset: pairs of (flat image, one-hot
||| label) tensors, both NoGrad (they're data, not parameters). Lifts the
||| idx C reader (the surviving engine) behind the Dataset surface —
||| `loadModel`-style "the module IS the adapter". inputDim = rows*cols.
||| The idx handle is opened once and captured for the dataset's lifetime
||| (no per-item free, matching the legacy loader).
export
idxDataset : {0 ex : Executor} -> {0 dt : Type} -> Backend ex dt =>
             (imgPath : String) -> (lblPath : String) ->
             (inputDim : Nat) -> (numClasses : Nat) ->
             Dataset (Tensor [inputDim] ex dt NoGrad, Tensor [numClasses] ex dt NoGrad)
idxDataset imgPath lblPath inputDim numClasses =
  let dsh   = prim__idxLoad imgPath lblPath
      count = cast {to=Nat} (cast {to=Integer} (prim__idxCount dsh))
  in fromIndexed count (\n =>
       let i       = cast {to=Int} (natToInteger n)
           flatImg = idxImage {ex} {t=dt} dsh i (cast {to=Int} inputDim)
           lbl     = prim__idxLabel dsh i
           lblBuf  = prim__setInt (prim__allocInts 1) 0 lbl
           tgt     = primOneHot {ex} lblBuf 1 (cast {to=Int} numClasses) (dtypeTag {t=dt})
       in pure (MkTensor flatImg Nothing, MkTensor tgt Nothing))
