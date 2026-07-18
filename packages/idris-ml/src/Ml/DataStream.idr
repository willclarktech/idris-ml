||| Pull-based data streams — the v1 ordering + batching layer over
||| `Dataset`. A `DataStream` yields elements on demand; `stream` iterates a
||| `Dataset` in (shuffled) index order reshuffling each epoch, `batched`
||| (see below, separate commit) collates samples into batch tensors, and
||| `generate` wraps a raw IO action for synthetic / RL-rollout feeds.
||| The `fit` driver pulls `next` once per epoch.
module Ml.DataStream

import Data.Fin
import Data.IORef
import Data.Nat
import Data.Vect

import Ml.Dataset
import Ml.Executor
import Ml.Tensor

----------------------------------------------------------------------
-- C FFI: seeded per-stream index array (Fisher-Yates over an embedded
-- xoshiro256++ state). Each stream's shuffle order is reproducible from
-- its seed and independent of the process-global rand() — so two streams
-- (e.g. train + val) shuffle independently, and a multi-seed campaign is
-- reproducible per stream. Declared PrimIO here — these mutate / read
-- mutable C state, so IO sequencing must force them.
----------------------------------------------------------------------

%foreign "C:create_seeded_index_array,libidrisml"
prim__createSeededIndexArray : Int -> Bits64 -> PrimIO AnyPtr

%foreign "C:seeded_index_array_shuffle,libidrisml"
prim__seededIndexArrayShuffle : AnyPtr -> PrimIO AnyPtr

%foreign "C:seeded_index_array_get,libidrisml"
prim__seededIndexArrayGet : AnyPtr -> Int -> PrimIO Int

----------------------------------------------------------------------
-- DataStream
----------------------------------------------------------------------

||| A pull-based data source. `next` yields the next element (a batch
||| once `batched` is applied). `epochLen` is `Just n` for finite
||| sources (from `stream`/`batched`) and `Nothing` for infinite ones
||| (from `generate`); the driver uses it as advisory metadata.
public export
record DataStream (a : Type) where
  constructor MkDataStream
  next     : IO a
  epochLen : Maybe Nat

public export
Functor DataStream where
  map f (MkDataStream nxt el) = MkDataStream (map f nxt) el

||| Shuffle policy for `stream`. `Shuffle seed` permutes via a per-stream
||| xoshiro256++ RNG seeded by `seed` (reproducible, independent of the
||| process-global rand() and of any other stream); `NoShuffle` iterates
||| in index order.
public export
data ShuffleSpec = NoShuffle | Shuffle Bits64

||| Wrap an IO action as an infinite stream — synthetic tasks (copy /
||| recall), RL rollout feeds, on-the-fly augmentation. This is exactly
||| the `dataSrc : IO dp` that the legacy runner consumed, now typed.
export
generate : {0 a : Type} -> IO a -> DataStream a
generate act = MkDataStream act Nothing

-- Pull one sample from the shuffled index array, reshuffling (or, for
-- NoShuffle, restarting in order) on epoch wrap. The index array only
-- ever holds values in [0, size), so the natToFin always succeeds; the
-- Nothing arm is an unreachable defensive crash (also covers the
-- nonsensical size-0 stream).
pullSample : {0 a : Type} -> ShuffleSpec -> Dataset a -> AnyPtr -> Int -> IORef Int -> IO a
pullSample spec (MkDataset sz itm) arr sizeI posRef = do
  pos <- readIORef posRef
  pos' <- if pos >= sizeI
            then do case spec of
                      Shuffle _ => ignore $ primIO (prim__seededIndexArrayShuffle arr)
                      NoShuffle => pure ()
                    writeIORef posRef 0
                    pure 0
            else pure pos
  rawIdx <- primIO (prim__seededIndexArrayGet arr pos')
  let natIdx = cast {to=Nat} (cast {to=Integer} rawIdx)
  writeIORef posRef (pos' + 1)
  case natToFin natIdx sz of
    Just fin => itm fin
    Nothing  => assert_total $ idris_crash "DataStream.stream: index out of range"

||| Iterate a `Dataset` as a stream of single samples in (shuffled)
||| index order, reshuffling at each epoch wrap (`Shuffle`) or restarting
||| in order (`NoShuffle`). One index array per `stream` call, reshuffled
||| in place — matches the legacy `mkIndexedLoader` lifecycle (incl. the
||| accepted one-allocation-per-stream, never freed). Compose with
||| `batched` for batch tensors.
export
stream : {0 a : Type} -> ShuffleSpec -> Dataset a -> IO (DataStream a)
stream spec ds@(MkDataset sz _) = do
  let sizeI : Int := cast sz
      seed  : Bits64 := case spec of
                          Shuffle s => s
                          NoShuffle => 0
  arr <- primIO (prim__createSeededIndexArray sizeI seed)
  case spec of
    Shuffle _ => ignore $ primIO (prim__seededIndexArrayShuffle arr)
    NoShuffle => pure ()
  posRef <- newIORef (the Int 0)
  pure $ MkDataStream (pullSample spec ds arr sizeI posRef) (Just sz)

----------------------------------------------------------------------
-- Batching / collation (C-side, no host readback)
----------------------------------------------------------------------

-- Pull n elements from an IO action into a Vect.
pullN : (n : Nat) -> IO a -> IO (Vect n a)
pullN Z     _   = pure []
pullN (S k) act = do
  x <- act
  rest <- pullN k act
  pure (x :: rest)

-- ceil(n / b); the (advisory) batched epoch length.
batchEpochs : Nat -> Nat -> Nat
batchEpochs _ Z     = 0
batchEpochs n (S k) = divNatNZ (n + k) (S k) ItIsSucc

-- Sequentially store `count` raw handles into a C `TensorHandle*` buffer,
-- threading the array pointer through each `prim__ptrArraySet` so the
-- pure-typed FFI can't be reordered/elided (the `_return` shim threads
-- the buffer for exactly this reason). One `ioRerun` per store keeps the
-- stores ordered under IO sequencing.
fillHandleArray : AnyPtr -> Int -> List AnyPtr -> IO AnyPtr
fillHandleArray a _ []        = pure a
fillHandleArray a i (p :: ps) = do
  a' <- ioRerun (\_ => prim__ptrArraySet a i p)
  fillHandleArray a' (i + 1) ps

-- Stack b single-[n] tensor handles into one [b, n] tensor in a single
-- collation FFI: pack the wrapped handles into a `TensorHandle*` buffer,
-- then `primBatch` (tensor_batch: B × [...] -> [B, ...] along a new
-- leading axis) does the stack C-side with no host readback. This
-- replaces the former b-1 pairwise `primCat2` chain + reshape; on
-- large-batch mlx that collapses b-1 cat kernel launches into one stack.
-- The collation tensor is data (NoGrad in practice), and tape's
-- tensor_batch is a no-grad memcpy — matching the data-only contract.
-- The handle buffer is consumed eagerly by tensor_batch (read into a
-- vector / memcpy'd before return), so it is freed immediately after.
collate : {0 ex : Executor} -> {0 dt : DType} -> {0 g : GradMode} -> {b, n : Nat} ->
          UserExecutorLinear ex => Vect b (Tensor [n] ex dt g) -> IO (Tensor [b, n] ex dt g)
collate {b} samples = do
  arr0 <- ioRerun (\_ => prim__ptrArrayAlloc (cast b))
  arr  <- fillHandleArray arr0 0 (toList (map tensorPtr samples))
  raw  <- ioRerun (\_ => primBatch {ex} arr (cast b))
  primIO (prim__ptrArrayFree arr)
  pure (MkTensor raw Nothing)

||| Collate a stream of single [i] tensors into batches of [b, i].
||| Matches the north-star shape; for supervised (input, target) pairs
||| use `batched`.
export
batched1 : {0 ex : Executor} -> {0 dt : DType} -> {0 g : GradMode} -> {b, i : Nat} ->
           UserExecutorLinear ex =>
           DataStream (Tensor [i] ex dt g) -> DataStream (Tensor [b, i] ex dt g)
batched1 {b} (MkDataStream nxt el) =
  MkDataStream (do xs <- pullN b nxt; collate {b} xs) (map (\m => batchEpochs m b) el)

||| Collate a stream of (input, target) tensor pairs into batch pairs
||| ([b, i], [b, o]) — the supervised default the `fit` Step consumes.
export
batched : {0 ex : Executor} -> {0 dt : DType} -> {0 g : GradMode} -> {b, i, o : Nat} ->
          UserExecutorLinear ex =>
          DataStream (Tensor [i] ex dt g, Tensor [o] ex dt g) ->
          DataStream (Tensor [b, i] ex dt g, Tensor [b, o] ex dt g)
batched {b} (MkDataStream nxt el) =
  MkDataStream
    (do pairs <- pullN b nxt
        xb <- collate {b} (map fst pairs)
        yb <- collate {b} (map snd pairs)
        pure (xb, yb))
    (map (\m => batchEpochs m b) el)
