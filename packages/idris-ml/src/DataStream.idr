||| Pull-based data streams — the v1 ordering + batching layer over
||| `Dataset`. A `DataStream` yields elements on demand; `stream` iterates a
||| `Dataset` in (shuffled) index order reshuffling each epoch, `batched`
||| (see below, separate commit) collates samples into batch tensors, and
||| `generate` wraps a raw IO action for synthetic / RL-rollout feeds.
||| The `fit` driver pulls `next` once per epoch.
module DataStream

import Data.IORef
import Data.Fin

import Dataset


----------------------------------------------------------------------
-- C FFI: index array with Fisher-Yates shuffle (the surviving engine
-- behind shuffled iteration). Declared PrimIO here — these mutate /
-- read mutable C state, so IO sequencing must force them (DataLoader's
-- legacy pure typing needs a manual read-to-force; PrimIO is cleaner).
----------------------------------------------------------------------

%foreign "C:create_index_array,libidrisml"
prim__createIndexArray : Int -> PrimIO AnyPtr

%foreign "C:shuffle_index_array,libidrisml"
prim__shuffleIndexArray : AnyPtr -> Int -> PrimIO AnyPtr

%foreign "C:index_array_get,libidrisml"
prim__indexArrayGet : AnyPtr -> Int -> PrimIO Int


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

||| Shuffle policy for `stream`. `Shuffle` uses the process-global
||| Fisher-Yates RNG (seed via `srand` upstream, as examples already do);
||| a per-stream seed would need a C signature change (deferred).
public export
data ShuffleSpec = NoShuffle | Shuffle

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
                      Shuffle   => ignore $ primIO (prim__shuffleIndexArray arr sizeI)
                      NoShuffle => pure ()
                    writeIORef posRef 0
                    pure 0
            else pure pos
  rawIdx <- primIO (prim__indexArrayGet arr pos')
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
  arr <- primIO (prim__createIndexArray sizeI)
  case spec of
    Shuffle   => ignore $ primIO (prim__shuffleIndexArray arr sizeI)
    NoShuffle => pure ()
  posRef <- newIORef (the Int 0)
  pure $ MkDataStream (pullSample spec ds arr sizeI posRef) (Just sz)
