module Layer.Core

import Data.Vect
import System
import System.File

import Device
import Tensor


----------------------------------------------------------------------
-- Path C P3-1 spike: rank-aware layer interface
----------------------------------------------------------------------
--
-- Parallel to `Layer.Core.LayerLike`, but operates on rank-aware
-- `Tensor` directly. No Vect-of-Vect packing; no scalar boundaries
-- between layers. Spike-only — only the methods needed for a
-- single Linear + chained forward pass are present. Full
-- migration widens this surface (toDouble, debug, batched, etc.).

public export
interface LayerLike (l : Nat -> Nat -> (0 _ : Device) -> Type) where
  ||| Array-level forward: `Tensor [i] d -> Tensor [o] d`.
  applyVar : {0 d : Device} -> {i, o : Nat} ->
              l i o d -> Tensor [i] d WithGrad -> (l i o d, Tensor [o] d WithGrad)

  ||| Auto-naming prefix (e.g. "llv2" for Linear).
  layerPrefix : {0 d : Device} -> {i, o : Nat} -> l i o d -> String
  layerPrefix _ = ""

  ||| Reset per-sequence state (recurrent layers override; default = id).
  ||| Used by `resetNetwork` between sequences in recurrent training.
  resetState : {0 d : Device} -> {i, o : Nat} -> l i o d -> l i o d
  resetState = id

  ||| Batched tensor-level forward: `Tensor [b, i] d -> Tensor [b, o] d`.
  ||| Default crashes — layers that participate in batched training
  ||| (Linear, Activation, Dropout) MUST override. Stateful layers
  ||| (LSTM/RNN/GRU/NTM/DNC) keep the default; batched-cell semantics
  ||| are not supported in this surface (use sequence-level batching
  ||| at the example level instead).
  applyVarBatch : {0 d : Device} -> {i, o : Nat} -> {b : Nat} ->
                   l i o d -> Tensor [b, i] d WithGrad -> (l i o d, Tensor [b, o] d WithGrad)
  applyVarBatch _ _ =
    idris_crash "applyVarBatch: layer does not support batched forward"


----------------------------------------------------------------------
-- AnyLayer (existential wrapper)
----------------------------------------------------------------------

public export
data AnyLayer : Nat -> Nat -> (0 _ : Device) -> Type where
  MkAnyLayer : (l : Nat -> Nat -> (0 _ : Device) -> Type) -> LayerLike l =>
                 l i o d -> AnyLayer i o d

export
applyVarAny : {0 d : Device} -> {i, o : Nat} ->
               AnyLayer i o d -> Tensor [i] d WithGrad -> (AnyLayer i o d, Tensor [o] d WithGrad)
applyVarAny (MkAnyLayer l @{dict} layer) input =
  case applyVar @{dict} layer input of
    (layer', out) => (MkAnyLayer l @{dict} layer', out)

export
applyVarBatchAny : {0 d : Device} -> {i, o : Nat} -> {b : Nat} ->
                    AnyLayer i o d -> Tensor [b, i] d WithGrad ->
                    (AnyLayer i o d, Tensor [b, o] d WithGrad)
applyVarBatchAny (MkAnyLayer l @{dict} layer) input =
  case applyVarBatch @{dict} layer input of
    (layer', out) => (MkAnyLayer l @{dict} layer', out)


----------------------------------------------------------------------
-- Network
----------------------------------------------------------------------

public export
data Network : (i : Nat) -> (hs : List Nat) -> (o : Nat) -> (0 _ : Device) -> (0 _ : GradMode) -> Type where
  OutputLayer : AnyLayer i o d -> Network i [] o d WithGrad
  (~~>) : AnyLayer i h d -> Network h hs o d WithGrad -> Network i (h :: hs) o d WithGrad

export infixr 5 ~~>

||| Array-level forward through a Network.
export
forwardVar : {0 d : Device} -> {i, o : Nat} -> {hs : List Nat} ->
              Network i hs o d WithGrad -> Tensor [i] d WithGrad -> (Network i hs o d WithGrad, Tensor [o] d WithGrad)
forwardVar (OutputLayer l) input =
  case applyVarAny l input of
    (l', out) => (OutputLayer l', out)
forwardVar {hs = h :: _} (l ~~> rest) input =
  case applyVarAny l input of
    (l', mid) =>
      case forwardVar rest mid of
        (rest', out) => (l' ~~> rest', out)

||| Mark a Network as no-grad at the type level. Pure cast — the
||| `g` parameter is 0-quantity, so the runtime value is byte-identical;
||| only the static promise changes. After `freezeNetwork`, the network
||| can't be passed to `runBackward` / `nativeTrainStep` (once Phase 4
||| lands).
|||
||| Asymmetric with `weakenGrad` (which DOES flip the C-side
||| `requires_grad` flag): `Network` is opaque from Idris (the
||| `LayerLike` interface doesn't expose individual params) and the
||| C-side param registry is process-global. A per-network runtime
||| freeze would either freeze every network in the process or require
||| a structural change to `LayerLike` / the registry. Both are deferred.
||| For runtime tape gating, combine `freezeNetwork` with a `withNoGrad`
||| block around the inference path.
export
freezeNetwork : Network i hs o d g -> Network i hs o d NoGrad
freezeNetwork = believe_me

||| Reset per-sequence state on every layer in the network. Use
||| between training sequences for recurrent layers (Lstm, Rnn,
||| Gru). Stateless layers' default `resetState` is identity.
export
resetNetwork : {0 d : Device} -> {i, o : Nat} -> {hs : List Nat} ->
                 Network i hs o d WithGrad -> Network i hs o d WithGrad
resetNetwork (OutputLayer (MkAnyLayer l @{dict} layer)) =
  OutputLayer (MkAnyLayer l @{dict} (resetState @{dict} layer))
resetNetwork ((MkAnyLayer l @{dict} layer) ~~> rest) =
  MkAnyLayer l @{dict} (resetState @{dict} layer) ~~> resetNetwork rest

||| Batched tensor-level forward through a Network: each layer's
||| `applyVarBatch` runs on the threaded `[b, _]` tensor. Linear /
||| Activation / Dropout override; other layers crash via the
||| interface default.
export
forwardVarBatch : {0 d : Device} -> {i, o : Nat} -> {b : Nat} ->
                   {hs : List Nat} ->
                   Network i hs o d WithGrad -> Tensor [b, i] d WithGrad ->
                   (Network i hs o d WithGrad, Tensor [b, o] d WithGrad)
forwardVarBatch (OutputLayer l) input =
  case applyVarBatchAny l input of
    (l', out) => (OutputLayer l', out)
forwardVarBatch {hs = h :: _} (l ~~> rest) input =
  case applyVarBatchAny l input of
    (l', mid) =>
      case forwardVarBatch rest mid of
        (rest', out) => (l' ~~> rest', out)


----------------------------------------------------------------------
-- Lightweight forward tracer
----------------------------------------------------------------------

||| Walks the Network like `forwardVar`, printing each layer's output
||| `min` / `max` / `mean` to stderr as it goes. Useful for "where
||| did the NaN come from?" debugging without committing to a
||| structured DebugEntry surface.
|||
||| The autograd graph is preserved — this just adds side-effecting
||| reads between layer applications. The returned Tensor is the same
||| one a plain `forwardVar` would produce. The min / max / mean
||| reductions create non-grad-tracking tape entries that get released
||| at the next `tape_reset`; they don't affect training numerics.
|||
||| Usage: swap `forwardVar` for `forwardVarTraced "epoch5"` at any
||| call site to get per-layer trace lines. Output goes to stderr so
||| training stdout stays clean. Lines look like:
|||
|||     epoch5:0 min=-0.123 max=0.456 mean=0.012
|||     epoch5:1 min=-0.234 max=0.567 mean=0.099
|||     epoch5:out min=-0.300 max=0.700 mean=0.150  [NaN]
export
forwardVarTraced : {0 d : Device} -> {i, o : Nat} -> {hs : List Nat} ->
                   (label : String) ->
                   Network i hs o d WithGrad -> Tensor [i] d WithGrad ->
                   IO (Network i hs o d WithGrad, Tensor [o] d WithGrad)
forwardVarTraced label net input = go 0 net input
  where
    -- Take the raw AnyPtr so we don't have to thread `d` through
    -- the implicit-binding nest. The reductions are non-grad anyway.
    summarize : (idxLabel : String) -> AnyPtr -> IO ()
    summarize idxLabel ptr = do
      let mn = prim__item (prim__tensorMin ptr)
          mx = prim__item (prim__tensorMax ptr)
          me = prim__item (prim__mean ptr)
          isNaN : Double -> Bool
          isNaN x = x /= x
          tag = if isNaN mn || isNaN mx || isNaN me then "  [NaN]" else ""
      ignore $ fPutStrLn stderr $
        label ++ ":" ++ idxLabel
          ++ " min=" ++ show mn
          ++ " max=" ++ show mx
          ++ " mean=" ++ show me ++ tag

    go : {0 d : Device} -> {i, o : Nat} -> {hs : List Nat} ->
         Nat ->
         Network i hs o d WithGrad -> Tensor [i] d WithGrad ->
         IO (Network i hs o d WithGrad, Tensor [o] d WithGrad)
    go idx (OutputLayer l) inp =
      case applyVarAny l inp of
        (l', out) => do
          summarize (show idx ++ "(out)") out.tensorPtr
          pure (OutputLayer l', out)
    go {hs = h :: _} idx (l ~~> rest) inp =
      case applyVarAny l inp of
        (l', mid) => do
          summarize (show idx) mid.tensorPtr
          (rest', out) <- go (idx + 1) rest mid
          pure (l' ~~> rest', out)
