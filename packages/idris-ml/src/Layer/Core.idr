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
interface LayerLike (l : Nat -> Nat -> (0 _ : Device) -> (0 _ : DType) -> (0 _ : GradMode) -> Type) where
  ||| Array-level forward: `Tensor [i] d dt g-> Tensor [o] d dt g`, IO-typed
  ||| because the forward pass triggers FFI side effects (tape append,
  ||| tensor allocation). IO sequencing controls when those fire —
  ||| critical for `withNoGrad` to correctly bracket eval-phase work.
  ||| Polymorphic in `g` so forwarding a `NoGrad` input through a
  ||| frozen layer yields a `NoGrad` output naturally.
  applyVar : {0 d : Device} -> UserDeviceTape d => {0 g : GradMode} -> {i, o : Nat} ->
              l i o d dt g -> Tensor [i] d dt g -> IO (l i o d dt g, Tensor [o] d dt g)

  ||| Auto-naming prefix (e.g. "llv2" for Linear).
  layerPrefix : {0 d : Device} -> {0 g : GradMode} -> {i, o : Nat} -> l i o d dt g -> String
  layerPrefix _ = ""

  ||| Reset per-sequence state (recurrent layers override; default = id).
  ||| Used by `resetNetwork` between sequences in recurrent training.
  resetState : {0 d : Device} -> {0 g : GradMode} -> {i, o : Nat} -> l i o d dt g -> l i o d dt g
  resetState = id

  ||| Batched tensor-level forward: `Tensor [b, i] d dt g-> Tensor [b, o] d dt g`.
  ||| Default crashes — layers that participate in batched training
  ||| (Linear, Activation, Dropout) MUST override. Stateful layers
  ||| (LSTM/RNN/GRU/NTM/DNC) keep the default; batched-cell semantics
  ||| are not supported in this surface (use sequence-level batching
  ||| at the example level instead).
  applyVarBatch : {0 d : Device} -> UserDeviceTape d => {0 g : GradMode} -> {i, o : Nat} -> {b : Nat} ->
                   l i o d dt g -> Tensor [b, i] d dt g -> IO (l i o d dt g, Tensor [b, o] d dt g)
  applyVarBatch _ _ =
    idris_crash "applyVarBatch: layer does not support batched forward"

  ||| Freeze: flip C-side `requires_grad=false` on every parameter
  ||| tensor in the layer state. Linear in input — consumes the old
  ||| reference so the caller can't keep using the WithGrad-typed
  ||| value after the C-side flags have been mutated. Returns the
  ||| layer retyped as `NoGrad`. Optimizer steps won't update frozen
  ||| params (their gradients don't accumulate on rg=false leaves).
  freezeLayer : {0 d : Device} -> {0 g : GradMode} -> {i, o : Nat} ->
                (1 _ : l i o d dt g) -> IO (l i o d dt NoGrad)

  ||| Inverse of `freezeLayer`. Sets `requires_grad=true` on every
  ||| parameter and retypes the layer as `WithGrad`. The result is
  ||| trainable again. Linear in input.
  unfreezeLayer : {0 d : Device} -> {i, o : Nat} ->
                  (1 _ : l i o d dt NoGrad) -> IO (l i o d dt WithGrad)


----------------------------------------------------------------------
-- AnyLayer (existential wrapper)
----------------------------------------------------------------------

public export
data AnyLayer : Nat -> Nat -> (0 _ : Device) -> (0 _ : DType) -> (0 _ : GradMode) -> Type where
  MkAnyLayer : (l : Nat -> Nat -> (0 _ : Device) -> (0 _ : DType) -> (0 _ : GradMode) -> Type) -> LayerLike l =>
                 l i o d dt g -> AnyLayer i o d dt g

export
applyVarAny : {0 d : Device} -> UserDeviceTape d => {0 g : GradMode} -> {i, o : Nat} ->
               AnyLayer i o d dt g -> Tensor [i] d dt g -> IO (AnyLayer i o d dt g, Tensor [o] d dt g)
applyVarAny (MkAnyLayer l @{dict} layer) input = do
  (layer', out) <- applyVar @{dict} layer input
  pure (MkAnyLayer l @{dict} layer', out)

export
applyVarBatchAny : {0 d : Device} -> UserDeviceTape d => {0 g : GradMode} -> {i, o : Nat} -> {b : Nat} ->
                    AnyLayer i o d dt g -> Tensor [b, i] d dt g ->
                    IO (AnyLayer i o d dt g, Tensor [b, o] d dt g)
applyVarBatchAny (MkAnyLayer l @{dict} layer) input = do
  (layer', out) <- applyVarBatch @{dict} layer input
  pure (MkAnyLayer l @{dict} layer', out)

export
freezeAnyLayer : {0 d : Device} -> {0 g : GradMode} -> {i, o : Nat} ->
                  (1 _ : AnyLayer i o d dt g) -> IO (AnyLayer i o d dt NoGrad)
freezeAnyLayer (MkAnyLayer l @{dict} layer) = do
  layer' <- freezeLayer @{dict} layer
  pure (MkAnyLayer l @{dict} layer')

export
unfreezeAnyLayer : {0 d : Device} -> {i, o : Nat} ->
                    (1 _ : AnyLayer i o d dt NoGrad) -> IO (AnyLayer i o d dt WithGrad)
unfreezeAnyLayer (MkAnyLayer l @{dict} layer) = do
  layer' <- unfreezeLayer @{dict} layer
  pure (MkAnyLayer l @{dict} layer')


----------------------------------------------------------------------
-- Network
----------------------------------------------------------------------

public export
data Network : (i : Nat) -> (hs : List Nat) -> (o : Nat) -> (0 _ : Device) -> (0 _ : DType) -> (0 _ : GradMode) -> Type where
  OutputLayer : AnyLayer i o d dt g -> Network i [] o d dt g
  (~~>) : AnyLayer i h d dt g -> Network h hs o d dt g -> Network i (h :: hs) o d dt g

export infixr 5 ~~>

||| Array-level forward through a Network. Polymorphic in `g`:
||| forwarding a `NoGrad` input through a frozen network yields a
||| `NoGrad` output naturally.
export
forwardVar : {0 d : Device} -> UserDeviceTape d => {0 g : GradMode} -> {i, o : Nat} -> {hs : List Nat} ->
              Network i hs o d dt g -> Tensor [i] d dt g -> IO (Network i hs o d dt g, Tensor [o] d dt g)
forwardVar (OutputLayer l) input = do
  (l', out) <- applyVarAny l input
  pure (OutputLayer l', out)
forwardVar {hs = h :: _} (l ~~> rest) input = do
  (l', mid) <- applyVarAny l input
  (rest', out) <- forwardVar rest mid
  pure (l' ~~> rest', out)

||| Freeze a Network: walks each layer and calls `freezeLayer` on it,
||| which flips C-side `requires_grad=false` on every parameter tensor.
||| Linear in input — the original WithGrad-typed reference is consumed
||| so the user can't accidentally train through it (the C-side flags
||| have been mutated under the original Idris variable).
||| Returns the network retyped as `NoGrad`.
|||
||| Frozen networks remain usable with `forwardVar` (now polymorphic in
||| `g`) — output adopts `NoGrad` and the type system prevents feeding
||| it back to `runBackward` / `nativeTrainStep`.
export
freezeNetwork : {0 d : Device} -> {0 g : GradMode} -> {i, o : Nat} -> {hs : List Nat} ->
                 (1 _ : Network i hs o d dt g) -> IO (Network i hs o d dt NoGrad)
freezeNetwork (OutputLayer l) = do
  l' <- freezeAnyLayer l
  pure (OutputLayer l')
freezeNetwork {hs = h :: _} (l ~~> rest) = do
  l' <- freezeAnyLayer l
  rest' <- freezeNetwork rest
  pure (l' ~~> rest')

||| Inverse of `freezeNetwork`: sets `requires_grad=true` on every
||| parameter and retypes the network as `WithGrad`. Linear in input.
||| Use for progressive fine-tuning workflows (train head with backbone
||| frozen, then unfreeze backbone for joint fine-tuning).
export
unfreezeNetwork : {0 d : Device} -> {i, o : Nat} -> {hs : List Nat} ->
                   (1 _ : Network i hs o d dt NoGrad) -> IO (Network i hs o d dt WithGrad)
unfreezeNetwork (OutputLayer l) = do
  l' <- unfreezeAnyLayer l
  pure (OutputLayer l')
unfreezeNetwork {hs = h :: _} (l ~~> rest) = do
  l' <- unfreezeAnyLayer l
  rest' <- unfreezeNetwork rest
  pure (l' ~~> rest')

||| Reset per-sequence state on every layer in the network. Use
||| between training sequences for recurrent layers (Lstm, Rnn,
||| Gru). Stateless layers' default `resetState` is identity.
export
resetNetwork : {0 d : Device} -> {0 g : GradMode} -> {i, o : Nat} -> {hs : List Nat} ->
                 Network i hs o d dt g -> Network i hs o d dt g
resetNetwork (OutputLayer (MkAnyLayer l @{dict} layer)) =
  OutputLayer (MkAnyLayer l @{dict} (resetState @{dict} layer))
resetNetwork ((MkAnyLayer l @{dict} layer) ~~> rest) =
  MkAnyLayer l @{dict} (resetState @{dict} layer) ~~> resetNetwork rest

||| Batched tensor-level forward through a Network: each layer's
||| `applyVarBatch` runs on the threaded `[b, _]` tensor. Linear /
||| Activation / Dropout override; other layers crash via the
||| interface default.
export
forwardVarBatch : {0 d : Device} -> UserDeviceTape d => {0 g : GradMode} -> {i, o : Nat} -> {b : Nat} ->
                   {hs : List Nat} ->
                   Network i hs o d dt g -> Tensor [b, i] d dt g ->
                   IO (Network i hs o d dt g, Tensor [b, o] d dt g)
forwardVarBatch (OutputLayer l) input = do
  (l', out) <- applyVarBatchAny l input
  pure (OutputLayer l', out)
forwardVarBatch {hs = h :: _} (l ~~> rest) input = do
  (l', mid) <- applyVarBatchAny l input
  (rest', out) <- forwardVarBatch rest mid
  pure (l' ~~> rest', out)


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
forwardVarTraced : {0 d : Device} -> UserDeviceTape d => {0 g : GradMode} -> {i, o : Nat} -> {hs : List Nat} ->
                   (label : String) ->
                   Network i hs o d dt g -> Tensor [i] d dt g ->
                   IO (Network i hs o d dt g, Tensor [o] d dt g)
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

    go : {0 d : Device} -> UserDeviceTape d => {0 g : GradMode} -> {i, o : Nat} -> {hs : List Nat} ->
         Nat ->
         Network i hs o d dt g -> Tensor [i] d dt g ->
         IO (Network i hs o d dt g, Tensor [o] d dt g)
    go idx (OutputLayer l) inp = do
      (l', out) <- applyVarAny l inp
      summarize (show idx ++ "(out)") out.tensorPtr
      pure (OutputLayer l', out)
    go {hs = h :: _} idx (l ~~> rest) inp = do
      (l', mid) <- applyVarAny l inp
      summarize (show idx) mid.tensorPtr
      (rest', out) <- go (idx + 1) rest mid
      pure (l' ~~> rest', out)
