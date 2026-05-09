module Layer.Core

import Data.Vect

import Device
import Variable


----------------------------------------------------------------------
-- Path C P3-1 spike: rank-aware layer interface
----------------------------------------------------------------------
--
-- Parallel to `Layer.Core.LayerLike`, but operates on rank-aware
-- `Variable` directly. No Vect-of-Vect packing; no scalar boundaries
-- between layers. Spike-only — only the methods needed for a
-- single Linear + chained forward pass are present. Full
-- migration widens this surface (toDouble, debug, batched, etc.).

public export
interface LayerLike (l : Nat -> Nat -> (0 _ : Device) -> Type) where
  ||| Tensor-level forward: `Variable [i] d -> Variable [o] d`.
  applyVar : {0 d : Device} -> {i, o : Nat} ->
              l i o d -> Variable [i] d -> (l i o d, Variable [o] d)

  ||| Auto-naming prefix (e.g. "llv2" for Linear).
  layerPrefix : {0 d : Device} -> {i, o : Nat} -> l i o d -> String
  layerPrefix _ = ""

  ||| Reset per-sequence state (recurrent layers override; default = id).
  ||| Used by `resetNetwork` between sequences in recurrent training.
  resetState : {0 d : Device} -> {i, o : Nat} -> l i o d -> l i o d
  resetState = id

  ||| Batched tensor-level forward: `Variable [b, i] d -> Variable [b, o] d`.
  ||| Default crashes — layers that participate in batched training
  ||| (Linear, Activation, Dropout) MUST override. Stateful layers
  ||| (LSTM/RNN/GRU/NTM/DNC) keep the default; batched-cell semantics
  ||| are not supported in this surface (use sequence-level batching
  ||| at the example level instead).
  applyVarBatch : {0 d : Device} -> {i, o : Nat} -> {b : Nat} ->
                   l i o d -> Variable [b, i] d -> (l i o d, Variable [b, o] d)
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
               AnyLayer i o d -> Variable [i] d -> (AnyLayer i o d, Variable [o] d)
applyVarAny (MkAnyLayer l @{dict} layer) input =
  case applyVar @{dict} layer input of
    (layer', out) => (MkAnyLayer l @{dict} layer', out)

export
applyVarBatchAny : {0 d : Device} -> {i, o : Nat} -> {b : Nat} ->
                    AnyLayer i o d -> Variable [b, i] d ->
                    (AnyLayer i o d, Variable [b, o] d)
applyVarBatchAny (MkAnyLayer l @{dict} layer) input =
  case applyVarBatch @{dict} layer input of
    (layer', out) => (MkAnyLayer l @{dict} layer', out)


----------------------------------------------------------------------
-- Network
----------------------------------------------------------------------

public export
data Network : (i : Nat) -> (hs : List Nat) -> (o : Nat) -> (0 _ : Device) -> Type where
  OutputLayer : AnyLayer i o d -> Network i [] o d
  (~~>) : AnyLayer i h d -> Network h hs o d -> Network i (h :: hs) o d

export infixr 5 ~~>

||| Tensor-level forward through a Network.
export
forwardVar : {0 d : Device} -> {i, o : Nat} -> {hs : List Nat} ->
              Network i hs o d -> Variable [i] d -> (Network i hs o d, Variable [o] d)
forwardVar (OutputLayer l) input =
  case applyVarAny l input of
    (l', out) => (OutputLayer l', out)
forwardVar {hs = h :: _} (l ~~> rest) input =
  case applyVarAny l input of
    (l', mid) =>
      case forwardVar rest mid of
        (rest', out) => (l' ~~> rest', out)

||| Reset per-sequence state on every layer in the network. Use
||| between training sequences for recurrent layers (Lstm, Rnn,
||| Gru). Stateless layers' default `resetState` is identity.
export
resetNetwork : {0 d : Device} -> {i, o : Nat} -> {hs : List Nat} ->
                 Network i hs o d -> Network i hs o d
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
                   Network i hs o d -> Variable [b, i] d ->
                   (Network i hs o d, Variable [b, o] d)
forwardVarBatch (OutputLayer l) input =
  case applyVarBatchAny l input of
    (l', out) => (OutputLayer l', out)
forwardVarBatch {hs = h :: _} (l ~~> rest) input =
  case applyVarBatchAny l input of
    (l', mid) =>
      case forwardVarBatch rest mid of
        (rest', out) => (l' ~~> rest', out)
