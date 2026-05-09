module Layer.CoreV2

import Data.Vect

import Device
import Variable


----------------------------------------------------------------------
-- Path C P3-1 spike: rank-aware layer interface
----------------------------------------------------------------------
--
-- Parallel to `Layer.Core.LayerLike`, but operates on rank-aware
-- `TVar` directly. No Vect-of-Vect packing; no scalar boundaries
-- between layers. Spike-only — only the methods needed for a
-- single LinearV2 + chained forward pass are present. Full
-- migration widens this surface (toDouble, debug, batched, etc.).

public export
interface LayerLikeV2 (l : Nat -> Nat -> (0 _ : Device) -> Type) where
  ||| Tensor-level forward: `TVar [i] d -> TVar [o] d`.
  applyTVar : {0 d : Device} -> {i, o : Nat} ->
              l i o d -> TVar [i] d -> (l i o d, TVar [o] d)

  ||| Auto-naming prefix (e.g. "llv2" for LinearV2).
  layerPrefixV2 : {0 d : Device} -> {i, o : Nat} -> l i o d -> String
  layerPrefixV2 _ = ""

  ||| Reset per-sequence state (recurrent layers override; default = id).
  ||| Used by `resetNetworkV2` between sequences in recurrent training.
  resetStateV2 : {0 d : Device} -> {i, o : Nat} -> l i o d -> l i o d
  resetStateV2 = id

  ||| Batched tensor-level forward: `TVar [b, i] d -> TVar [b, o] d`.
  ||| Default crashes — layers that participate in batched training
  ||| (Linear, Activation, Dropout) MUST override. Stateful layers
  ||| (LSTM/RNN/GRU/NTM/DNC) keep the default; batched-cell semantics
  ||| are not supported in this surface (use sequence-level batching
  ||| at the example level instead).
  applyTVarBatch : {0 d : Device} -> {i, o : Nat} -> {b : Nat} ->
                   l i o d -> TVar [b, i] d -> (l i o d, TVar [b, o] d)
  applyTVarBatch _ _ =
    idris_crash "applyTVarBatch: layer does not support batched forward"


----------------------------------------------------------------------
-- AnyLayerV2 (existential wrapper)
----------------------------------------------------------------------

public export
data AnyLayerV2 : Nat -> Nat -> (0 _ : Device) -> Type where
  MkAnyLayerV2 : (l : Nat -> Nat -> (0 _ : Device) -> Type) -> LayerLikeV2 l =>
                 l i o d -> AnyLayerV2 i o d

export
applyTVarAny : {0 d : Device} -> {i, o : Nat} ->
               AnyLayerV2 i o d -> TVar [i] d -> (AnyLayerV2 i o d, TVar [o] d)
applyTVarAny (MkAnyLayerV2 l @{dict} layer) input =
  case applyTVar @{dict} layer input of
    (layer', out) => (MkAnyLayerV2 l @{dict} layer', out)

export
applyTVarBatchAny : {0 d : Device} -> {i, o : Nat} -> {b : Nat} ->
                    AnyLayerV2 i o d -> TVar [b, i] d ->
                    (AnyLayerV2 i o d, TVar [b, o] d)
applyTVarBatchAny (MkAnyLayerV2 l @{dict} layer) input =
  case applyTVarBatch @{dict} layer input of
    (layer', out) => (MkAnyLayerV2 l @{dict} layer', out)


----------------------------------------------------------------------
-- NetworkV2
----------------------------------------------------------------------

public export
data NetworkV2 : (i : Nat) -> (hs : List Nat) -> (o : Nat) -> (0 _ : Device) -> Type where
  OutputLayerV2 : AnyLayerV2 i o d -> NetworkV2 i [] o d
  (~~>) : AnyLayerV2 i h d -> NetworkV2 h hs o d -> NetworkV2 i (h :: hs) o d

export infixr 5 ~~>

||| Tensor-level forward through a NetworkV2.
export
forwardTVar : {0 d : Device} -> {i, o : Nat} -> {hs : List Nat} ->
              NetworkV2 i hs o d -> TVar [i] d -> (NetworkV2 i hs o d, TVar [o] d)
forwardTVar (OutputLayerV2 l) input =
  case applyTVarAny l input of
    (l', out) => (OutputLayerV2 l', out)
forwardTVar {hs = h :: _} (l ~~> rest) input =
  case applyTVarAny l input of
    (l', mid) =>
      case forwardTVar rest mid of
        (rest', out) => (l' ~~> rest', out)

||| Reset per-sequence state on every layer in the network. Use
||| between training sequences for recurrent layers (LstmV2, RnnV2,
||| GruV2). Stateless layers' default `resetStateV2` is identity.
export
resetNetworkV2 : {0 d : Device} -> {i, o : Nat} -> {hs : List Nat} ->
                 NetworkV2 i hs o d -> NetworkV2 i hs o d
resetNetworkV2 (OutputLayerV2 (MkAnyLayerV2 l @{dict} layer)) =
  OutputLayerV2 (MkAnyLayerV2 l @{dict} (resetStateV2 @{dict} layer))
resetNetworkV2 ((MkAnyLayerV2 l @{dict} layer) ~~> rest) =
  MkAnyLayerV2 l @{dict} (resetStateV2 @{dict} layer) ~~> resetNetworkV2 rest

||| Batched tensor-level forward through a NetworkV2: each layer's
||| `applyTVarBatch` runs on the threaded `[b, _]` tensor. Linear /
||| Activation / Dropout override; other layers crash via the
||| interface default.
export
forwardTVarBatch : {0 d : Device} -> {i, o : Nat} -> {b : Nat} ->
                   {hs : List Nat} ->
                   NetworkV2 i hs o d -> TVar [b, i] d ->
                   (NetworkV2 i hs o d, TVar [b, o] d)
forwardTVarBatch (OutputLayerV2 l) input =
  case applyTVarBatchAny l input of
    (l', out) => (OutputLayerV2 l', out)
forwardTVarBatch {hs = h :: _} (l ~~> rest) input =
  case applyTVarBatchAny l input of
    (l', mid) =>
      case forwardTVarBatch rest mid of
        (rest', out) => (l' ~~> rest', out)
