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
