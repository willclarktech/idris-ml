||| `groupOf` — the exact param set of a (sub)model, by registry name. The
||| v1 replacement for substring-prefix scoping (`adam {scope="actor_"}`,
||| `groups := [("bert.", lr)]`, `freezeByPrefix`): instead of matching a
||| string against every registered name (where `"actor"` also catches
||| `"actor_critic_*"` — the silent gradient-leak bug class), you hand the
||| optimizer the precise names a submodel owns, derived structurally from
||| its `Params` traversal.
module Ml.Nn.Group

import Data.Linear
import Data.List

import Ml.Executor
import Ml.Nn.Module
import Ml.Tensor

||| The registry names of every param owned by `m`, in traversal order (params
||| without a registry name are dropped). Uses the read-only flat `params`
||| accessor, so it does not consume the model.
export
groupOf : {0 ex : Executor} -> {0 dt : DType} -> {0 g : GradMode} -> {0 i, o : Nat} ->
          {l : Nat -> Nat -> (0 _ : Executor) -> (0 _ : DType) -> (0 _ : GradMode) -> Type} ->
          Params l => l i o ex dt g -> List String
groupOf m = mapMaybe paramName (params m)

||| The linear twin of `groupOf`: take a single-owner (linear) model, return its
||| exact registry names beside the model threaded back, so it can be used at a
||| site where the model is a `1`-quantity resource (inside `Control.Linear.LIO`).
||| Built on the linear `reflect`, so it consumes-and-rebuilds rather than
||| ω-projecting. The leak-free source for optimizer ownership: pair with
||| `Train.Freeze.restrictTo` to scope an optimizer to one net's exact params.
export
reflectNames : {0 ex : Executor} -> {0 dt : DType} -> {0 g : GradMode} -> {0 i, o : Nat} ->
               {l : Nat -> Nat -> (0 _ : Executor) -> (0 _ : DType) -> (0 _ : GradMode) -> Type} ->
               Params l => (1 _ : l i o ex dt g) -> LPair (!* (List String)) (l i o ex dt g)
reflectNames m = let (MkBang ps # m') = reflect m in MkBang (mapMaybe paramName ps) # m'
