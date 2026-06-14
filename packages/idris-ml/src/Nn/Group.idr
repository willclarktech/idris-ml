||| `groupOf` — the exact param set of a (sub)model, by registry name. The
||| v1 replacement for substring-prefix scoping (`adam {scope="actor_"}`,
||| `groups := [("bert.", lr)]`, `freezeByPrefix`): instead of matching a
||| string against every registered name (where `"actor"` also catches
||| `"actor_critic_*"` — the silent gradient-leak bug class), you hand the
||| optimizer the precise names a submodel owns, derived structurally from
||| its `Params` traversal.
module Nn.Group

import Data.List

import Executor
import Nn.Module

||| The registry names of every param owned by `m`, in traversal order.
||| Params without a registry name (intermediates) are dropped.
export
groupOf : {0 ex : Executor} -> {0 dt : DType} -> {0 i, o : Nat} ->
          {l : Nat -> Nat -> (0 _ : Executor) -> (0 _ : DType) -> Type} ->
          Params l => l i o ex dt -> List String
groupOf m = mapMaybe paramName (params m)
