||| Type-safe integral index API: argsort / gather / scatter-add.
module Tensor.Index

import Data.Vect

import DType.Core
import Executor
import GradMode
import Tensor.Core

----------------------------------------------------------------------
-- Type-safe integral index API — argsort / gather / scatterAdd
--
-- The C `tensor_argsort` (torch) materializes its sort permutation in an
-- *integer* dtype (kLong/I64), and `gather`/`scatter_add` consume integral
-- indices. These wrappers lift that into the type: `targsort` returns an
-- `I64`-dtyped tensor, `tgather`/`tscatterAdd` require an `IsIntegral`
-- index, so "this tensor holds indices, not reals" is checked at compile
-- time rather than papered over by a float round-trip.
--
-- Torch-only by construction: an integer-dtyped tensor only exists where
-- `Compatible ex I64` holds (TorchExecutor TCpu / TCuda — Metal has no F64/int
-- gating, tape/mlx store F64 only). Calling these on tape/mlx is a type
-- error, not a runtime dtype mismatch. The untyped `primArgsort` /
-- `primGather` / `primScatterAdd` stay available for the F64 DNC path,
-- which runs on every backend and can't spell an integral index.
----------------------------------------------------------------------

||| Argument-sort: the permutation that sorts `t` along `axis`, returned as
||| an `I64` index tensor (the type captures "these are indices"). Set
||| `descending` for largest-first. Not autograd-tracked (indices have no
||| gradient), hence `NoGrad`.
export %inline
targsort : {0 ex : Executor} -> UserExecutorLinear ex => Compatible ex I64 =>
           (axis : Nat) -> (descending : Bool) ->
           Tensor dims ex dt g -> IO (Tensor dims ex I64 NoGrad)
targsort axis descending t = ioRerun (\_ =>
  MkTensor (primArgsort {ex} t.tensorPtr (cast axis) (if descending then 1 else 0)) Nothing)

||| Gather rows of `src` along axis 0 at the given integral indices
||| (torch `index_select`). `IsIntegral idt` rejects a float "index"
||| tensor. Differentiable w.r.t. `src`, so the result carries `src`'s
||| grad mode.
export %inline
tgather : {0 ex : Executor} -> UserExecutorLinear ex => IsIntegral idt =>
          {m, n : Nat} -> {0 r : Nat} -> {0 rest : Vect r Nat} ->
          Tensor (m :: rest) ex dt g -> Tensor [n] ex idt NoGrad ->
          IO (Tensor (n :: rest) ex dt g)
tgather src idx = ioRerun (\_ =>
  MkTensor (primGather {ex} src.tensorPtr idx.tensorPtr (cast n)) Nothing)

||| Scatter-add `src` into a fresh `[outSize]` zero vector at the given
||| integral indices along axis 0 (torch `scatter_add_`). `IsIntegral idt`
||| rejects a float "index" tensor. Differentiable w.r.t. `src`.
export %inline
tscatterAdd : {0 ex : Executor} -> UserExecutorLinear ex => IsIntegral idt =>
              {n : Nat} -> (outSize : Nat) ->
              Tensor [n] ex idt NoGrad -> Tensor [n] ex dt g ->
              IO (Tensor [outSize] ex dt g)
tscatterAdd outSize idx src = ioRerun (\_ =>
  MkTensor (primScatterAdd {ex} idx.tensorPtr src.tensorPtr (cast outSize)) Nothing)
