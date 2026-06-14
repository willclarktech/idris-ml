||| `LayerNorm` — stateless-shape (`i = o = n`) but param-bearing (learnable
||| gamma/beta) layer on the v1 `Nn` surface. `forward` normalises along the
||| feature dim of a batched `[b,n]` input via the fused `primLayerNorm2d`
||| (the same C op the legacy 1D path reshaped into) — no reshape needed at
||| batch rank. PyTorch parameter names: `weight` (gamma), `bias` (beta).
module Nn.LayerNorm

import Data.Vect

import Executor
import Nn.Init
import Nn.Module
import Tensor

%default total

||| Layer normalisation with learnable scale + shift (`i = o = n`).
public export
data LayerNorm : Nat -> Nat -> (0 _ : Executor) -> (0 _ : DType) -> (0 _ : GradMode) -> Type where
  MkLayerNorm : TVec n ex dt g -> TVec n ex dt g -> LayerNorm n n ex dt g

public export
Module LayerNorm where
  forward (MkLayerNorm gamma beta) x = ioRerun (\_ =>
    MkTensor (primLayerNorm2d {ex} x.tensorPtr gamma.tensorPtr beta.tensorPtr 1.0e-5) Nothing)

public export
Params LayerNorm where
  params (MkLayerNorm gamma beta)   = [toParam gamma, toParam beta]
  castGrad (MkLayerNorm gamma beta) = MkLayerNorm (retypeGrad gamma) (retypeGrad beta)

||| Construct a `LayerNorm n n` inside an `Init` derivation. Registers
||| `<scope>.layer_norm_<n>.weight` (gamma, init 1) / `.bias` (beta, init 0).
export
layerNorm : {0 ex : Executor} -> Backend ex dt => {n : Nat} -> Init (LayerNorm n n ex dt WithGrad)
layerNorm = do
  name  <- freshChild "layer_norm"
  gamma <- liftIO $ tparam1dConst {ex} {dt} {n} (name ++ ".weight") 1.0
  beta  <- liftIO $ tparam1dConst {ex} {dt} {n} (name ++ ".bias")   0.0
  pure (MkLayerNorm gamma beta)
