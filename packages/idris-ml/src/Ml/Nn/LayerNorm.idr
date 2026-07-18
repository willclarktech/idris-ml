||| `LayerNorm` — stateless-shape (`i = o = n`) but param-bearing (learnable
||| gamma/beta) layer on the v1 `Nn` surface. `forward` normalises along the
||| feature dim of a batched `[b,n]` input via the fused `primLayerNorm2d`
||| (the same C op the legacy 1D path reshaped into) — no reshape needed at
||| batch rank. PyTorch parameter names: `weight` (gamma), `bias` (beta).
module Ml.Nn.LayerNorm

import Control.Linear.LIO
import Data.Linear
import Data.Vect

import Ml.Executor
import Ml.Nn.Init
import Ml.Nn.Module
import Ml.Tensor

%default total

||| Layer normalisation with learnable scale + shift (`i = o = n`).
public export
data LayerNorm : Nat -> Nat -> (0 _ : Executor) -> (0 _ : DType) -> (0 _ : GradMode) -> Type where
  MkLayerNorm : TVec n ex dt g -> TVec n ex dt g -> LayerNorm n n ex dt g

||| Params. `gamma`/`beta` are ω tensors: they feed the reflected param list
||| and the rebuild.
public export
Params LayerNorm where
  params (MkLayerNorm gamma beta)   = [toParam gamma, toParam beta]
  reflect (MkLayerNorm gamma beta)  = MkBang [toParam gamma, toParam beta] # MkLayerNorm gamma beta
  castGrad (MkLayerNorm gamma beta) = MkLayerNorm (retypeGrad gamma) (retypeGrad beta)
  discard (MkLayerNorm _ _)         = pure ()

||| `Module` — sequences the fused `L IO` layer-norm op directly.
public export
Module LayerNorm where
  forward (MkLayerNorm gamma beta) x = do
    y <- ioRerunL (\_ =>
      MkTensor (primLayerNorm2d {ex} x.tensorPtr gamma.tensorPtr beta.tensorPtr 1.0e-5) Nothing)
    pure1 (MkBang y # MkLayerNorm gamma beta)

||| Construct a `LayerNorm n n` inside an `Init` derivation. Registers
||| `<scope>.layer_norm_<n>.weight` (gamma, init 1) / `.bias` (beta, init 0).
export
layerNorm : KnownGrad g => {0 ex : Executor} -> Backend ex dt => {n : Nat} -> Init (LayerNorm n n ex dt g)
layerNorm = do
  name  <- freshChild "layer_norm"
  gamma <- liftIO $ tparam1dConst {ex} {dt} {n} (name ++ ".weight") 1.0
  beta  <- liftIO $ tparam1dConst {ex} {dt} {n} (name ++ ".bias")   0.0
  case sgrad {g} of
    SWithGrad => pure (MkLayerNorm gamma beta)
    SNoGrad   => do gamma' <- liftIO (weakenGrad gamma)
                    beta'  <- liftIO (weakenGrad beta)
                    pure (MkLayerNorm gamma' beta')
