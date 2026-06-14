||| `TransformerBlock` — a pre-norm transformer block, the composable unit
||| of a transformer. It is a batched `Module` mapping `[seqLen, dModel] →
||| [seqLen, dModel]` (b = seqLen, i = o = dModel), so blocks **stack via
||| `Seq`** like any other layer — the elegant decomposition the monolithic
||| legacy `Transformer` couldn't express. Like `Conv2D`, the extra config
||| Nats (numHeads/headDim) lead and the `(dModel, dModel)` i/o pin trails,
||| fitting the `Module`/`Params` kind.
|||
||| Forward (pre-norm, GPT-style): `h₁ = h + attn(LN₁ h)`, then
||| `out = h₁ + FFN(LN₂ h₁)`, where FFN is a bias-free
||| `relu(x·W₁ᵀ)·W₂ᵀ` with a 4× hidden. Composes `Nn.Attention` +
||| `Nn.LayerNorm`; assemble the full model (embedding → block `Seq` →
||| norm → head) at the example level.
module Nn.Transformer

import Control.Linear.LIO
import Data.Linear
import Data.Vect

import Executor
import Nn.Attention
import Nn.Init
import Nn.LayerNorm
import Nn.Module
import Tensor

%default total

||| A pre-norm transformer block. `numHeads`/`headDim` lead; the trailing
||| `dModel dModel` are the Module i/o.
public export
data TransformerBlock : (numHeads : Nat) -> (headDim : Nat) ->
                        Nat -> Nat -> (0 _ : Executor) -> (0 _ : DType) -> (0 _ : GradMode) -> Type where
  MkTransformerBlock :
    {dModel : Nat} ->
    Attention dModel numHeads headDim ex dt g ->
    LayerNorm dModel dModel ex dt g ->
    LayerNorm dModel dModel ex dt g ->
    TMat (4 * dModel) dModel ex dt g ->   -- ff1 (bias-free)
    TMat dModel (4 * dModel) ex dt g ->   -- ff2 (bias-free)
    TransformerBlock numHeads headDim dModel dModel ex dt g

public export
{numHeads, headDim : Nat} -> Module (TransformerBlock numHeads headDim) where
  forward (MkTransformerBlock attn n1 n2 ff1 ff2) h = assert_total $ do
    normed1 <- forward n1 h
    attnOut <- attentionForward attn normed1
    h1      <- tadd attnOut h
    normed2 <- forward n2 h1
    ioRerun (\_ =>
      let ffHidden = primClampMin {ex} (primMm {ex} normed2.tensorPtr (primTranspose2d {ex} ff1.tensorPtr)) 0.0
          ffOut    = primMm {ex} ffHidden (primTranspose2d {ex} ff2.tensorPtr)
      in MkTensor (primAdd {ex} ffOut h1.tensorPtr) Nothing)

public export
{numHeads, headDim : Nat} -> Params (TransformerBlock numHeads headDim) where
  params (MkTransformerBlock attn n1 n2 ff1 ff2) =
    attentionParams attn ++ params n1 ++ params n2 ++ [toParam ff1, toParam ff2]
  castGrad (MkTransformerBlock attn n1 n2 ff1 ff2) =
    MkTransformerBlock (attentionCastGrad attn) (castGrad n1) (castGrad n2)
                       (retypeGrad ff1) (retypeGrad ff2)

||| Linear-resource params. attn/norms/ff weights bind at ω, so the splice
||| reuses the IO `attentionParams`/`params` for both the reflected list and
||| the rebuild.
public export
{numHeads, headDim : Nat} -> ParamsL (TransformerBlock numHeads headDim) where
  reflectL (MkTransformerBlock attn n1 n2 ff1 ff2) =
    MkBang (attentionParams attn ++ params n1 ++ params n2 ++ [toParam ff1, toParam ff2])
      # MkTransformerBlock attn n1 n2 ff1 ff2
  castGradL (MkTransformerBlock attn n1 n2 ff1 ff2) =
    MkTransformerBlock (attentionCastGrad attn) (castGrad n1) (castGrad n2)
                       (retypeGrad ff1) (retypeGrad ff2)
  discardL (MkTransformerBlock _ _ _ _ _) = pure ()

||| Linear-resource `Module`. The forward is read-only on every sub-layer
||| (the block is returned unchanged), so it consumes the linear block,
||| delegates the multi-step body to the IO `forward` at the linear boundary,
||| and rebuilds the unchanged block beside the banged output.
public export
{numHeads, headDim : Nat} -> ModuleL (TransformerBlock numHeads headDim) where
  forwardL (MkTransformerBlock attn n1 n2 ff1 ff2) h = do
    out <- liftIO1 (forward (MkTransformerBlock attn n1 n2 ff1 ff2) h)
    pure1 (MkBang out # MkTransformerBlock attn n1 n2 ff1 ff2)

||| Construct a pre-norm `TransformerBlock` inside an `Init` derivation.
||| Nests `attn.*` (per-head projections), `norm1`/`norm2`, and the
||| bias-free `ff1`/`ff2` (~ N(0, 1/√fan_in)) under the current scope.
export
transformerBlock : {0 ex : Executor} -> Backend ex dt => {dModel, numHeads, headDim : Nat} ->
                   Init (TransformerBlock numHeads headDim dModel dModel ex dt WithGrad)
transformerBlock = do
  a   <- scopedChild "attn" attention
  n1  <- named "norm1" (layerNorm {n = dModel})
  n2  <- named "norm2" (layerNorm {n = dModel})
  f1n <- freshChild "ff1"
  ff1 <- liftIO $ tparam2dNormal {ex} {dt} {o = 4 * dModel} {i = dModel}     (f1n ++ ".weight") 0.0 (1.0 / sqrt (cast {to=Double} dModel))
  f2n <- freshChild "ff2"
  ff2 <- liftIO $ tparam2dNormal {ex} {dt} {o = dModel}     {i = 4 * dModel} (f2n ++ ".weight") 0.0 (1.0 / sqrt (cast {to=Double} (4 * dModel)))
  pure (MkTransformerBlock a n1 n2 ff1 ff2)
