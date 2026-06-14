module Layer.RmsNorm

import Data.Vect

import Executor
import Layer.Core
import Tensor

----------------------------------------------------------------------
-- RMSNorm — root-mean-square layer normalisation
----------------------------------------------------------------------
--
-- Standard formulation (Zhang & Sennrich 2019, the variant Llama / T5
-- / Falcon use):
--
--   out[i] = (x[i] / sqrt(mean_i(x²) + eps)) * weight[i]
--
-- vs LayerNorm's `(x - mean) / sqrt(var + eps) * gamma + beta`,
-- RMSNorm drops the mean-centring + bias. One learnable param
-- (weight), no beta. Cheaper to compute and matches the empirical
-- claim that the re-centring isn't load-bearing for modern LLM-class
-- training.
--
-- HF naming: Llama's `model.norm.weight` and `model.layers.{i}.input_layernorm.weight`
-- both register with `_weight` suffix only (no `_bias`). This module
-- registers under `<prefix>_weight`; HF-aligned modules (HfLlama)
-- re-bind to the exact HF on-disk name at registration time.
--
-- No fused C primitive — composes `primMul` / `primSum` /
-- `primMulScalar` / `primAddScalar` / `primSqrt` / `primDiv` with
-- scalar broadcasting through primDiv (same broadcasting story as
-- primAdd; verified working on all three backends). Per the plan
-- ("C-side: no new primitive needed"), this is the v1 shape. A
-- fused `primRmsNorm2d` is the obvious Phase-4 perf optimisation if
-- the composed form is a hot path.

public export
data RmsNormState : Nat -> Nat -> (0 _ : Executor) -> (0 _ : DType) -> (0 _ : GradMode) -> Type where
  MkRmsNorm : TVec n ex dt g -> RmsNormState n n ex dt g

----------------------------------------------------------------------
-- Forward
----------------------------------------------------------------------

%default partial

-- Llama 3 uses 1e-5; older Llama 2 / T5 / Falcon used 1e-6. The HF
-- config's `rms_norm_eps` carries the per-model value — callers that
-- match a specific HF checkpoint should pass it through. This
-- module's default is 1e-5 (matches the model we actually load).
public export
defaultRmsNormEps : Double
defaultRmsNormEps = 1.0e-5

||| Apply RMSNorm with a caller-supplied epsilon. The model-construction
||| smart constructors below pin `defaultRmsNormEps` for the LayerLike
||| instance; HfLlama uses this form directly with the config's eps.
export
applyRmsNormEps : {0 ex : Executor} -> UserExecutorTraining ex => UserExecutorCore ex => {n : Nat} ->
                  (eps : Double) -> RmsNormState n n ex dt g ->
                  TVec n ex dt g -> IO (RmsNormState n n ex dt g, TVec n ex dt g)
applyRmsNormEps {n} eps st@(MkRmsNorm weight) input = ioRerun (\_ =>
  let nD = cast {to=Double} n
      sq      = primMul {ex} input.tensorPtr input.tensorPtr
      tot     = primSum {ex} sq
      mean    = primMulScalar {ex} tot (1.0 / nD)
      meanEps = primAddScalar {ex} mean eps
      rms     = primSqrt {ex} meanEps
      normed  = primDiv {ex} input.tensorPtr rms
      scaled  = primMul {ex} normed weight.tensorPtr
  in (st, MkTensor scaled Nothing))

----------------------------------------------------------------------
-- Constructor
----------------------------------------------------------------------

||| Build a `RmsNormState n n` with `weight` initialised to 1.0
||| (HF-default — same as LayerNorm's gamma). Registers as a C param
||| under `<prefix>_weight`. HF-aligned modules (HfLlama) re-bind the
||| name at registration to `<model.layers.i.input_layernorm.weight>`.
export
rmsNormLayer : Backend ex dt =>
               {n : Nat} -> (paramPrefix : String) ->
               IO (RmsNormState n n ex dt WithGrad)
rmsNormLayer paramPrefix = do
  let wName = paramPrefix ++ "_weight"
  weight <- tparam1dConst {ex} {dt} {n} wName 1.0
  pure $ MkRmsNorm weight

----------------------------------------------------------------------
-- LayerLike instance
----------------------------------------------------------------------

public export
LayerLike RmsNormState where
  applyVar st@(MkRmsNorm _) input = applyRmsNormEps defaultRmsNormEps st input
  layerPrefix _                   = "rms"

  freezeLayer (MkRmsNorm w) = do
    w' <- weakenGrad w
    pure (MkRmsNorm w')

  unfreezeLayer (MkRmsNorm w) = do
    primIO (primSetRequiresGrad {ex} w.tensorPtr 1)
    pure (MkRmsNorm (retypeGrad w))

||| Wrap an RmsNorm in `AnyLayer`.
export
rmsNormLayerAny : Backend ex dt =>
                  {n : Nat} -> (paramPrefix : String) ->
                  IO (AnyLayer n n ex dt WithGrad)
rmsNormLayerAny pid = map (MkAnyLayer RmsNormState) (rmsNormLayer pid)
