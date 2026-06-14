||| `RmsNorm` — root-mean-square layer norm (Zhang & Sennrich 2019; the
||| Llama/T5 variant): `out[i] = x[i] / sqrt(mean(x²) + eps) * weight[i]`,
||| no mean-centring or bias. One learnable param (`weight`).
|||
||| Params-only, NOT a batched `Module`: the differentiable form composes
||| `primSum` (a GLOBAL reduction), so it is correct only per-vector (1-D).
||| A batched `[b,n]` Module would need a per-row differentiable reduction,
||| which no current prim provides (the fused `primRmsNorm2d` is
||| inference-only — no tape backward). `rmsNormForward` is the 1-D
||| differentiable forward; transformers apply it per position. A fused
||| differentiable `primRmsNorm2d` is the follow-up that would let RmsNorm
||| become a real `Module`.
module Nn.RmsNorm

import Data.Vect

import Executor
import Tensor
import Nn.Init
import Nn.Module

%default total

||| Llama 3 default; Llama 2 / T5 / Falcon used 1e-6. Pass the HF config's
||| `rms_norm_eps` when matching a specific checkpoint.
public export
defaultRmsNormEps : Double
defaultRmsNormEps = 1.0e-5

||| RMSNorm with a single learnable scale (`i = o = n`).
public export
data RmsNorm : Nat -> Nat -> (0 _ : Executor) -> (0 _ : DType) -> Type where
  MkRmsNorm : TVec n ex dt WithGrad -> RmsNorm n n ex dt

public export
Params RmsNorm where
  params (MkRmsNorm w) = [toParam w]

||| 1-D differentiable RMSNorm forward (per vector). `primSum` reduces the
||| whole vector, so this is single-vector only — see the module header.
export
rmsNormForward : {0 ex : Executor} -> Backend ex dt => {n : Nat} ->
                 (eps : Double) -> RmsNorm n n ex dt -> TVec n ex dt g -> IO (TVec n ex dt g)
rmsNormForward {n} eps (MkRmsNorm weight) input = ioRerun (\_ =>
  let sq      = primMul {ex} input.tensorPtr input.tensorPtr
      tot     = primSum {ex} sq
      mean    = primMulScalar {ex} tot (1.0 / cast {to=Double} n)
      meanEps = primAddScalar {ex} mean eps
      rms     = primSqrt {ex} meanEps
      normed  = primDiv {ex} input.tensorPtr rms
      scaled  = primMul {ex} normed weight.tensorPtr
  in MkTensor scaled Nothing)

||| Construct an `RmsNorm n n` inside an `Init` derivation; registers
||| `<scope>.rms_norm_<n>.weight` (init 1, HF default).
export
rmsNorm : {0 ex : Executor} -> Backend ex dt => {n : Nat} -> Init (RmsNorm n n ex dt)
rmsNorm = do
  name   <- freshChild "rms_norm"
  weight <- liftIO $ tparam1dConst {ex} {dt} {n} (name ++ ".weight") 1.0
  pure (MkRmsNorm weight)
