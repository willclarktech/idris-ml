||| BitNet b1.58 quantized-linear surface (ternary create + forward +
||| the load-time quantization recipe).
module Ml.Tensor.Quant

import Data.Vect

import Ml.DType.Core
import Ml.Executor
import Ml.GradMode
import Ml.Tensor.Core

----------------------------------------------------------------------
-- BitNet b1.58 quantized linear (#411 B2)
--
-- `tCreateTernaryPacked2d` constructs a `Tensor [o, i] ex Ternary NoGrad`
-- from a host buffer of packed 2-bit ternary codes (4 values/byte, see
-- the C-side `tensor_create_ternary_packed_2d` docstring + design-
-- decisions.md "Per-backend ternary storage" for the per-backend
-- storage layouts).
--
-- `tBitlinearFwd` runs y = (W_ternary .* scale[:, None]) @ x + bias on
-- the device, decoding W inline (tape) or via int8-cast (torch/mlx).
-- Weight is NoGrad by construction (BitNet b1.58 freezes the ternary
-- params); bias is grad-mode-parametric so callers can attach the
-- usual autograd edge for fine-tuning.
--
-- Both go through the opt-in `UserExecutorQuant ex =>` typeclass —
-- built-in backends (tape/torch/mlx) implement it; BYO backends opt
-- in only if they want BitNet. Dispatches on `d` to the suffixed C
-- symbol on each backend (the unified-name alias machinery was
-- removed earlier so unprefixed dispatch isn't available).
----------------------------------------------------------------------

||| Build a `Tensor [o, i] ex Ternary NoGrad` from a host buffer of
||| packed 2-bit ternary codes. The buffer layout is row-major with
||| each row padded to `(i + 3) / 4` bytes (trailing slots zero-padded
||| as ternary 0). See the C docstring for the bit-level encoding.
|||
||| `bytesPtr` must be backed by `prim__allocBytes` or another host
||| buffer the caller keeps alive across the call; the C side copies
||| into the device arena, so the buffer is freeable on return.
export
tCreateTernaryPacked2d : {0 ex : Executor} -> UserExecutorQuant ex =>
                         {o, i : Nat} ->
                         AnyPtr -> (byteCount : Int) ->
                         IO (Tensor [o, i] ex Ternary NoGrad)
tCreateTernaryPacked2d bytesPtr byteCount = ioRerun (\_ =>
  MkTensor (primCreateTernaryPacked2d {ex} bytesPtr byteCount
              (cast o) (cast i) 0)
           Nothing)

||| Build a Ternary tensor from HF's `[(o + 3) / 4, i]` uint8 packed
||| buffer (microsoft/bitnet-b1.58-2B-4T-style layout). One-shot at
||| safetensors load; layout repack + encoding remap happens inside
||| the C primitive. `bytesPtr` must point at `((o + 3) / 4) * i`
||| HF-format bytes the caller keeps alive across the call (the C
||| side copies into device storage).
export
tCreateTernaryFromHfPacked2d : {0 ex : Executor} -> UserExecutorQuant ex =>
                               {o, i : Nat} -> AnyPtr ->
                               IO (Tensor [o, i] ex Ternary NoGrad)
tCreateTernaryFromHfPacked2d bytesPtr = ioRerun (\_ =>
  MkTensor (primCreateTernaryFromHfPacked2d {ex} bytesPtr (cast o) (cast i))
           Nothing)

||| Fused HF BitLinear forward — RMSNorm + per-token int8 act-quant +
||| matmul + scalar dequant + bias. Matches HF transformers'
||| `BitLinear.forward` semantics for microsoft/bitnet-b1.58-2B-4T-
||| style checkpoints. Equivalent to (and ~2x faster than) composing
||| `tActivationQuantInt8` + `tBitlinearFwd` from Idris.
|||
||| `useRmsNorm = True` applies RMSNorm with `rmsNormWeight` + `eps`
|||  to `x` before the activation quant; `rmsNormWeight` must be
|||  non-null when this flag is set. With `useRmsNorm = False`, the
|||  `rmsNormWeight` tensor is ignored — pass any [i]-shaped tensor
|||  (e.g. the same `x` placeholder) and the C side won't read it.
export
tBitlinearFwdHfQuant : {0 ex : Executor} -> UserExecutorQuant ex =>
                      {o, i : Nat} ->
                      Tensor [o, i] ex Ternary NoGrad ->
                      (weightScale : Double) ->
                      Tensor [i] ex cDt g ->
                      Tensor [o] ex cDt g ->
                      (useRmsNorm : Bool) ->
                      Tensor [i] ex cDt NoGrad ->
                      (rmsNormEps : Double) ->
                      IO (Tensor [o] ex cDt g)
tBitlinearFwdHfQuant w wScale x bias useRmsNorm rmsW rmsEps = ioRerun (\_ =>
  MkTensor (primBitlinearFwdHfQuant {ex} w.tensorPtr wScale
              x.tensorPtr bias.tensorPtr
              (if useRmsNorm then 1 else 0) rmsW.tensorPtr rmsEps)
           Nothing)

||| BitLinear forward: y = (W_ternary .* scale[:, None]) @ x + bias.
||| W and scale are NoGrad (BitNet b1.58 freezes both the ternary
||| weight and the per-row dequant scale); x and bias share an
||| arbitrary grad mode so gradients flow through the rest of the
||| chain. Inference-only in this commit (#411 B2); the training
||| path with STE backward is filed under #411 B5.
export
tBitlinearFwd : {0 ex : Executor} -> UserExecutorQuant ex =>
                Tensor [o, i] ex Ternary NoGrad ->
                Tensor [o] ex cDt NoGrad -> Tensor [i] ex cDt g ->
                Tensor [o] ex cDt g -> IO (Tensor [o] ex cDt g)
tBitlinearFwd w s x b = ioRerun (\_ =>
  MkTensor (primBitlinearFwd {ex} w.tensorPtr s.tensorPtr x.tensorPtr b.tensorPtr)
           Nothing)

||| Per-row absmean of a 2D float weight: scale[j] = mean_k(|w[j, k]|).
||| One half of the load-time BitNet quantization recipe (the other is
||| `tTernaryQuantWithScale2d` below). NoGrad; the pair runs once per
||| linear at checkpoint load. Matches `absmean_ternary_quant` in
||| `packages/pytorch/torch_ref/models/bitlinear.py`.
export
tAbsmeanPerRow2d : {0 ex : Executor} -> UserExecutorQuant ex =>
                   {o, i : Nat} ->
                   Tensor [o, i] ex cDt NoGrad ->
                   IO (Tensor [o] ex cDt NoGrad)
tAbsmeanPerRow2d w = ioRerun (\_ =>
  MkTensor (primAbsmeanPerRow2d {ex} w.tensorPtr) Nothing)

||| Quantize a 2D float weight to ternary via a per-row divisor:
||| t[j, k] = round(w[j, k] / scale[j]).clamp(-1, +1)
||| (rows with scale == 0 produce all-zero ternary, no /0 trap).
||| Storage is per-backend packed/int8 (see design-decisions.md
||| "Per-backend ternary storage"). NoGrad.
export
tTernaryQuantWithScale2d : {0 ex : Executor} -> UserExecutorQuant ex =>
                           {o, i : Nat} ->
                           Tensor [o, i] ex cDt NoGrad ->
                           Tensor [o] ex cDt NoGrad ->
                           IO (Tensor [o, i] ex Ternary NoGrad)
tTernaryQuantWithScale2d w scale = ioRerun (\_ =>
  MkTensor (primTernaryQuantWithScale2d {ex} w.tensorPtr scale.tensorPtr)
           Nothing)

||| Combined load-time recipe: per-row absmean + ternary-quant. Returns
||| (ternary_weight, scale) ready to drop into a `BitLinearState`. The
||| caller is responsible for the up-stream load of `w` from
||| safetensors / a host buffer / etc.
export
tAbsmeanTernaryQuant2d : {0 ex : Executor} -> UserExecutorQuant ex =>
                         {o, i : Nat} ->
                         Tensor [o, i] ex cDt NoGrad ->
                         IO (Tensor [o, i] ex Ternary NoGrad,
                             Tensor [o] ex cDt NoGrad)
tAbsmeanTernaryQuant2d w = do
  scale <- tAbsmeanPerRow2d w
  ternary <- tTernaryQuantWithScale2d w scale
  pure (ternary, scale)
