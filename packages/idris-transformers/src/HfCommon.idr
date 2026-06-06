||| Shared helpers across HF-aligned model adapters
||| (`HfBert` / `HfGpt2` / `HfLlama` / `HfBitNet`).
|||
||| Per `packages/idris-transformers/CONVENTIONS.md`:
|||   - Param-name string literals stay per-adapter (load-bearing
|||     contract with the HF on-disk format).
|||   - Per-arch storage-shape catalogues stay per-adapter.
|||   - Pure-Idris composition that doesn't depend on either of the
|||     above can live here; the per-arch `Hf*.idr` modules import this
|||     module and thin-wrap the helpers behind their arch-specific
|||     newtype records.
|||
||| Don't add cross-imports between `Hf*` modules. If you need
||| something shared, lift it here.
module HfCommon

import Data.Vect

import Executor
import Tensor


----------------------------------------------------------------------
-- 2D RMSNorm (fused C primitive)
----------------------------------------------------------------------

||| Per-position RmsNorm on a `[seqLen, hidden]` tensor. The weight is
||| a raw `[hidden]` tensor; per-arch wrappers in `HfLlama` /
||| `HfBitNet` thin-wrap this to take their own RmsNorm record types.
|||
||| One fused FFI call (`primRmsNorm2d` on `UserExecutorTraining`) per
||| invocation. Replaces a per-row 7-primitive chain (narrow / mul /
||| sum / mul_scalar / add_scalar / sqrt / div / mul); ~600 tape entries
||| per Llama forward become ~32. Each backend's impl matches the HF
||| LlamaRMSNorm formula exactly:
|||     rstd_i = 1 / sqrt(mean(input[i, :]^2) + eps)
|||     out[i, j] = input[i, j] * rstd_i * weight[j]
export
applyRmsNorm2dRaw : {0 ex : Executor} -> UserExecutorTraining ex => UserExecutorCore ex =>
                    {seqLen, hidden : Nat} ->
                    (eps : Double) ->
                    Tensor [hidden] ex dt g ->
                    Tensor [seqLen, hidden] ex dt g ->
                    IO (Tensor [seqLen, hidden] ex dt g)
applyRmsNorm2dRaw eps weight input = ioRerun (\_ =>
  let out = primRmsNorm2d {ex} input.tensorPtr weight.tensorPtr eps
  in MkTensor out Nothing)


----------------------------------------------------------------------
-- Pre-norm decoder block skeleton
----------------------------------------------------------------------

||| The standard Llama-shaped pre-norm decoder block:
|||
|||   x'  = x  + attn(preAttnNorm(x))
|||   y   = x' + mlp(preMlpNorm(x'))
|||
||| Parameterised by the per-arch attention and MLP closures. Both
||| HfLlama and HfBitNet's `applyBlock` reduce to this skeleton; the
||| arch-specific bits (BitNet's `attn_sub_norm` / `ffn_sub_norm`,
||| different linear primitives, GQA tiling) live entirely inside the
||| `attn` / `mlp` closures the caller passes in.
export
decoderBlockPreNorm
  : {0 ex : Executor} -> UserExecutorTraining ex => UserExecutorCore ex
 => RuntimeDType dt => Linked ex => Compatible ex dt
 => {seq, hidden : Nat}
 -> (preAttnNorm : Tensor [seq, hidden] ex dt g -> IO (Tensor [seq, hidden] ex dt g))
 -> (attn        : Tensor [seq, hidden] ex dt g -> IO (Tensor [seq, hidden] ex dt g))
 -> (preMlpNorm  : Tensor [seq, hidden] ex dt g -> IO (Tensor [seq, hidden] ex dt g))
 -> (mlp         : Tensor [seq, hidden] ex dt g -> IO (Tensor [seq, hidden] ex dt g))
 -> Tensor [seq, hidden] ex dt g
 -> IO (Tensor [seq, hidden] ex dt g)
decoderBlockPreNorm preAttnNorm attn preMlpNorm mlp x = do
  xLn1 <- preAttnNorm x
  aOut <- attn xLn1
  xMid <- tadd x aOut
  xLn2 <- preMlpNorm xMid
  mOut <- mlp xLn2
  tadd xMid mOut
