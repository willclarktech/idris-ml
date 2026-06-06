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
applyRmsNorm2dRaw : {0 d : Executor} -> UserExecutorTraining d => UserExecutorCore d =>
                    {seqLen, hidden : Nat} ->
                    (eps : Double) ->
                    Tensor [hidden] d dt g ->
                    Tensor [seqLen, hidden] d dt g ->
                    IO (Tensor [seqLen, hidden] d dt g)
applyRmsNorm2dRaw eps weight input = ioRerun (\_ =>
  let out = primRmsNorm2d {d} input.tensorPtr weight.tensorPtr eps
  in MkTensor out Nothing)
