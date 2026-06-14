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
module Transformers.Common

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

----------------------------------------------------------------------
-- Tied LM-head projection
----------------------------------------------------------------------

||| Tied LM-head: project the final hidden state `[seq, hidden]`
||| against the (shared) word-embedding weight `[vocab, hidden]` and
||| return per-position vocab logits `[seq, vocab]`. The bias is a
||| run-time zero `[vocab]` (calloc-backed, never registered as a
||| param), since the standard `tlinear2d` smart constructor expects
||| `y = x @ W^T + b` and HF's tied head has no bias.
|||
||| Used by HfLlama (forward + cached step), HfBitNet (forward), and
||| HfGpt2 (forward). Bert's MLM head has a real learnable bias and
||| does NOT use this — `applyMlmHead` keeps its own path.
export
projectTiedLmHead
  : {0 ex : Executor} -> UserExecutorTraining ex => UserExecutorCore ex
 => RuntimeDType dt => Linked ex => Compatible ex dt
 => {seqLen, vocab, hidden : Nat}
 -> (embedWeight : Tensor [vocab, hidden] ex dt g)
 -> (hFinal      : Tensor [seqLen, hidden] ex dt g)
 -> IO (Tensor [seqLen, vocab] ex dt g)
projectTiedLmHead embedWeight hFinal =
  let vI = cast {to=Int} vocab
      zBuf = prim__allocDoubles vI    -- calloc-backed → already zeros
      zeroBias : Tensor [vocab] ex dt g
      zeroBias = MkTensor (dtCreateState1d {ex} {t=dt} vI zBuf (deviceStreamTag {ex})) Nothing
  in tlinear2d embedWeight hFinal zeroBias

----------------------------------------------------------------------
-- Per-block fan-out helper
----------------------------------------------------------------------

||| `forBlocks n mk = mk 0 ++ mk 1 ++ … ++ mk (n - 1)`.
|||
||| Every adapter's `hfXxxParamNames` catalogue has the same skeleton:
||| an opening section (embeddings / wte / model.embed_tokens), then a
||| per-block fan-out, then a closing section (final norm). This
||| helper centralises the fan-out so each adapter only owns its per-
||| block component-name list — the strings themselves stay
||| per-adapter (load-bearing HF on-disk contract).
export
forBlocks : Nat -> (Nat -> List String) -> List String
forBlocks n mk = go n 0
  where
    go : Nat -> Nat -> List String
    go Z     _ = []
    go (S k) i = mk i ++ go k (S i)
