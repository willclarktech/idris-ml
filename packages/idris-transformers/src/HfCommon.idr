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

import Device
import Tensor


----------------------------------------------------------------------
-- 2D RMSNorm (per-row fold)
----------------------------------------------------------------------

||| One row of `applyRmsNorm2dRaw`. Lifted to top-level so its body
||| elaborates once at module compile, not per call site of the outer
||| `applyRmsNorm2dRaw` (which is called O(numLayers × 4) times inside
||| the recursive decoder-block chain).
private
rmsNorm2dProcessRow : {0 d : Device} -> UserDeviceLinear d =>
                      (inPtr : AnyPtr) -> (wPtr : AnyPtr) ->
                      (hD : Double) -> (eps : Double) ->
                      (r : Int) -> AnyPtr
rmsNorm2dProcessRow inPtr wPtr hD eps r =
  let row     = primNarrow {d} inPtr 0 r 1                -- [1, hidden]
      sq      = primMul {d} row row
      tot     = primSum {d} sq
      mean    = primMulScalar {d} tot (1.0 / hD)
      meanEps = primAddScalar {d} mean eps
      rms     = primSqrt {d} meanEps
      normed  = primDiv {d} row rms
      scaled  = primMul {d} normed wPtr                   -- broadcasts [hidden]
  in scaled

||| Row-folding helper for `applyRmsNorm2dRaw`. Lifted to top-level
||| (see `rmsNorm2dProcessRow`).
private
rmsNorm2dFoldRows : {0 d : Device} -> UserDeviceLinear d =>
                    (inPtr : AnyPtr) -> (wPtr : AnyPtr) ->
                    (hD : Double) -> (eps : Double) ->
                    (seqLenI : Int) -> (r : Int) -> (acc : AnyPtr) -> AnyPtr
rmsNorm2dFoldRows inPtr wPtr hD eps seqLenI r acc =
  if r >= seqLenI
    then acc
    else rmsNorm2dFoldRows {d} inPtr wPtr hD eps seqLenI (r + 1)
           (primCat2 {d} acc (rmsNorm2dProcessRow {d} inPtr wPtr hD eps r))

||| Per-position RmsNorm on a `[seqLen, hidden]` tensor. The weight is
||| a raw `[hidden]` tensor; per-arch wrappers in `HfLlama` /
||| `HfBitNet` thin-wrap this to take their own RmsNorm record types.
|||
||| Loops over rows because tape's `primSumDim` is a stub
||| (full-reduction); the 1D RmsNorm formula composes cleanly per-row.
||| Each row pays ~7 primitive calls. A fused 2D `tensor_rms_norm` C
||| primitive is the natural perf follow-up (filed as `#4 Fusion 1`).
export
applyRmsNorm2dRaw : {0 d : Device} -> UserDeviceTraining d => UserDeviceCore d =>
                    {seqLen, hidden : Nat} ->
                    (eps : Double) ->
                    Tensor [hidden] d dt g ->
                    Tensor [seqLen, hidden] d dt g ->
                    IO (Tensor [seqLen, hidden] d dt g)
applyRmsNorm2dRaw {seqLen} {hidden} eps weight input = ioRerun (\_ =>
  let hD       = cast {to=Double} hidden
      inPtr    = input.tensorPtr
      wPtr     = weight.tensorPtr
      seqLenI  = cast {to=Int} seqLen
      out      = if seqLen == 0
                   then inPtr  -- impossible at well-typed call sites
                   else rmsNorm2dFoldRows {d} inPtr wPtr hD eps seqLenI 1
                          (rmsNorm2dProcessRow {d} inPtr wPtr hD eps 0)
  in MkTensor out Nothing)
