||| `Nn.RoPE` — Rotary Position Embeddings (Su et al 2021, Llama variant).
|||
||| RoPE has NO learnable parameters, so this is the `Nn`-surface home for
||| RoPE as a set of parameter-free free functions — NOT a `Module`/`Params`
||| instance. It relocates the table builders + forward rotation from
||| `Layer.RoPE` (which dies with the rest of `Layer/` once the migration
||| sweep completes) and additionally hosts `ropeAllHeadsFlat`, the
||| flat↔rank-3 reshape wrapper the HF model files (`Transformers.Llama` /
||| `Transformers.BitNet`) used to each define privately.
module Nn.RoPE

import Control.Linear.LIO as LIO
import Data.Vect

import Executor
import Tensor

----------------------------------------------------------------------
-- RoPE — Rotary Position Embeddings (Su et al 2021, Llama variant)
----------------------------------------------------------------------
--
-- Stateless layer (no learnable params). Precomputes cos / sin tables
-- of shape `[maxPos, headDim / 2]` at construction time; the forward
-- pass applies the rotation by indexing into those tables.
--
-- The Llama variant uses the "split-half" pair convention rather than
-- "interleaved": the rotation on a head-dim vector `q[0..d)` is
--
--   q'[i]       = q[i]       * cos[m, i] - q[i + d/2] * sin[m, i]
--   q'[i + d/2] = q[i + d/2] * cos[m, i] + q[i]       * sin[m, i]
--
-- where `m` is the absolute position and `i in 0..d/2`.
--
-- Llama 3 (and later) adds NTK-aware scaling so the model can extend
-- beyond the original 8192-token training context. The frequency
-- adjustment is the `_compute_llama3_parameters` formula from HF's
-- `modeling_llama.py`:
--
--   inv_freq[i] = 1 / base^(2i/d)
--   wavelen[i]  = 2 * pi / inv_freq[i]
--   low_band  = original_max_position / high_freq_factor
--   high_band = original_max_position / low_freq_factor
--
--   if wavelen < low_band:  freq[i] = inv_freq[i]              (high-freq: no scale)
--   if wavelen > high_band: freq[i] = inv_freq[i] / factor     (low-freq: full scale-down)
--   else:                   smooth interpolation between the two
--
-- Then cos[m, i] = cos(m * freq[i]), sin[m, i] = sin(m * freq[i]).
--
-- This module ships the TABLE BUILDER + LlamaRopeScaling config. The
-- table is host-side Double math; uploads as the destination dtype
-- (F32 / F64 / BF16) via dtCreateState2d. A paired Python oracle
-- (`packages/idris-transformers/scripts/save_rope_oracle.py`) validates
-- the table values against PyTorch's reference within 1e-12 F64.

----------------------------------------------------------------------
-- Config
----------------------------------------------------------------------

||| Llama 3+ NTK-aware RoPE scaling. The "no scaling" case (Llama 2 /
||| GPT-NeoX) is recoverable as `MkRopeScaling 1.0 1.0 1.0 0` — see
||| `noScaling` below.
public export
record LlamaRopeScaling where
  constructor MkRopeScaling
  factor              : Double   -- HF "factor" — divide low-freq by this
  lowFreqFactor       : Double   -- HF "low_freq_factor"
  highFreqFactor      : Double   -- HF "high_freq_factor"
  originalMaxPosition : Nat      -- HF "original_max_position"

||| Llama 3.2 defaults: factor=32, low=1.0, high=4.0, orig=8192.
public export
llama3Scaling : LlamaRopeScaling
llama3Scaling = MkRopeScaling 32.0 1.0 4.0 8192

||| Plain RoPE without NTK scaling — produces inv_freq[i] unchanged.
||| `applyLlamaFreqScaling` reduces to identity when factor=1.
public export
noScaling : LlamaRopeScaling
noScaling = MkRopeScaling 1.0 1.0 4.0 8192

----------------------------------------------------------------------
-- Frequency computation (host-side Double math)
----------------------------------------------------------------------

-- pi : Double. Idris-2 prelude doesn't ship Math.Pi at a convenient
-- name; compute via 4 * atan(1) (which is exact to F64 ulps).
pi64 : Double
pi64 = 4.0 * atan 1.0

||| Base inverse-frequency for RoPE position m, dimension index i:
|||   inv_freq[i] = 1 / base^(2i/d)
|||
||| Returns a Vect of length headDim/2 (the unique frequency table).
||| Pure host-side computation; no Tensor / FFI involvement.
export
baseInvFreq : (headDim : Nat) -> (base : Double) -> Vect (div headDim 2) Double
baseInvFreq headDim base =
  -- Pin `len` via the signature's `div headDim 2` directly so Idris
  -- doesn't have to reconcile a let-bound `halfDim = div headDim 2`
  -- against the return-type `div headDim 2` (they don't unify
  -- automatically — same gotcha as `length idsList` in the GPT-2
  -- example's genOneStep).
  let dD = cast {to=Double} headDim
  in tabulate {len=div headDim 2} (\fi =>
       let i = cast {to=Double} (finToNat fi)
       in 1.0 / pow base (2.0 * i / dD))

||| Apply Llama 3's NTK-aware scaling to a single inv_freq value.
||| Pure-Idris translation of HF's `_compute_llama3_parameters`.
||| Verified against PyTorch by the Test.Nn.RoPE oracle table.
export
applyLlamaFreqScaling : LlamaRopeScaling -> (invFreq : Double) -> Double
applyLlamaFreqScaling (MkRopeScaling factor lowFF highFF origMaxPos) invFreq =
  -- factor=1 is the "no-op" path — identity, regardless of band.
  -- Catches floating noise that would otherwise drift inv_freq by
  -- ~1 ulp due to the smooth-band interpolation.
  if factor == 1.0
    then invFreq
    else
      let wavelen = 2.0 * pi64 / invFreq
          origD    = cast {to=Double} origMaxPos
          lowBand  = origD / highFF
          highBand = origD / lowFF
      in if wavelen < lowBand
           then invFreq                                  -- high-freq: no scale
           else if wavelen > highBand
             then invFreq / factor                       -- low-freq: full scale
             else                                        -- medium-freq: smooth interp
               let smoothFactor = (origD / wavelen - lowFF) / (highFF - lowFF)
                   scaled       = invFreq / factor
               in (1.0 - smoothFactor) * scaled + smoothFactor * invFreq

||| Llama 3 inverse-frequency table after NTK scaling.
||| Shape: `Vect (headDim/2) Double`.
export
llamaInvFreq : (headDim : Nat) -> (base : Double) -> LlamaRopeScaling
            -> Vect (div headDim 2) Double
llamaInvFreq headDim base scaling =
  map (applyLlamaFreqScaling scaling) (baseInvFreq headDim base)

----------------------------------------------------------------------
-- cos / sin table builders (host-side, ready to upload as tensors)
----------------------------------------------------------------------

-- Write cos(m * freq[i]) at offset (m * halfDim + i) into the buffer.
-- Recursive over (pos, fi) — same pattern as Transformer.idr's writePE.
-- Caller materialises the buffer via prim__allocDoubles.
writeCosTable : AnyPtr -> (halfDim : Int) -> (sLen : Int)
             -> (freqs : List Double)
             -> (pos : Int) -> (i : Int)
             -> AnyPtr
writeCosTable buf halfDim sLen freqs pos i =
  if pos >= sLen then buf
  else if i >= halfDim then writeCosTable buf halfDim sLen freqs (pos + 1) 0
  else case getAt freqs i of
    Nothing  => buf  -- impossible; freqs has halfDim entries
    Just inv =>
      let val  = cos (cast {to=Double} pos * inv)
          buf' = prim__setDouble buf (pos * halfDim + i) val
      in writeCosTable buf' halfDim sLen freqs pos (i + 1)
  where
    getAt : List a -> Int -> Maybe a
    getAt []        _ = Nothing
    getAt (x :: xs) k = if k <= 0 then Just x else getAt xs (k - 1)

writeSinTable : AnyPtr -> (halfDim : Int) -> (sLen : Int)
             -> (freqs : List Double)
             -> (pos : Int) -> (i : Int)
             -> AnyPtr
writeSinTable buf halfDim sLen freqs pos i =
  if pos >= sLen then buf
  else if i >= halfDim then writeSinTable buf halfDim sLen freqs (pos + 1) 0
  else case getAt freqs i of
    Nothing  => buf
    Just inv =>
      let val  = sin (cast {to=Double} pos * inv)
          buf' = prim__setDouble buf (pos * halfDim + i) val
      in writeSinTable buf' halfDim sLen freqs pos (i + 1)
  where
    getAt : List a -> Int -> Maybe a
    getAt []        _ = Nothing
    getAt (x :: xs) k = if k <= 0 then Just x else getAt xs (k - 1)

||| Build `[maxPos, headDim/2]` cos and sin tables for RoPE rotation.
||| Both materialised as persistent-state tensors (`dtCreateState2d`)
||| so they survive `tape_reset` between training/inference calls.
|||
||| For Llama 3.2 1B (headDim=64, maxPos=131072 max but typically used
||| at 8192 for our small-context demos): table is 8192*32 = 262144
||| doubles per table = 2 MB at F64. Llama 3 max context (131072): 33 MB
||| per table. Both manageable.
public export
record RoPETables (maxPos : Nat) (headDim : Nat)
                  (0 ex : Executor) (0 dt : DType) (0 g : GradMode) where
  constructor MkRoPETables
  cosTable : Tensor [maxPos, div headDim 2] ex dt g
  sinTable : Tensor [maxPos, div headDim 2] ex dt g

||| Construct Llama-3 RoPE tables.
|||   `maxPos`   : maximum sequence position the tables cover (rows).
|||   `headDim`  : per-head dim (cols / 2).
|||   `base`     : rope_theta (Llama 3 = 500000).
|||   `scaling`  : NTK scaling params (use `llama3Scaling` for default).
export
buildLlamaRoPETables : KnownGrad g => {0 ex : Executor} -> Backend ex dt
                    => {maxPos, headDim : Nat}
                    -> (base : Double)
                    -> (scaling : LlamaRopeScaling)
                    -> IO (RoPETables maxPos headDim ex dt g)
buildLlamaRoPETables base scaling = do
  -- Build the (paramId-less) cos/sin state tables at WithGrad, then on the
  -- NoGrad branch weaken both so a NoGrad Llama is genuinely tape-free.
  tbl <- the (IO (RoPETables maxPos headDim ex dt WithGrad)) $ ioRerun (\_ =>
    let halfDimI = cast {to=Int} (div headDim 2)
        sLenI   = cast {to=Int} maxPos
        nElts   = halfDimI * sLenI
        freqs   = toList (llamaInvFreq headDim base scaling)
        cosBuf  = prim__allocDoubles nElts
        sinBuf  = prim__allocDoubles nElts
        cosBuf' = writeCosTable cosBuf halfDimI sLenI freqs 0 0
        sinBuf' = writeSinTable sinBuf halfDimI sLenI freqs 0 0
        cosPtr  = dtCreateState2d {ex} {t=dt} sLenI halfDimI cosBuf' (deviceStreamTag {ex})
        sinPtr  = dtCreateState2d {ex} {t=dt} sLenI halfDimI sinBuf' (deviceStreamTag {ex})
    in MkRoPETables (MkTensor cosPtr Nothing) (MkTensor sinPtr Nothing))
  case sgrad {g} of
    SWithGrad => pure tbl
    SNoGrad   => do let MkRoPETables cosT sinT = tbl
                    cosT' <- weakenGrad cosT
                    sinT' <- weakenGrad sinT
                    pure (MkRoPETables cosT' sinT')

----------------------------------------------------------------------
-- applyRope — rotate a [seq, headDim] tensor in place
----------------------------------------------------------------------
--
-- Llama's split-half pair convention: split the input's last axis
-- into halves, apply the 2D rotation per (firstHalf[i], secondHalf[i])
-- pair using cos[m, i] / sin[m, i] sliced for the current positions.
--
--   firstOut  = firstHalf  * cos - secondHalf * sin
--   secondOut = secondHalf * cos + firstHalf  * sin
--   out       = concat firstOut secondOut along axis=1
--
-- Composes existing 2D primitives — primNarrow (axis-aware), primMul /
-- primSub / primAdd (elementwise), and primConcat2dAxis1. No new core C
-- surface needed.
--
-- `positionOffset` is the absolute starting position. For prefill
-- (no KV cache, processing the full prompt) it's 0. For incremental
-- decode step k after a prefill of L tokens it's L + k - 1 (single-
-- token input). The cos/sin tables are sliced via primNarrow on axis
-- 0 to pull the [seq, halfDim] rows aligned with the current input.

||| Apply Llama-style RoPE to a single head's `[seq, headDim]` tensor.
||| Assumes `seq + positionOffset <= maxPos` (caller's responsibility).
||| Idris's type system can't catch this today without a runtime
||| bounds-check.
public export
applyRope : {0 ex : Executor} -> UserExecutorTraining ex =>
            {seq, headDim, maxPos : Nat} ->
            RoPETables maxPos headDim ex dt g ->
            (positionOffset : Nat) ->
            Tensor [seq, headDim] ex dt g ->
            IO (Tensor [seq, headDim] ex dt g)
applyRope {seq} {headDim} (MkRoPETables cosT sinT) positionOffset input = ioRerun (\_ =>
  let halfDimI = cast {to=Int} (div headDim 2)
      seqI    = cast {to=Int} seq
      offsetI = cast {to=Int} positionOffset
      inPtr   = input.tensorPtr
      cosPtr  = cosT.tensorPtr
      sinPtr  = sinT.tensorPtr
      -- Split the input along axis=1 (head_dim) into the two halves.
      firstHalf  = primNarrow {ex} inPtr 1 0        halfDimI
      secondHalf = primNarrow {ex} inPtr 1 halfDimI halfDimI
      -- Slice cos/sin tables to [seq, halfDim] starting at offset.
      cosSlice = primNarrow {ex} cosPtr 0 offsetI seqI
      sinSlice = primNarrow {ex} sinPtr 0 offsetI seqI
      -- Rotation per pair (firstHalf[m, i], secondHalf[m, i]).
      firstCos  = primMul {ex} firstHalf  cosSlice
      secondSin = primMul {ex} secondHalf sinSlice
      firstOut  = primSub {ex} firstCos secondSin
      secondCos = primMul {ex} secondHalf cosSlice
      firstSin  = primMul {ex} firstHalf  sinSlice
      secondOut = primAdd {ex} secondCos firstSin
      -- Concat halves back to [seq, headDim].
      result    = primConcat2dAxis1 {ex} firstOut secondOut
  in MkTensor result Nothing)

||| `L IO` twin of `applyRope`, for use inside a model `forward` block
||| without a `liftIO1` seam. Same deferral semantics; kept a free function
||| (RoPE holds no learnable parameter and its true signature — Q+K per-head
||| + a position offset — can't pass through `Module.forward`).
export
applyRopeL : {0 ex : Executor} -> UserExecutorTraining ex =>
             {seq, headDim, maxPos : Nat} ->
             RoPETables maxPos headDim ex dt g ->
             (positionOffset : Nat) ->
             Tensor [seq, headDim] ex dt g ->
             LIO.L IO (Tensor [seq, headDim] ex dt g)
applyRopeL tables positionOffset input = liftIO1 (applyRope tables positionOffset input)

----------------------------------------------------------------------
-- applyRopeAllHeads — vectorized rotation across the head axis
----------------------------------------------------------------------
--
-- Input layout `[seq, numHeads, headDim]`; output same shape. The
-- rotation math is identical to `applyRope`, lifted one rank via
-- cos/sin unsqueeze + broadcast:
--
--   cos / sin slice : [seq, halfDim]
--   reshape         : [seq, 1, halfDim]   (rank-3 view with unit head axis)
--   primMul against : [seq, numHeads, halfDim] half slices
--
-- numpy-style broadcasting fills the unit head dim. Replaces the
-- Idris-side per-head loop that, on Llama-3.2-1B (32 heads × 2 ops
-- × 16 layers), accounted for ~80% of forward op count — see
-- `docs/develop/perf-changes.md` 2026-05-30 entry on #399.
--
-- Caveat: relies on the elementwise primitives' rank-3 × rank-3
-- broadcast support. Tape's general_bcast path handles this (up to
-- MAX_BCAST_RANK=8); torch's `at::mul` is shape-generic; mlx's
-- `mx::multiply` is too.

||| All-heads variant of `applyRope`. Input layout
||| `[seq, numHeads, headDim]`; rotates each head's headDim chunk
||| via the same cos/sin tables broadcast across the head axis.
||| Caller's responsibility: `seq + positionOffset <= maxPos`.
public export
applyRopeAllHeads : {0 ex : Executor} -> UserExecutorTraining ex =>
                    {seq, numHeads, headDim, maxPos : Nat} ->
                    RoPETables maxPos headDim ex dt g ->
                    (positionOffset : Nat) ->
                    Tensor [seq, numHeads, headDim] ex dt g ->
                    IO (Tensor [seq, numHeads, headDim] ex dt g)
applyRopeAllHeads {seq} {numHeads} {headDim} (MkRoPETables cosT sinT) positionOffset input = ioRerun (\_ =>
  let halfDimI = cast {to=Int} (div headDim 2)
      seqI    = cast {to=Int} seq
      numHI   = cast {to=Int} numHeads
      headDI  = cast {to=Int} headDim
      offsetI = cast {to=Int} positionOffset
      inPtr   = input.tensorPtr
      cosPtr  = cosT.tensorPtr
      sinPtr  = sinT.tensorPtr
      -- Split [seq, numHeads, headDim] along axis=2 into halves
      firstHalf  = primNarrow {ex} inPtr 2 0        halfDimI  -- [seq, nH, halfDim]
      secondHalf = primNarrow {ex} inPtr 2 halfDimI halfDimI
      -- Slice cos/sin tables to [seq, halfDim] starting at offset
      cosSlice2 = primNarrow {ex} cosPtr 0 offsetI seqI
      sinSlice2 = primNarrow {ex} sinPtr 0 offsetI seqI
      -- Unsqueeze cos/sin to [seq, 1, halfDim] for broadcast against
      -- [seq, numHeads, halfDim] halves.
      cosSlice = primReshape3d {ex} cosSlice2 seqI 1 halfDimI
      sinSlice = primReshape3d {ex} sinSlice2 seqI 1 halfDimI
      -- Rotation per pair (cos/sin broadcast across head axis).
      firstCos  = primMul {ex} firstHalf  cosSlice
      secondSin = primMul {ex} secondHalf sinSlice
      firstOut  = primSub {ex} firstCos secondSin
      secondCos = primMul {ex} secondHalf cosSlice
      firstSin  = primMul {ex} firstHalf  sinSlice
      secondOut = primAdd {ex} secondCos firstSin
      -- Concat halves back along axis=2. No rank-3 concat primitive
      -- today — flatten to 2D, use primConcat2dAxis1, reshape back.
      flat       = seqI * numHI
      firstOut2  = primReshape2d {ex} firstOut  flat halfDimI
      secondOut2 = primReshape2d {ex} secondOut flat halfDimI
      concat2    = primConcat2dAxis1 {ex} firstOut2 secondOut2  -- [seq*nH, headDim]
      result     = primReshape3d {ex} concat2 seqI numHI headDI
  in MkTensor result Nothing)

||| `L IO` twin of `applyRopeAllHeads`, for use inside a model `forward`
||| block without a `liftIO1` seam. Same deferral semantics; free function
||| for the same reason as `applyRopeL`.
export
applyRopeAllHeadsL : {0 ex : Executor} -> UserExecutorTraining ex =>
                     {seq, numHeads, headDim, maxPos : Nat} ->
                     RoPETables maxPos headDim ex dt g ->
                     (positionOffset : Nat) ->
                     Tensor [seq, numHeads, headDim] ex dt g ->
                     LIO.L IO (Tensor [seq, numHeads, headDim] ex dt g)
applyRopeAllHeadsL tables positionOffset input =
  liftIO1 (applyRopeAllHeads tables positionOffset input)

----------------------------------------------------------------------
-- ropeAllHeadsFlat — flat [seq, numH*headDim] convenience wrapper
----------------------------------------------------------------------
--
-- The HF model files store Q/K projections as a flat [seq, numH*headDim]
-- matmul output; RoPE wants a rank-3 [seq, numH, headDim] view. This
-- wraps reshape-up → `applyRopeAllHeads` → reshape-back-down so the
-- model code stays at the flat 2D layout. The flat↔rank-3 reshapes are
-- metadata-only on torch + mlx (view-with-strides) and copy-free on
-- tape (shape metadata only).
--
-- Previously defined privately+identically in `HfLlama.idr` and
-- `HfBitNet.idr`; consolidated here so the migrated `Transformers.*`
-- files share one definition.

||| Apply all-heads RoPE to a flat `[seq, numH * headDim]` projection
||| pointer, returning a flat `[seq, numH * headDim]` pointer. Reshapes
||| up to rank-3, rotates via `applyRopeAllHeads`, reshapes back.
export
ropeAllHeadsFlat :
     {0 ex : Executor} -> UserExecutorTraining ex =>
     {seq, numH, headDim, maxPos : Nat} ->
     RoPETables maxPos headDim ex dt g ->
     (full : AnyPtr) ->                     -- [seq, numH * headDim]
     (sI, nHI, hdI : Int) ->
     (positionOffset : Nat) ->
     IO AnyPtr
ropeAllHeadsFlat {ex} {seq} {numH} {headDim} {maxPos} tables full sI nHI hdI offset = do
  full3 <- ioRerun (\_ =>
            the (Tensor [seq, numH, headDim] ex dt g)
                (MkTensor (primReshape3d {ex} full sI nHI hdI) Nothing))
  rot3 <- applyRopeAllHeads {seq} {numHeads=numH} {headDim} {maxPos} tables offset full3
  ioRerun (\_ => primReshape2d {ex} rot3.tensorPtr sI (nHI * hdI))

----------------------------------------------------------------------
-- applyRopeInverse — inverse rotation
----------------------------------------------------------------------
--
-- Llama-style RoPE rotates each `(q[i], q[i + d/2])` pair by angle
-- `θ = m · inv_freq[i]`. The forward rotation matrix is
--
--   [ cos  -sin ]
--   [ sin   cos ]
--
-- Its inverse — rotation by `-θ` — flips the sign of `sin`:
--
--   q[i]       = q'[i]       * cos[m, i] + q'[i + d/2] * sin[m, i]
--   q[i + d/2] = q'[i + d/2] * cos[m, i] - q'[i]       * sin[m, i]
--
-- i.e. `applyRopeInverse cos sin (applyRope cos sin x) ≡ x` up to
-- F64 rounding. Provided so round-trip / commutativity properties
-- can be expressed without FFI-driven Hedgehog generators.

||| Inverse of `applyRope` — rotates each `[seq, headDim]` pair by
||| `-θ` so that round-tripping returns the input within F64 ULP.
||| Caller's responsibility: same `seq + positionOffset <= maxPos`
||| bound as `applyRope`.
public export
applyRopeInverse : {0 ex : Executor} -> UserExecutorTraining ex =>
                   {seq, headDim, maxPos : Nat} ->
                   RoPETables maxPos headDim ex dt g ->
                   (positionOffset : Nat) ->
                   Tensor [seq, headDim] ex dt g ->
                   IO (Tensor [seq, headDim] ex dt g)
applyRopeInverse {seq} {headDim} (MkRoPETables cosT sinT) positionOffset input = ioRerun (\_ =>
  let halfDimI = cast {to=Int} (div headDim 2)
      seqI       = cast {to=Int} seq
      offsetI    = cast {to=Int} positionOffset
      inPtr      = input.tensorPtr
      cosPtr     = cosT.tensorPtr
      sinPtr     = sinT.tensorPtr
      firstHalf  = primNarrow {ex} inPtr 1 0        halfDimI
      secondHalf = primNarrow {ex} inPtr 1 halfDimI halfDimI
      cosSlice   = primNarrow {ex} cosPtr 0 offsetI seqI
      sinSlice   = primNarrow {ex} sinPtr 0 offsetI seqI
      -- Inverse rotation: sign-flipped relative to `applyRope`.
      firstCos  = primMul {ex} firstHalf  cosSlice
      secondSin = primMul {ex} secondHalf sinSlice
      firstOut  = primAdd {ex} firstCos secondSin       -- forward: primSub
      secondCos = primMul {ex} secondHalf cosSlice
      firstSin  = primMul {ex} firstHalf  sinSlice
      secondOut = primSub {ex} secondCos firstSin       -- forward: primAdd
      result    = primConcat2dAxis1 {ex} firstOut secondOut
  in MkTensor result Nothing)

||| All-heads variant of `applyRopeInverse`. Same inverse rotation,
||| lifted across `[seq, numHeads, headDim]` via cos/sin broadcast on
||| the head axis.
public export
applyRopeInverseAllHeads : {0 ex : Executor} -> UserExecutorTraining ex =>
                           {seq, numHeads, headDim, maxPos : Nat} ->
                           RoPETables maxPos headDim ex dt g ->
                           (positionOffset : Nat) ->
                           Tensor [seq, numHeads, headDim] ex dt g ->
                           IO (Tensor [seq, numHeads, headDim] ex dt g)
applyRopeInverseAllHeads {seq} {numHeads} {headDim} (MkRoPETables cosT sinT) positionOffset input = ioRerun (\_ =>
  let halfDimI = cast {to=Int} (div headDim 2)
      seqI       = cast {to=Int} seq
      numHI      = cast {to=Int} numHeads
      headDI     = cast {to=Int} headDim
      offsetI    = cast {to=Int} positionOffset
      inPtr      = input.tensorPtr
      cosPtr     = cosT.tensorPtr
      sinPtr     = sinT.tensorPtr
      firstHalf  = primNarrow {ex} inPtr 2 0        halfDimI
      secondHalf = primNarrow {ex} inPtr 2 halfDimI halfDimI
      cosSlice2  = primNarrow {ex} cosPtr 0 offsetI seqI
      sinSlice2  = primNarrow {ex} sinPtr 0 offsetI seqI
      cosSlice   = primReshape3d {ex} cosSlice2 seqI 1 halfDimI
      sinSlice   = primReshape3d {ex} sinSlice2 seqI 1 halfDimI
      -- Inverse rotation: sign-flipped relative to `applyRopeAllHeads`.
      firstCos   = primMul {ex} firstHalf  cosSlice
      secondSin  = primMul {ex} secondHalf sinSlice
      firstOut   = primAdd {ex} firstCos secondSin      -- forward: primSub
      secondCos  = primMul {ex} secondHalf cosSlice
      firstSin   = primMul {ex} firstHalf  sinSlice
      secondOut  = primSub {ex} secondCos firstSin      -- forward: primAdd
      flat       = seqI * numHI
      firstOut2  = primReshape2d {ex} firstOut  flat halfDimI
      secondOut2 = primReshape2d {ex} secondOut flat halfDimI
      concat2    = primConcat2dAxis1 {ex} firstOut2 secondOut2
      result     = primReshape3d {ex} concat2 seqI numHI headDI
  in MkTensor result Nothing)
