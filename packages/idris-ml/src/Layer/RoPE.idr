module Layer.RoPE

import Data.Vect

import Device
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
--
-- The `applyRope` forward (rotation on a `[seq, headDim]` tensor) is
-- a Phase-4 follow-up in this file — it composes the existing 2D
-- primitives (`primNarrow`, `primMul`, `primAdd`, `primSub`,
-- `primConcat2dAxis1`) so no new core C surface is needed.


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
||| Verified against PyTorch by the Test.RoPE oracle table.
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
          origD   = cast {to=Double} origMaxPos
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
                  (0 d : Device) (0 dt : DType) (0 g : GradMode) where
  constructor MkRoPETables
  cosTable : Tensor [maxPos, div headDim 2] d dt g
  sinTable : Tensor [maxPos, div headDim 2] d dt g

||| Construct Llama-3 RoPE tables.
|||   `maxPos`   : maximum sequence position the tables cover (rows).
|||   `headDim`  : per-head dim (cols / 2).
|||   `base`     : rope_theta (Llama 3 = 500000).
|||   `scaling`  : NTK scaling params (use `llama3Scaling` for default).
export
buildLlamaRoPETables : {0 d : Device} -> UserDeviceTraining d =>
                       RuntimeDType dt => Linked d => Compatible d dt
                    => {maxPos, headDim : Nat}
                    -> (base : Double)
                    -> (scaling : LlamaRopeScaling)
                    -> IO (RoPETables maxPos headDim d dt NoGrad)
buildLlamaRoPETables base scaling = ioRerun (\_ =>
  let halfDimI = cast {to=Int} (div headDim 2)
      sLenI    = cast {to=Int} maxPos
      nElts    = halfDimI * sLenI
      freqs    = toList (llamaInvFreq headDim base scaling)
      cosBuf   = prim__allocDoubles nElts
      sinBuf   = prim__allocDoubles nElts
      cosBuf'  = writeCosTable cosBuf halfDimI sLenI freqs 0 0
      sinBuf'  = writeSinTable sinBuf halfDimI sLenI freqs 0 0
      cosPtr   = dtCreateState2d {d} {t=dt} sLenI halfDimI cosBuf' (deviceStreamTag {d})
      sinPtr   = dtCreateState2d {d} {t=dt} sLenI halfDimI sinBuf' (deviceStreamTag {d})
  in MkRoPETables (MkTensor cosPtr Nothing) (MkTensor sinPtr Nothing))
