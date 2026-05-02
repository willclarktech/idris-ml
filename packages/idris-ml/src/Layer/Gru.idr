-- | GRU (Gated Recurrent Unit) layer.
-- |
-- | 2-gate recurrent: reset (r), update (z), candidate (n).
-- | Lighter than LSTM (no cell state, 3*o gates vs 4*o).
-- |
-- | Input: i, Output: o (hidden size), recurrent state: hidden [o].

module Layer.Gru

import Data.Nat
import Data.Vect

import Device
import Endofunctor
import Floating
import Init
import Layer.Core
import Layer.Linear
import Math
import Tensor
import Util
import Variable


----------------------------------------------------------------------
-- Gate splitting (mirrors Lstm.lstmSplitGates)
----------------------------------------------------------------------

coerceLastGate : {o : Nat} -> Vector (o + 0) ty -> Vector o ty
coerceLastGate {o} v = rewrite sym (plusZeroRightNeutral o) in v

||| Split the combined GRU gate vector into (z, r, n) gates. Order
||| matches the C kernel `tensor_gru_cell` (z first, r second, n
||| third). Note: the simplified GRU variant computes r but does NOT
||| use it to mask n — see `applyGeneric`.
export
gruSplitGates :
    {o : Nat} -> Vector (3 * o) ty
    -> (Vector o ty, Vector o ty, Vector o ty)
gruSplitGates {o} combined =
  let s1 = Tensor.splitAt o combined
      s2 = Tensor.splitAt o (snd s1)
  in (fst s1, fst s2, coerceLastGate (snd s2))


----------------------------------------------------------------------
-- GRU State
----------------------------------------------------------------------

public export
record GruState (inputSize : Nat) (outputSize : Nat) (ty : Type) where
  constructor MkGru
  wIH : LinearState inputSize (3 * outputSize) ty    -- input -> gates
  wHH : LinearState outputSize (3 * outputSize) ty   -- hidden -> gates
  hidden : Vector outputSize ty                        -- recurrent state
  hiddenTensor : Maybe AnyPtr


----------------------------------------------------------------------
-- LayerLike Instance
----------------------------------------------------------------------

%default partial
export
LayerLike GruState where
  -- Pure-Double forward path used by `evaluateRecurrent` /
  -- `calculateLossRecurrent` after `toDoubleNetwork`. Mirrors the C
  -- kernel `tensor_gru_cell` exactly: simplified GRU variant where r
  -- is computed but NOT used to mask n.
  applyGeneric {o} (MkGru wih whh h _) xs =
    let ihPart = matrixVectorMultiply (wih.weights) xs + (wih.bias)
        hhPart = matrixVectorMultiply (whh.weights) h  + (whh.bias)
        combined = ihPart + hhPart
        gates = gruSplitGates {o} combined
        zGate = fst gates
        nGate = snd (snd gates)  -- r (= fst (snd gates)) intentionally unused
        z = map sig zGate
        n = map tanh nGate
        ones = map (const (fromDouble 1.0)) (the (Vector o ty) zeros)
        newH = (ones - z) * n + z * h
    in (MkGru wih whh newH Nothing, newH)
    where
      sig : ty -> ty
      sig x = 1 / (1 + exp (-x))

  applyVar {d} {i} {o} st xs =
    let (VTensor xElems) = xs
        inputT = vecStackTensor {n=i} xElems
        (st', outT) = applyVarTensor st inputT
    in (st', VTensor $ tensorToScalars outT 0 o)

  applyVarTensor {d} {i} {o} st inputT =
    case (extractWeightTensor (wIH st), extractBiasTensor (wIH st),
          extractWeightTensor (wHH st), extractBiasTensor (wHH st),
          st.hiddenTensor) of
      (Just wihW, Just wihB, Just whhW, Just whhB, Just hT) =>
        let oI = cast {to=Int} o
            -- gates = W_ih @ x + b_ih + W_hh @ h + b_hh
            ihPart = tensorAdd (tensorMv wihW inputT) wihB
            hhPart = tensorAdd (tensorMv whhW hT) whhB
            combined = tensorAdd ihPart hhPart
            -- GRU cell: z,r,n gates -> new hidden
            newH = prim__gruCell combined hT oI
        in ({ hiddenTensor := Just newH } st, newH)
      _ => idris_crash "GRU: weight tensors not initialized"

  emapLayer f (MkGru wih whh h ht) =
    MkGru (emapLayer f wih) (emapLayer f whh) (map f h) ht

  showLayer {i} {o} _ = "GRU<" ++ show i ++ ":" ++ show o ++ ">"

  nameLayer {d} {i} {o} pfx (MkGru wih whh h _) =
    let wih' = nameLayer (pfx ++ "_wih") wih
        whh' = nameLayer (pfx ++ "_whh") whh
        oI = cast {to=Int} o
        hBuf = prim__allocDoubles oI
        hBuf' = packScalarVals hBuf 0 h
        hT = prim__createState1d oI hBuf'
    in MkGru wih' whh' h (Just hT)
    where
      packScalarVals : AnyPtr -> Int -> Vector n (Variable d) -> AnyPtr
      packScalarVals buf _ (VTensor []) = buf
      packScalarVals buf idx (VTensor (STensor v :: rest)) =
        packScalarVals (prim__setDouble buf idx v.value) (idx + 1) (VTensor rest)

  layerPrefix _ = "gru"

  toDoubleLayer {d} (MkGru wih whh h _) =
    MkGru (toDoubleLayer wih) (toDoubleLayer whh) (map value h) Nothing

  resetState {d} {o} (MkGru wih whh h ht) =
    case ht of
      Just t =>
        let oI = cast {to=Int} o
            buf = prim__allocDoubles oI
            buf' = packZeros buf 0 oI
            newHT = prim__createState1d oI buf'
        in MkGru wih whh h (Just newHT)
      Nothing => MkGru wih whh h Nothing
    where
      packZeros : AnyPtr -> Int -> Int -> AnyPtr
      packZeros buf idx n = if idx >= n then buf
        else packZeros (prim__setDouble buf idx 0.0) (idx + 1) n

  debugApply _ _ = idris_crash "GRU: use tensor path"


----------------------------------------------------------------------
-- Constructor
----------------------------------------------------------------------

||| Create a GRU layer with Xavier initialization.
export
gruLayer : {i, o : Nat} -> (Num ty, FromDouble ty) => IO (AnyLayer i o ty)
gruLayer = do
  wih <- mkLinear {i, o = 3 * o}
  whh <- mkLinear {i = o, o = 3 * o}
  let h = the (Vector o ty) zeros
  pure $ MkAnyLayer GruState (MkGru wih whh h Nothing)
