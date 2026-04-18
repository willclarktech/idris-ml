-- | GRU (Gated Recurrent Unit) layer.
-- |
-- | 2-gate recurrent: reset (r), update (z), candidate (n).
-- | Lighter than LSTM (no cell state, 3*o gates vs 4*o).
-- |
-- | Input: i, Output: o (hidden size), recurrent state: hidden [o].

module Layer.Gru

import Data.Vect

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
  applyGeneric _ _ = idris_crash "GRU: use tensor path"
  applyVar _ _ = idris_crash "GRU: use tensor path"

  applyVarTensor {i} {o} st inputT =
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

  nameLayer {i} {o} pfx (MkGru wih whh h _) =
    let wih' = nameLayer (pfx ++ "_wih") wih
        whh' = nameLayer (pfx ++ "_whh") whh
        oI = cast {to=Int} o
        hBuf = prim__allocDoubles oI
        hBuf' = packScalarVals hBuf 0 h
        hT = prim__createState1d oI hBuf'
    in MkGru wih' whh' h (Just hT)
    where
      packScalarVals : AnyPtr -> Int -> Vector n Variable -> AnyPtr
      packScalarVals buf _ (VTensor []) = buf
      packScalarVals buf idx (VTensor (STensor v :: rest)) =
        packScalarVals (prim__setDouble buf idx v.value) (idx + 1) (VTensor rest)

  layerPrefix _ = "gru"

  toDoubleLayer (MkGru wih whh h _) =
    MkGru (toDoubleLayer wih) (toDoubleLayer whh) (map value h) Nothing

  resetState {o} (MkGru wih whh h ht) =
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
