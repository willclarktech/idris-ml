module Layer.Ntm

import Data.Vect

import Compat.Random
import Device
import Init
import Layer.Core
import Layer.Linear
import Layer.Lstm
import Sampler
import Tensor


----------------------------------------------------------------------
-- NTM constants (mirror V1)
----------------------------------------------------------------------

public export
ShiftKernelSize : Nat
ShiftKernelSize = 3

public export
ReadParamWidth : Nat -> Nat
ReadParamWidth m = (m + ShiftKernelSize) + 3

public export
WriteParamWidth : Nat -> Nat
WriteParamWidth m = ReadParamWidth m + m


----------------------------------------------------------------------
-- NtmState — typed-surface NTM controller (Path C)
----------------------------------------------------------------------
--
-- Mirrors V1 `Layer/Ntm.idr`'s `applyVarTensor` path. The LSTM
-- controller ingests `cat(readOutput, input)`; its cell state feeds
-- read/write FCs which produce head parameters; addressing is
-- composed from generic Tensor primitives via `ntmReadHeadIdris`
-- / `ntmInterpWriteIdris` below.
--
-- Architecture:
--   inputSize -> Lstm(m+inputSize, h) -> readFc/writeFc/outputFc
--                                       -> Memory[n,m] -> outputSize
--
-- State (memT, readAddrT, writeAddrT, readOutT) is persistent C
-- tensor handles, reset between sequences via `resetNtmState`.

public export
data NtmState :
  (n : Nat) -> (m : Nat) -> (h : Nat) ->
  Nat -> Nat -> (0 _ : Device) -> Type
  where
  MkNtm :
    LstmState (m + i) h d ->
    LinearState h (ReadParamWidth m) d ->
    LinearState h (WriteParamWidth m) d ->
    LinearState (h + m) o d ->
    Maybe (Tensor [n, m] d) ->                          -- memory state
    Maybe (TVec n d) ->                               -- read addr
    Maybe (TVec n d) ->                               -- write addr
    Maybe (TVec m d) ->                               -- last read output
    NtmState n m h i o d


----------------------------------------------------------------------
-- Forward
----------------------------------------------------------------------

%default partial

-- Allocate a fresh persistent zero-state of the given size.
zeroState1d : (n : Nat) -> AnyPtr
zeroState1d n =
  let nI = cast {to=Int} n
      buf = prim__allocDoubles nI
  in prim__createState1d nI buf

zeroState2d : (n, m : Nat) -> AnyPtr
zeroState2d n m =
  let nI = cast {to=Int} n
      mI = cast {to=Int} m
      buf = prim__allocDoubles (nI * mI)
  in prim__createState2d nI mI buf

-- NTM read head decomposition (Graves et al. 2014, §3.3).
-- Returns (newReadAddr [n], readOutput [m]) given memory [n,m],
-- prevWeights [n], key [m], beta [], g [], gamma [], shift [k].
%inline
ntmReadHeadIdris : (memT, prevWT, keyT, betaT, gT, gammaT, shiftT : AnyPtr) ->
                   (AnyPtr, AnyPtr)
ntmReadHeadIdris memT prevWT keyT betaT gT gammaT shiftT =
  let -- 1. Content addressing: cosine sim per memory row vs key.
      keyT2d        = prim__unsqueeze keyT 0           -- [1, m]
      cosScoresT    = prim__cosineSimilarity memT keyT2d 1   -- [n]
      scaledScoresT = prim__mul betaT cosScoresT       -- broadcast [] × [n]
      contentWT     = prim__softmax scaledScoresT 0    -- [n]
      -- 2. Interpolation: g · content + (1 - g) · prev
      oneMinusG     = prim__addScalar (prim__neg gT) 1.0
      interpT       = prim__add (prim__mul gT contentWT)
                                (prim__mul oneMinusG prevWT)
      -- 3. Circular shift convolution.
      shiftedT      = prim__conv1dCircular interpT shiftT
      -- 4. Sharpening: pow(max(x, 1e-10), gamma); then normalize.
      shiftedClampedT = prim__clampMin shiftedT 1.0e-10
      poweredT      = prim__pow shiftedClampedT gammaT
      normSumT      = prim__addScalar (prim__sum poweredT) 1.0e-10
      focusedT      = prim__div poweredT normSumT
      -- 5. Read: focused [n] @ memory [n,m] -> [m]
      readOutT      = prim__matmul focusedT memT
  in (focusedT, readOutT)

-- NTM interpolated write (Graves et al. 2014, §3.3).
-- memory' = memory + outer(weights, addVector). The Graves erase term
-- is omitted to match the existing C `tensor_ntm_interp_write` (which
-- has only ever implemented add, not erase).
%inline
ntmInterpWriteIdris : (memT, weightsT, addVecT : AnyPtr) -> AnyPtr
ntmInterpWriteIdris memT weightsT addVecT =
  prim__add memT (prim__outer weightsT addVecT)

export
applyNtm : {n, m, h, i, o : Nat} ->
             NtmState n m h i o d ->
             TVec i d ->
             (NtmState n m h i o d, TVec o d)
applyNtm {n} {m} {h} {i} {o}
           (MkNtm lstm readFc writeFc outputFc memT raT waT roT) input =
  let memTPtr = case memT of
                  Just t => t.tensorPtr
                  Nothing => zeroState2d n m
      raTPtr = case raT of
                 Just t => t.tensorPtr
                 Nothing => zeroState1d n
      waTPtr = case waT of
                 Just t => t.tensorPtr
                 Nothing => zeroState1d n
      roTPtr = case roT of
                 Just t => t.tensorPtr
                 Nothing => zeroState1d m
      -- 1. cat(readOut, input) -> [m + i]
      lstmInputPtr = prim__cat2 roTPtr input.tensorPtr
      lstmInputV = the (TVec (m + i) d) (MkTensor lstmInputPtr Nothing)
      -- 2. LSTM forward
      (updLstm, hiddenV) = applyLstm lstm lstmInputV
      -- 3. Extract cell tensor (post-LSTM cell state)
      cellPtr = case updLstm.cellT of
                  Just c => c.tensorPtr
                  Nothing => idris_crash "Ntm: cell tensor missing post-LSTM"
      -- Sub-layer weight tensor handles
      rfcW = readFc.weightT.tensorPtr
      rfcB = readFc.biasT.tensorPtr
      wfcW = writeFc.weightT.tensorPtr
      wfcB = writeFc.biasT.tensorPtr
      ofcW = outputFc.weightT.tensorPtr
      ofcB = outputFc.biasT.tensorPtr
      mI = cast {to=Int} m
      skI = cast {to=Int} ShiftKernelSize
      -- 4. Read FC: cell -> [ReadParamWidth m]
      readResultT = prim__linear rfcW cellPtr rfcB
      keyT = prim__narrow readResultT 0 0 mI
      shiftT = prim__softmax (prim__narrow readResultT 0 mI skI) 0
      betaT = prim__softplus (prim__select readResultT 0 (mI + skI))
      gT = prim__sigmoid (prim__select readResultT 0 (mI + skI + 1))
      gammaT = prim__addScalar (prim__softplus
                  (prim__select readResultT 0 (mI + skI + 2))) 1.0
      (newReadAddrT, newReadOutT) = ntmReadHeadIdris memTPtr raTPtr keyT betaT gT gammaT shiftT
      -- 5. Write FC: cell -> [WriteParamWidth m]
      writeResultT = prim__linear wfcW cellPtr wfcB
      rpw = cast {to=Int} (ReadParamWidth m)
      wKeyT = prim__narrow writeResultT 0 0 mI
      wShiftT = prim__softmax (prim__narrow writeResultT 0 mI skI) 0
      wBetaT = prim__softplus (prim__select writeResultT 0 (mI + skI))
      wGT = prim__sigmoid (prim__select writeResultT 0 (mI + skI + 1))
      wGammaT = prim__addScalar (prim__softplus
                   (prim__select writeResultT 0 (mI + skI + 2))) 1.0
      (newWriteAddrT, _) = ntmReadHeadIdris memTPtr waTPtr wKeyT wBetaT wGT wGammaT wShiftT
      addT = prim__narrow writeResultT 0 rpw mI
      newMemT = ntmInterpWriteIdris memTPtr newWriteAddrT addT
      -- 6. Output FC: cat(hidden, readOut) -> [o]
      concatPtr = prim__cat2 hiddenV.tensorPtr newReadOutT
      outputPtr = prim__linear ofcW concatPtr ofcB
  in ( MkNtm updLstm readFc writeFc outputFc
        (Just (MkTensor newMemT Nothing))
        (Just (MkTensor newReadAddrT Nothing))
        (Just (MkTensor newWriteAddrT Nothing))
        (Just (MkTensor newReadOutT Nothing))
     , MkTensor outputPtr Nothing )


----------------------------------------------------------------------
-- Constructor
----------------------------------------------------------------------

-- Memory init: small constant (1e-6) per V1 NTM convention (Collier
-- & Beel 2018 — improves stability vs zero init).
fillConst : AnyPtr -> Int -> Int -> Double -> AnyPtr
fillConst buf _ 0 _ = buf
fillConst buf off i v =
  fillConst (prim__setDouble buf off v) (off + 1) (i - 1) v

||| Build an `NtmState n m h inputSize outputSize CPU` with default
||| init: LSTM weights via Xavier (Lstm default), FC layers Xavier
||| (Linear default), memory init to 1e-6 across [n,m], all
||| addresses and read output zero. State tensors are persistent.
export
ntmLayer : {n, m, h, i, o : Nat} ->
             (paramPrefix : String) ->
             IO (NtmState n m h i o CPU)
ntmLayer pfx = do
  lstm <- lstmLayer {i = m + i} {o = h} (pfx ++ "_lstm")
  rfc  <- linearLayer {i = h} {o = ReadParamWidth m} (pfx ++ "_readFc")
  wfc  <- linearLayer {i = h} {o = WriteParamWidth m} (pfx ++ "_writeFc")
  ofc  <- linearLayer {i = h + m} {o = o} (pfx ++ "_outputFc")
  let nI = cast {to=Int} n
      mI = cast {to=Int} m
      memBuf = fillConst (prim__allocDoubles (nI * mI)) 0 (nI * mI) 1.0e-6
      memT : Tensor [n, m] CPU
      memT = MkTensor (prim__createState2d nI mI memBuf) Nothing
      raT : TVec n CPU
      raT = MkTensor (zeroState1d n) Nothing
      waT : TVec n CPU
      waT = MkTensor (zeroState1d n) Nothing
      roT : TVec m CPU
      roT = MkTensor (zeroState1d m) Nothing
  pure $ MkNtm lstm rfc wfc ofc (Just memT) (Just raT) (Just waT) (Just roT)

||| Reset NTM state to fresh persistent zero/init tensors. Use
||| between training sequences (memory + addresses are not learned).
export
resetNtmState : {n, m, h : Nat} -> NtmState n m h i o d -> NtmState n m h i o d
resetNtmState (MkNtm lstm rfc wfc ofc _ _ _ _) =
  let nI = cast {to=Int} n
      mI = cast {to=Int} m
      memBuf = fillConst (prim__allocDoubles (nI * mI)) 0 (nI * mI) 1.0e-6
      memT : Tensor [n, m] _
      memT = MkTensor (prim__createState2d nI mI memBuf) Nothing
      raT : TVec n _
      raT = MkTensor (zeroState1d n) Nothing
      waT : TVec n _
      waT = MkTensor (zeroState1d n) Nothing
      roT : TVec m _
      roT = MkTensor (zeroState1d m) Nothing
  in MkNtm (resetLstmState lstm) rfc wfc ofc
             (Just memT) (Just raT) (Just waT) (Just roT)


----------------------------------------------------------------------
-- LayerLike instance
----------------------------------------------------------------------

public export
{n, m, h : Nat} ->
  LayerLike (NtmState n m h) where
  applyVar st@(MkNtm _ _ _ _ _ _ _ _) input = applyNtm st input
  layerPrefix _ = "ntm"
  resetState st = resetNtmState st

export
ntmLayerAny : {n, m, h, i, o : Nat} ->
                (paramPrefix : String) ->
                IO (AnyLayer i o CPU)
ntmLayerAny pid =
  map (MkAnyLayer (NtmState n m h)) (ntmLayer {n} {m} {h} {i} {o} pid)
