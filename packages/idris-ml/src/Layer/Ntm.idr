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
  Nat -> Nat -> (0 _ : Device) -> (0 _ : GradMode) -> Type
  where
  MkNtm :
    LstmState (m + i) h d g ->
    LinearState h (ReadParamWidth m) d g ->
    LinearState h (WriteParamWidth m) d g ->
    LinearState (h + m) o d g ->
    TVec (m * n) d g ->                          -- memoryInit (LEARNED, raw flat)
    TVec m d g ->                                -- initialReadOut (Kaiming, NON-learned)
    Maybe (Tensor [n, m] d g) ->                          -- memory state
    Maybe (TVec n d g) ->                               -- read addr
    Maybe (TVec n d g) ->                               -- write addr
    Maybe (TVec m d g) ->                               -- last read output
    NtmState n m h i o d g


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

-- NTM interpolation write (Graves et al. 2014, §3.3, with the Collier &
-- Beel 2018 single-vector simplification: no separate erase vector — the
-- write weight itself controls how much old memory to keep).
--
--   memory' = w·addVec + (1-w)·memory
--
-- Each row i of memory' is `w[i]*addVec + (1-w[i])*memory[i]`. Bounded
-- by construction: a row never grows beyond max(memory[i], addVec).
--
-- Mirrors `torch_ref/ntm/memory.py:write_memory`.
%inline
ntmInterpWriteIdris : {n : Nat} -> (memT, weightsT, addVecT : AnyPtr) -> AnyPtr
ntmInterpWriteIdris {n} memT weightsT addVecT =
  let writeAdd = prim__outer weightsT addVecT              -- (n,m) — w[i]*a[j]
      wCol     = prim__reshape2d weightsT (cast n) 1       -- (n,1) view of w
      keep     = prim__addScalar (prim__neg wCol) 1.0      -- (n,1) — 1-w[i]
      kept     = prim__mul keep memT                       -- (n,m) — (n,1)·(n,m) bcast
  in prim__add kept writeAdd

export
applyNtm : {0 d : Device} -> UserDeviceLinear d => {n, m, h, i, o : Nat} ->
             NtmState n m h i o d g ->
             TVec i d g ->
             (NtmState n m h i o d g, TVec o d g)
applyNtm {n} {m} {h} {i} {o}
           (MkNtm lstm readFc writeFc outputFc memInitT initReadOutT memT raT waT roT) input =
  let nI = cast {to=Int} n
      mI = cast {to=Int} m
      -- Initial memory at sequence start: sigmoid(memoryInit).reshape(n, m).
      -- Mirrors `torch_ref/ntm/layer.py:96`. Gradient flows through sigmoid+
      -- reshape back to the registered memInitT parameter.
      initMemPtr = prim__reshape2d (prim__sigmoid memInitT.tensorPtr) nI mI
      memTPtr = case memT of
                  Just t => t.tensorPtr
                  Nothing => initMemPtr
      raTPtr = case raT of
                 Just t => t.tensorPtr
                 Nothing => zeroState1d n
      waTPtr = case waT of
                 Just t => t.tensorPtr
                 Nothing => zeroState1d n
      -- Initial read output at sequence start: fixed Kaiming-uniform sample,
      -- non-learnable. Mirrors `torch_ref/ntm/layer.py:84-86, 102`.
      roTPtr = case roT of
                 Just t => t.tensorPtr
                 Nothing => initReadOutT.tensorPtr
      -- 1. cat(readOut, input) -> [m + i]
      lstmInputPtr = prim__cat2 roTPtr input.tensorPtr
      lstmInputV = the (TVec (m + i) d g) (MkTensor lstmInputPtr Nothing)
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
      newMemT = ntmInterpWriteIdris {n} memTPtr newWriteAddrT addT
      -- 6. Output FC: cat(hidden, readOut) -> [o]
      concatPtr = prim__cat2 hiddenV.tensorPtr newReadOutT
      outputPtr = prim__linear ofcW concatPtr ofcB
  in ( MkNtm updLstm readFc writeFc outputFc memInitT initReadOutT
        (Just (MkTensor newMemT Nothing))
        (Just (MkTensor newReadAddrT Nothing))
        (Just (MkTensor newWriteAddrT Nothing))
        (Just (MkTensor newReadOutT Nothing))
     , MkTensor outputPtr Nothing )


----------------------------------------------------------------------
-- Constructor
----------------------------------------------------------------------

||| Build an `NtmState n m h inputSize outputSize CPU` matching the
||| PyTorch reference's `NTMLayer.__init__` (`torch_ref/ntm/layer.py`)
||| line-for-line. All inits mirror PyTorch's `nn.init` calls:
|||
||| - LSTM controller:        Idris's `lstmLayer` default (LSTMCell default)
||| - read/write FC weights:  `xavier_uniform_(gain=1.4)`
||| - output FC weight:       `kaiming_uniform_` (PyTorch default ≈ LeCun)
||| - all FC biases:          `normal_(std=0.01)`
||| - memory_init param:      `xavier_uniform_(view(n, m))`, sigmoid'd at
|||                           sequence-start in `applyNtm`
||| - initial read output:    `kaiming_uniform_((1, m))`, non-learnable,
|||                           sampled once at construction
export
ntmLayer : {n, m, h, i, o : Nat} ->
             (paramPrefix : String) ->
             IO (NtmState n m h i o CPU WithGrad)
ntmLayer pfx = do
  lstm <- lstmLayer {i = m + i} {o = h} (pfx ++ "_lstm")
  rfc  <- mkLinearWith {i = h} {o = ReadParamWidth m}
            (pfx ++ "_readFc")  (xavierGain 1.4 uniform) (normal 0.0001)
  wfc  <- mkLinearWith {i = h} {o = WriteParamWidth m}
            (pfx ++ "_writeFc") (xavierGain 1.4 uniform) (normal 0.0001)
  ofc  <- mkLinearWith {i = h + m} {o = o}
            (pfx ++ "_outputFc") (ptKaimingDefault uniform) (normal 0.0001)
  -- memoryInit: shape (n, m) Xavier — fan_in=m, fan_out=n.
  let mnI = cast {to=Int} (m * n)
      mI  = cast {to=Int} m
  memInitVals <- traverse (\_ => xavier uniform m n) (Vect.replicate (m * n) ())
  let miBuf = prim__allocDoubles mnI
      miBuf' = packDoubles miBuf 0 memInitVals
      memInitT : TVec (m * n) CPU WithGrad
      memInitT = tparam1d (pfx ++ "_memoryInit") miBuf'
  -- initialReadOut: PyTorch default kaiming_uniform on (1, m) — fan_in=m.
  -- Sampled once, non-learnable (state tensor handle).
  let iroBound = 1.0 / prim__doubleSqrt (cast m)
  iroVals <- traverse (\_ => randomRIO (-iroBound, iroBound)) (Vect.replicate m ())
  let iroBuf = prim__allocDoubles mI
      iroBuf' = packDoubles iroBuf 0 iroVals
      initReadOutT : TVec m CPU WithGrad
      initReadOutT = MkTensor (prim__createState1d mI iroBuf') Nothing
  -- Per-sequence runtime state starts as Nothing — applyNtm computes the
  -- actual initial memT and roT from memInitT/initReadOutT on first call.
  pure $ MkNtm lstm rfc wfc ofc memInitT initReadOutT
                Nothing Nothing Nothing Nothing

||| Reset NTM state between sequences. Keeps the learned `memInitT` /
||| Kaiming-fixed `initReadOutT` parameters; clears per-sequence runtime
||| state so the next `applyNtm` re-derives initial memory + read output.
export
resetNtmState : {n, m, h : Nat} -> {0 g : GradMode} -> NtmState n m h i o d g -> NtmState n m h i o d g
resetNtmState (MkNtm lstm rfc wfc ofc memInitT initReadOutT _ _ _ _) =
  MkNtm (resetLstmState lstm) rfc wfc ofc memInitT initReadOutT
        Nothing Nothing Nothing Nothing


----------------------------------------------------------------------
-- LayerLike instance
----------------------------------------------------------------------

public export
{n, m, h : Nat} ->
  LayerLike (NtmState n m h) where
  applyVar st@(MkNtm _ _ _ _ _ _ _ _ _ _) input = applyNtm st input
  layerPrefix _ = "ntm"
  resetState st = resetNtmState st

  freezeLayer (MkNtm lstm rfc wfc ofc memInit iro mem ra wa ro) = do
    lstm'   <- freezeLayer lstm
    rfc'    <- freezeLayer rfc
    wfc'    <- freezeLayer wfc
    ofc'    <- freezeLayer ofc
    memInit' <- weakenGrad memInit
    iro'    <- weakenGrad iro
    mem' <- case mem of
      Nothing => pure Nothing
      Just t  => Just <$> weakenGrad t
    ra'  <- case ra of
      Nothing => pure Nothing
      Just t  => Just <$> weakenGrad t
    wa'  <- case wa of
      Nothing => pure Nothing
      Just t  => Just <$> weakenGrad t
    ro'  <- case ro of
      Nothing => pure Nothing
      Just t  => Just <$> weakenGrad t
    pure (MkNtm lstm' rfc' wfc' ofc' memInit' iro' mem' ra' wa' ro')

  unfreezeLayer (MkNtm lstm rfc wfc ofc memInit iro mem ra wa ro) = do
    lstm'   <- unfreezeLayer lstm
    rfc'    <- unfreezeLayer rfc
    wfc'    <- unfreezeLayer wfc
    ofc'    <- unfreezeLayer ofc
    primIO (prim__setRequiresGrad memInit.tensorPtr 1)
    primIO (prim__setRequiresGrad iro.tensorPtr 1)
    case mem of
      Nothing => pure ()
      Just t  => primIO (prim__setRequiresGrad t.tensorPtr 1)
    case ra of
      Nothing => pure ()
      Just t  => primIO (prim__setRequiresGrad t.tensorPtr 1)
    case wa of
      Nothing => pure ()
      Just t  => primIO (prim__setRequiresGrad t.tensorPtr 1)
    case ro of
      Nothing => pure ()
      Just t  => primIO (prim__setRequiresGrad t.tensorPtr 1)
    pure (MkNtm lstm' rfc' wfc' ofc'
                (retypeGrad memInit) (retypeGrad iro)
                (map retypeGrad mem) (map retypeGrad ra)
                (map retypeGrad wa) (map retypeGrad ro))

export
ntmLayerAny : {n, m, h, i, o : Nat} ->
                (paramPrefix : String) ->
                IO (AnyLayer i o CPU WithGrad)
ntmLayerAny pid =
  map (MkAnyLayer (NtmState n m h)) (ntmLayer {n} {m} {h} {i} {o} pid)
