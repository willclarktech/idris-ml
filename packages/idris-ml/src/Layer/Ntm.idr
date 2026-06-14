module Layer.Ntm

import Data.Vect

import Compat.Random
import Executor
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
  Nat -> Nat -> (0 _ : Executor) -> (0 _ : DType) -> (0 _ : GradMode) -> Type
  where
  MkNtm :
    LstmState (m + i) h ex dt g ->
    LinearState h (ReadParamWidth m) ex dt g ->
    LinearState h (WriteParamWidth m) ex dt g ->
    LinearState (h + m) o ex dt g ->
    TVec (m * n) ex dt g ->                          -- memoryInit (LEARNED, raw flat)
    TVec m ex dt g ->                                -- initialReadOut (Kaiming, NON-learned)
    Maybe (Tensor [n, m] ex dt g) ->                          -- memory state
    Maybe (TVec n ex dt g) ->                               -- read addr
    Maybe (TVec n ex dt g) ->                               -- write addr
    Maybe (TVec m ex dt g) ->                               -- last read output
    NtmState n m h i o ex dt g

----------------------------------------------------------------------
-- Forward
----------------------------------------------------------------------

%default partial

-- Allocate a fresh per-sequence zero-state of the given size. Refcount-
-- managed: the Tensor lives as long as tape entries reference it and any
-- Idris-wrapped Tensor handle is alive; freed once both let go. Without
-- this management the per-sequence state leaks unboundedly across eval-
-- phase forwards on mlx (see docs/develop/tensor-lifecycle.md).
zeroState1d : {0 ex : Executor} -> Backend ex dt => (n : Nat) -> AnyPtr
zeroState1d n =
  let nI = cast {to=Int} n
      buf = prim__allocDoubles nI
  in dtCreateState1d {ex} {t=dt} nI buf (deviceStreamTag {ex})

zeroState2d : {0 ex : Executor} -> Backend ex dt => (n, m : Nat) -> AnyPtr
zeroState2d n m =
  let nI = cast {to=Int} n
      mI  = cast {to=Int} m
      buf = prim__allocDoubles (nI * mI)
  in dtCreateState2d {ex} {t=dt} nI mI buf (deviceStreamTag {ex})

-- NTM read head decomposition (Graves et al. 2014, §3.3).
-- Returns (newReadAddr [n], readOutput [m]) given memory [n,m],
-- prevWeights [n], key [m], beta [], g [], gamma [], shift [k].
%inline
ntmReadHeadIdris : {0 ex : Executor} -> UserExecutorTraining ex =>
                   (memT, prevWT, keyT, betaT, gT, gammaT, shiftT : AnyPtr) ->
                   (AnyPtr, AnyPtr)
ntmReadHeadIdris memT prevWT keyT betaT gT gammaT shiftT =
  let -- 1. Content addressing: cosine sim per memory row vs key.
      keyT2d        = primUnsqueeze {ex} keyT 0           -- [1, m]
      cosScoresT    = primCosineSimilarity {ex} memT keyT2d 1   -- [n]
      scaledScoresT = primMul {ex} betaT cosScoresT       -- broadcast [] × [n]
      contentWT     = primSoftmax {ex} scaledScoresT 0    -- [n]
      -- 2. Interpolation: g · content + (1 - g) · prev
      oneMinusG = primAddScalar {ex} (primNeg {ex} gT) 1.0
      interpT   = primAdd {ex} (primMul {ex} gT contentWT)
                                (primMul {ex} oneMinusG prevWT)
      -- 3. Circular shift convolution.
      shiftedT      = primConv1dCircular {ex} interpT shiftT
      -- 4. Sharpening: pow(max(x, 1e-10), gamma); then normalize.
      shiftedClampedT = primClampMin {ex} shiftedT 1.0e-10
      poweredT        = primPow {ex} shiftedClampedT gammaT
      normSumT        = primAddScalar {ex} (primSum {ex} poweredT) 1.0e-10
      focusedT        = primDiv {ex} poweredT normSumT
      -- 5. Read: focused [n] @ memory [n,m] -> [m]
      readOutT      = primMatmul {ex} focusedT memT
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
ntmInterpWriteIdris : {0 ex : Executor} -> UserExecutorTraining ex => {n : Nat} -> (memT, weightsT, addVecT : AnyPtr) -> AnyPtr
ntmInterpWriteIdris {n} memT weightsT addVecT =
  let writeAdd = primOuter {ex} weightsT addVecT              -- (n,m) — w[i]*a[j]
      wCol = primReshape2d {ex} weightsT (cast n) 1       -- (n,1) view of w
      keep = primAddScalar {ex} (primNeg {ex} wCol) 1.0      -- (n,1) — 1-w[i]
      kept = primMul {ex} keep memT                       -- (n,m) — (n,1)·(n,m) bcast
  in primAdd {ex} kept writeAdd

export
applyNtm : {0 ex : Executor} -> Backend ex dt => {n, m, h, i, o : Nat} ->
             NtmState n m h i o ex dt g ->
             TVec i ex dt g ->
             IO (NtmState n m h i o ex dt g, TVec o ex dt g)
applyNtm {n} {m} {h} {i} {o}
           (MkNtm lstm readFc writeFc outputFc memInitT initReadOutT memT raT waT roT) input = do
  let nI = cast {to=Int} n
      mI         = cast {to=Int} m
      initMemPtr = primReshape2d {ex} (primSigmoid {ex} memInitT.tensorPtr) nI mI
      memTPtr    = case memT of
                  Just t  => t.tensorPtr
                  Nothing => initMemPtr
      raTPtr = case raT of
                 Just t  => t.tensorPtr
                 Nothing => zeroState1d {ex} {dt} n
      waTPtr = case waT of
                 Just t  => t.tensorPtr
                 Nothing => zeroState1d {ex} {dt} n
      roTPtr = case roT of
                 Just t  => t.tensorPtr
                 Nothing => initReadOutT.tensorPtr
      lstmInputPtr = primCat2 {ex} roTPtr input.tensorPtr
      lstmInputV   = the (TVec (m + i) ex dt g) (MkTensor lstmInputPtr Nothing)
  -- 2. LSTM forward (IO)
  (updLstm, hiddenV) <- applyLstm lstm lstmInputV
  let cellPtr = case updLstm.cellT of
                  Just c  => c.tensorPtr
                  Nothing => idris_crash "Ntm: cell tensor missing post-LSTM"
      rfcW        = readFc.weightT.tensorPtr
      rfcB        = readFc.biasT.tensorPtr
      wfcW        = writeFc.weightT.tensorPtr
      wfcB        = writeFc.biasT.tensorPtr
      ofcW        = outputFc.weightT.tensorPtr
      ofcB        = outputFc.biasT.tensorPtr
      skI         = cast {to=Int} ShiftKernelSize
      readResultT = primLinear {ex} rfcW cellPtr rfcB
      keyT        = primNarrow {ex} readResultT 0 0 mI
      shiftT      = primSoftmax {ex} (primNarrow {ex} readResultT 0 mI skI) 0
      betaT       = primSoftplus {ex} (primSelect {ex} readResultT 0 (mI + skI))
      gT          = primSigmoid {ex} (primSelect {ex} readResultT 0 (mI + skI + 1))
      gammaT = primAddScalar {ex} (primSoftplus {ex}
                  (primSelect {ex} readResultT 0 (mI + skI + 2))) 1.0
      (newReadAddrT, newReadOutT) = ntmReadHeadIdris {ex} memTPtr raTPtr keyT betaT gT gammaT shiftT
      writeResultT                = primLinear {ex} wfcW cellPtr wfcB
      rpw                         = cast {to=Int} (ReadParamWidth m)
      wKeyT                       = primNarrow {ex} writeResultT 0 0 mI
      wShiftT                     = primSoftmax {ex} (primNarrow {ex} writeResultT 0 mI skI) 0
      wBetaT                      = primSoftplus {ex} (primSelect {ex} writeResultT 0 (mI + skI))
      wGT                         = primSigmoid {ex} (primSelect {ex} writeResultT 0 (mI + skI + 1))
      wGammaT = primAddScalar {ex} (primSoftplus {ex}
                   (primSelect {ex} writeResultT 0 (mI + skI + 2))) 1.0
      (newWriteAddrT, _) = ntmReadHeadIdris {ex} memTPtr waTPtr wKeyT wBetaT wGT wGammaT wShiftT
      addT               = primNarrow {ex} writeResultT 0 rpw mI
      newMemT            = ntmInterpWriteIdris {ex} {n} memTPtr newWriteAddrT addT
      concatPtr          = primCat2 {ex} hiddenV.tensorPtr newReadOutT
      outputPtr          = primLinear {ex} ofcW concatPtr ofcB
  pure ( MkNtm updLstm readFc writeFc outputFc memInitT initReadOutT
          (Just (MkTensor newMemT Nothing))
          (Just (MkTensor newReadAddrT Nothing))
          (Just (MkTensor newWriteAddrT Nothing))
          (Just (MkTensor newReadOutT Nothing))
       , MkTensor outputPtr Nothing )

----------------------------------------------------------------------
-- Constructor
----------------------------------------------------------------------

||| Build an `NtmState n m h inputSize outputSize TapeExecutor` matching the
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
ntmLayer : Backend ex dt => {n, m, h, i, o : Nat} ->
             (paramPrefix : String) ->
             IO (NtmState n m h i o ex dt WithGrad)
ntmLayer pfx = do
  lstm <- lstmLayer {i = m + i} {o = h} (pfx ++ "_lstm")
  -- xavierGain 1.4 uniform → equivalent normal std = 1.4 * sqrt(2/(i+o)).
  -- bias ~ N(0, 0.0001).
  let xavStd : (i, o : Nat) -> Double
      xavStd i' o' = 1.4 * sqrt (2.0 / cast {to=Double} (i' + o'))
  rfc  <- mkLinearWith {i = h} {o = ReadParamWidth m}
            (pfx ++ "_readFc")  (xavStd h (ReadParamWidth m))  0.0001
  wfc  <- mkLinearWith {i = h} {o = WriteParamWidth m}
            (pfx ++ "_writeFc") (xavStd h (WriteParamWidth m)) 0.0001
  -- Output FC: PyTorch nn.Linear default (kaiming-uniform-as-normal),
  -- std = 1/sqrt(fan_in) ≈ 1/sqrt(h+m); bias ~ N(0, 0.0001).
  ofc  <- mkLinearWith {i = h + m} {o = o}
            (pfx ++ "_outputFc") (1.0 / sqrt (cast {to=Double} (h + m))) 0.0001
  -- memoryInit: learnable [m * n] (flat shape; underlying shape is [n, m]
  -- where fan_in=m, fan_out=n). Xavier-normal-via-uniform std = sqrt(2/(m+n)).
  let memStd = sqrt (2.0 / cast {to=Double} (m + n))
      mI  = cast {to=Int} m
  memInitT <- tparam1dNormal {n = m * n} (pfx ++ "_memoryInit") 0.0 memStd
  -- initialReadOut: PyTorch default kaiming_uniform on (1, m) — fan_in=m.
  -- Sampled once, non-learnable (state tensor handle).
  let iroBound = 1.0 / prim__doubleSqrt (cast m)
  iroVals <- traverse (\_ => randomRIO (-iroBound, iroBound)) (Vect.replicate m ())
  let iroBuf = prim__allocDoubles mI
      iroBuf' = packDoubles iroBuf 0 iroVals
      initReadOutT : TVec m ex dt WithGrad
      initReadOutT = MkTensor (dtCreateState1d {ex} {t=dt} mI iroBuf' (deviceStreamTag {ex})) Nothing
  -- Per-sequence runtime state starts as Nothing — applyNtm computes the
  -- actual initial memT and roT from memInitT/initReadOutT on first call.
  pure $ MkNtm lstm rfc wfc ofc memInitT initReadOutT
                Nothing Nothing Nothing Nothing

||| Reset NTM state between sequences. Keeps the learned `memInitT` /
||| Kaiming-fixed `initReadOutT` parameters; clears per-sequence runtime
||| state so the next `applyNtm` re-derives initial memory + read output.
export
resetNtmState : {n, m, h : Nat} -> {0 g : GradMode} -> NtmState n m h i o ex dt g -> NtmState n m h i o ex dt g
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
  layerPrefix _                                 = "ntm"
  resetState st                                 = resetNtmState st

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
    primIO (primSetRequiresGrad {ex} memInit.tensorPtr 1)
    primIO (primSetRequiresGrad {ex} iro.tensorPtr 1)
    case mem of
      Nothing => pure ()
      Just t  => primIO (primSetRequiresGrad {ex} t.tensorPtr 1)
    case ra of
      Nothing => pure ()
      Just t  => primIO (primSetRequiresGrad {ex} t.tensorPtr 1)
    case wa of
      Nothing => pure ()
      Just t  => primIO (primSetRequiresGrad {ex} t.tensorPtr 1)
    case ro of
      Nothing => pure ()
      Just t  => primIO (primSetRequiresGrad {ex} t.tensorPtr 1)
    pure (MkNtm lstm' rfc' wfc' ofc'
                (retypeGrad memInit) (retypeGrad iro)
                (map retypeGrad mem) (map retypeGrad ra)
                (map retypeGrad wa) (map retypeGrad ro))

export
ntmLayerAny : Backend ex dt => {n, m, h, i, o : Nat} ->
                (paramPrefix : String) ->
                IO (AnyLayer i o ex dt WithGrad)
ntmLayerAny pid =
  map (MkAnyLayer (NtmState n m h)) (ntmLayer {n} {m} {h} {i} {o} pid)
