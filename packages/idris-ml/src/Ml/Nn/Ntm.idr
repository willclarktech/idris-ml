||| `Ntm` — Neural Turing Machine (Graves/Wayne/Danihelka 2014, with the
||| Collier & Beel 2018 single-vector write) on the v1 `Nn` surface,
||| implementing `Recurrent`. The first genuinely *composite* recurrent
||| layer: it embeds an `Nn.Lstm` controller + three `Nn.Linear` heads
||| (read/write/output) and carries an external memory matrix + read/write
||| addressing weights as per-sequence state. `params` composes the
||| sub-layers' params plus the learned memory-init; the smart constructor
||| nests everything under a numbered `ntm_<n>` scope via `scopedChild`,
||| with `named` sub-modules (controller / read_fc / write_fc / output_fc).
module Ml.Nn.Ntm

import Control.Linear.LIO
import Data.Linear
import Data.Vect

import Ml.Compat.Random
import Ml.Executor
import Ml.Nn.Init
import Ml.Nn.Linear
import Ml.Nn.Lstm
import Ml.Nn.Module
import Ml.Nn.Recurrent
import Ml.Tensor

%default total

----------------------------------------------------------------------
-- Head-parameter widths (type level)
----------------------------------------------------------------------

public export
ShiftKernelSize : Nat
ShiftKernelSize = 3

||| Read head emits: key[m] ++ shift[k] ++ beta[1] ++ g[1] ++ gamma[1].
public export
ReadParamWidth : Nat -> Nat
ReadParamWidth m = (m + ShiftKernelSize) + 3

||| Write head = read params + the add vector[m].
public export
WriteParamWidth : Nat -> Nat
WriteParamWidth m = ReadParamWidth m + m

----------------------------------------------------------------------
-- Raw per-step helpers (pure prim composition, transcribed from the
-- legacy Layer.Ntm — addressing math is unchanged)
----------------------------------------------------------------------

zeroState1d : {0 ex : Executor} -> Backend ex dt => (n : Nat) -> AnyPtr
zeroState1d n =
  let nI = cast {to=Int} n
  in dtCreateState1d {ex} {t=dt} nI (prim__allocDoubles nI) (deviceStreamTag {ex})

-- Pack a Vect of Doubles into a buffer at offset (for the fixed read-out).
packDoubles : AnyPtr -> Int -> Vect k Double -> AnyPtr
packDoubles buf _ []            = buf
packDoubles buf off (x :: rest) = packDoubles (prim__setDouble buf off x) (off + 1) rest

-- NTM read head (Graves §3.3): content addressing → interpolation →
-- circular shift → sharpening → read. Returns (newAddr[n], readOut[m]).
%inline
ntmReadHead : {0 ex : Executor} -> UserExecutorTraining ex =>
              (memT, prevWT, keyT, betaT, gT, gammaT, shiftT : AnyPtr) -> (AnyPtr, AnyPtr)
ntmReadHead memT prevWT keyT betaT gT gammaT shiftT =
  let keyT2d        = primUnsqueeze {ex} keyT 0
      cosScoresT     = primCosineSimilarity {ex} memT keyT2d 1
      scaledScoresT  = primMul {ex} betaT cosScoresT
      contentWT      = primSoftmax {ex} scaledScoresT 0
      oneMinusG      = primAddScalar {ex} (primNeg {ex} gT) 1.0
      interpT        = primAdd {ex} (primMul {ex} gT contentWT) (primMul {ex} oneMinusG prevWT)
      shiftedT       = primConv1dCircular {ex} interpT shiftT
      shiftedClamped = primClampMin {ex} shiftedT 1.0e-10
      poweredT       = primPow {ex} shiftedClamped gammaT
      normSumT       = primAddScalar {ex} (primSum {ex} poweredT) 1.0e-10
      focusedT       = primDiv {ex} poweredT normSumT
      readOutT       = primMatmul {ex} focusedT memT
  in (focusedT, readOutT)

-- NTM interpolation write: memory' = w·addVec + (1-w)·memory (row-wise).
%inline
ntmWrite : {0 ex : Executor} -> UserExecutorTraining ex => {n : Nat} ->
           (memT, weightsT, addVecT : AnyPtr) -> AnyPtr
ntmWrite {n} memT weightsT addVecT =
  let writeAdd = primOuter {ex} weightsT addVecT
      wCol = primReshape2d {ex} weightsT (cast n) 1
      keep = primAddScalar {ex} (primNeg {ex} wCol) 1.0
      kept = primMul {ex} keep memT
  in primAdd {ex} kept writeAdd

----------------------------------------------------------------------
-- The layer
----------------------------------------------------------------------

||| NTM cell. Controller + 3 heads + learned memory-init are params;
||| memT / read+write addresses / last read-out are per-sequence state.
public export
record Ntm (n : Nat) (m : Nat) (h : Nat) (i : Nat) (o : Nat) (0 ex : Executor) (0 dt : DType) (0 g : GradMode) where
  constructor MkNtm
  controller   : Lstm (m + i) h ex dt g
  readFc       : Linear h (ReadParamWidth m) ex dt g
  writeFc      : Linear h (WriteParamWidth m) ex dt g
  outputFc     : Linear (h + m) o ex dt g
  memInitT     : TVec (m * n) ex dt g
  initReadOutT : TVec m ex dt g
  memT         : Maybe (Tensor [n, m] ex dt g)
  readAddrT    : Maybe (TVec n ex dt g)
  writeAddrT   : Maybe (TVec n ex dt g)
  readOutT     : Maybe (TVec m ex dt g)

-- IO step body for the NTM cell, shared by the (linear) `recurStep`. Threads
-- the LSTM controller through ω fields; the addressing math is unchanged from
-- the legacy Layer.Ntm. Kept as a top-level helper (not an interface method)
-- so the linear `Recurrent` instance can delegate to it at the IO boundary.
ntmStepIO : {0 ex : Executor} -> Backend ex dt => {n, m, h : Nat} -> {i, o : Nat} ->
            Ntm n m h i o ex dt WithGrad -> Tensor [i] ex dt WithGrad ->
            IO (Ntm n m h i o ex dt WithGrad, Tensor [o] ex dt WithGrad)
ntmStepIO {i} {o} st input = do
    let nI = cast {to=Int} n
        mI         = cast {to=Int} m
        initMemPtr = primReshape2d {ex} (primSigmoid {ex} st.memInitT.tensorPtr) nI mI
        memTPtr    = maybe initMemPtr (.tensorPtr) st.memT
        raTPtr     = maybe (zeroState1d {ex} {dt} n) (.tensorPtr) st.readAddrT
        waTPtr     = maybe (zeroState1d {ex} {dt} n) (.tensorPtr) st.writeAddrT
        roTPtr     = maybe st.initReadOutT.tensorPtr (.tensorPtr) st.readOutT
        lstmInputV = the (TVec (m + i) ex dt WithGrad)
                         (MkTensor (primCat2 {ex} roTPtr input.tensorPtr) Nothing)
    -- Step the LSTM controller via `lstmStepIO` (ω in/out; the controller is
    -- threaded ω internally, the cell's external handle is the single-owner
    -- linear resource).
    (updCtrl, hiddenV) <- lstmStepIO st.controller lstmInputV
    -- The LSTM step always sets cellT; the zero fallback is unreachable
    -- (kept total instead of crashing).
    let cellPtr = maybe (zeroState1d {ex} {dt} h) (.tensorPtr) updCtrl.cellT
        skI                         = cast {to=Int} ShiftKernelSize
        readResultT                 = primLinear {ex} st.readFc.weightT.tensorPtr cellPtr st.readFc.biasT.tensorPtr
        keyT                        = primNarrow {ex} readResultT 0 0 mI
        shiftT                      = primSoftmax {ex} (primNarrow {ex} readResultT 0 mI skI) 0
        betaT                       = primSoftplus {ex} (primSelect {ex} readResultT 0 (mI + skI))
        gT                          = primSigmoid {ex} (primSelect {ex} readResultT 0 (mI + skI + 1))
        gammaT                      = primAddScalar {ex} (primSoftplus {ex} (primSelect {ex} readResultT 0 (mI + skI + 2))) 1.0
        (newReadAddrT, newReadOutT) = ntmReadHead {ex} memTPtr raTPtr keyT betaT gT gammaT shiftT
        writeResultT                = primLinear {ex} st.writeFc.weightT.tensorPtr cellPtr st.writeFc.biasT.tensorPtr
        rpw                         = cast {to=Int} (ReadParamWidth m)
        wKeyT                       = primNarrow {ex} writeResultT 0 0 mI
        wShiftT                     = primSoftmax {ex} (primNarrow {ex} writeResultT 0 mI skI) 0
        wBetaT                      = primSoftplus {ex} (primSelect {ex} writeResultT 0 (mI + skI))
        wGT                         = primSigmoid {ex} (primSelect {ex} writeResultT 0 (mI + skI + 1))
        wGammaT                     = primAddScalar {ex} (primSoftplus {ex} (primSelect {ex} writeResultT 0 (mI + skI + 2))) 1.0
        (newWriteAddrT, _)          = ntmReadHead {ex} memTPtr waTPtr wKeyT wBetaT wGT wGammaT wShiftT
        addT                        = primNarrow {ex} writeResultT 0 rpw mI
        newMemT                     = ntmWrite {ex} {n} memTPtr newWriteAddrT addT
        outputPtr                   = primLinear {ex} st.outputFc.weightT.tensorPtr
                      (primCat2 {ex} hiddenV.tensorPtr newReadOutT) st.outputFc.biasT.tensorPtr
    pure ( { controller := updCtrl
           , memT       := Just (MkTensor newMemT Nothing)
           , readAddrT  := Just (MkTensor newReadAddrT Nothing)
           , writeAddrT := Just (MkTensor newWriteAddrT Nothing)
           , readOutT   := Just (MkTensor newReadOutT Nothing) } st
         , MkTensor outputPtr Nothing )

||| Params for the NTM cell. The controller + 3 heads + memory-init/read-out
||| all bind at ω, so the sub-models reuse their `Params` methods for both the
||| reflected list and the rebuild; the per-sequence state fields ride at ω.
public export
{n, m, h : Nat} -> Params (Ntm n m h) where
  params (MkNtm ctrl rfc wfc ofc memInit iro memS raS waS roS) =
    params ctrl ++ params rfc ++ params wfc ++ params ofc ++ [toParam memInit, toParam iro]
  reflect (MkNtm ctrl rfc wfc ofc memInit iro memS raS waS roS) =
    let (MkBang pc # ctrl') = reflect ctrl
        (MkBang p1 # rfc') = reflect rfc
        (MkBang p2 # wfc') = reflect wfc
        (MkBang p3 # ofc') = reflect ofc in
    MkBang (pc ++ p1 ++ p2 ++ p3 ++ [toParam memInit, toParam iro])
      # MkNtm ctrl' rfc' wfc' ofc' memInit iro memS raS waS roS
  castGrad (MkNtm ctrl rfc wfc ofc memInit iro memS raS waS roS) =
    MkNtm (castGrad ctrl) (castGrad rfc) (castGrad wfc) (castGrad ofc)
          (retypeGrad memInit) (retypeGrad iro)
          (map retypeGrad memS) (map retypeGrad raS) (map retypeGrad waS) (map retypeGrad roS)
  discard (MkNtm _ _ _ _ _ _ _ _ _ _) = pure ()

||| Linear-resource recurrent step. The body is large and raw-prim-heavy and
||| threads the LSTM controller through an ω record field, so for now it
||| consumes the linear cell and delegates to the IO step helper (`ntmStepIO`)
||| at the linear boundary (the handle-level guarantee — no reuse of a stale
||| NTM — holds; the internal controller threading stays ω in IO). The inline
||| `L IO` body lands when the controller field can become linear.
public export
{n, m, h : Nat} -> Recurrent (Ntm n m h) where
  -- Pattern-match to discharge linearity (fields bind at ω), rebuild an ω
  -- cell, and delegate to the IO step; the returned cell rides the linear pair.
  recurStep (MkNtm ctrl rfc wfc ofc memInit iro memS raS waS roS) input = do
    (updSt, out) <- liftIO1
      (ntmStepIO (MkNtm ctrl rfc wfc ofc memInit iro memS raS waS roS) input)
    pure1 (MkBang out # updSt)
  recurReset (MkNtm ctrl rfc wfc ofc memInit iro _ _ _ _) =
    MkNtm (recurReset ctrl) rfc wfc ofc memInit iro Nothing Nothing Nothing Nothing

||| Construct an `Ntm` inside an `Init` derivation, mirroring the PyTorch
||| reference inits (LSTM default; read/write heads xavier-1.4 + bias
||| N(0,0.01); output head LeCun-ish; memory-init xavier-normal; fixed
||| Kaiming read-out). Nests sub-modules under `<scope>.ntm_<n>.…`.
export partial
ntm : KnownGrad g => {0 ex : Executor} -> Backend ex dt => {n, m, h, i, o : Nat} -> Init (Ntm n m h i o ex dt g)
ntm = scopedChild "ntm" $ do
  let xavStd : (a, b : Nat) -> Double
      xavStd a b = 1.4 * sqrt (2.0 / cast {to=Double} (a + b))
      memStd     = sqrt (2.0 / cast {to=Double} (m + n))
  -- Sub-modules built at the requested `g` (annotated so `g` flows up front,
  -- as in transformerBlock); the directly-created memInit param + iro state
  -- tensor are built WithGrad then weakened on the NoGrad branch.
  ctrl <- the (Init (Lstm (m + i) h ex dt g)) (named "controller" (lstmWithBias {i = m + i} {o = h}
                                    (1.0 / sqrt (cast {to=Double} h))))
  rfc  <- the (Init (Linear h (ReadParamWidth m) ex dt g))
              (named "read_fc"  (linearWith {i = h} {o = ReadParamWidth m}  (xavStd h (ReadParamWidth m))  0.01))
  wfc  <- the (Init (Linear h (WriteParamWidth m) ex dt g))
              (named "write_fc" (linearWith {i = h} {o = WriteParamWidth m} (xavStd h (WriteParamWidth m)) 0.01))
  ofc  <- the (Init (Linear (h + m) o ex dt g))
              -- He-uniform + normal(0, 0.01) bias, matching the reference's
              -- `kaiming_uniform_(output_fc.weight)` (default a=0, ReLU gain
              -- √2 → bound √6/√fan_in). NOT the shared dense contract:
              -- narrowing it to U(±1/√fan_in) slowed recall convergence past
              -- its bar, and NTM init is the tuned part of the architecture.
              (named "output_fc" (linearUniformWith {i = h + m} {o = o}
                                    (sqrt (6.0 / cast {to=Double} (h + m))) 0.01))
  mname <- freshChild "memory_init"
  memInit <- liftIO $ tparam1dNormal {ex} {dt} {n = m * n} mname 0.0 memStd
  iro <- liftIO $ do
    let iroBound = 1.0 / sqrt (cast {to=Double} m)
    iroVals <- traverse (\_ => randomRIO (-iroBound, iroBound)) (Vect.replicate m ())
    let buf = packDoubles (prim__allocDoubles (cast m)) 0 iroVals
    pure (the (TVec m ex dt WithGrad)
              (MkTensor (dtCreateState1d {ex} {t=dt} (cast m) buf (deviceStreamTag {ex})) Nothing))
  case sgrad {g} of
    SWithGrad => pure (MkNtm ctrl rfc wfc ofc memInit iro Nothing Nothing Nothing Nothing)
    SNoGrad   => do memInit' <- liftIO (weakenGrad memInit)
                    iro'     <- liftIO (weakenGrad iro)
                    pure (MkNtm ctrl rfc wfc ofc memInit' iro' Nothing Nothing Nothing Nothing)
