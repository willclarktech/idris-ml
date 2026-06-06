module Layer.Dnc

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
-- DNC constants (mirror V1)
----------------------------------------------------------------------

public export
DncControllerInput : Nat -> Nat -> Nat -> Nat
DncControllerInput r m i = r * m + i

public export
DncOutputInput : Nat -> Nat -> Nat -> Nat
DncOutputInput h r m = h + r * m


----------------------------------------------------------------------
-- DNC tensor-level helpers (top-level — mirrors V1 design to avoid
-- where-clause + implicit-args interactions)
----------------------------------------------------------------------

-- Concat read-output tensors followed by the input tensor.
catReadOutsAndInput : {0 ex : Executor} -> UserExecutorTraining ex => {k : Nat} -> Vect k AnyPtr -> AnyPtr -> AnyPtr
catReadOutsAndInput [] inp = inp
catReadOutsAndInput (ro :: rest) inp =
  primCat2 {ex} ro (catReadOutsAndInput {ex} rest inp)

-- Concat r read-output tensors. Crashes on r=0.
%default partial

catReadOuts : {0 ex : Executor} -> UserExecutorTraining ex => {k : Nat} -> Vect k AnyPtr -> AnyPtr
catReadOuts [] = idris_crash "Dnc: catReadOuts r=0"
catReadOuts (h :: t) = catRest h t
  where
    catRest : AnyPtr -> {k' : Nat} -> Vect k' AnyPtr -> AnyPtr
    catRest acc [] = acc
    catRest acc (h' :: rest) = catRest (primCat2 {ex} acc h') rest

-- Compute prod_j (1 - free_gate_j * prev_read_w_j) over r heads.
-- `onesScalar` is the precomputed scalar 1.0 (passed in to avoid
-- one dtCreateScalar {t=dt} call per (deviceStreamTag {ex}) recursion).
dncRetention : {0 ex : Executor} -> UserExecutorTraining ex => {k : Nat} -> AnyPtr -> Int -> AnyPtr -> Vect k AnyPtr -> AnyPtr -> AnyPtr
dncRetention _ _ _ [] acc = acc
dncRetention onesScalar idx freeGatesT (rw :: rws) acc =
  let fg = primSelect {ex} freeGatesT 0 idx
      factor = primSub {ex} onesScalar (primMul {ex} fg rw)
  in dncRetention {ex} onesScalar (idx + 1) freeGatesT rws (primMul {ex} acc factor)

-- Build a [n,n] non-diagonal mask once (1 off-diagonal, 0 on-diagonal).
-- Stored persistently in DncState; reused every timestep instead of
-- being rebuilt in `dncZeroDiag` (was 1 + n*n + 1 prim FFI calls
-- per timestep, dominated DNC forward overhead at ~1k prims/step
-- for n=32 — close to 200ms/epoch wasted on a constant). Fix moves
-- those 1027 prims out of the hot path entirely.
buildNonDiagMask : {0 ex : Executor} -> UserExecutorTraining ex => RuntimeDType dt => Linked ex => Compatible ex dt => (n : Nat) -> AnyPtr
buildNonDiagMask n =
  let nI = cast {to=Int} n
      numElems = nI * nI
      buf = prim__allocDoubles numElems
      buf' = fillOffDiag buf 0 nI numElems
  in dtCreateState2d {ex} {t=dt} nI nI buf' (deviceStreamTag {ex})
  where
    fillOffDiag : AnyPtr -> Int -> Int -> Int -> AnyPtr
    fillOffDiag b i nn numE = if i >= numE then b else
      let row = i `div` nn
          col = i `mod` nn
          val = if row == col then 0.0 else 1.0
          b' = prim__setDouble b i val
      in fillOffDiag b' (i + 1) nn numE

-- Zero the diagonal of a [n,n] matrix using a precomputed mask.
-- The mask is built once at DncState construction (`buildNonDiagMask`)
-- and stored in the state; this is now just the multiply.
dncZeroDiag : {0 ex : Executor} -> UserExecutorTraining ex => AnyPtr -> AnyPtr -> AnyPtr
dncZeroDiag maskPtr matT = primMul {ex} matT maskPtr

-- Per-head read processing for r heads.
-- `linkTransT` is the transposed link matrix, computed ONCE by the
-- caller and threaded in — used to be `primTranspose2d {ex} linkT` per
-- head, R redundant FFI calls on a head-invariant value.
dncReadHeads : {0 ex : Executor} -> UserExecutorTraining ex => {k : Nat} -> Int -> Vect k AnyPtr ->
                  AnyPtr -> AnyPtr -> AnyPtr ->
                  AnyPtr -> AnyPtr -> AnyPtr ->
                  Int ->
                  (Vect k AnyPtr, Vect k AnyPtr)
dncReadHeads _ [] _ _ _ _ _ _ _ = ([], [])
dncReadHeads idx (prevRw :: restRws) linkT linkTransT memT keysT betasT modesT mI =
  let headKeyT      = primNarrow {ex} keysT 0 (idx * mI) mI
      headBetaPtr   = primSelect {ex} betasT 0 idx
      headBetaT     = primSoftplus {ex} headBetaPtr
      headModesRawT = primNarrow {ex} modesT 0 (idx * 3) 3
      headModesT    = primSoftmax {ex} headModesRawT 0
      cosScoresT    = primCosineSimilarity {ex} memT (primUnsqueeze {ex} headKeyT 0) 1
      scaledScoresT = primMul {ex} headBetaT cosScoresT
      contentRwT    = primSoftmax {ex} scaledScoresT 0
      forwardT      = primMatmul {ex} linkT prevRw
      backwardT     = primMatmul {ex} linkTransT prevRw
      pi0           = primSelect {ex} headModesT 0 0
      pi1           = primSelect {ex} headModesT 0 1
      pi2           = primSelect {ex} headModesT 0 2
      scaledBack    = primMul {ex} pi0 backwardT
      scaledContent = primMul {ex} pi1 contentRwT
      scaledForward = primMul {ex} pi2 forwardT
      rwSumT        = primAdd {ex} (primAdd {ex} scaledBack scaledContent) scaledForward
      rwClampedT    = primClampMin {ex} rwSumT 1.0e-10
      rwNormSumT    = primAddScalar {ex} (primSum {ex} rwClampedT) 1.0e-10
      rwT           = primDiv {ex} rwClampedT rwNormSumT
      roT           = primMatmul {ex} rwT memT
      (restRws', restRos') =
        dncReadHeads {ex} (idx + 1) restRws linkT linkTransT memT keysT betasT modesT mI
  in (rwT :: restRws', roT :: restRos')


----------------------------------------------------------------------
-- DncState — typed-surface DNC (Path C)
----------------------------------------------------------------------
--
-- Mirrors V1 `Layer/Dnc.idr`'s `applyVarTensor`. 11 FCs + LSTM
-- controller + 7 state tensors (memory, usage, write weights,
-- precedence, link, R read weights, R read outputs).

public export
data DncState :
  (r : Nat) -> (n : Nat) -> (m : Nat) -> (h : Nat) ->
  Nat -> Nat -> (0 _ : Executor) -> (0 _ : DType) -> (0 _ : GradMode) -> Type
  where
  MkDnc :
    LstmState (DncControllerInput r m i) h ex dt g ->
    LinearState h m ex dt g ->                  -- writeKeyFc
    LinearState h 1 ex dt g ->                  -- writeBetaFc
    LinearState h m ex dt g ->                  -- eraseFc
    LinearState h m ex dt g ->                  -- addFc
    LinearState h r ex dt g ->                  -- freeGatesFc
    LinearState h 1 ex dt g ->                  -- allocGateFc
    LinearState h 1 ex dt g ->                  -- writeGateFc
    LinearState h (r * m) ex dt g ->            -- readKeysFc
    LinearState h r ex dt g ->                  -- readBetasFc
    LinearState h (r * 3) ex dt g ->            -- readModesFc
    LinearState (DncOutputInput h r m) o ex dt g ->  -- outputFc
    TVec (m * n) ex dt g ->                          -- memInit (LEARNED, raw flat)
    Vect r AnyPtr ->                          -- initReadOuts (Kaiming, NON-learned)
    AnyPtr ->                                 -- nonDiagMask: [n,n] (1 - I), precomputed once
    Maybe (Tensor [n, m] ex dt g) ->                -- memT
    Maybe (TVec n ex dt g) ->                     -- usageT
    Maybe (TVec n ex dt g) ->                     -- writeWtT
    Maybe (TVec n ex dt g) ->                     -- precedenceT
    Maybe (Tensor [n, n] ex dt g) ->                -- linkT
    Maybe (Vect r AnyPtr) ->                -- read weight tensor handles
    Maybe (Vect r AnyPtr) ->                -- read output tensor handles
    DncState r n m h i o ex dt g


----------------------------------------------------------------------
-- State init helpers
----------------------------------------------------------------------

-- Per-sequence transient state. Refcount-managed: lives as long as tape
-- entries or wrapped Idris Tensors reference it, freed when both let go.
-- See docs/develop/tensor-lifecycle.md and `Layer/Ntm.idr`'s
-- zeroState comment.
zeroState1d : {0 ex : Executor} -> UserExecutorTraining ex => RuntimeDType dt => Linked ex => Compatible ex dt => (n : Nat) -> AnyPtr
zeroState1d {ex} {dt} n =
  let nI = cast {to=Int} n
      buf = prim__allocDoubles nI
  in dtCreateState1d {ex} {t=dt} nI buf (deviceStreamTag {ex})

constState1d : {0 ex : Executor} -> UserExecutorTraining ex => RuntimeDType dt => Linked ex => Compatible ex dt => (n : Nat) -> Double -> AnyPtr
constState1d n v =
  let nI = cast {to=Int} n
      buf = fillBuf (prim__allocDoubles nI) 0 nI v
  in dtCreateState1d {ex} {t=dt} nI buf (deviceStreamTag {ex})
  where
    fillBuf : AnyPtr -> Int -> Int -> Double -> AnyPtr
    fillBuf b i n v = if i >= n then b
      else fillBuf (prim__setDouble b i v) (i + 1) n v

zeroState2d : {0 ex : Executor} -> UserExecutorTraining ex => RuntimeDType dt => Linked ex => Compatible ex dt => (a, b : Nat) -> AnyPtr
zeroState2d a b =
  let aI = cast {to=Int} a
      bI = cast {to=Int} b
      buf = prim__allocDoubles (aI * bI)
  in dtCreateState2d {ex} {t=dt} aI bI buf (deviceStreamTag {ex})

constState2d : {0 ex : Executor} -> UserExecutorTraining ex => RuntimeDType dt => Linked ex => Compatible ex dt => (a, b : Nat) -> Double -> AnyPtr
constState2d a b v =
  let aI = cast {to=Int} a
      bI = cast {to=Int} b
      buf = fillBuf (prim__allocDoubles (aI * bI)) 0 (aI * bI) v
  in dtCreateState2d {ex} {t=dt} aI bI buf (deviceStreamTag {ex})
  where
    fillBuf : AnyPtr -> Int -> Int -> Double -> AnyPtr
    fillBuf b i n v = if i >= n then b
      else fillBuf (prim__setDouble b i v) (i + 1) n v

-- Vect r of zero-state [n] handles (for read weights and read outputs).
mkZeroVectN : {0 ex : Executor} -> UserExecutorTraining ex => RuntimeDType dt => Linked ex => Compatible ex dt => (r : Nat) -> Nat -> Vect r AnyPtr
mkZeroVectN Z _ = []
mkZeroVectN (S k) n = zeroState1d {ex} {dt} n :: mkZeroVectN {ex} {dt} k n

mkZeroVectM : {0 ex : Executor} -> UserExecutorTraining ex => RuntimeDType dt => Linked ex => Compatible ex dt => (r : Nat) -> Nat -> Vect r AnyPtr
mkZeroVectM Z _ = []
mkZeroVectM (S k) m = zeroState1d {ex} {dt} m :: mkZeroVectM {ex} {dt} k m


----------------------------------------------------------------------
-- Forward
----------------------------------------------------------------------

export
applyDnc : {0 ex : Executor} -> UserExecutorTraining ex => UserExecutorCore ex => RuntimeDType dt => Linked ex => Compatible ex dt => {r, n, m, h, i, o : Nat} ->
             DncState r n m h i o ex dt g ->
             TVec i ex dt g ->
             IO (DncState r n m h i o ex dt g, TVec o ex dt g)
applyDnc {r} {n} {m}
           (MkDnc lstm wkFc wbFc eFc aFc fgFc agFc wgFc rkFc rbFc rmFc oFc
                    memInitT initReadOutsT nonDiagMaskT
                    memT usageT wwT precT linkT rwTs roTs) input = do
  let nI = cast {to=Int} n
      mI = cast {to=Int} m
      -- Initial memory at sequence start: sigmoid(memInit).reshape(n, m).
      -- Mirrors `torch_ref/dnc/layer.py:111`. Gradient flows back to memInitT.
      initMemPtr = primReshape2d {ex} (primSigmoid {ex} memInitT.tensorPtr) nI mI
      memTPtr = case memT of
                  Just t => t.tensorPtr
                  Nothing => initMemPtr
      usageTPtr = case usageT of
                    Just t => t.tensorPtr
                    Nothing => zeroState1d {ex} {dt} n
      wwTPtr = case wwT of
                 Just t => t.tensorPtr
                 Nothing => zeroState1d {ex} {dt} n
      precTPtr = case precT of
                   Just t => t.tensorPtr
                   Nothing => zeroState1d {ex} {dt} n
      linkTPtr = case linkT of
                   Just t => t.tensorPtr
                   Nothing => zeroState2d {ex} {dt} n n
      rwTsPtrs = the (Vect r AnyPtr) $ case rwTs of
                   Just ts => ts
                   Nothing => mkZeroVectN {ex} {dt} r n
      -- Initial read outputs: Kaiming-uniform samples, fixed at construction.
      -- Mirrors `torch_ref/dnc/layer.py:104, 117`.
      roTsPtrs = the (Vect r AnyPtr) $ case roTs of
                   Just ts => ts
                   Nothing => initReadOutsT
      -- 1. cat(readOuts, input) -> [r*m + i]
      lstmInputPtr = catReadOutsAndInput {ex} roTsPtrs input.tensorPtr
      lstmInputV = the (TVec (DncControllerInput r m i) ex dt g) (MkTensor lstmInputPtr Nothing)
  -- 2. LSTM forward (IO)
  (updLstm, hiddenV) <- applyLstm lstm lstmInputV
  -- 3. Cell-state for FCs
  let cellPtr = case updLstm.cellT of
                  Just c => c.tensorPtr
                  Nothing => idris_crash "Dnc: cell tensor missing post-LSTM"
      -- Sub-layer weight handles
      wkW = wkFc.weightT.tensorPtr; wkB = wkFc.biasT.tensorPtr
      wbW = wbFc.weightT.tensorPtr; wbB = wbFc.biasT.tensorPtr
      eW  = eFc.weightT.tensorPtr;  eB  = eFc.biasT.tensorPtr
      aW  = aFc.weightT.tensorPtr;  aB  = aFc.biasT.tensorPtr
      fgW = fgFc.weightT.tensorPtr; fgB = fgFc.biasT.tensorPtr
      agW = agFc.weightT.tensorPtr; agB = agFc.biasT.tensorPtr
      wgW = wgFc.weightT.tensorPtr; wgB = wgFc.biasT.tensorPtr
      rkW = rkFc.weightT.tensorPtr; rkB = rkFc.biasT.tensorPtr
      rbW = rbFc.weightT.tensorPtr; rbB = rbFc.biasT.tensorPtr
      rmW = rmFc.weightT.tensorPtr; rmB = rmFc.biasT.tensorPtr
      oW  = oFc.weightT.tensorPtr;  oB  = oFc.biasT.tensorPtr
      onesScalar = dtCreateScalar {ex} {t=dt} 1.0 0 (deviceStreamTag {ex})
      -- 4. 11 FCs (mv+add fused into primLinear {ex} — collapses two
      --    FFI hops into one per FC, ~10x FFI overhead reduction here)
      writeKeyT      = primLinear {ex} wkW cellPtr wkB
      writeBetaRawT  = primLinear {ex} wbW cellPtr wbB
      eraseRawT      = primLinear {ex} eW  cellPtr eB
      addVecT        = primLinear {ex} aW  cellPtr aB
      freeGatesRawT  = primLinear {ex} fgW cellPtr fgB
      allocGateRawT  = primLinear {ex} agW cellPtr agB
      writeGateRawT  = primLinear {ex} wgW cellPtr wgB
      readKeysFlatT  = primLinear {ex} rkW cellPtr rkB
      readBetasRawT  = primLinear {ex} rbW cellPtr rbB
      readModesFlatT = primLinear {ex} rmW cellPtr rmB
      -- 5. Activations
      writeBetaT  = primSoftplus {ex} writeBetaRawT
      eraseVecT   = primSigmoid {ex} eraseRawT
      freeGatesT  = primSigmoid {ex} freeGatesRawT
      allocGateT  = primSigmoid {ex} allocGateRawT
      writeGateT  = primSigmoid {ex} writeGateRawT
      -- 6. Usage update
      writeUsageT = primSub {ex} (primAdd {ex} usageTPtr wwTPtr) (primMul {ex} usageTPtr wwTPtr)
      retentionT  = dncRetention {ex} onesScalar 0 freeGatesT rwTsPtrs onesScalar
      retClampedT = primClampMin {ex} retentionT 1.0e-10
      newUsageT   = primMul {ex} writeUsageT retClampedT
      -- 7. Allocation
      indicesT      = primArgsort {ex} newUsageT 0 0
      sortedUsageT  = primClampMin {ex} (primGather {ex} newUsageT indicesT nI) 1.0e-6
      cumprodT      = primCumprod {ex} sortedUsageT 0
      slicedT       = primNarrow {ex} cumprodT 0 0 (nI - 1)
      shiftedT      = primCat2 {ex} (primUnsqueeze {ex} onesScalar 0) slicedT
      oneMinusUsageT = primSub {ex} onesScalar sortedUsageT
      sortedAllocT  = primMul {ex} oneMinusUsageT shiftedT
      allocT        = primScatterAdd {ex} indicesT sortedAllocT nI
      -- 8. Write content addressing
      cosScoresT    = primCosineSimilarity {ex} memTPtr (primUnsqueeze {ex} writeKeyT 0) 1
      scaledScoresT = primMul {ex} writeBetaT cosScoresT
      contentWriteWT = primSoftmax {ex} scaledScoresT 0
      -- 9. Write weighting
      oneMinusAGT   = primSub {ex} onesScalar allocGateT
      blendT        = primAdd {ex} (primMul {ex} allocGateT allocT)
                                 (primMul {ex} oneMinusAGT contentWriteWT)
      newWriteWT    = primMul {ex} writeGateT blendT
      -- 10. Memory write
      eraseGateT    = primOuter {ex} newWriteWT eraseVecT
      keepGateT     = primSub {ex} onesScalar eraseGateT
      erasedT       = primMul {ex} memTPtr keepGateT
      addGateT      = primOuter {ex} newWriteWT addVecT
      newMemT       = primAdd {ex} erasedT addGateT
      -- 11. Link matrix update
      wiT           = primUnsqueeze {ex} newWriteWT 1
      wjT           = primUnsqueeze {ex} newWriteWT 0
      pjT           = primUnsqueeze {ex} precTPtr 0
      decayT        = primSub {ex} (primSub {ex} onesScalar wiT) wjT
      decayClampT   = primClampMin {ex} decayT 0.0
      newLinkRawT   = primAdd {ex} (primMul {ex} decayClampT linkTPtr) (primMul {ex} wiT pjT)
      newLinkT      = primClampMin {ex} (dncZeroDiag {ex} nonDiagMaskT newLinkRawT) 0.0
      -- 12. Precedence update
      wSumT         = primSum {ex} newWriteWT
      oneMinusWSumT = primSub {ex} onesScalar wSumT
      newPrecT      = primAdd {ex} (primMul {ex} oneMinusWSumT precTPtr) newWriteWT
      -- 13. Read heads. Compute the link transpose ONCE outside the
      -- per-head recursion (was being computed R times — head-invariant).
      newLinkTransT = primTranspose2d {ex} newLinkT
      (newRwTs, newRoTs) = dncReadHeads {ex} 0 rwTsPtrs newLinkT newLinkTransT newMemT
                              readKeysFlatT readBetasRawT readModesFlatT mI
      -- 14. Output FC
      allNewReadsT  = catReadOuts {ex} newRoTs
      outputInputT  = primCat2 {ex} hiddenV.tensorPtr allNewReadsT
      outputT       = primLinear {ex} oW outputInputT oB
  pure ( MkDnc updLstm wkFc wbFc eFc aFc fgFc agFc wgFc rkFc rbFc rmFc oFc
          memInitT initReadOutsT nonDiagMaskT
          (Just (MkTensor newMemT Nothing))
          (Just (MkTensor newUsageT Nothing))
          (Just (MkTensor newWriteWT Nothing))
          (Just (MkTensor newPrecT Nothing))
          (Just (MkTensor newLinkT Nothing))
          (Just newRwTs)
          (Just newRoTs)
       , MkTensor outputT Nothing )


----------------------------------------------------------------------
-- Constructor
----------------------------------------------------------------------

-- Build r Kaiming-uniform read-output state tensors (one per read head).
-- PyTorch default kaiming_uniform on (1, m) per head: bound = 1/sqrt(m).
-- Sampled once at construction; non-learnable.
mkKaimingReadOuts : {0 ex : Executor} -> UserExecutorTraining ex => RuntimeDType dt => Linked ex => Compatible ex dt => (r : Nat) -> (m : Nat) -> Double -> IO (Vect r AnyPtr)
mkKaimingReadOuts Z _ _ = pure []
mkKaimingReadOuts (S k) m bound = do
  vals <- traverse (\_ => randomRIO (-bound, bound)) (Vect.replicate m ())
  let mI = cast {to=Int} m
      buf = prim__allocDoubles mI
      buf' = packDoubles buf 0 vals
      ptr = dtCreateState1d {ex} {t=dt} mI buf' (deviceStreamTag {ex})
  rest <- mkKaimingReadOuts {ex} {dt} k m bound
  pure (ptr :: rest)

||| Build a `DncState r n m h i o TapeExecutor` matching the PyTorch reference's
||| `DNCLayer.__init__` (`torch_ref/dnc/layer.py`) line-for-line:
|||
||| - LSTM controller:        Idris's `lstmLayer` (now with learned h0/c0)
||| - all 10 head FC weights: `xavier_uniform_(gain=1.4)`
||| - output FC weight:       `kaiming_uniform_` (PyTorch default ≈ LeCun)
||| - all FC biases:          `normal_(std=0.01)`
||| - memory_init param:      `xavier_uniform_(view(n, m))`, sigmoid'd at
|||                           sequence-start in `applyDnc`
||| - initial read outputs:   `kaiming_uniform_((R, m))`, non-learnable,
|||                           sampled once at construction
export
dncLayer : UserExecutorTraining ex => RuntimeDType dt => Linked ex => Compatible ex dt => {r, n, m, h, i, o : Nat} ->
             (paramPrefix : String) ->
             IO (DncState r n m h i o ex dt WithGrad)
dncLayer pfx = do
  lstm <- lstmLayer {i = DncControllerInput r m i} {o = h} (pfx ++ "_lstm")
  -- 10 head FCs: xavier-normal-via-uniform (gain=1.4) → std = 1.4 * sqrt(2/(i+o));
  -- biases ~ N(0, 0.0001). Output FC: PyTorch nn.Linear default
  -- (1/sqrt(fan_in)); same N(0, 0.0001) biases.
  let xavStd : (i, o : Nat) -> Double
      xavStd i' o' = 1.4 * sqrt (2.0 / cast {to=Double} (i' + o'))
  wkFc <- mkLinearWith {i = h} {o = m}     (pfx ++ "_writeKey")
            (xavStd h m)       0.0001
  wbFc <- mkLinearWith {i = h} {o = 1}     (pfx ++ "_writeBeta")
            (xavStd h 1)       0.0001
  eFc  <- mkLinearWith {i = h} {o = m}     (pfx ++ "_erase")
            (xavStd h m)       0.0001
  aFc  <- mkLinearWith {i = h} {o = m}     (pfx ++ "_add")
            (xavStd h m)       0.0001
  fgFc <- mkLinearWith {i = h} {o = r}     (pfx ++ "_freeGates")
            (xavStd h r)       0.0001
  agFc <- mkLinearWith {i = h} {o = 1}     (pfx ++ "_allocGate")
            (xavStd h 1)       0.0001
  wgFc <- mkLinearWith {i = h} {o = 1}     (pfx ++ "_writeGate")
            (xavStd h 1)       0.0001
  rkFc <- mkLinearWith {i = h} {o = r * m} (pfx ++ "_readKeys")
            (xavStd h (r*m))   0.0001
  rbFc <- mkLinearWith {i = h} {o = r}     (pfx ++ "_readBetas")
            (xavStd h r)       0.0001
  rmFc <- mkLinearWith {i = h} {o = r * 3} (pfx ++ "_readModes")
            (xavStd h (r*3))   0.0001
  oFc  <- mkLinearWith {i = DncOutputInput h r m} {o = o}
            (pfx ++ "_output")
            (1.0 / sqrt (cast {to=Double} (DncOutputInput h r m))) 0.0001
  -- memoryInit: learnable [m * n] (flat shape; underlying shape is [n, m]
  -- where fan_in=m, fan_out=n). Xavier-normal-via-uniform std = sqrt(2/(m+n)).
  let memStd = sqrt (2.0 / cast {to=Double} (m + n))
  memInitT <- tparam1dNormal {n = m * n} (pfx ++ "_memoryInit") 0.0 memStd
  -- initialReadOuts: PyTorch default kaiming_uniform on (R, m), bound=1/sqrt(m)
  let iroBound = 1.0 / prim__doubleSqrt (cast m)
  initReadOutsT <- mkKaimingReadOuts {ex} {dt} r m iroBound
  -- nonDiagMask: [n,n] (1 - I), built once and reused every timestep
  -- inside the link-matrix update. Saves ~1 + n*n + 1 prim FFI calls
  -- per step.
  let nonDiagMaskT = buildNonDiagMask {ex} {dt} n
  -- Per-sequence runtime state starts as Nothing — applyDnc computes the
  -- actual initial memT and roTs from memInitT/initReadOutsT on first call.
  pure $ MkDnc lstm wkFc wbFc eFc aFc fgFc agFc wgFc rkFc rbFc rmFc oFc
                memInitT initReadOutsT nonDiagMaskT
                Nothing Nothing Nothing Nothing Nothing Nothing Nothing

||| Reset DNC state between sequences. Keeps the learned `memInitT` and
||| Kaiming-fixed `initReadOutsT`; clears per-sequence runtime state so
||| the next `applyDnc` re-derives initial memory + read outputs.
export
resetDncState : {r, n, m, h : Nat} -> {0 g : GradMode} ->
                  DncState r n m h i o ex dt g -> DncState r n m h i o ex dt g
resetDncState (MkDnc lstm wkFc wbFc eFc aFc fgFc agFc wgFc rkFc rbFc rmFc oFc
                          memInitT initReadOutsT nonDiagMaskT _ _ _ _ _ _ _) =
  MkDnc (resetLstmState lstm) wkFc wbFc eFc aFc fgFc agFc wgFc rkFc rbFc rmFc oFc
        memInitT initReadOutsT nonDiagMaskT
        Nothing Nothing Nothing Nothing Nothing Nothing Nothing


----------------------------------------------------------------------
-- LayerLike instance
----------------------------------------------------------------------

public export
{r, n, m, h : Nat} ->
  LayerLike (DncState r n m h) where
  applyVar st@(MkDnc _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _) input =
    applyDnc st input
  layerPrefix _ = "dnc"
  resetState st = resetDncState st

  freezeLayer (MkDnc lstm wkFc wbFc eFc aFc fgFc agFc wgFc rkFc rbFc rmFc oFc
                     memInit iro nonDiag mem usage ww prec link rwTs roTs) = do
    lstm'  <- freezeLayer lstm
    wkFc'  <- freezeLayer wkFc
    wbFc'  <- freezeLayer wbFc
    eFc'   <- freezeLayer eFc
    aFc'   <- freezeLayer aFc
    fgFc'  <- freezeLayer fgFc
    agFc'  <- freezeLayer agFc
    wgFc'  <- freezeLayer wgFc
    rkFc'  <- freezeLayer rkFc
    rbFc'  <- freezeLayer rbFc
    rmFc'  <- freezeLayer rmFc
    oFc'   <- freezeLayer oFc
    memInit' <- weakenGrad memInit
    mem'  <- case mem  of Nothing => pure Nothing; Just t => Just <$> weakenGrad t
    usage' <- case usage of Nothing => pure Nothing; Just t => Just <$> weakenGrad t
    ww'   <- case ww   of Nothing => pure Nothing; Just t => Just <$> weakenGrad t
    prec' <- case prec of Nothing => pure Nothing; Just t => Just <$> weakenGrad t
    link' <- case link of Nothing => pure Nothing; Just t => Just <$> weakenGrad t
    pure (MkDnc lstm' wkFc' wbFc' eFc' aFc' fgFc' agFc' wgFc'
                rkFc' rbFc' rmFc' oFc'
                memInit' iro nonDiag mem' usage' ww' prec' link' rwTs roTs)

  unfreezeLayer (MkDnc lstm wkFc wbFc eFc aFc fgFc agFc wgFc rkFc rbFc rmFc oFc
                       memInit iro nonDiag mem usage ww prec link rwTs roTs) = do
    lstm'  <- unfreezeLayer lstm
    wkFc'  <- unfreezeLayer wkFc
    wbFc'  <- unfreezeLayer wbFc
    eFc'   <- unfreezeLayer eFc
    aFc'   <- unfreezeLayer aFc
    fgFc'  <- unfreezeLayer fgFc
    agFc'  <- unfreezeLayer agFc
    wgFc'  <- unfreezeLayer wgFc
    rkFc'  <- unfreezeLayer rkFc
    rbFc'  <- unfreezeLayer rbFc
    rmFc'  <- unfreezeLayer rmFc
    oFc'   <- unfreezeLayer oFc
    primIO (primSetRequiresGrad {ex} memInit.tensorPtr 1)
    case mem   of Nothing => pure (); Just t => primIO (primSetRequiresGrad {ex} t.tensorPtr 1)
    case usage of Nothing => pure (); Just t => primIO (primSetRequiresGrad {ex} t.tensorPtr 1)
    case ww    of Nothing => pure (); Just t => primIO (primSetRequiresGrad {ex} t.tensorPtr 1)
    case prec  of Nothing => pure (); Just t => primIO (primSetRequiresGrad {ex} t.tensorPtr 1)
    case link  of Nothing => pure (); Just t => primIO (primSetRequiresGrad {ex} t.tensorPtr 1)
    pure (MkDnc lstm' wkFc' wbFc' eFc' aFc' fgFc' agFc' wgFc'
                rkFc' rbFc' rmFc' oFc'
                (retypeGrad memInit) iro nonDiag
                (map retypeGrad mem) (map retypeGrad usage)
                (map retypeGrad ww) (map retypeGrad prec)
                (map retypeGrad link) rwTs roTs)

export
dncLayerAny : UserExecutorTraining ex => RuntimeDType dt => Linked ex => Compatible ex dt => {r, n, m, h, i, o : Nat} ->
                (paramPrefix : String) ->
                IO (AnyLayer i o ex dt WithGrad)
dncLayerAny pid =
  map (MkAnyLayer (DncState r n m h)) (dncLayer {r} {n} {m} {h} {i} {o} pid)
