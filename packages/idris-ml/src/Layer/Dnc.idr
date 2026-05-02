module Layer.Dnc

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
catReadOutsAndInput : {k : Nat} -> Vect k AnyPtr -> AnyPtr -> AnyPtr
catReadOutsAndInput [] inp = inp
catReadOutsAndInput (ro :: rest) inp =
  prim__cat2 ro (catReadOutsAndInput rest inp)

-- Concat r read-output tensors. Crashes on r=0.
%default partial

catReadOuts : {k : Nat} -> Vect k AnyPtr -> AnyPtr
catReadOuts [] = idris_crash "Dnc: catReadOuts r=0"
catReadOuts (h :: t) = catRest h t
  where
    catRest : AnyPtr -> {k' : Nat} -> Vect k' AnyPtr -> AnyPtr
    catRest acc [] = acc
    catRest acc (h' :: rest) = catRest (prim__cat2 acc h') rest

-- Compute prod_j (1 - free_gate_j * prev_read_w_j) over r heads.
-- `onesScalar` is the precomputed scalar 1.0 (passed in to avoid
-- one prim__createScalar call per recursion).
dncRetention : {k : Nat} -> AnyPtr -> Int -> AnyPtr -> Vect k AnyPtr -> AnyPtr -> AnyPtr
dncRetention _ _ _ [] acc = acc
dncRetention onesScalar idx freeGatesT (rw :: rws) acc =
  let fg = prim__select freeGatesT 0 idx
      factor = prim__sub onesScalar (prim__mul fg rw)
  in dncRetention onesScalar (idx + 1) freeGatesT rws (prim__mul acc factor)

-- Build a [n,n] non-diagonal mask once (1 off-diagonal, 0 on-diagonal).
-- Stored persistently in DncState; reused every timestep instead of
-- being rebuilt in `dncZeroDiag` (was 1 + n*n + 1 prim FFI calls
-- per timestep, dominated DNC forward overhead at ~1k prims/step
-- for n=32 — close to 200ms/epoch wasted on a constant). Fix moves
-- those 1027 prims out of the hot path entirely.
buildNonDiagMask : (n : Nat) -> AnyPtr
buildNonDiagMask n =
  let nI = cast {to=Int} n
      numElems = nI * nI
      buf = prim__allocDoubles numElems
      buf' = fillOffDiag buf 0 nI numElems
  in prim__createState2d nI nI buf'
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
dncZeroDiag : AnyPtr -> AnyPtr -> AnyPtr
dncZeroDiag maskPtr matT = prim__mul matT maskPtr

-- Per-head read processing for r heads.
-- `linkTransT` is the transposed link matrix, computed ONCE by the
-- caller and threaded in — used to be `prim__transpose2d linkT` per
-- head, R redundant FFI calls on a head-invariant value.
dncReadHeads : {k : Nat} -> Int -> Vect k AnyPtr ->
                  AnyPtr -> AnyPtr -> AnyPtr ->
                  AnyPtr -> AnyPtr -> AnyPtr ->
                  Int ->
                  (Vect k AnyPtr, Vect k AnyPtr)
dncReadHeads _ [] _ _ _ _ _ _ _ = ([], [])
dncReadHeads idx (prevRw :: restRws) linkT linkTransT memT keysT betasT modesT mI =
  let headKeyT      = prim__narrow keysT 0 (idx * mI) mI
      headBetaPtr   = prim__select betasT 0 idx
      headBetaT     = prim__softplus headBetaPtr
      headModesRawT = prim__narrow modesT 0 (idx * 3) 3
      headModesT    = prim__softmax headModesRawT 0
      cosScoresT    = prim__cosineSimilarity memT (prim__unsqueeze headKeyT 0) 1
      scaledScoresT = prim__mul headBetaT cosScoresT
      contentRwT    = prim__softmax scaledScoresT 0
      forwardT      = prim__matmul linkT prevRw
      backwardT     = prim__matmul linkTransT prevRw
      pi0           = prim__select headModesT 0 0
      pi1           = prim__select headModesT 0 1
      pi2           = prim__select headModesT 0 2
      scaledBack    = prim__mul pi0 backwardT
      scaledContent = prim__mul pi1 contentRwT
      scaledForward = prim__mul pi2 forwardT
      rwSumT        = prim__add (prim__add scaledBack scaledContent) scaledForward
      rwClampedT    = prim__clampMin rwSumT 1.0e-10
      rwNormSumT    = prim__addScalar (prim__sum rwClampedT) 1.0e-10
      rwT           = prim__div rwClampedT rwNormSumT
      roT           = prim__matmul rwT memT
      (restRws', restRos') =
        dncReadHeads (idx + 1) restRws linkT linkTransT memT keysT betasT modesT mI
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
  Nat -> Nat -> (0 _ : Device) -> (0 _ : DType) -> (0 _ : GradMode) -> Type
  where
  MkDnc :
    LstmState (DncControllerInput r m i) h d dt g ->
    LinearState h m d dt g ->                  -- writeKeyFc
    LinearState h 1 d dt g ->                  -- writeBetaFc
    LinearState h m d dt g ->                  -- eraseFc
    LinearState h m d dt g ->                  -- addFc
    LinearState h r d dt g ->                  -- freeGatesFc
    LinearState h 1 d dt g ->                  -- allocGateFc
    LinearState h 1 d dt g ->                  -- writeGateFc
    LinearState h (r * m) d dt g ->            -- readKeysFc
    LinearState h r d dt g ->                  -- readBetasFc
    LinearState h (r * 3) d dt g ->            -- readModesFc
    LinearState (DncOutputInput h r m) o d dt g ->  -- outputFc
    TVec (m * n) d dt g ->                          -- memInit (LEARNED, raw flat)
    Vect r AnyPtr ->                          -- initReadOuts (Kaiming, NON-learned)
    AnyPtr ->                                 -- nonDiagMask: [n,n] (1 - I), precomputed once
    Maybe (Tensor [n, m] d dt g) ->                -- memT
    Maybe (TVec n d dt g) ->                     -- usageT
    Maybe (TVec n d dt g) ->                     -- writeWtT
    Maybe (TVec n d dt g) ->                     -- precedenceT
    Maybe (Tensor [n, n] d dt g) ->                -- linkT
    Maybe (Vect r AnyPtr) ->                -- read weight tensor handles
    Maybe (Vect r AnyPtr) ->                -- read output tensor handles
    DncState r n m h i o d dt g


----------------------------------------------------------------------
-- State init helpers
----------------------------------------------------------------------

-- Per-sequence transient state. Refcount-managed: lives as long as tape
-- entries or wrapped Idris Tensors reference it, freed when both let go.
-- See docs/develop/tensor-lifecycle.md and `Layer/Ntm.idr`'s
-- zeroState comment.
zeroState1d : (n : Nat) -> AnyPtr
zeroState1d n =
  let nI = cast {to=Int} n
      buf = prim__allocDoubles nI
  in prim__createState1d nI buf

constState1d : (n : Nat) -> Double -> AnyPtr
constState1d n v =
  let nI = cast {to=Int} n
      buf = fillBuf (prim__allocDoubles nI) 0 nI v
  in prim__createState1d nI buf
  where
    fillBuf : AnyPtr -> Int -> Int -> Double -> AnyPtr
    fillBuf b i n v = if i >= n then b
      else fillBuf (prim__setDouble b i v) (i + 1) n v

zeroState2d : (a, b : Nat) -> AnyPtr
zeroState2d a b =
  let aI = cast {to=Int} a
      bI = cast {to=Int} b
      buf = prim__allocDoubles (aI * bI)
  in prim__createState2d aI bI buf

constState2d : (a, b : Nat) -> Double -> AnyPtr
constState2d a b v =
  let aI = cast {to=Int} a
      bI = cast {to=Int} b
      buf = fillBuf (prim__allocDoubles (aI * bI)) 0 (aI * bI) v
  in prim__createState2d aI bI buf
  where
    fillBuf : AnyPtr -> Int -> Int -> Double -> AnyPtr
    fillBuf b i n v = if i >= n then b
      else fillBuf (prim__setDouble b i v) (i + 1) n v

-- Vect r of zero-state [n] handles (for read weights and read outputs).
mkZeroVectN : (r : Nat) -> Nat -> Vect r AnyPtr
mkZeroVectN Z _ = []
mkZeroVectN (S k) n = zeroState1d n :: mkZeroVectN k n

mkZeroVectM : (r : Nat) -> Nat -> Vect r AnyPtr
mkZeroVectM Z _ = []
mkZeroVectM (S k) m = zeroState1d m :: mkZeroVectM k m


----------------------------------------------------------------------
-- Forward
----------------------------------------------------------------------

export
applyDnc : {0 d : Device} -> UserDeviceTape d => RuntimeDType dt => {r, n, m, h, i, o : Nat} ->
             DncState r n m h i o d dt g ->
             TVec i d dt g ->
             IO (DncState r n m h i o d dt g, TVec o d dt g)
applyDnc {r} {n} {m}
           (MkDnc lstm wkFc wbFc eFc aFc fgFc agFc wgFc rkFc rbFc rmFc oFc
                    memInitT initReadOutsT nonDiagMaskT
                    memT usageT wwT precT linkT rwTs roTs) input = do
  let nI = cast {to=Int} n
      mI = cast {to=Int} m
      -- Initial memory at sequence start: sigmoid(memInit).reshape(n, m).
      -- Mirrors `torch_ref/dnc/layer.py:111`. Gradient flows back to memInitT.
      initMemPtr = prim__reshape2d (prim__sigmoid memInitT.tensorPtr) nI mI
      memTPtr = case memT of
                  Just t => t.tensorPtr
                  Nothing => initMemPtr
      usageTPtr = case usageT of
                    Just t => t.tensorPtr
                    Nothing => zeroState1d n
      wwTPtr = case wwT of
                 Just t => t.tensorPtr
                 Nothing => zeroState1d n
      precTPtr = case precT of
                   Just t => t.tensorPtr
                   Nothing => zeroState1d n
      linkTPtr = case linkT of
                   Just t => t.tensorPtr
                   Nothing => zeroState2d n n
      rwTsPtrs = the (Vect r AnyPtr) $ case rwTs of
                   Just ts => ts
                   Nothing => mkZeroVectN r n
      -- Initial read outputs: Kaiming-uniform samples, fixed at construction.
      -- Mirrors `torch_ref/dnc/layer.py:104, 117`.
      roTsPtrs = the (Vect r AnyPtr) $ case roTs of
                   Just ts => ts
                   Nothing => initReadOutsT
      -- 1. cat(readOuts, input) -> [r*m + i]
      lstmInputPtr = catReadOutsAndInput roTsPtrs input.tensorPtr
      lstmInputV = the (TVec (DncControllerInput r m i) d dt g) (MkTensor lstmInputPtr Nothing)
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
      onesScalar = prim__createScalar 1.0 0
      -- 4. 11 FCs (mv+add fused into prim__linear — collapses two
      --    FFI hops into one per FC, ~10x FFI overhead reduction here)
      writeKeyT      = prim__linear wkW cellPtr wkB
      writeBetaRawT  = prim__linear wbW cellPtr wbB
      eraseRawT      = prim__linear eW  cellPtr eB
      addVecT        = prim__linear aW  cellPtr aB
      freeGatesRawT  = prim__linear fgW cellPtr fgB
      allocGateRawT  = prim__linear agW cellPtr agB
      writeGateRawT  = prim__linear wgW cellPtr wgB
      readKeysFlatT  = prim__linear rkW cellPtr rkB
      readBetasRawT  = prim__linear rbW cellPtr rbB
      readModesFlatT = prim__linear rmW cellPtr rmB
      -- 5. Activations
      writeBetaT  = prim__softplus writeBetaRawT
      eraseVecT   = prim__sigmoid eraseRawT
      freeGatesT  = prim__sigmoid freeGatesRawT
      allocGateT  = prim__sigmoid allocGateRawT
      writeGateT  = prim__sigmoid writeGateRawT
      -- 6. Usage update
      writeUsageT = prim__sub (prim__add usageTPtr wwTPtr) (prim__mul usageTPtr wwTPtr)
      retentionT  = dncRetention onesScalar 0 freeGatesT rwTsPtrs onesScalar
      retClampedT = prim__clampMin retentionT 1.0e-10
      newUsageT   = prim__mul writeUsageT retClampedT
      -- 7. Allocation
      indicesT      = prim__argsort newUsageT 0 0
      sortedUsageT  = prim__clampMin (prim__gather newUsageT indicesT nI) 1.0e-6
      cumprodT      = prim__cumprod sortedUsageT 0
      slicedT       = prim__narrow cumprodT 0 0 (nI - 1)
      shiftedT      = prim__cat2 (prim__unsqueeze onesScalar 0) slicedT
      oneMinusUsageT = prim__sub onesScalar sortedUsageT
      sortedAllocT  = prim__mul oneMinusUsageT shiftedT
      allocT        = prim__scatterAdd indicesT sortedAllocT nI
      -- 8. Write content addressing
      cosScoresT    = prim__cosineSimilarity memTPtr (prim__unsqueeze writeKeyT 0) 1
      scaledScoresT = prim__mul writeBetaT cosScoresT
      contentWriteWT = prim__softmax scaledScoresT 0
      -- 9. Write weighting
      oneMinusAGT   = prim__sub onesScalar allocGateT
      blendT        = prim__add (prim__mul allocGateT allocT)
                                 (prim__mul oneMinusAGT contentWriteWT)
      newWriteWT    = prim__mul writeGateT blendT
      -- 10. Memory write
      eraseGateT    = prim__outer newWriteWT eraseVecT
      keepGateT     = prim__sub onesScalar eraseGateT
      erasedT       = prim__mul memTPtr keepGateT
      addGateT      = prim__outer newWriteWT addVecT
      newMemT       = prim__add erasedT addGateT
      -- 11. Link matrix update
      wiT           = prim__unsqueeze newWriteWT 1
      wjT           = prim__unsqueeze newWriteWT 0
      pjT           = prim__unsqueeze precTPtr 0
      decayT        = prim__sub (prim__sub onesScalar wiT) wjT
      decayClampT   = prim__clampMin decayT 0.0
      newLinkRawT   = prim__add (prim__mul decayClampT linkTPtr) (prim__mul wiT pjT)
      newLinkT      = prim__clampMin (dncZeroDiag nonDiagMaskT newLinkRawT) 0.0
      -- 12. Precedence update
      wSumT         = prim__sum newWriteWT
      oneMinusWSumT = prim__sub onesScalar wSumT
      newPrecT      = prim__add (prim__mul oneMinusWSumT precTPtr) newWriteWT
      -- 13. Read heads. Compute the link transpose ONCE outside the
      -- per-head recursion (was being computed R times — head-invariant).
      newLinkTransT = prim__transpose2d newLinkT
      (newRwTs, newRoTs) = dncReadHeads 0 rwTsPtrs newLinkT newLinkTransT newMemT
                              readKeysFlatT readBetasRawT readModesFlatT mI
      -- 14. Output FC
      allNewReadsT  = catReadOuts newRoTs
      outputInputT  = prim__cat2 hiddenV.tensorPtr allNewReadsT
      outputT       = prim__linear oW outputInputT oB
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
mkKaimingReadOuts : (r : Nat) -> (m : Nat) -> Double -> IO (Vect r AnyPtr)
mkKaimingReadOuts Z _ _ = pure []
mkKaimingReadOuts (S k) m bound = do
  vals <- traverse (\_ => randomRIO (-bound, bound)) (Vect.replicate m ())
  let mI = cast {to=Int} m
      buf = prim__allocDoubles mI
      buf' = packDoubles buf 0 vals
      ptr = prim__createState1d mI buf'
  rest <- mkKaimingReadOuts k m bound
  pure (ptr :: rest)

||| Build a `DncState r n m h i o CPU` matching the PyTorch reference's
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
dncLayer : RuntimeDType dt => {r, n, m, h, i, o : Nat} ->
             (paramPrefix : String) ->
             IO (DncState r n m h i o d dt WithGrad)
dncLayer pfx = do
  lstm <- lstmLayer {i = DncControllerInput r m i} {o = h} (pfx ++ "_lstm")
  -- 10 head FCs: xavier_uniform(gain=1.4) weights, normal(std=0.01) biases
  wkFc <- mkLinearWith {i = h} {o = m}     (pfx ++ "_writeKey")
            (xavierGain 1.4 uniform) (normal 0.0001)
  wbFc <- mkLinearWith {i = h} {o = 1}     (pfx ++ "_writeBeta")
            (xavierGain 1.4 uniform) (normal 0.0001)
  eFc  <- mkLinearWith {i = h} {o = m}     (pfx ++ "_erase")
            (xavierGain 1.4 uniform) (normal 0.0001)
  aFc  <- mkLinearWith {i = h} {o = m}     (pfx ++ "_add")
            (xavierGain 1.4 uniform) (normal 0.0001)
  fgFc <- mkLinearWith {i = h} {o = r}     (pfx ++ "_freeGates")
            (xavierGain 1.4 uniform) (normal 0.0001)
  agFc <- mkLinearWith {i = h} {o = 1}     (pfx ++ "_allocGate")
            (xavierGain 1.4 uniform) (normal 0.0001)
  wgFc <- mkLinearWith {i = h} {o = 1}     (pfx ++ "_writeGate")
            (xavierGain 1.4 uniform) (normal 0.0001)
  rkFc <- mkLinearWith {i = h} {o = r * m} (pfx ++ "_readKeys")
            (xavierGain 1.4 uniform) (normal 0.0001)
  rbFc <- mkLinearWith {i = h} {o = r}     (pfx ++ "_readBetas")
            (xavierGain 1.4 uniform) (normal 0.0001)
  rmFc <- mkLinearWith {i = h} {o = r * 3} (pfx ++ "_readModes")
            (xavierGain 1.4 uniform) (normal 0.0001)
  -- Output FC: kaiming_uniform default (LeCun), normal(std=0.01) bias
  oFc  <- mkLinearWith {i = DncOutputInput h r m} {o = o}
            (pfx ++ "_output") (ptKaimingDefault uniform) (normal 0.0001)
  -- memoryInit: shape (n, m) Xavier — fan_in=m, fan_out=n.
  let mnI = cast {to=Int} (m * n)
  memInitVals <- traverse (\_ => xavier uniform m n) (Vect.replicate (m * n) ())
  let miBuf = prim__allocDoubles mnI
      miBuf' = packDoubles miBuf 0 memInitVals
  memInitT <- tparam1d {n = m * n} (pfx ++ "_memoryInit") miBuf'
  -- initialReadOuts: PyTorch default kaiming_uniform on (R, m), bound=1/sqrt(m)
  let iroBound = 1.0 / prim__doubleSqrt (cast m)
  initReadOutsT <- mkKaimingReadOuts r m iroBound
  -- nonDiagMask: [n,n] (1 - I), built once and reused every timestep
  -- inside the link-matrix update. Saves ~1 + n*n + 1 prim FFI calls
  -- per step.
  let nonDiagMaskT = buildNonDiagMask n
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
                  DncState r n m h i o d dt g -> DncState r n m h i o d dt g
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
    primIO (prim__setRequiresGrad memInit.tensorPtr 1)
    case mem   of Nothing => pure (); Just t => primIO (prim__setRequiresGrad t.tensorPtr 1)
    case usage of Nothing => pure (); Just t => primIO (prim__setRequiresGrad t.tensorPtr 1)
    case ww    of Nothing => pure (); Just t => primIO (prim__setRequiresGrad t.tensorPtr 1)
    case prec  of Nothing => pure (); Just t => primIO (prim__setRequiresGrad t.tensorPtr 1)
    case link  of Nothing => pure (); Just t => primIO (prim__setRequiresGrad t.tensorPtr 1)
    pure (MkDnc lstm' wkFc' wbFc' eFc' aFc' fgFc' agFc' wgFc'
                rkFc' rbFc' rmFc' oFc'
                (retypeGrad memInit) iro nonDiag
                (map retypeGrad mem) (map retypeGrad usage)
                (map retypeGrad ww) (map retypeGrad prec)
                (map retypeGrad link) rwTs roTs)

export
dncLayerAny : RuntimeDType dt => {r, n, m, h, i, o : Nat} ->
                (paramPrefix : String) ->
                IO (AnyLayer i o d dt WithGrad)
dncLayerAny pid =
  map (MkAnyLayer (DncState r n m h)) (dncLayer {r} {n} {m} {h} {i} {o} pid)
