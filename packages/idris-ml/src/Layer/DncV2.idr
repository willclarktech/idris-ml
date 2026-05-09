module Layer.DncV2

import Data.Vect

import Compat.Random
import Device
import Init
import Layer.CoreV2
import Layer.LinearV2
import Layer.LstmV2
import Sampler
import Variable


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
catReadOutsAndInputTV2 : {k : Nat} -> Vect k AnyPtr -> AnyPtr -> AnyPtr
catReadOutsAndInputTV2 [] inp = inp
catReadOutsAndInputTV2 (ro :: rest) inp =
  prim__cat2 ro (catReadOutsAndInputTV2 rest inp)

-- Concat r read-output tensors. Crashes on r=0.
%default partial

catReadOutsTV2 : {k : Nat} -> Vect k AnyPtr -> AnyPtr
catReadOutsTV2 [] = idris_crash "DncV2: catReadOutsTV2 r=0"
catReadOutsTV2 (h :: t) = catRest h t
  where
    catRest : AnyPtr -> {k' : Nat} -> Vect k' AnyPtr -> AnyPtr
    catRest acc [] = acc
    catRest acc (h' :: rest) = catRest (prim__cat2 acc h') rest

-- Compute prod_j (1 - free_gate_j * prev_read_w_j) over r heads.
dncRetentionTV2 : {k : Nat} -> Int -> AnyPtr -> Vect k AnyPtr -> AnyPtr -> AnyPtr
dncRetentionTV2 _ _ [] acc = acc
dncRetentionTV2 idx freeGatesT (rw :: rws) acc =
  let fg = prim__select freeGatesT 0 idx
      factor = prim__sub (prim__createScalar 1.0 0) (prim__mul fg rw)
  in dncRetentionTV2 (idx + 1) freeGatesT rws (prim__mul acc factor)

-- Zero the diagonal of a [n,n] matrix.
dncZeroDiagTV2 : Int -> AnyPtr -> AnyPtr
dncZeroDiagTV2 nI matT =
  let numElems = nI * nI
      buf = prim__allocDoubles numElems
      buf' = fillMaskOffDiag buf 0 nI numElems
      maskT = prim__create2d nI nI buf' 0
  in prim__mul matT maskT
  where
    fillMaskOffDiag : AnyPtr -> Int -> Int -> Int -> AnyPtr
    fillMaskOffDiag b i nn numE = if i >= numE then b else
      let row = i `div` nn
          col = i `mod` nn
          val = if row == col then 0.0 else 1.0
          b' = prim__setDouble b i val
      in fillMaskOffDiag b' (i + 1) nn numE

-- Per-head read processing for r heads.
dncReadHeadsTV2 : {k : Nat} -> Int -> Vect k AnyPtr ->
                  AnyPtr -> AnyPtr ->
                  AnyPtr -> AnyPtr -> AnyPtr ->
                  Int ->
                  (Vect k AnyPtr, Vect k AnyPtr)
dncReadHeadsTV2 _ [] _ _ _ _ _ _ = ([], [])
dncReadHeadsTV2 idx (prevRw :: restRws) linkT memT keysT betasT modesT mI =
  let headKeyT      = prim__narrow keysT 0 (idx * mI) mI
      headBetaPtr   = prim__select betasT 0 idx
      headBetaT     = prim__softplus headBetaPtr
      headModesRawT = prim__narrow modesT 0 (idx * 3) 3
      headModesT    = prim__softmax headModesRawT 0
      cosScoresT    = prim__cosineSimilarity memT (prim__unsqueeze headKeyT 0) 1
      scaledScoresT = prim__mul headBetaT cosScoresT
      contentRwT    = prim__softmax scaledScoresT 0
      forwardT      = prim__matmul linkT prevRw
      linkTransT    = prim__transpose2d linkT
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
      (restRws', restRos') = dncReadHeadsTV2 (idx + 1) restRws linkT memT keysT betasT modesT mI
  in (rwT :: restRws', roT :: restRos')


----------------------------------------------------------------------
-- DncStateV2 — typed-surface DNC (Path C)
----------------------------------------------------------------------
--
-- Mirrors V1 `Layer/Dnc.idr`'s `applyVarTensor`. 11 FCs + LSTM
-- controller + 7 state tensors (memory, usage, write weights,
-- precedence, link, R read weights, R read outputs).

public export
data DncStateV2 :
  (r : Nat) -> (n : Nat) -> (m : Nat) -> (h : Nat) ->
  Nat -> Nat -> (0 _ : Device) -> Type
  where
  MkDncV2 :
    LstmStateV2 (DncControllerInput r m i) h d ->
    LinearStateV2 h m d ->                  -- writeKeyFc
    LinearStateV2 h 1 d ->                  -- writeBetaFc
    LinearStateV2 h m d ->                  -- eraseFc
    LinearStateV2 h m d ->                  -- addFc
    LinearStateV2 h r d ->                  -- freeGatesFc
    LinearStateV2 h 1 d ->                  -- allocGateFc
    LinearStateV2 h 1 d ->                  -- writeGateFc
    LinearStateV2 h (r * m) d ->            -- readKeysFc
    LinearStateV2 h r d ->                  -- readBetasFc
    LinearStateV2 h (r * 3) d ->            -- readModesFc
    LinearStateV2 (DncOutputInput h r m) o d ->  -- outputFc
    Maybe (TVar [n, m] d) ->                -- memT
    Maybe (TVec n d) ->                     -- usageT
    Maybe (TVec n d) ->                     -- writeWtT
    Maybe (TVec n d) ->                     -- precedenceT
    Maybe (TVar [n, n] d) ->                -- linkT
    Maybe (Vect r AnyPtr) ->                -- read weight tensor handles
    Maybe (Vect r AnyPtr) ->                -- read output tensor handles
    DncStateV2 r n m h i o d


----------------------------------------------------------------------
-- State init helpers
----------------------------------------------------------------------

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
applyDncV2 : {r, n, m, h, i, o : Nat} ->
             DncStateV2 r n m h i o d ->
             TVec i d ->
             (DncStateV2 r n m h i o d, TVec o d)
applyDncV2 {r} {n} {m}
           (MkDncV2 lstm wkFc wbFc eFc aFc fgFc agFc wgFc rkFc rbFc rmFc oFc
                    memT usageT wwT precT linkT rwTs roTs) input =
  let memTPtr = case memT of
                  Just t => t.tensorPtr
                  Nothing => constState2d n m 1.0e-6
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
      roTsPtrs = the (Vect r AnyPtr) $ case roTs of
                   Just ts => ts
                   Nothing => mkZeroVectM r m
      -- 1. cat(readOuts, input) -> [r*m + i]
      lstmInputPtr = catReadOutsAndInputTV2 roTsPtrs input.tensorPtr
      lstmInputV = the (TVec (DncControllerInput r m i) d) (MkTVar lstmInputPtr Nothing)
      -- 2. LSTM forward
      (updLstm, hiddenV) = applyLstmV2 lstm lstmInputV
      -- 3. Cell-state for FCs
      cellPtr = case updLstm.cellT of
                  Just c => c.tensorPtr
                  Nothing => idris_crash "DncV2: cell tensor missing post-LSTM"
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
      mI = cast {to=Int} m
      nI = cast {to=Int} n
      onesScalar = prim__createScalar 1.0 0
      -- 4. 11 FCs
      writeKeyT      = prim__add (prim__mv wkW cellPtr) wkB
      writeBetaRawT  = prim__add (prim__mv wbW cellPtr) wbB
      eraseRawT      = prim__add (prim__mv eW  cellPtr) eB
      addVecT        = prim__add (prim__mv aW  cellPtr) aB
      freeGatesRawT  = prim__add (prim__mv fgW cellPtr) fgB
      allocGateRawT  = prim__add (prim__mv agW cellPtr) agB
      writeGateRawT  = prim__add (prim__mv wgW cellPtr) wgB
      readKeysFlatT  = prim__add (prim__mv rkW cellPtr) rkB
      readBetasRawT  = prim__add (prim__mv rbW cellPtr) rbB
      readModesFlatT = prim__add (prim__mv rmW cellPtr) rmB
      -- 5. Activations
      writeBetaT  = prim__softplus writeBetaRawT
      eraseVecT   = prim__sigmoid eraseRawT
      freeGatesT  = prim__sigmoid freeGatesRawT
      allocGateT  = prim__sigmoid allocGateRawT
      writeGateT  = prim__sigmoid writeGateRawT
      -- 6. Usage update
      writeUsageT = prim__sub (prim__add usageTPtr wwTPtr) (prim__mul usageTPtr wwTPtr)
      retentionT  = dncRetentionTV2 0 freeGatesT rwTsPtrs onesScalar
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
      newLinkT      = prim__clampMin (dncZeroDiagTV2 nI newLinkRawT) 0.0
      -- 12. Precedence update
      wSumT         = prim__sum newWriteWT
      oneMinusWSumT = prim__sub onesScalar wSumT
      newPrecT      = prim__add (prim__mul oneMinusWSumT precTPtr) newWriteWT
      -- 13. Read heads
      (newRwTs, newRoTs) = dncReadHeadsTV2 0 rwTsPtrs newLinkT newMemT
                              readKeysFlatT readBetasRawT readModesFlatT mI
      -- 14. Output FC
      allNewReadsT  = catReadOutsTV2 newRoTs
      outputInputT  = prim__cat2 hiddenV.tensorPtr allNewReadsT
      outputT       = prim__add (prim__mv oW outputInputT) oB
  in ( MkDncV2 updLstm wkFc wbFc eFc aFc fgFc agFc wgFc rkFc rbFc rmFc oFc
        (Just (MkTVar newMemT Nothing))
        (Just (MkTVar newUsageT Nothing))
        (Just (MkTVar newWriteWT Nothing))
        (Just (MkTVar newPrecT Nothing))
        (Just (MkTVar newLinkT Nothing))
        (Just newRwTs)
        (Just newRoTs)
     , MkTVar outputT Nothing )


----------------------------------------------------------------------
-- Constructor
----------------------------------------------------------------------

||| Build a `DncStateV2 r n m h i o CPU` with default init: LSTM
||| controller + 11 FCs (Xavier via LinearV2), memory init to 1e-6,
||| addresses/usage/precedence/link/reads zero. State persistent.
export
dncLayerV2 : {r, n, m, h, i, o : Nat} ->
             (paramPrefix : String) ->
             IO (DncStateV2 r n m h i o CPU)
dncLayerV2 pfx = do
  lstm <- lstmLayerV2 {i = DncControllerInput r m i} {o = h} (pfx ++ "_lstm")
  wkFc <- linearLayerV2 {i = h} {o = m}        (pfx ++ "_writeKey")
  wbFc <- linearLayerV2 {i = h} {o = 1}        (pfx ++ "_writeBeta")
  eFc  <- linearLayerV2 {i = h} {o = m}        (pfx ++ "_erase")
  aFc  <- linearLayerV2 {i = h} {o = m}        (pfx ++ "_add")
  fgFc <- linearLayerV2 {i = h} {o = r}        (pfx ++ "_freeGates")
  agFc <- linearLayerV2 {i = h} {o = 1}        (pfx ++ "_allocGate")
  wgFc <- linearLayerV2 {i = h} {o = 1}        (pfx ++ "_writeGate")
  rkFc <- linearLayerV2 {i = h} {o = r * m}    (pfx ++ "_readKeys")
  rbFc <- linearLayerV2 {i = h} {o = r}        (pfx ++ "_readBetas")
  rmFc <- linearLayerV2 {i = h} {o = r * 3}    (pfx ++ "_readModes")
  oFc  <- linearLayerV2 {i = DncOutputInput h r m} {o = o} (pfx ++ "_output")
  let memTV : TVar [n, m] CPU
      memTV = MkTVar (constState2d n m 1.0e-6) Nothing
      usageTV : TVec n CPU
      usageTV = MkTVar (zeroState1d n) Nothing
      wwTV : TVec n CPU
      wwTV = MkTVar (zeroState1d n) Nothing
      precTV : TVec n CPU
      precTV = MkTVar (zeroState1d n) Nothing
      linkTV : TVar [n, n] CPU
      linkTV = MkTVar (zeroState2d n n) Nothing
  pure $ MkDncV2 lstm wkFc wbFc eFc aFc fgFc agFc wgFc rkFc rbFc rmFc oFc
                 (Just memTV) (Just usageTV) (Just wwTV) (Just precTV)
                 (Just linkTV) (Just (mkZeroVectN r n)) (Just (mkZeroVectM r m))

||| Reset DNC state to fresh persistent zero/init tensors.
export
resetDncStateV2 : {r, n, m, h : Nat} ->
                  DncStateV2 r n m h i o d -> DncStateV2 r n m h i o d
resetDncStateV2 (MkDncV2 lstm wkFc wbFc eFc aFc fgFc agFc wgFc rkFc rbFc rmFc oFc
                          _ _ _ _ _ _ _) =
  let memTV : TVar [n, m] _
      memTV = MkTVar (constState2d n m 1.0e-6) Nothing
      usageTV : TVec n _
      usageTV = MkTVar (zeroState1d n) Nothing
      wwTV : TVec n _
      wwTV = MkTVar (zeroState1d n) Nothing
      precTV : TVec n _
      precTV = MkTVar (zeroState1d n) Nothing
      linkTV : TVar [n, n] _
      linkTV = MkTVar (zeroState2d n n) Nothing
  in MkDncV2 (resetLstmStateV2 lstm) wkFc wbFc eFc aFc fgFc agFc wgFc rkFc rbFc rmFc oFc
             (Just memTV) (Just usageTV) (Just wwTV) (Just precTV)
             (Just linkTV) (Just (mkZeroVectN r n)) (Just (mkZeroVectM r m))


----------------------------------------------------------------------
-- LayerLikeV2 instance
----------------------------------------------------------------------

public export
{r, n, m, h : Nat} ->
  LayerLikeV2 (DncStateV2 r n m h) where
  applyTVar st@(MkDncV2 _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _) input =
    applyDncV2 st input
  layerPrefixV2 _ = "dncV2"
  resetStateV2 st = resetDncStateV2 st

export
dncLayerV2Any : {r, n, m, h, i, o : Nat} ->
                (paramPrefix : String) ->
                IO (AnyLayerV2 i o CPU)
dncLayerV2Any pid =
  map (MkAnyLayerV2 (DncStateV2 r n m h)) (dncLayerV2 {r} {n} {m} {h} {i} {o} pid)
