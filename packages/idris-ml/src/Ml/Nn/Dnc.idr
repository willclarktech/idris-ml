||| `Dnc` — Differentiable Neural Computer (Graves et al. 2016) on the v1
||| `Nn` surface, implementing `Recurrent`. The largest composite: an
||| `Nn.Lstm` controller + eleven `Nn.Linear` heads (write key/beta/erase/
||| add, free/alloc/write gates, read keys/betas/modes, output) plus a
||| dynamic external memory with usage-based allocation and a temporal link
||| matrix. `params` composes the sub-layers + the learned memory-init;
||| `scopedChild` + `named` nest everything under `dnc_<n>.…`. The memory
||| dynamics are transcribed unchanged from the legacy.
module Ml.Nn.Dnc

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

----------------------------------------------------------------------
-- Shapes (type level)
----------------------------------------------------------------------

public export
DncControllerInput : Nat -> Nat -> Nat -> Nat
DncControllerInput r m i = r * m + i

public export
DncOutputInput : Nat -> Nat -> Nat -> Nat
DncOutputInput h r m = h + r * m

----------------------------------------------------------------------
-- Raw per-step helpers (pure prim composition, transcribed from the
-- legacy Layer.Dnc)
----------------------------------------------------------------------

zeroState1d : {0 ex : Executor} -> Backend ex dt => (n : Nat) -> AnyPtr
zeroState1d n =
  let nI = cast {to=Int} n
  in dtCreateState1d {ex} {t=dt} nI (prim__allocDoubles nI) (deviceStreamTag {ex})

zeroState2d : {0 ex : Executor} -> Backend ex dt => (a, b : Nat) -> AnyPtr
zeroState2d a b =
  let aI = cast {to=Int} a; bI = cast {to=Int} b
  in dtCreateState2d {ex} {t=dt} aI bI (prim__allocDoubles (aI * bI)) (deviceStreamTag {ex})

packDoubles : AnyPtr -> Int -> Vect k Double -> AnyPtr
packDoubles buf _ []            = buf
packDoubles buf off (x :: rest) = packDoubles (prim__setDouble buf off x) (off + 1) rest

mkZeroVect : {0 ex : Executor} -> Backend ex dt => (r : Nat) -> Nat -> Vect r AnyPtr
mkZeroVect Z _     = []
mkZeroVect (S k) n = zeroState1d {ex} {dt} n :: mkZeroVect {ex} {dt} k n

catReadOutsAndInput : {0 ex : Executor} -> UserExecutorTraining ex => {k : Nat} -> Vect k AnyPtr -> AnyPtr -> AnyPtr
catReadOutsAndInput [] inp           = inp
catReadOutsAndInput (ro :: rest) inp = primCat2 {ex} ro (catReadOutsAndInput {ex} rest inp)

partial
catReadOuts : {0 ex : Executor} -> UserExecutorTraining ex => {k : Nat} -> Vect k AnyPtr -> AnyPtr
catReadOuts []       = idris_crash "Dnc: catReadOuts r=0"
catReadOuts (h :: t) = catRest h t
  where
    catRest : AnyPtr -> {k' : Nat} -> Vect k' AnyPtr -> AnyPtr
    catRest acc []           = acc
    catRest acc (h' :: rest) = catRest (primCat2 {ex} acc h') rest

dncRetention : {0 ex : Executor} -> UserExecutorTraining ex => {k : Nat} -> AnyPtr -> Int -> AnyPtr -> Vect k AnyPtr -> AnyPtr -> AnyPtr
dncRetention _ _ _ [] acc                              = acc
dncRetention onesScalar idx freeGatesT (rw :: rws) acc =
  let fg = primSelect {ex} freeGatesT 0 idx
      factor = primSub {ex} onesScalar (primMul {ex} fg rw)
  in dncRetention {ex} onesScalar (idx + 1) freeGatesT rws (primMul {ex} acc factor)

buildNonDiagMask : {0 ex : Executor} -> Backend ex dt => (n : Nat) -> AnyPtr
buildNonDiagMask n =
  let nI = cast {to=Int} n
      numElems = nI * nI
      buf'     = fillOffDiag (prim__allocDoubles numElems) 0 nI numElems
  in dtCreateState2d {ex} {t=dt} nI nI buf' (deviceStreamTag {ex})
  where
    fillOffDiag : AnyPtr -> Int -> Int -> Int -> AnyPtr
    fillOffDiag b idx nn numE = if idx >= numE then b else
      let val = if (idx `div` nn) == (idx `mod` nn) then 0.0 else 1.0
      in fillOffDiag (prim__setDouble b idx val) (idx + 1) nn numE

dncZeroDiag : {0 ex : Executor} -> UserExecutorTraining ex => AnyPtr -> AnyPtr -> AnyPtr
dncZeroDiag maskPtr matT = primMul {ex} matT maskPtr

dncReadHeads : {0 ex : Executor} -> UserExecutorTraining ex => {k : Nat} -> Int -> Vect k AnyPtr ->
               AnyPtr -> AnyPtr -> AnyPtr -> AnyPtr -> AnyPtr -> AnyPtr -> Int ->
               (Vect k AnyPtr, Vect k AnyPtr)
dncReadHeads _ [] _ _ _ _ _ _ _                                                   = ([], [])
dncReadHeads idx (prevRw :: restRws) linkT linkTransT memT keysT betasT modesT mI =
  let headKeyT      = primNarrow {ex} keysT 0 (idx * mI) mI
      headBetaT            = primSoftplus {ex} (primSelect {ex} betasT 0 idx)
      headModesT           = primSoftmax {ex} (primNarrow {ex} modesT 0 (idx * 3) 3) 0
      cosScoresT           = primCosineSimilarity {ex} memT (primUnsqueeze {ex} headKeyT 0) 1
      contentRwT           = primSoftmax {ex} (primMul {ex} headBetaT cosScoresT) 0
      forwardT             = primMatmul {ex} linkT prevRw
      backwardT            = primMatmul {ex} linkTransT prevRw
      scaledBack           = primMul {ex} (primSelect {ex} headModesT 0 0) backwardT
      scaledContent        = primMul {ex} (primSelect {ex} headModesT 0 1) contentRwT
      scaledForward        = primMul {ex} (primSelect {ex} headModesT 0 2) forwardT
      rwSumT               = primAdd {ex} (primAdd {ex} scaledBack scaledContent) scaledForward
      rwClampedT           = primClampMin {ex} rwSumT 1.0e-10
      rwNormSumT           = primAddScalar {ex} (primSum {ex} rwClampedT) 1.0e-10
      rwT                  = primDiv {ex} rwClampedT rwNormSumT
      roT                  = primMatmul {ex} rwT memT
      (restRws', restRos') = dncReadHeads {ex} (idx + 1) restRws linkT linkTransT memT keysT betasT modesT mI
  in (rwT :: restRws', roT :: restRos')

-- Apply a head FC to the controller cell vector (W·cell + b).
fcApply : {0 ex : Executor} -> UserExecutorTraining ex => {0 a, b : Nat} -> {0 g : GradMode} ->
          AnyPtr -> Linear a b ex dt g -> AnyPtr
fcApply cellPtr fc = primLinear {ex} fc.weightT.tensorPtr cellPtr fc.biasT.tensorPtr

----------------------------------------------------------------------
-- The layer
----------------------------------------------------------------------

||| DNC cell. Controller + 11 heads + learned memory-init are params;
||| memory / usage / write-weights / precedence / link / per-head read
||| weights + outputs are per-sequence state. `initReadOutsT` (fixed
||| Kaiming) + `nonDiagMaskT` (precomputed 1−I) are non-param buffers.
public export
record Dnc (r : Nat) (n : Nat) (m : Nat) (h : Nat) (i : Nat) (o : Nat) (0 ex : Executor) (0 dt : DType) (0 g : GradMode) where
  constructor MkDnc
  controller    : Lstm (DncControllerInput r m i) h ex dt g
  writeKeyFc    : Linear h m ex dt g
  writeBetaFc   : Linear h 1 ex dt g
  eraseFc       : Linear h m ex dt g
  addFc         : Linear h m ex dt g
  freeGatesFc   : Linear h r ex dt g
  allocGateFc   : Linear h 1 ex dt g
  writeGateFc   : Linear h 1 ex dt g
  readKeysFc    : Linear h (r * m) ex dt g
  readBetasFc   : Linear h r ex dt g
  readModesFc   : Linear h (r * 3) ex dt g
  outputFc      : Linear (DncOutputInput h r m) o ex dt g
  memInitT      : TVec (m * n) ex dt g
  initReadOutsT : Vect r AnyPtr
  nonDiagMaskT  : AnyPtr
  memT          : Maybe (Tensor [n, m] ex dt g)
  usageT        : Maybe (TVec n ex dt g)
  writeWtT      : Maybe (TVec n ex dt g)
  precedenceT   : Maybe (TVec n ex dt g)
  linkT         : Maybe (Tensor [n, n] ex dt g)
  readWtsT      : Maybe (Vect r AnyPtr)
  readOutsT     : Maybe (Vect r AnyPtr)

-- IO step body for the DNC cell, shared by the (linear) `recurStep`. Threads
-- the LSTM controller through ω fields; the memory dynamics are unchanged from
-- the legacy Layer.Dnc. Kept as a top-level helper (not an interface method)
-- so the linear `Recurrent` instance can delegate to it at the IO boundary.
-- partial: catReadOuts crashes on r=0 (a DNC needs ≥1 read head); the legacy
-- DNC was wholesale %default partial for the same reason.
partial
dncStepIO : {0 ex : Executor} -> Backend ex dt => {r, n, m, h : Nat} -> {i, o : Nat} ->
            Dnc r n m h i o ex dt WithGrad -> Tensor [i] ex dt WithGrad ->
            IO (Dnc r n m h i o ex dt WithGrad, Tensor [o] ex dt WithGrad)
dncStepIO {i} {o} st input = assert_total $ do
    let nI = cast {to=Int} n
        mI         = cast {to=Int} m
        initMemPtr = primReshape2d {ex} (primSigmoid {ex} st.memInitT.tensorPtr) nI mI
        memTPtr    = maybe initMemPtr (.tensorPtr) st.memT
        usagePtr   = maybe (zeroState1d {ex} {dt} n) (.tensorPtr) st.usageT
        wwPtr      = maybe (zeroState1d {ex} {dt} n) (.tensorPtr) st.writeWtT
        precPtr    = maybe (zeroState1d {ex} {dt} n) (.tensorPtr) st.precedenceT
        linkPtr    = maybe (zeroState2d {ex} {dt} n n) (.tensorPtr) st.linkT
        rwTsPtrs   = maybe (mkZeroVect {ex} {dt} r n) id st.readWtsT
        roTsPtrs   = maybe st.initReadOutsT id st.readOutsT
        lstmInputV = the (TVec (DncControllerInput r m i) ex dt WithGrad)
                         (MkTensor (catReadOutsAndInput {ex} roTsPtrs input.tensorPtr) Nothing)
    -- Step the LSTM controller via `lstmStepIO` (ω in/out; controller threaded
    -- ω internally, the cell handle is the single-owner linear resource).
    (updCtrl, hiddenV) <- lstmStepIO st.controller lstmInputV
    let cellPtr = maybe (zeroState1d {ex} {dt} h) (.tensorPtr) updCtrl.cellT
        onesScalar     = dtCreateScalar {ex} {t=dt} 1.0 0 (deviceStreamTag {ex})
        writeKeyT      = fcApply {ex} cellPtr st.writeKeyFc
        writeBetaT     = primSoftplus {ex} (fcApply {ex} cellPtr st.writeBetaFc)
        eraseVecT      = primSigmoid {ex} (fcApply {ex} cellPtr st.eraseFc)
        addVecT        = fcApply {ex} cellPtr st.addFc
        freeGatesT     = primSigmoid {ex} (fcApply {ex} cellPtr st.freeGatesFc)
        allocGateT     = primSigmoid {ex} (fcApply {ex} cellPtr st.allocGateFc)
        writeGateT     = primSigmoid {ex} (fcApply {ex} cellPtr st.writeGateFc)
        readKeysFlatT  = fcApply {ex} cellPtr st.readKeysFc
        readBetasRawT  = fcApply {ex} cellPtr st.readBetasFc
        readModesFlatT = fcApply {ex} cellPtr st.readModesFc
        -- Usage update
        writeUsageT = primSub {ex} (primAdd {ex} usagePtr wwPtr) (primMul {ex} usagePtr wwPtr)
        retentionT  = dncRetention {ex} onesScalar 0 freeGatesT rwTsPtrs onesScalar
        newUsageT   = primMul {ex} writeUsageT (primClampMin {ex} retentionT 1.0e-10)
        -- Allocation
        indicesT     = primArgsort {ex} newUsageT 0 0
        sortedUsageT = primClampMin {ex} (primGather {ex} newUsageT indicesT nI) 1.0e-6
        slicedT      = primNarrow {ex} (primCumprod {ex} sortedUsageT 0) 0 0 (nI - 1)
        shiftedT     = primCat2 {ex} (primUnsqueeze {ex} onesScalar 0) slicedT
        sortedAllocT = primMul {ex} (primSub {ex} onesScalar sortedUsageT) shiftedT
        allocT       = primScatterAdd {ex} indicesT sortedAllocT nI
        -- Write content addressing + weighting
        cwScoresT      = primMul {ex} writeBetaT (primCosineSimilarity {ex} memTPtr (primUnsqueeze {ex} writeKeyT 0) 1)
        contentWriteWT = primSoftmax {ex} cwScoresT 0
        blendT         = primAdd {ex} (primMul {ex} allocGateT allocT)
                                   (primMul {ex} (primSub {ex} onesScalar allocGateT) contentWriteWT)
        newWriteWT    = primMul {ex} writeGateT blendT
        -- Memory write
        erasedT = primMul {ex} memTPtr (primSub {ex} onesScalar (primOuter {ex} newWriteWT eraseVecT))
        newMemT = primAdd {ex} erasedT (primOuter {ex} newWriteWT addVecT)
        -- Link + precedence
        wiT         = primUnsqueeze {ex} newWriteWT 1
        wjT         = primUnsqueeze {ex} newWriteWT 0
        decayClampT = primClampMin {ex} (primSub {ex} (primSub {ex} onesScalar wiT) wjT) 0.0
        newLinkRawT = primAdd {ex} (primMul {ex} decayClampT linkPtr) (primMul {ex} wiT (primUnsqueeze {ex} precPtr 0))
        newLinkT    = primClampMin {ex} (dncZeroDiag {ex} st.nonDiagMaskT newLinkRawT) 0.0
        newPrecT    = primAdd {ex} (primMul {ex} (primSub {ex} onesScalar (primSum {ex} newWriteWT)) precPtr) newWriteWT
        -- Read heads
        newLinkTransT      = primTranspose2d {ex} newLinkT
        (newRwTs, newRoTs) = dncReadHeads {ex} 0 rwTsPtrs newLinkT newLinkTransT newMemT
                               readKeysFlatT readBetasRawT readModesFlatT mI
        outputT = primLinear {ex} st.outputFc.weightT.tensorPtr
                    (primCat2 {ex} hiddenV.tensorPtr (catReadOuts {ex} newRoTs)) st.outputFc.biasT.tensorPtr
    pure ( { controller := updCtrl
           , memT := Just (MkTensor newMemT Nothing), usageT := Just (MkTensor newUsageT Nothing)
           , writeWtT := Just (MkTensor newWriteWT Nothing), precedenceT := Just (MkTensor newPrecT Nothing)
           , linkT := Just (MkTensor newLinkT Nothing), readWtsT := Just newRwTs, readOutsT := Just newRoTs } st
         , MkTensor outputT Nothing )

||| Params for the DNC cell. The controller + 11 heads + memory-init all bind
||| at ω, so the sub-models reuse their `Params` methods for both the reflected
||| list and the rebuild; the buffers + per-sequence state ride at ω.
public export
{r, n, m, h : Nat} -> Params (Dnc r n m h) where
  params (MkDnc ctrl wk wb er ad fg ag wg rk rb rm outFc mi iros ndm memS usS wwS prS lkS rwS roS) =
    params ctrl ++ params wk ++ params wb ++ params er ++ params ad ++ params fg
      ++ params ag ++ params wg ++ params rk ++ params rb ++ params rm ++ params outFc
      ++ [toParam mi]
  reflect (MkDnc ctrl wk wb er ad fg ag wg rk rb rm outFc mi iros ndm memS usS wwS prS lkS rwS roS) =
    let (MkBang pc # ctrl')  = reflect ctrl
        (MkBang pwk # wk')    = reflect wk
        (MkBang pwb # wb')    = reflect wb
        (MkBang per # er')    = reflect er
        (MkBang pad # ad')    = reflect ad
        (MkBang pfg # fg')    = reflect fg
        (MkBang pag # ag')    = reflect ag
        (MkBang pwg # wg')    = reflect wg
        (MkBang prk # rk')    = reflect rk
        (MkBang prb # rb')    = reflect rb
        (MkBang prm # rm')    = reflect rm
        (MkBang pof # outFc') = reflect outFc in
    MkBang (pc ++ pwk ++ pwb ++ per ++ pad ++ pfg
              ++ pag ++ pwg ++ prk ++ prb ++ prm ++ pof
              ++ [toParam mi])
      # MkDnc ctrl' wk' wb' er' ad' fg' ag' wg' rk' rb' rm' outFc' mi iros ndm memS usS wwS prS lkS rwS roS
  castGrad (MkDnc ctrl wk wb er ad fg ag wg rk rb rm outFc mi iros ndm memS usS wwS prS lkS rwS roS) =
    MkDnc (castGrad ctrl) (castGrad wk) (castGrad wb) (castGrad er) (castGrad ad)
          (castGrad fg) (castGrad ag) (castGrad wg) (castGrad rk) (castGrad rb)
          (castGrad rm) (castGrad outFc) (retypeGrad mi) iros ndm
          (map retypeGrad memS) (map retypeGrad usS) (map retypeGrad wwS)
          (map retypeGrad prS) (map retypeGrad lkS) rwS roS
  discard (MkDnc _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _) = pure ()

||| Linear-resource recurrent step. As with NTM, the body is large and
||| raw-prim-heavy and threads the LSTM controller through an ω record field,
||| so it consumes the linear cell and delegates to the IO step helper
||| (`dncStepIO`) at the linear boundary (the handle-level guarantee holds;
||| controller threading stays ω in IO). The inline `L IO` body lands later.
public export
{r, n, m, h : Nat} -> Recurrent (Dnc r n m h) where
  -- Pattern-match to discharge linearity (fields bind at ω), rebuild an ω
  -- cell, and delegate to the IO step; the returned cell rides the linear pair.
  recurStep (MkDnc ctrl wk wb er ad fg ag wg rk rb rm outFc mi iros ndm memS usS wwS prS lkS rwS roS) input = do
    (updSt, out) <- liftIO1
      (dncStepIO (MkDnc ctrl wk wb er ad fg ag wg rk rb rm outFc mi iros ndm
                        memS usS wwS prS lkS rwS roS) input)
    pure1 (MkBang out # updSt)
  recurReset (MkDnc ctrl wk wb er ad fg ag wg rk rb rm outFc mi iros ndm _ _ _ _ _ _ _) =
    MkDnc (recurReset ctrl) wk wb er ad fg ag wg rk rb rm outFc mi iros ndm
          Nothing Nothing Nothing Nothing Nothing Nothing Nothing

-- Build r fixed Kaiming-uniform read-output buffers (non-learnable).
mkKaimingReadOuts : {0 ex : Executor} -> Backend ex dt => (r : Nat) -> (m : Nat) -> Double -> IO (Vect r AnyPtr)
mkKaimingReadOuts Z _ _         = pure []
mkKaimingReadOuts (S k) m bound = do
  vals <- traverse (\_ => randomRIO (-bound, bound)) (Vect.replicate m ())
  let buf = packDoubles (prim__allocDoubles (cast m)) 0 vals
      ptr = dtCreateState1d {ex} {t=dt} (cast m) buf (deviceStreamTag {ex})
  rest <- mkKaimingReadOuts {ex} {dt} k m bound
  pure (ptr :: rest)

||| Construct a `Dnc` inside an `Init` derivation, mirroring the PyTorch
||| reference inits (LSTM default; head FCs xavier-1.4 + bias N(0,0.01);
||| output head LeCun-ish; memory-init xavier-normal; fixed Kaiming
||| read-outs; precomputed 1−I mask). Nests under `<scope>.dnc_<n>.…`.
export partial
dnc : KnownGrad g => {0 ex : Executor} -> Backend ex dt => {r, n, m, h, i, o : Nat} -> Init (Dnc r n m h i o ex dt g)
dnc = scopedChild "dnc" $ do
  let xavStd : (a, b : Nat) -> Double
      xavStd a b = 1.4 * sqrt (2.0 / cast {to=Double} (a + b))
      biasStd    = 0.01
  -- Sub-modules built at the requested `g`, each `the`-annotated so `g` (+ ex/dt)
  -- flow up front (a bare `{g}` pin leaves Backend ?ex ?dt unsolved at the bind,
  -- as in transformerBlock). Only the directly-created memInit param needs
  -- explicit weakening (read-out buffers + mask are raw `AnyPtr`, not g-typed;
  -- state fields are Nothing).
  ctrl <- the (Init (Lstm (DncControllerInput r m i) h ex dt g))
              (named "controller" (lstmWithBias {i = DncControllerInput r m i} {o = h}
                                    (1.0 / sqrt (cast {to=Double} h))))
  wkFc <- the (Init (Linear h m ex dt g)) (named "write_key"  (linearWith {i=h} {o=m} (xavStd h m) biasStd))
  wbFc <- the (Init (Linear h 1 ex dt g)) (named "write_beta" (linearWith {i=h} {o=1} (xavStd h 1) biasStd))
  eFc  <- the (Init (Linear h m ex dt g)) (named "erase"      (linearWith {i=h} {o=m} (xavStd h m) biasStd))
  aFc  <- the (Init (Linear h m ex dt g)) (named "add"        (linearWith {i=h} {o=m} (xavStd h m) biasStd))
  fgFc <- the (Init (Linear h r ex dt g)) (named "free_gates" (linearWith {i=h} {o=r} (xavStd h r) biasStd))
  agFc <- the (Init (Linear h 1 ex dt g)) (named "alloc_gate" (linearWith {i=h} {o=1} (xavStd h 1) biasStd))
  wgFc <- the (Init (Linear h 1 ex dt g)) (named "write_gate" (linearWith {i=h} {o=1} (xavStd h 1) biasStd))
  rkFc <- the (Init (Linear h (r * m) ex dt g)) (named "read_keys"  (linearWith {i=h} {o=r * m} (xavStd h (r*m)) biasStd))
  rbFc <- the (Init (Linear h r ex dt g))       (named "read_betas" (linearWith {i=h} {o=r}     (xavStd h r)     biasStd))
  rmFc <- the (Init (Linear h (r * 3) ex dt g)) (named "read_modes" (linearWith {i=h} {o=r * 3} (xavStd h (r*3)) biasStd))
  oFc  <- the (Init (Linear (DncOutputInput h r m) o ex dt g))
              -- He-uniform + normal bias, matching the reference's
              -- `kaiming_uniform_(output_fc.weight)`; see Ntm.idr for why this
              -- is not the shared dense contract.
              (named "output" (linearUniformWith {i=DncOutputInput h r m} {o=o}
                                 (sqrt (6.0 / cast {to=Double} (DncOutputInput h r m)))
                                 biasStd))
  mname <- freshChild "memory_init"
  memInit <- liftIO $ tparam1dNormal {ex} {dt} {n = m * n} mname 0.0 (sqrt (2.0 / cast {to=Double} (m + n)))
  iros <- liftIO $ mkKaimingReadOuts {ex} {dt} r m (1.0 / sqrt (cast {to=Double} m))
  case sgrad {g} of
    SWithGrad => pure (MkDnc ctrl wkFc wbFc eFc aFc fgFc agFc wgFc rkFc rbFc rmFc oFc
                             memInit iros (buildNonDiagMask {ex} {dt} n)
                             Nothing Nothing Nothing Nothing Nothing Nothing Nothing)
    SNoGrad   => do memInit' <- liftIO (weakenGrad memInit)
                    pure (MkDnc ctrl wkFc wbFc eFc aFc fgFc agFc wgFc rkFc rbFc rmFc oFc
                                memInit' iros (buildNonDiagMask {ex} {dt} n)
                                Nothing Nothing Nothing Nothing Nothing Nothing Nothing)
