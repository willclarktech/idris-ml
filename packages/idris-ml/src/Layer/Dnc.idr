module Layer.Dnc

import Data.Vect
import Data.Zippable
import Compat.Random

import Device
import Floating
import Init
import Layer.Core
import Layer.Linear
import Layer.Lstm
import Math
import Memory
import Tensor
import Util
import Variable


----------------------------------------------------------------------
-- DNC Width Calculations
----------------------------------------------------------------------

||| Controller input: R read outputs (each m wide) + external input.
public export
DncControllerInput : (r : Nat) -> (m : Nat) -> (inputSize : Nat) -> Nat
DncControllerInput r m inputSize = r * m + inputSize

||| Output FC input: controller hidden + R read outputs.
public export
DncOutputInput : (h : Nat) -> (r : Nat) -> (m : Nat) -> Nat
DncOutputInput h r m = h + r * m


----------------------------------------------------------------------
-- DNC State
----------------------------------------------------------------------

public export
record DncState (r : Nat) (n : Nat) (m : Nat) (h : Nat)
               (inputSize : Nat) (outputSize : Nat) (ty : Type) where
  constructor MkDnc
  -- Sub-layers (learnable)
  lstm        : LstmState (DncControllerInput r m inputSize) h ty
  writeKeyFc  : LinearState h m ty
  writeBetaFc : LinearState h 1 ty
  eraseFc     : LinearState h m ty
  addFc       : LinearState h m ty
  freeGatesFc : LinearState h r ty
  allocGateFc : LinearState h 1 ty
  writeGateFc : LinearState h 1 ty
  readKeysFc  : LinearState h (r * m) ty
  readBetasFc : LinearState h r ty
  readModesFc : LinearState h (r * 3) ty
  outputFc    : LinearState (DncOutputInput h r m) outputSize ty
  -- State (non-learnable, reset per sequence)
  dncMemory    : Matrix n m ty
  usage        : Vector n ty
  writeWeights : Vector n ty
  readWeights  : Vect r (Vector n ty)
  readOutputs  : Vect r (Vector m ty)
  linkMatrix   : Matrix n n ty
  precedence   : Vector n ty
  -- Tensor handles (Just after nameLayer)
  memTensor    : Maybe AnyPtr
  usageTensor  : Maybe AnyPtr
  writeWtTensor   : Maybe AnyPtr
  precedenceTensor : Maybe AnyPtr
  linkTensor   : Maybe AnyPtr
  readWtTensors  : Maybe (Vect r AnyPtr)
  readOutTensors : Maybe (Vect r AnyPtr)


----------------------------------------------------------------------
-- Helper: Concatenate R read outputs into a single vector
----------------------------------------------------------------------

||| Flatten Vect r (Vector m ty) to Vector (r * m) ty.
concatReads : {r, m : Nat} -> Vect r (Vector m ty) -> Vector (r * m) ty
concatReads {r = Z} {m} [] = VTensor []
concatReads {r = S k} {m} ((VTensor v) :: rest) =
  let VTensor restFlat = concatReads {r = k} {m} rest
  in VTensor (v ++ restFlat)


----------------------------------------------------------------------
-- Addressing Weight Projection
----------------------------------------------------------------------

||| Project addressing weights onto the probability simplex after
||| gradient updates. Clamps to [epsilon, inf) and renormalizes.
projectWeights : {d : Device} -> {n : Nat} -> Vector n (Variable d) -> Vector n (Variable d)
projectWeights (VTensor vs) =
  let clamp : Tensor [] (Variable d) -> Tensor [] (Variable d)
      clamp (STensor v) = STensor ({ value $= max 0.00000001 } v)
      clamped = map clamp vs
      s : Double
      s = foldl (\acc, (STensor v) => acc + v.value) 0.0 clamped
      normalize : Tensor [] (Variable d) -> Tensor [] (Variable d)
      normalize (STensor v) = STensor ({ value $= (/ s) } v)
  in VTensor (map normalize clamped)


----------------------------------------------------------------------
-- DNC-specific Variable Operations
----------------------------------------------------------------------

||| Update usage vector.
||| u_t = (u_{t-1} + w^w_{t-1} - u_{t-1} * w^w_{t-1}) * retention
||| retention = prod_j(1 - f^j_t * w^{r,j}_{t-1})
dncUsageUpdate : {d : Device} -> {r, n : Nat} ->
    Vector n (Variable d) -> Vector n (Variable d) ->
    Vect r (Variable d) -> Vect r (Vector n (Variable d)) ->
    Vector n (Variable d)
dncUsageUpdate {r} {n} prevUsage prevWriteW freeGates prevReadWs =
  let -- write_usage = u + w - u*w
      writeUsage = prevUsage + prevWriteW - prevUsage * prevWriteW
      -- retention = prod_j(1 - f_j * w^r_j)
      ones = the (Vector n (Variable d)) (map (\_ => fromDouble 1.0) prevUsage)
      retention = computeRetention ones freeGates prevReadWs ones
      -- Clamp retention to [1e-10, inf) to prevent usage zeroing
      VTensor retElems = retention
      retT = vecStackTensor retElems
      retClamped = prim__clampMin retT 1.0e-10
      clampedRetention = VTensor $ tensorToScalars retClamped 0 n
  in writeUsage * clampedRetention
  where
    computeRetention : Vector n (Variable d) -> Vect k (Variable d) -> Vect k (Vector n (Variable d)) ->
                       Vector n (Variable d) -> Vector n (Variable d)
    computeRetention _ [] [] acc = acc
    computeRetention ones' (fg :: fgs) (rw :: rws) acc =
      let -- (1 - f_j * w^r_j)
          scaled = map (* fg) rw
          factor = ones' - scaled
      in computeRetention ones' fgs rws (acc * factor)

||| Allocation weighting from usage vector.
||| Uses argsort + cumprod at tensor level.
dncAllocate : {d : Device} -> {n : Nat} -> Vector n (Variable d) -> Vector n (Variable d)
dncAllocate {n} (VTensor usageElems) =
  let nI = cast {to=Int} n
      usageT = vecStackTensor usageElems
      -- Sort ascending
      indicesT = prim__argsort usageT 0 0
      -- Gather sorted usage, clamp to [1e-6, inf) to prevent cumprod underflow
      -- (1e-6 not 1e-10 because cumprod backward divides by these values)
      sortedUsageT = prim__clampMin (prim__gather usageT indicesT nI) 1.0e-6
      -- Cumprod of sorted usage
      cumprodT = prim__cumprod sortedUsageT 0
      -- Shifted cumprod: [1, cp[0], cp[1], ..., cp[n-2]]
      oneT = prim__createScalar 1.0 0
      slicedT = prim__narrow cumprodT 0 0 (nI - 1)
      shiftedT = prim__cat2 (prim__unsqueeze oneT 0) slicedT
      -- sorted_alloc = (1 - sorted_usage) * shifted_cumprod
      onesT = prim__createScalar 1.0 0
      oneMinusUsage = prim__sub onesT sortedUsageT
      sortedAllocT = prim__mul oneMinusUsage shiftedT
      -- Unsort: scatter back to original positions
      allocT = prim__scatterAdd indicesT sortedAllocT nI
  in VTensor $ tensorToScalars allocT 0 n

||| Write weighting: w^w = g^w * (g^a * a + (1-g^a) * c^w)
dncWriteWeight : {d : Device} -> {n : Nat} ->
    Vector n (Variable d) -> Vector n (Variable d) ->
    Variable d -> Variable d -> Vector n (Variable d)
dncWriteWeight contentW allocW writeGate allocGate =
  let oneMinusAG = fromDouble 1.0 - allocGate
      blended = map (* allocGate) allocW + map (* oneMinusAG) contentW
  in map (* writeGate) blended

||| Erase+add write: M' = M * (1 - outer(w, e)) + outer(w, a)
dncEraseAddWrite : {d : Device} -> {n, m : Nat} ->
    Matrix n m (Variable d) -> Vector n (Variable d) ->
    Vector m (Variable d) -> Vector m (Variable d) -> Matrix n m (Variable d)
dncEraseAddWrite {n} {m} (VTensor memRows) (VTensor wts) (VTensor eraseElems) (VTensor addElems) =
  let wtTensor = vecStackTensor wts
      memTensor = matStackTensor memRows
      eraseTensor = vecStackTensor eraseElems
      addTensor = vecStackTensor addElems
      eraseGate = prim__outer wtTensor eraseTensor
      ones = prim__createScalar 1.0 0
      keepGate = prim__sub ones eraseGate
      erased = prim__mul memTensor keepGate
      addGate = prim__outer wtTensor addTensor
      result = prim__add erased addGate
  in VTensor $ buildMatrixRows result 0 n m
  where
    buildMatrixRows : AnyPtr -> Int -> (rows : Nat) -> (cols : Nat) -> Vect rows (Vector cols (Variable d))
    buildMatrixRows _ _ Z _ = []
    buildMatrixRows t row (S r') cols =
      let rowTensor = prim__select t 0 row
      in VTensor (tensorToScalars rowTensor 0 cols) :: buildMatrixRows t (row + 1) r' cols

||| Link matrix update:
||| L'[i,j] = (1 - w[i] - w[j]) * L[i,j] + w[i] * p[j]
||| L'[i,i] = 0
dncLinkUpdate : {d : Device} -> {n : Nat} ->
    Matrix n n (Variable d) -> Vector n (Variable d) -> Vector n (Variable d) ->
    (Matrix n n (Variable d), Vector n (Variable d))
dncLinkUpdate {n} (VTensor linkRows) (VTensor wElems) precedenceVec =
  let nI = cast {to=Int} n
      wT = vecStackTensor wElems
      linkT = matStackTensor linkRows
      VTensor precElems = precedenceVec
      precT = vecStackTensor precElems
      -- w_i [n,1] and w_j [1,n]
      wiT = prim__unsqueeze wT 1  -- [n, 1]
      wjT = prim__unsqueeze wT 0  -- [1, n]
      pjT = prim__unsqueeze precT 0  -- [1, n]
      -- (1 - w_i - w_j) * L + w_i * p_j
      -- Clamp decay to [0, inf) — prevents negative decay when w_i + w_j > 1
      ones = prim__createScalar 1.0 0
      decay = prim__sub (prim__sub ones wiT) wjT  -- [n, n]
      decayClamped = prim__clampMin decay 0.0
      newLinkT = prim__add (prim__mul decayClamped linkT) (prim__mul wiT pjT)
      -- Zero diagonal and clamp entries non-negative
      diagMask = zeroDiag nI newLinkT
      linkClamped = prim__clampMin diagMask 0.0
      -- Precedence update: p' = (1 - sum(w)) * p + w
      wSum = prim__sum wT
      oneMinusWSum = prim__sub (prim__createScalar 1.0 0) wSum
      newPrecT = prim__add (prim__mul oneMinusWSum precT) wT
      newLink = VTensor $ buildMatrixRows linkClamped 0 n n
      newPrec = VTensor $ tensorToScalars newPrecT 0 n
  in (newLink, newPrec)
  where
    buildMatrixRows : AnyPtr -> Int -> (rows : Nat) -> (cols : Nat) -> Vect rows (Vector cols (Variable d))
    buildMatrixRows _ _ Z _ = []
    buildMatrixRows t row (S r') cols =
      let rowTensor = prim__select t 0 row
      in VTensor (tensorToScalars rowTensor 0 cols) :: buildMatrixRows t (row + 1) r' cols
    -- Zero the diagonal: L[i,i] = 0.
    -- Build (1-I) mask [n,n] using C allocation, then element-wise multiply.
    zeroDiag : Int -> AnyPtr -> AnyPtr
    zeroDiag nI linkT =
      let numElems = nI * nI
          buf = prim__allocDoubles numElems
          buf' = fillMask buf 0 nI numElems
          maskT = prim__create2d nI nI buf' 0
      in prim__mul linkT maskT
      where
        fillMask : AnyPtr -> Int -> Int -> Int -> AnyPtr
        fillMask b i nn numE = if i >= numE then b else
          let row = i `div` nn
              col = i `mod` nn
              val = if row == col then 0.0 else 1.0
              b' = prim__setDouble b i val
          in fillMask b' (i + 1) nn numE

||| Read weighting for one head:
||| w^r = pi[0]*backward + pi[1]*content + pi[2]*forward
dncReadWeight : {d : Device} -> {n : Nat} ->
    Matrix n n (Variable d) -> Vector n (Variable d) ->
    Vector n (Variable d) -> Vector 3 (Variable d) ->
    Vector n (Variable d)
dncReadWeight {n} (VTensor linkRows) (VTensor prevRwElems) contentW (VTensor [STensor pi0, STensor pi1, STensor pi2]) =
  let nI = cast {to=Int} n
      linkT = matStackTensor linkRows
      prevRwT = vecStackTensor prevRwElems
      VTensor cwElems = contentW
      cwT = vecStackTensor cwElems
      -- forward = L @ prev_rw
      forwardT = prim__matmul linkT prevRwT
      -- backward = L^T @ prev_rw
      linkTransT = prim__transpose2d linkT
      backwardT = prim__matmul linkTransT prevRwT
      -- weighted sum
      scaledBack = prim__mul pi0.tensorPtr backwardT
      scaledContent = prim__mul pi1.tensorPtr cwT
      scaledForward = prim__mul pi2.tensorPtr forwardT
      result = prim__add (prim__add scaledBack scaledContent) scaledForward
      -- Clamp and normalize read weights (mirrors NTM's focusVar pattern)
      resultClamped = prim__clampMin result 1.0e-10
      resultSum = prim__addScalar (prim__sum resultClamped) 1.0e-10
      normalized = prim__div resultClamped resultSum
  in VTensor $ tensorToScalars normalized 0 n


----------------------------------------------------------------------
-- Tensor-level helpers for applyVarTensor (P1, 2026-05-02)
--
-- Top-level recursive helpers used by the LayerLike applyVarTensor.
-- Defined here (not inside the instance's where-clause) to avoid the
-- Idris2 elaborator hang observed when implicit-r' inside a where
-- clause interacts with the LayerLike's outer {r : Nat}. Mirrors the
-- structure NTM uses for forwardReadHeadUnboundedVar etc.
----------------------------------------------------------------------

||| Concat read outputs (Vect r AnyPtr) followed by the input tensor,
||| producing a single 1D tensor for the LSTM controller's input.
||| Purely tensor — no scalar Variable round-trips.
catReadOutsAndInputT : {k : Nat} -> Vect k AnyPtr -> AnyPtr -> AnyPtr
catReadOutsAndInputT [] inp = inp
catReadOutsAndInputT (ro :: rest) inp =
  prim__cat2 ro (catReadOutsAndInputT rest inp)

||| Concat r read output tensors into a single [r*m] tensor for the
||| output FC input. Crashes on r=0 (DNC always has at least 1 head).
partial
catReadOutsT : {k : Nat} -> Vect k AnyPtr -> AnyPtr
catReadOutsT [] = idris_crash "DNC: catReadOutsT called with r=0"
catReadOutsT (h :: t) = catRest h t
  where
    catRest : AnyPtr -> {k' : Nat} -> Vect k' AnyPtr -> AnyPtr
    catRest acc [] = acc
    catRest acc (h' :: rest) = catRest (prim__cat2 acc h') rest

||| Compute prod_j (1 - free_gate_j * prev_read_w_j) over r heads.
||| free gates passed as a [r] tensor; we select scalars via prim__select.
||| prev read weights passed as Vect r AnyPtr (each [n]).
||| Threads an accumulator (start with rank-0 ones scalar from caller).
dncRetentionT : {k : Nat} -> Int -> AnyPtr -> Vect k AnyPtr -> AnyPtr -> AnyPtr
dncRetentionT _ _ [] acc = acc
dncRetentionT idx freeGatesT (rw :: rws) acc =
  let fg = prim__select freeGatesT 0 idx
      factor = prim__sub (prim__createScalar 1.0 0) (prim__mul fg rw)
  in dncRetentionT (idx + 1) freeGatesT rws (prim__mul acc factor)

||| Zero the diagonal of a [n,n] matrix by elementwise-multiplying with
||| an off-diagonal mask.
dncZeroDiagT : Int -> AnyPtr -> AnyPtr
dncZeroDiagT nI matT =
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

||| Per-head read processing: for each head j in 0..r-1, compute the
||| new read weight (via temporal-link forward + backward + content
||| addressing mode mixture, clamped + normalized) and read output
||| (read_w @ memory). Returns the new read weights and outputs as
||| two parallel Vects.
|||
||| Inputs: idx (head start = 0), prevRws (Vect r prev read weights,
||| each [n]), linkT [n,n], memT [n,m], readKeysFlatT [r*m],
||| readBetasRawT [r], readModesFlatT [r*3], mI (= cast m).
dncReadHeadsT : {k : Nat} -> Int -> Vect k AnyPtr ->
                AnyPtr -> AnyPtr ->
                AnyPtr -> AnyPtr -> AnyPtr ->
                Int ->
                (Vect k AnyPtr, Vect k AnyPtr)
dncReadHeadsT _ [] _ _ _ _ _ _ = ([], [])
dncReadHeadsT idx (prevRw :: restRws) linkT memT keysT betasT modesT mI =
  let headKeyT      = prim__narrow keysT 0 (idx * mI) mI
      headBetaPtr   = prim__select betasT 0 idx
      headBetaT     = prim__log (prim__addScalar (prim__exp headBetaPtr) 1.0)
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
      (restRws', restRos') = dncReadHeadsT (idx + 1) restRws linkT memT keysT betasT modesT mI
  in (rwT :: restRws', roT :: restRos')


----------------------------------------------------------------------
-- LayerLike Instance
----------------------------------------------------------------------

%default partial
export
{r, n, m, h : Nat} -> LayerLike (DncState r n m h) where
  -- Generic forward pass (Double-based) — used for eval after toDoubleNetwork.
  -- Simplified: uses content addressing for read/write, skips temporal links
  -- and allocation (those mainly affect training dynamics, not eval output).
  applyGeneric st inp =
    let allReads = concatReads {r} {m} st.readOutputs
        controllerInput = allReads ++ inp
        (updLstm, hidden) = applyGeneric st.lstm controllerInput
        cell = extractCellState updLstm
        (_, writeKey) = applyGeneric st.writeKeyFc cell
        (_, writeBetaRaw) = applyGeneric st.writeBetaFc cell
        (_, eraseRaw) = applyGeneric st.eraseFc cell
        (_, addVec) = applyGeneric st.addFc cell
        (_, readKeysFlat) = applyGeneric st.readKeysFc cell
        (_, readBetasRaw) = applyGeneric st.readBetasFc cell
        -- Write: content-based addressing only
        writeBeta = softplus (headScalarG writeBetaRaw)
        contentWriteW = getContentAddress softmax writeBeta st.dncMemory writeKey
        eraseVec = map sig eraseRaw
        wh = MkWriteHead (MkReadHead contentWriteW)
        newMem = writeOp wh st.dncMemory eraseVec addVec
        -- Read: content-based addressing only (no temporal links for eval)
        newReadOuts = readHeadsGeneric st.readWeights readKeysFlat readBetasRaw newMem
        -- Output
        allNewReads = concatReads {r} {m} (map snd newReadOuts)
        output = snd (applyGeneric st.outputFc (hidden ++ allNewReads))
    in (MkDnc updLstm st.writeKeyFc st.writeBetaFc st.eraseFc st.addFc
              st.freeGatesFc st.allocGateFc st.writeGateFc
              st.readKeysFc st.readBetasFc st.readModesFc st.outputFc
              newMem st.usage contentWriteW (map fst newReadOuts) (map snd newReadOuts)
              st.linkMatrix st.precedence
              Nothing Nothing Nothing Nothing Nothing Nothing Nothing, output)
    where
      headScalarG : Vector 1 ty -> ty
      headScalarG (VTensor [STensor v]) = v

      -- Read heads: content-based only (simplified for eval).
      -- Uses splitAt to peel off m elements at a time for keys.
      readHeadsGeneric :
                         Vect k (Vector n ty) -> Vector (k * m) ty -> Vector k ty ->
                         Matrix n m ty -> Vect k (Vector n ty, Vector m ty)
      readHeadsGeneric [] _ _ _ = []
      readHeadsGeneric {k = S j} (prevRw :: rest) keysFlat (VTensor (STensor headBetaRaw :: restBetas)) mem =
        let beta = the ty (softplus headBetaRaw)
            (headKey, restKeys) = Tensor.splitAt m keysFlat
            rw = getContentAddress softmax beta mem headKey
            ro = readOp (MkReadHead rw) mem
        in (rw, ro) :: readHeadsGeneric rest restKeys (VTensor restBetas) mem

  -- Variable forward pass: delegates to applyVarTensor when state tensors
  -- are initialized (post-nameLayer). Tensor path keeps state on AnyPtr
  -- handles end-to-end, eliminating per-op vecStackTensor / tensorToScalars
  -- round-trips (mirrors NTM's pattern, Layer/Ntm.idr:189-194).
  applyVar {d} {i} {o} st@(MkDnc _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _
                                (Just _) (Just _) (Just _) (Just _) (Just _)
                                (Just _) (Just _)) xs =
    let (VTensor xElems) = xs
        inputT = vecStackTensor xElems
        (st', outT) = applyVarTensor st inputT
        output = VTensor (tensorToScalars outT 0 o)
    in (st', output)
  -- Scalar fallback: when state tensors not initialized (pre-nameLayer
  -- or after resetState).
  applyVar {d} st inp =
    let -- 1. Controller input: concat all read outputs + input
        allReads = concatReads {r} {m} st.readOutputs
        controllerInput = allReads ++ inp
        -- 2. LSTM forward
        (updLstm, hidden) = applyVar st.lstm controllerInput
        cell = extractCellState updLstm
        -- 3. Interface vector from cell state
        (_, writeKey) = applyVar st.writeKeyFc cell
        (_, writeBetaRaw) = applyVar st.writeBetaFc cell
        (_, eraseRaw) = applyVar st.eraseFc cell
        (_, addVec) = applyVar st.addFc cell
        (_, freeGatesRaw) = applyVar st.freeGatesFc cell
        (_, allocGateRaw) = applyVar st.allocGateFc cell
        (_, writeGateRaw) = applyVar st.writeGateFc cell
        (_, readKeysFlat) = applyVar st.readKeysFc cell
        (_, readBetasRaw) = applyVar st.readBetasFc cell
        (_, readModesFlat) = applyVar st.readModesFc cell
        -- 4. Activate params
        writeBeta = softplus (headScalar writeBetaRaw)
        eraseVec = map sigmoidVar eraseRaw
        freeGates = map sigmoidVar freeGatesRaw
        allocGate = sigmoidVar (headScalar allocGateRaw)
        writeGate = sigmoidVar (headScalar writeGateRaw)
        -- 5. Usage update
        freeGateList = toVectScalars freeGates
        prevReadWList = st.readWeights
        newUsage = dncUsageUpdate st.usage st.writeWeights freeGateList prevReadWList
        -- 6. Allocation
        allocW = dncAllocate newUsage
        -- 7. Write content addressing
        contentWriteW = let scores = batchCosineSimilarityVar writeBeta st.dncMemory writeKey
                        in softmaxVar scores
        -- 8. Write weighting
        newWriteW = dncWriteWeight contentWriteW allocW writeGate allocGate
        -- 9. Memory write (erase + add)
        newMem = dncEraseAddWrite st.dncMemory newWriteW eraseVec addVec
        -- 10. Link matrix + precedence update
        (newLink, newPrec) = dncLinkUpdate st.linkMatrix newWriteW st.precedence
        -- 11. Read heads — convert to tensor pointers for narrow-based slicing
        VTensor rkElems = readKeysFlat
        VTensor rbElems = readBetasRaw
        VTensor rmElems = readModesFlat
        keysTensor = vecStackTensor rkElems
        betasTensor = vecStackTensor rbElems
        modesTensor = vecStackTensor rmElems
        readResults = computeReads 0 st.readWeights keysTensor betasTensor modesTensor newLink newMem
        newReadWs = map fst readResults
        newReadOuts = map snd readResults
        -- 12. Output: use CURRENT read outputs (not previous timestep's)
        allNewReads = concatReads {r} {m} newReadOuts
        (_, outputVec) = applyVar st.outputFc (hidden ++ allNewReads)
    in (MkDnc updLstm st.writeKeyFc st.writeBetaFc st.eraseFc st.addFc
              st.freeGatesFc st.allocGateFc st.writeGateFc
              st.readKeysFc st.readBetasFc st.readModesFc st.outputFc
              newMem newUsage newWriteW newReadWs newReadOuts newLink newPrec
              Nothing Nothing Nothing Nothing Nothing Nothing Nothing, outputVec)
    where
      headScalar : Vector 1 (Variable d) -> Variable d
      headScalar (VTensor [STensor v]) = v

      toVectScalars : {k : Nat} -> Vector k (Variable d) -> Vect k (Variable d)
      toVectScalars (VTensor vs) = map (\(STensor v) => v) vs

      -- Process each read head using tensor-level slicing
      computeReads : Int -> Vect k (Vector n (Variable d)) ->
                     AnyPtr -> AnyPtr -> AnyPtr ->
                     Matrix n n (Variable d) -> Matrix n m (Variable d) ->
                     Vect k (Vector n (Variable d), Vector m (Variable d))
      computeReads _ [] _ _ _ _ _ = []
      computeReads idx (prevRw :: restRws) keysTensor betasTensor modesTensor link mem =
        let mI = cast {to=Int} m
            -- Extract this head's key [m] via narrow
            headKeyT = prim__narrow keysTensor 0 (idx * mI) mI
            headKey = VTensor $ tensorToScalars headKeyT 0 m
            -- Extract this head's beta (scalar)
            headBetaT = prim__select betasTensor 0 idx
            headBeta = softplus (Var headBetaT Nothing (prim__item headBetaT))
            -- Extract this head's modes [3] via narrow
            headModesT = prim__narrow modesTensor 0 (idx * 3) 3
            headModes = VTensor $ tensorToScalars headModesT 0 3
            -- Content addressing
            scores = batchCosineSimilarityVar headBeta mem headKey
            contentRW = softmaxVar scores
            -- Softmax mode params
            headModesSm = softmaxVar headModes
            -- Read weighting (forward + backward + content mode mix)
            rw = dncReadWeight link prevRw contentRW headModesSm
            -- Read from memory
            ro = readOpVar rw mem
        in (rw, ro) :: computeReads (idx + 1) restRws keysTensor betasTensor modesTensor link mem

  -- Tensor-level forward pass — keeps state on AnyPtr handles end-to-end.
  -- Eliminates the ~30K small-op tape entries / 2.4M SELECT calls per epoch
  -- that dominated DNC's pre-rewrite forward time. Mirrors NTM's pattern
  -- (Layer/Ntm.idr:217-267) scaled to DNC's 11 FCs + allocation/link/multi-
  -- head read mechanisms. Helpers are top-level (above) — see comment there.
  applyVarTensor {d} (MkDnc lstm wkFc wbFc eFc aFc fgFc agFc wgFc rkFc rbFc rmFc oFc
                            memory usage ww rws ros link prec
                            (Just memT) (Just usageT) (Just wwT)
                            (Just precT) (Just linkT)
                            (Just rwTs) (Just roTs)) inputT =
    let Just wkW = extractWeightTensor wkFc | _ => idris_crash "DNC: writeKeyFc W"
        Just wkB = extractBiasTensor wkFc   | _ => idris_crash "DNC: writeKeyFc B"
        Just wbW = extractWeightTensor wbFc | _ => idris_crash "DNC: writeBetaFc W"
        Just wbB = extractBiasTensor wbFc   | _ => idris_crash "DNC: writeBetaFc B"
        Just eW  = extractWeightTensor eFc  | _ => idris_crash "DNC: eraseFc W"
        Just eB  = extractBiasTensor eFc    | _ => idris_crash "DNC: eraseFc B"
        Just aW  = extractWeightTensor aFc  | _ => idris_crash "DNC: addFc W"
        Just aB  = extractBiasTensor aFc    | _ => idris_crash "DNC: addFc B"
        Just fgW = extractWeightTensor fgFc | _ => idris_crash "DNC: freeGatesFc W"
        Just fgB = extractBiasTensor fgFc   | _ => idris_crash "DNC: freeGatesFc B"
        Just agW = extractWeightTensor agFc | _ => idris_crash "DNC: allocGateFc W"
        Just agB = extractBiasTensor agFc   | _ => idris_crash "DNC: allocGateFc B"
        Just wgW = extractWeightTensor wgFc | _ => idris_crash "DNC: writeGateFc W"
        Just wgB = extractBiasTensor wgFc   | _ => idris_crash "DNC: writeGateFc B"
        Just rkW = extractWeightTensor rkFc | _ => idris_crash "DNC: readKeysFc W"
        Just rkB = extractBiasTensor rkFc   | _ => idris_crash "DNC: readKeysFc B"
        Just rbW = extractWeightTensor rbFc | _ => idris_crash "DNC: readBetasFc W"
        Just rbB = extractBiasTensor rbFc   | _ => idris_crash "DNC: readBetasFc B"
        Just rmW = extractWeightTensor rmFc | _ => idris_crash "DNC: readModesFc W"
        Just rmB = extractBiasTensor rmFc   | _ => idris_crash "DNC: readModesFc B"
        Just oW  = extractWeightTensor oFc  | _ => idris_crash "DNC: outputFc W"
        Just oB  = extractBiasTensor oFc    | _ => idris_crash "DNC: outputFc B"
        nI = cast {to=Int} n
        mI = cast {to=Int} m
        onesScalar = prim__createScalar 1.0 0
    in
        let -- 1. Concat read outputs + input → LSTM input
            lstmInputT = catReadOutsAndInputT roTs inputT
            -- 2. LSTM forward at tensor level
            (updLstm, hiddenT) = applyVarTensor lstm lstmInputT
            -- 3. Cell-state input for FCs.
            -- IMPORTANT: scalar applyVar feeds the FCs `extractCellState updLstm`
            -- which is the LSTM's `cellState : Vector h (Variable d)` field — a
            -- VIEW into the c0 parameter. LSTM's applyVarTensor does NOT update
            -- this Vector field (only the cellTensor handle), so the FCs see
            -- the c0-param view across all timesteps within a sequence (and
            -- across the whole epoch, since c0 only updates between optimizer
            -- steps). This is mirrored here by stacking updLstm.cellState into
            -- a tensor at each timestep — preserves the autograd connection
            -- to c0 and matches the scalar baseline's training dynamics
            -- exactly. Using updLstm.cellTensor (the evolving hidden state)
            -- here changes the model's effective architecture and breaks
            -- convergence equivalence (verified in P1 spike).
            (VTensor csElems) = updLstm.cellState
            cellT = vecStackTensor csElems
            -- 3. 11 FCs as direct C ops (no scalar round-trips)
            writeKeyT      = tensorAdd (tensorMv wkW cellT) wkB
            writeBetaRawT  = tensorAdd (tensorMv wbW cellT) wbB
            eraseRawT      = tensorAdd (tensorMv eW  cellT) eB
            addVecT        = tensorAdd (tensorMv aW  cellT) aB
            freeGatesRawT  = tensorAdd (tensorMv fgW cellT) fgB
            allocGateRawT  = tensorAdd (tensorMv agW cellT) agB
            writeGateRawT  = tensorAdd (tensorMv wgW cellT) wgB
            readKeysFlatT  = tensorAdd (tensorMv rkW cellT) rkB
            readBetasRawT  = tensorAdd (tensorMv rbW cellT) rbB
            readModesFlatT = tensorAdd (tensorMv rmW cellT) rmB
            -- 4. Activations
            writeBetaT  = prim__log (prim__addScalar (prim__exp writeBetaRawT) 1.0)
            eraseVecT   = prim__sigmoid eraseRawT
            freeGatesT  = prim__sigmoid freeGatesRawT
            allocGateT  = prim__sigmoid allocGateRawT
            writeGateT  = prim__sigmoid writeGateRawT
            -- 5. Usage update: write_usage = u + w - u*w; retention = prod(1 - f_j * w^r_j)
            writeUsageT = prim__sub (prim__add usageT wwT) (prim__mul usageT wwT)
            retentionT  = dncRetentionT 0 freeGatesT rwTs onesScalar
            retClampedT = prim__clampMin retentionT 1.0e-10
            newUsageT   = prim__mul writeUsageT retClampedT
            -- 6. Allocation: argsort + cumprod + scatter
            indicesT      = prim__argsort newUsageT 0 0
            sortedUsageT  = prim__clampMin (prim__gather newUsageT indicesT nI) 1.0e-6
            cumprodT      = prim__cumprod sortedUsageT 0
            slicedT       = prim__narrow cumprodT 0 0 (nI - 1)
            shiftedT      = prim__cat2 (prim__unsqueeze onesScalar 0) slicedT
            oneMinusUsageT = prim__sub onesScalar sortedUsageT
            sortedAllocT  = prim__mul oneMinusUsageT shiftedT
            allocT        = prim__scatterAdd indicesT sortedAllocT nI
            -- 7. Write content addressing — match batchCosineSimilarityVar:
            -- unsqueeze key [m] → [1,m] before cosine_similarity along dim 1.
            cosScoresT    = prim__cosineSimilarity memT (prim__unsqueeze writeKeyT 0) 1
            scaledScoresT = prim__mul writeBetaT cosScoresT
            contentWriteWT = prim__softmax scaledScoresT 0
            -- 8. Write weighting: w^w = g^w * (g^a * a + (1-g^a) * c)
            oneMinusAGT   = prim__sub onesScalar allocGateT
            blendT        = prim__add (prim__mul allocGateT allocT)
                                       (prim__mul oneMinusAGT contentWriteWT)
            newWriteWT    = prim__mul writeGateT blendT
            -- 9. Memory write: M' = M*(1 - outer(w,e)) + outer(w,a)
            eraseGateT    = prim__outer newWriteWT eraseVecT
            keepGateT     = prim__sub onesScalar eraseGateT
            erasedT       = prim__mul memT keepGateT
            addGateT      = prim__outer newWriteWT addVecT
            newMemT       = prim__add erasedT addGateT
            -- 10. Link matrix update
            wiT           = prim__unsqueeze newWriteWT 1
            wjT           = prim__unsqueeze newWriteWT 0
            pjT           = prim__unsqueeze precT 0
            decayT        = prim__sub (prim__sub onesScalar wiT) wjT
            decayClampT   = prim__clampMin decayT 0.0
            newLinkRawT   = prim__add (prim__mul decayClampT linkT) (prim__mul wiT pjT)
            newLinkT      = prim__clampMin (dncZeroDiagT nI newLinkRawT) 0.0
            -- 11. Precedence update
            wSumT         = prim__sum newWriteWT
            oneMinusWSumT = prim__sub onesScalar wSumT
            newPrecT      = prim__add (prim__mul oneMinusWSumT precT) newWriteWT
            -- 12. Read heads (per-head tensor processing)
            (newRwTs, newRoTs) = dncReadHeadsT 0 rwTs newLinkT newMemT
                                    readKeysFlatT readBetasRawT readModesFlatT mI
            -- 13. Output FC
            allNewReadsT  = catReadOutsT newRoTs
            outputInputT  = prim__cat2 hiddenT allNewReadsT
            outputT       = tensorAdd (tensorMv oW outputInputT) oB
        in (MkDnc updLstm wkFc wbFc eFc aFc fgFc agFc wgFc rkFc rbFc rmFc oFc
                  memory usage ww rws ros link prec
                  (Just newMemT) (Just newUsageT) (Just newWriteWT)
                  (Just newPrecT) (Just newLinkT)
                  (Just newRwTs) (Just newRoTs), outputT)
  -- Fallback: when state tensors haven't been initialized yet (pre-nameLayer
  -- or after a sub-component reset), pack the input down to scalar Variables,
  -- run the existing scalar applyVar, and stack the output back to a tensor.
  -- Slow but correct — mirrors NTM's safety pattern.
  applyVarTensor {d} {i} {o} st inputT =
    let inputV : Vector i (Variable d)
        inputV = VTensor (tensorToScalars inputT 0 i)
    in case applyVar st inputV of
         (st', VTensor outElems) => (st', vecStackTensor outElems)

  emapLayer f (MkDnc lstm wkFc wbFc eFc aFc fgFc agFc wgFc rkFc rbFc rmFc oFc
               mem usage ww rws ros link prec _ _ _ _ _ _ _) =
    MkDnc (emapLayer f lstm) (emapLayer f wkFc) (emapLayer f wbFc)
           (emapLayer f eFc) (emapLayer f aFc) (emapLayer f fgFc)
           (emapLayer f agFc) (emapLayer f wgFc) (emapLayer f rkFc)
           (emapLayer f rbFc) (emapLayer f rmFc) (emapLayer f oFc)
           (map f mem) (map f usage) (map f ww)
           (map (map f) rws) (map (map f) ros)
           (map f link) (map f prec)
           Nothing Nothing Nothing Nothing Nothing Nothing Nothing

  showLayer {i} {o} _ =
    "Dnc<" ++ show i ++ ":" ++ show o
    ++ ", R=" ++ show r ++ ", mem=" ++ show n ++ "x" ++ show m ++ ", h=" ++ show h ++ ">"

  nameLayer {d} prefx (MkDnc lstm wkFc wbFc eFc aFc fgFc agFc wgFc rkFc rbFc rmFc oFc
                    mem usage ww rws ros link prec _ _ _ _ _ _ _) =
    let namedLstm   = nameLayer (prefx ++ "_lstm0") lstm
        namedWkFc   = nameLayer (prefx ++ "_writeKey_ll0") wkFc
        namedWbFc   = nameLayer (prefx ++ "_writeBeta_ll0") wbFc
        namedEFc    = nameLayer (prefx ++ "_erase_ll0") eFc
        namedAFc    = nameLayer (prefx ++ "_add_ll0") aFc
        namedFgFc   = nameLayer (prefx ++ "_freeGates_ll0") fgFc
        namedAgFc   = nameLayer (prefx ++ "_allocGate_ll0") agFc
        namedWgFc   = nameLayer (prefx ++ "_writeGate_ll0") wgFc
        namedRkFc   = nameLayer (prefx ++ "_readKeys_ll0") rkFc
        namedRbFc   = nameLayer (prefx ++ "_readBetas_ll0") rbFc
        namedRmFc   = nameLayer (prefx ++ "_readModes_ll0") rmFc
        namedOFc    = nameLayer (prefx ++ "_output_ll0") oFc
        -- Initialize state tensors. Persistent but NOT params (non-
        -- learnable, reset per-sequence). Mirrors NTM lines 283-298.
        (VTensor memRows)    = mem
        (VTensor usageElems) = usage
        (VTensor wwElems)    = ww
        (VTensor precElems)  = prec
        (VTensor linkRows)   = link
        nI = cast {to=Int} n
        mI = cast {to=Int} m
        memT   = prim__createState2d nI mI
                  (let buf = prim__allocDoubles (nI * mI) in packMatrixValues buf 0 {n=m} memRows)
        usageT = prim__createState1d nI
                  (let buf = prim__allocDoubles nI in packScalarValues buf 0 usageElems)
        wwT    = prim__createState1d nI
                  (let buf = prim__allocDoubles nI in packScalarValues buf 0 wwElems)
        precT  = prim__createState1d nI
                  (let buf = prim__allocDoubles nI in packScalarValues buf 0 precElems)
        linkT  = prim__createState2d nI nI
                  (let buf = prim__allocDoubles (nI * nI) in packMatrixValues buf 0 {n=n} linkRows)
        rwTs   = map (\(VTensor rwElems) =>
                       prim__createState1d nI
                         (let buf = prim__allocDoubles nI in packScalarValues buf 0 rwElems)) rws
        roTs   = map (\(VTensor roElems) =>
                       prim__createState1d mI
                         (let buf = prim__allocDoubles mI in packScalarValues buf 0 roElems)) ros
    in MkDnc namedLstm namedWkFc namedWbFc namedEFc namedAFc namedFgFc namedAgFc namedWgFc
             namedRkFc namedRbFc namedRmFc namedOFc
             mem usage ww rws ros link prec
             (Just memT) (Just usageT) (Just wwT) (Just precT) (Just linkT)
             (Just rwTs) (Just roTs)

  layerPrefix _ = "dnc"

  toDoubleLayer {d} (MkDnc lstm wkFc wbFc eFc aFc fgFc agFc wgFc rkFc rbFc rmFc oFc
                  mem usage ww rws ros link prec _ _ _ _ _ _ _) =
    MkDnc (toDoubleLayer lstm) (toDoubleLayer wkFc) (toDoubleLayer wbFc)
           (toDoubleLayer eFc) (toDoubleLayer aFc) (toDoubleLayer fgFc)
           (toDoubleLayer agFc) (toDoubleLayer wgFc) (toDoubleLayer rkFc)
           (toDoubleLayer rbFc) (toDoubleLayer rmFc) (toDoubleLayer oFc)
           (map value mem) (map value usage) (map value ww)
           (map (map value) rws) (map (map value) ros)
           (map value link) (map value prec)
           Nothing Nothing Nothing Nothing Nothing Nothing Nothing

  debugApply st inp =
    let (updated, output) = applyGeneric st inp
        entry = MkDebugEntry ("Dnc<" ++ show r ++ " heads>") []
    in (updated, output, entry)

  getParamIds {d} (MkDnc lstm wkFc wbFc eFc aFc fgFc agFc wgFc rkFc rbFc rmFc oFc _ _ _ _ _ _ _ _ _ _ _ _ _ _) =
    getParamIds lstm ++ getParamIds wkFc ++ getParamIds wbFc
    ++ getParamIds eFc ++ getParamIds aFc ++ getParamIds fgFc
    ++ getParamIds agFc ++ getParamIds wgFc ++ getParamIds rkFc
    ++ getParamIds rbFc ++ getParamIds rmFc ++ getParamIds oFc

  syncBuffers {d} (MkDnc lstm wkFc wbFc eFc aFc fgFc agFc wgFc rkFc rbFc rmFc oFc
               mem usage ww rws ros link prec
               memT usageT wwT precT linkT rwTs roTs) =
    MkDnc (syncBuffers lstm) (syncBuffers wkFc) (syncBuffers wbFc)
           (syncBuffers eFc) (syncBuffers aFc) (syncBuffers fgFc)
           (syncBuffers agFc) (syncBuffers wgFc) (syncBuffers rkFc)
           (syncBuffers rbFc) (syncBuffers rmFc) (syncBuffers oFc)
           mem usage (projectWeights ww)
           (map projectWeights rws) ros
           link prec
           memT usageT wwT precT linkT rwTs roTs

  applyDeltasAndSync {d} deltas (MkDnc lstm wkFc wbFc eFc aFc fgFc agFc wgFc rkFc rbFc rmFc oFc
                              mem usage ww rws ros link prec
                              memT usageT wwT precT linkT rwTs roTs) =
    MkDnc (applyDeltasAndSync deltas lstm) (applyDeltasAndSync deltas wkFc)
           (applyDeltasAndSync deltas wbFc) (applyDeltasAndSync deltas eFc)
           (applyDeltasAndSync deltas aFc) (applyDeltasAndSync deltas fgFc)
           (applyDeltasAndSync deltas agFc) (applyDeltasAndSync deltas wgFc)
           (applyDeltasAndSync deltas rkFc) (applyDeltasAndSync deltas rbFc)
           (applyDeltasAndSync deltas rmFc) (applyDeltasAndSync deltas oFc)
           mem usage (projectWeights ww)
           (map projectWeights rws) ros
           link prec
           memT usageT wwT precT linkT rwTs roTs

  -- Rebuild state with fresh zeros at each sequence start. Both the scalar
  -- Variable fields AND the tensor handles must be re-initialized:
  -- - Scalar Variables: needed for the scalar fallback path (and for
  --   Tensor* lifecycle on MLX where tape_reset deletes non-persistent
  --   Tensor*s wrapped by stale state Variables).
  -- - Tensor handles: persistent state tensors created via
  --   prim__createState{1d,2d}, freshly zeroed each sequence so the
  --   tensor-fast-path applyVarTensor can read them.
  resetState {d} (MkDnc lstm wkFc wbFc eFc aFc fgFc agFc wgFc rkFc rbFc rmFc oFc
                       _ _ _ _ _ _ _
                       _ _ _ _ _ _ _) =
    let zN    = the (Vector n (Variable d)) zeros
        zM    = the (Vector m (Variable d)) zeros
        memZ  = the (Matrix n m (Variable d)) zeros
        linkZ = the (Matrix n n (Variable d)) zeros
        zReadWs = replicate r zN
        zReadOuts = replicate r zM
        nI = cast {to=Int} n
        mI = cast {to=Int} m
        -- Destructure VTensors so packMatrixValues / packScalarValues see
        -- the inner Vect (matching the nameLayer pattern).
        (VTensor memZRows)    = memZ
        (VTensor zNElems)     = zN
        (VTensor linkZRows)   = linkZ
        -- Re-init tensor state handles with fresh zeros (mirror nameLayer)
        memT   = prim__createState2d nI mI
                  (let buf = prim__allocDoubles (nI * mI) in
                   packMatrixValues buf 0 {n=m} memZRows)
        usageT = prim__createState1d nI
                  (let buf = prim__allocDoubles nI in
                   packScalarValues buf 0 zNElems)
        wwT    = prim__createState1d nI
                  (let buf = prim__allocDoubles nI in
                   packScalarValues buf 0 zNElems)
        precT  = prim__createState1d nI
                  (let buf = prim__allocDoubles nI in
                   packScalarValues buf 0 zNElems)
        linkT  = prim__createState2d nI nI
                  (let buf = prim__allocDoubles (nI * nI) in
                   packMatrixValues buf 0 {n=n} linkZRows)
        rwTs   = map (\(VTensor rwElems) =>
                       prim__createState1d nI
                         (let buf = prim__allocDoubles nI in
                          packScalarValues buf 0 rwElems)) zReadWs
        roTs   = map (\(VTensor roElems) =>
                       prim__createState1d mI
                         (let buf = prim__allocDoubles mI in
                          packScalarValues buf 0 roElems)) zReadOuts
    in MkDnc (resetState lstm) wkFc wbFc eFc aFc fgFc agFc wgFc
             rkFc rbFc rmFc oFc
             memZ zN zN zReadWs zReadOuts linkZ zN
             (Just memT) (Just usageT) (Just wwT) (Just precT) (Just linkT)
             (Just rwTs) (Just roTs)


----------------------------------------------------------------------
-- Constructor
----------------------------------------------------------------------

||| Create a DNC layer.
||| r = read heads, n = memory slots, m = memory width, h = controller hidden.
export
dncLayer : {r, inputSize, outputSize, n, m, h : Nat} ->
           (Num ty, FromDouble ty) => IO (AnyLayer inputSize outputSize ty)
dncLayer = do
  lstm <- mkLstm {i = DncControllerInput r m inputSize, o = h}
  writeKeyFc <- mkLinearWithBias (xavierGain 1.4 uniform) 0.01 {i = h, o = m}
  writeBetaFc <- mkLinearWithBias (xavierGain 1.4 uniform) 0.01 {i = h, o = 1}
  eraseFc <- mkLinearWithBias (xavierGain 1.4 uniform) 0.01 {i = h, o = m}
  addFc <- mkLinearWithBias (xavierGain 1.4 uniform) 0.01 {i = h, o = m}
  freeGatesFc <- mkLinearWithBias (xavierGain 1.4 uniform) 0.01 {i = h, o = r}
  allocGateFc <- mkLinearWithBias (xavierGain 1.4 uniform) 0.01 {i = h, o = 1}
  writeGateFc <- mkLinearWithBias (xavierGain 1.4 uniform) 0.01 {i = h, o = 1}
  readKeysFc <- mkLinearWithBias (xavierGain 1.4 uniform) 0.01 {i = h, o = r * m}
  readBetasFc <- mkLinearWithBias (xavierGain 1.4 uniform) 0.01 {i = h, o = r}
  readModesFc <- mkLinearWithBias (xavierGain 1.4 uniform) 0.01 {i = h, o = r * 3}
  outputFc <- mkLinearWithBias (he uniform) 0.01 {i = DncOutputInput h r m, o = outputSize}
  -- Memory: sigmoid(random) ≈ values in [0,1]
  memInit <- traverse (\_ => map fromDouble (xavier uniform n m >>= \v => pure (1.0 / (1.0 + exp (negate v)))))
                      (the (Matrix n m ty) zeros)
  let usage = the (Vector n ty) zeros
  let writeW = the (Vector n ty) zeros
  let readWs = replicate r (the (Vector n ty) zeros)
  readOuts <- traverse (\_ => traverse (\_ => map fromDouble (he uniform m 1)) (the (Vector m ty) zeros)) (the (Vect r (Vector m ty)) (replicate r zeros))
  let link = the (Matrix n n ty) zeros
  let prec = the (Vector n ty) zeros
  pure $ MkAnyLayer (DncState r n m h) $ MkDnc lstm writeKeyFc writeBetaFc eraseFc addFc
    freeGatesFc allocGateFc writeGateFc readKeysFc readBetasFc readModesFc outputFc
    memInit usage writeW readWs readOuts link prec
    Nothing Nothing Nothing Nothing Nothing Nothing Nothing
