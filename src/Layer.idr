module Layer

import Data.Fin
import Data.List
import Data.SortedMap
import Data.Vect
import Data.Zippable
import System.Random

import DataPoint
import Endofunctor
import Floating
import Init
import Math
import Memory
import Tensor
import Util
import Variable


----------------------------------------------------------------------
-- NTM Width Calculations
----------------------------------------------------------------------

||| Read head output + input (legacy, used by old-style NTM controller input)
public export
NtmInputWidth : Nat -> Nat
NtmInputWidth w = w + w

||| Key vector + shift vector (ShiftKernelSize) + params (beta, g, gamma)
public export
ReadHeadInputWidth : Nat -> Nat -> Nat
ReadHeadInputWidth _ w = (w + ShiftKernelSize) + 3

||| Read head input + erase vector + add vector (legacy erase+add write)
public export
WriteHeadInputWidth : Nat -> Nat -> Nat
WriteHeadInputWidth n w = ReadHeadInputWidth n w + w + w

||| Read head input + Write head input + output (legacy)
public export
NtmOutputWidth : Nat -> Nat -> Nat
NtmOutputWidth n w = ReadHeadInputWidth n w + (WriteHeadInputWidth n w + w)

||| Read addressing params width: key(m) + shift(3) + beta + g + gamma
public export
ReadParamWidth : Nat -> Nat
ReadParamWidth m = (m + ShiftKernelSize) + 3

||| Write addressing params width: read params + add vector (no erase)
public export
WriteParamWidth : Nat -> Nat
WriteParamWidth m = ReadParamWidth m + m


----------------------------------------------------------------------
-- Layer and Network Types (mutually recursive)
----------------------------------------------------------------------

mutual
  public export
  data Layer : (inputSize : Nat) -> (outputSize : Nat) -> Type -> Type where
    LinearLayer : (weights : Matrix outputSize inputSize ty) -> (bias : Vector outputSize ty) -> (wBuf : Maybe AnyPtr) -> (bBuf : Maybe AnyPtr) -> Layer inputSize outputSize ty
    RnnLayer : (inputWeights : Matrix outputSize inputSize ty) -> (recurrentWeights : Matrix outputSize outputSize ty) -> (bias : Vector outputSize ty) -> (previousOutput : Vector outputSize ty) -> (iwBuf : Maybe AnyPtr) -> (rwBuf : Maybe AnyPtr) -> Layer inputSize outputSize ty
    ActivationLayer : (name : String) -> (f : ActivationFunction ty) -> Layer n n ty
    NormalizationLayer : (name : String) -> (f : NormalizationFunction ty) -> Layer n n ty
    LstmLayer : (inputWeights : Matrix (4 * outputSize) inputSize ty) ->
                (recurrentWeights : Matrix (4 * outputSize) outputSize ty) ->
                (bias : Vector (4 * outputSize) ty) ->
                (hiddenState : Vector outputSize ty) ->
                (cellState : Vector outputSize ty) ->
                (iwBuf : Maybe AnyPtr) -> (rwBuf : Maybe AnyPtr) ->
                (bBuf : Maybe AnyPtr) ->
                Layer inputSize outputSize ty
    ||| PyTorch-aligned NTM layer. LSTM controller with separate head FCs.
    ||| n = memory slots, m = memory width, h = controller hidden size.
    NtmLayer : {n, m, h : Nat} ->
               (lstm : Layer (m + inputSize) h ty) ->
               (readFc : Layer h (ReadParamWidth m) ty) ->
               (writeFc : Layer h (WriteParamWidth m) ty) ->
               (outputFc : Layer (h + m) outputSize ty) ->
               (memory : Matrix n m ty) ->
               (readAddr : Vector n ty) ->
               (writeAddr : Vector n ty) ->
               (readOutput : Vector m ty) ->
               (memBuf : Maybe AnyPtr) ->
               Layer inputSize outputSize ty

  public export
  data Network : (inputDims : Nat) -> (hiddenDims : List Nat) -> (outputDims : Nat) -> Type -> Type where
    OutputLayer : Layer i o ty -> Network i [] o ty
    (~>) : Layer i h ty -> Network h hs o ty -> Network i (h :: hs) o ty

export infixr 5 ~>


----------------------------------------------------------------------
-- Show Instances
----------------------------------------------------------------------

public export
implementation {inputSize : Nat} -> {outputSize : Nat} -> Show a => Show (Layer inputSize outputSize a) where
  show {inputSize} {outputSize} (LinearLayer _ _ _ _) = "Linear<" ++ show inputSize ++ ":" ++ show outputSize ++ ">"
  show {inputSize} {outputSize} (RnnLayer _ _ _ _ _ _) = "Rnn<" ++ show inputSize ++ ":" ++ show outputSize ++ ">"
  show {inputSize} {outputSize} (LstmLayer _ _ _ _ _ _ _ _) = "Lstm<" ++ show inputSize ++ ":" ++ show outputSize ++ ">"
  show (ActivationLayer name _) = "Activation<" ++ name ++ ">"
  show (NormalizationLayer name _) = "Normalization<" ++ name ++ ">"
  show {inputSize} {outputSize} (NtmLayer {n} {m} {h} _ _ _ _ _ _ _ _ _) = "Ntm<" ++ show inputSize ++ ":" ++ show outputSize ++ ", mem=" ++ show n ++ "x" ++ show m ++ ", h=" ++ show h ++ ">"

public export
implementation {i, o : Nat} -> Show ty => Show (Network i [] o ty) where
  show (OutputLayer layer) = show layer

public export
implementation {i, h : Nat} -> (Show ty, Show (Network h hs o ty)) => Show (Network i (h :: hs) o ty) where
  show (layer ~> layers) = show layer ++ " ~> " ++ show layers


----------------------------------------------------------------------
-- Endofunctor Instances
----------------------------------------------------------------------

mutual
  public export
  implementation Endofunctor (Layer i o) where
    emap f (LinearLayer w b wb bb) = LinearLayer (map f w) (map f b) wb bb
    emap f (RnnLayer iw rw b po iwb rwb) = RnnLayer (map f iw) (map f rw) (map f b) (map f po) iwb rwb
    emap f (LstmLayer iw rw b hs cs iwb rwb bb) = LstmLayer (map f iw) (map f rw) (map f b) (map f hs) (map f cs) iwb rwb bb
    emap f (NtmLayer lstm rfc wfc ofc mem ra wa ro mb) =
      NtmLayer (emap f lstm) (emap f rfc) (emap f wfc) (emap f ofc)
               (map f mem) (map f ra) (map f wa) (map f ro) mb
    emap _ l = l

  public export
  implementation Endofunctor (Network i hs o) where
    emap f (OutputLayer layer) = OutputLayer (emap f layer)
    emap f (layer ~> layers) = emap f layer ~> emap f layers


----------------------------------------------------------------------
-- Tanh Memory Bounding
----------------------------------------------------------------------

||| Bounds values to [-1, 1] via tanh. Uses integer literals to avoid
||| requiring a FromDouble constraint.
export
tanhBound : (Neg ty, Fractional ty, Floating ty) => ty -> ty
tanhBound x = 2 * (1 / (1 + exp (negate (2 * x)))) - 1


----------------------------------------------------------------------
-- LSTM Gate Helpers
----------------------------------------------------------------------

||| Coerce `Vector (o + 0) ty` to `Vector o ty`.
coerceLastGate : {o : Nat} -> Vector (o + 0) ty -> Vector o ty
coerceLastGate {o} v = rewrite sym (plusZeroRightNeutral o) in v

||| Helper to fix the type mismatch from splitAt on the last gate.
||| `4*o` normalizes as `o + (o + (o + (o + 0)))`, so three splitAts leave
||| `Vector (o + 0) ty` — this coerces it to `Vector o ty`.
lstmSplitGates :
    {o : Nat} -> Vector (4 * o) ty
    -> (Vector o ty, Vector o ty, Vector o ty, Vector o ty)
lstmSplitGates {o} combined =
  let s1 = Tensor.splitAt o combined
      s2 = Tensor.splitAt o (snd s1)
      s3 = Tensor.splitAt o (snd s2)
  in (fst s1, fst s2, fst s3, coerceLastGate (snd s3))


----------------------------------------------------------------------
-- Cell State Extraction
----------------------------------------------------------------------

||| Extract the cell state from an LSTM layer (for NTM head FC input).
||| Returns zeros if the layer is not an LSTM.
export
extractCellState : (Num ty) => {o : Nat} -> Layer i o ty -> Vector o ty
extractCellState (LstmLayer _ _ _ _ cell _ _ _) = cell
extractCellState _ = zeros


----------------------------------------------------------------------
-- Forward Pass
----------------------------------------------------------------------

mutual
  export
  applyLayer : (FromDouble ty, Floating ty, Fractional ty, Neg ty, Num ty, Ord ty) => {i, o : Nat} -> Layer i o ty -> Vector i ty -> (Layer i o ty, Vector o ty)
  applyLayer layer@(LinearLayer weights bias _ _) xs = (layer, matrixVectorMultiply {m=o, n=i} weights xs + bias)
  applyLayer (RnnLayer inputWeights recurrentWeights bias previousOutput iwb rwb) xs =
    let
      output = matrixVectorMultiply inputWeights xs + matrixVectorMultiply recurrentWeights previousOutput + bias
      updatedLayer = RnnLayer inputWeights recurrentWeights bias output iwb rwb
    in (updatedLayer, output)
  applyLayer layer@(ActivationLayer _ f) xs = (layer, map f xs)
  applyLayer layer@(NormalizationLayer _ f) xs = (layer, f xs)
  applyLayer {i} {o} (LstmLayer inputWeights recurrentWeights bias hiddenState cellState iwb rwb _) xs =
    let
      combined = matrixVectorMultiply inputWeights xs + matrixVectorMultiply recurrentWeights hiddenState + bias
      gates = lstmSplitGates {o} combined
      iGate = fst gates
      fGate = fst (snd gates)
      gGate = fst (snd (snd gates))
      oGate = snd (snd (snd gates))
      newCell = map sig fGate * cellState + map sig iGate * map tanhBound gGate
      newHidden = map sig oGate * map tanhBound newCell
      updatedLayer = LstmLayer inputWeights recurrentWeights bias newHidden newCell iwb rwb Nothing
    in (updatedLayer, newHidden)
  applyLayer (NtmLayer {n} {m} {h} lstm readFc writeFc outputFc memory readAddr writeAddr readOutput mb) inp =
    let
      -- 1. Controller: LSTM(readOutput ++ input)
      lstmResult = applyLayer lstm (readOutput ++ inp)
      hidden = snd lstmResult
      -- 2. Cell state for head FCs
      cell = extractCellState (fst lstmResult)
      -- 3. Head params from cell state
      readParams = snd (applyLayer readFc cell)
      writeParams = snd (applyLayer writeFc cell)
      -- 4. Read head (unbounded gamma)
      rh = MkReadHead readAddr
      readResult = forwardReadHeadUnbounded softmax memory rh readParams
      newReadAddr' = (fst readResult).addressingWeights
      newReadOutput = snd readResult
      -- 5. Write head (interpolation, no erase)
      wh = MkWriteHead (MkReadHead writeAddr)
      writeResult = forwardWriteHeadInterp softmax memory wh writeParams
      newWriteAddr' = (fst writeResult).readHead.addressingWeights
      newMemory = snd writeResult
      -- 6. Output FC(hidden ++ readOutput)
      output = snd (applyLayer outputFc (hidden ++ newReadOutput))
      newLayer = NtmLayer (fst lstmResult) readFc writeFc outputFc
                          newMemory newReadAddr' newWriteAddr' newReadOutput mb
    in (newLayer, output)

  export
  forward : (FromDouble ty, Floating ty, Fractional ty, Neg ty, Num ty, Ord ty) => {i, o : Nat} -> {hs : List Nat} -> Network i hs o ty -> Vector i ty -> (Network i hs o ty, Vector o ty)
  forward (OutputLayer layer) x =
    let (updatedLayer, output) = applyLayer layer x
    in (OutputLayer updatedLayer, output)
  forward {hs = h :: _} (layer ~> layers) x =
    let
      (updatedLayer, layerOutput) = applyLayer layer x
      (updatedNetwork, networkOutput) = forward layers layerOutput
    in (updatedLayer ~> updatedNetwork, networkOutput)



----------------------------------------------------------------------
-- Variable-specialized NTM Head Operations (C-backed memory ops)
----------------------------------------------------------------------

forwardReadHeadVar : {n, w : Nat} -> Matrix n w Variable -> ReadHead n Variable -> Vector ((w + ShiftKernelSize) + 3) Variable -> (ReadHead n Variable, Vector w Variable)
forwardReadHeadVar memory rh inp =
  let
    (mainInput, params) = splitAt (w + ShiftKernelSize) inp
    (keyVector, shiftVector) = splitAt w mainInput
    (betaVec, params') = splitAt 1 params
    (gVec, gammaVec) = splitAt 1 params'
    beta = softplus (sum betaVec)
    g = sigmoidVar (sum gVec)
    gamma = 1 + 4 * sigmoidVar (sum gammaVec)
    scores = batchCosineSimilarityVar beta memory keyVector
    contentWeights = softmaxVar scores
    interpolated = interpolate g contentWeights rh.addressingWeights
    shifted = shift softmaxVar interpolated shiftVector
    focused = focus gamma shifted
    newReadHead = { addressingWeights := focused } rh
    output = readOpVar newReadHead.addressingWeights memory
  in (newReadHead, output)

forwardWriteHeadVar : {n, w : Nat} -> Matrix n w Variable -> WriteHead n Variable -> Vector ((w + ShiftKernelSize) + 3 + w + w) Variable -> (WriteHead n Variable, Matrix n w Variable)
forwardWriteHeadVar memory (MkWriteHead readHead) inp =
  let
    inp' = rewrite plusAssociative ((w + ShiftKernelSize) + 3) w w in inp
    (readHeadInput, remainingInput) = Tensor.splitAt ((w + ShiftKernelSize) + 3) inp'
    (rawErase, rawAdd) = splitAt w remainingInput
    eraseVector = map sigmoidVar rawErase
    addVector = map (\x => 2 * sigmoidVar (2 * x) - 1) rawAdd
    (newReadHead, _) = forwardReadHeadVar memory readHead readHeadInput
    newWriteHead = MkWriteHead newReadHead
    newMemoryMatrix = writeOpVar newWriteHead.readHead.addressingWeights memory eraseVector addVector
  in (newWriteHead, newMemoryMatrix)


||| Variable-specialized read head with unbounded gamma (softplus).
forwardReadHeadUnboundedVar : {n, w : Nat} -> Matrix n w Variable -> ReadHead n Variable -> Vector ((w + ShiftKernelSize) + 3) Variable -> (ReadHead n Variable, Vector w Variable)
forwardReadHeadUnboundedVar memory rh inp =
  let
    (mainInput, params) = splitAt (w + ShiftKernelSize) inp
    (keyVector, shiftVector) = splitAt w mainInput
    (betaVec, params') = splitAt 1 params
    (gVec, gammaVec) = splitAt 1 params'
    beta = softplus (sum betaVec)
    g = sigmoidVar (sum gVec)
    gamma = 1 + softplus (sum gammaVec)
    scores = batchCosineSimilarityVar beta memory keyVector
    contentWeights = softmaxVar scores
    interpolated = interpolateVar g contentWeights rh.addressingWeights
    shiftKernel = softmaxVar shiftVector
    shifted = shiftVar interpolated shiftKernel
    focused = focusVar gamma shifted
    newReadHead = { addressingWeights := focused } rh
    output = readOpVar newReadHead.addressingWeights memory
  in (newReadHead, output)

||| Variable-specialized write head with interpolation write (no erase) and unbounded gamma.
||| Input: addressing params ((w + ShiftKernelSize) + 3) + add vector (w)
forwardWriteHeadInterpVar : {n, w : Nat} -> Matrix n w Variable -> WriteHead n Variable -> Vector (((w + ShiftKernelSize) + 3) + w) Variable -> (WriteHead n Variable, Matrix n w Variable)
forwardWriteHeadInterpVar memory (MkWriteHead readHead) inp =
  let
    (readHeadInput, rawAdd) = Tensor.splitAt ((w + ShiftKernelSize) + 3) inp
    addVector = rawAdd
    (newReadHead, _) = forwardReadHeadUnboundedVar memory readHead readHeadInput
    newWriteHead = MkWriteHead newReadHead
    newMemoryMatrix = interpolationWriteVar newWriteHead.readHead.addressingWeights memory addVector
  in (newWriteHead, newMemoryMatrix)


||| Buffer-aware read head: uses NtmMemBuf for memory (C memcpy pack).
||| Buffer-passing chain: intermediates stay in C buffers, only endpoints materialized.
forwardReadHeadUnboundedVarBuf : {n, w : Nat} -> AnyPtr -> ReadHead n Variable -> Vector ((w + ShiftKernelSize) + 3) Variable -> (ReadHead n Variable, Vector w Variable)
forwardReadHeadUnboundedVarBuf memBuf rh inp =
  let
    (mainInput, params) = splitAt (w + ShiftKernelSize) inp
    (keyVector, shiftVector) = splitAt w mainInput
    (betaVec, params') = splitAt 1 params
    (gVec, gammaVec) = splitAt 1 params'
    beta = softplus (sum betaVec)
    g = sigmoidVar (sum gVec)
    gamma = 1 + softplus (sum gammaVec)
    -- Buffer-passing chain: no Variable materialization for intermediates
    scoresBuf = batchCosineSimilarityVarBufBufOut {n} beta memBuf keyVector
    contentBuf = softmaxVarBufIO {n} scoresBuf
    interpBuf = interpolateVarBufIO {n} g contentBuf rh.addressingWeights
    shiftKBuf = softmaxVarBufOut shiftVector
    shiftedBuf = shiftVarBufIO {n} interpBuf shiftKBuf
    -- Materialize at endpoints: addressing weights stored as state
    focused = focusVarFromBuf {n} gamma shiftedBuf
    newReadHead = { addressingWeights := focused } rh
    output = readOpVarBuf newReadHead.addressingWeights memBuf
  in (newReadHead, output)

||| Buffer-aware write head: uses NtmMemBuf, returns (head, updated buffer wrapper).
forwardWriteHeadInterpVarBuf : {n, w : Nat} -> AnyPtr -> WriteHead n Variable -> Vector (((w + ShiftKernelSize) + 3) + w) Variable -> (WriteHead n Variable, AnyPtr)
forwardWriteHeadInterpVarBuf memBuf (MkWriteHead readHead) inp =
  let
    (readHeadInput, rawAdd) = Tensor.splitAt ((w + ShiftKernelSize) + 3) inp
    addVector = rawAdd
    (newReadHead, _) = forwardReadHeadUnboundedVarBuf memBuf readHead readHeadInput
    newWriteHead = MkWriteHead newReadHead
    mb' = interpolationWriteVarBuf newWriteHead.readHead.addressingWeights memBuf addVector
  in (newWriteHead, mb')


----------------------------------------------------------------------
-- Variable-specialized Forward Pass (C-backed matvec/dot)
----------------------------------------------------------------------

mutual
  -- Extract weight and bias buffers from a LinearLayer.
  getLinearBufs : Layer i o Variable -> (Maybe AnyPtr, Maybe AnyPtr)
  getLinearBufs (LinearLayer _ _ wBuf bBuf) = (wBuf, bBuf)
  getLinearBufs _ = (Nothing, Nothing)

  -- LSTM forward that also returns the raw output buffer + cell const start.
  -- Returns (updatedLayer, hidden, Just (outBuf, cellConstStart)) on the full-buffer path,
  -- or (updatedLayer, hidden, Nothing) on fallback.
  applyLstmGetBuf : {i, o : Nat} -> Layer i o Variable -> Vector i Variable
    -> (Layer i o Variable, Vector o Variable, Maybe (AnyPtr, Int))
  -- DIAGNOSTIC: use lstmCellVarFromBufs (non-Ext) directly
  applyLstmGetBuf {i} {o} (LstmLayer inputWeights recurrentWeights bias hiddenState cellState iwBuf rwBuf bBuf) xs =
    if i * o <= 4
      then let r = applyLayer (LstmLayer inputWeights recurrentWeights bias hiddenState cellState iwBuf rwBuf bBuf) xs
           in (fst r, snd r, Nothing)
      else case (iwBuf, rwBuf, bBuf) of
        (Just iwb, Just rwb, Just bb) =>
          let gateSize : Nat
              gateSize = 4 * o
              mulIWResult = matrixVectorMultiplyVarBufOut {m=gateSize, n=i} iwb xs
              mulRWResult = matrixVectorMultiplyVarBufOut {m=gateSize, n=o} rwb hiddenState
          in case lstmCellVarFromBufsExt
                    (fst mulIWResult) (snd mulIWResult)
                    (fst mulRWResult) (snd mulRWResult) bb cellState of
            (newCell, newHidden, outBuf, cellConstStart) =>
              let updatedLayer = LstmLayer inputWeights recurrentWeights bias newHidden newCell iwBuf rwBuf bBuf
              in (updatedLayer, newHidden, Just (outBuf, cellConstStart))
        _ =>
          let gateSize : Nat
              gateSize = 4 * o
              mulIW : Vector gateSize Variable
              mulIW = maybe (matrixVectorMultiplyVar {m=gateSize, n=i} inputWeights xs)
                            (\wb => matrixVectorMultiplyVarBuf {m=gateSize, n=i} wb xs) iwBuf
              mulRW : Vector gateSize Variable
              mulRW = maybe (matrixVectorMultiplyVar {m=gateSize, n=o} recurrentWeights hiddenState)
                            (\wb => matrixVectorMultiplyVarBuf {m=gateSize, n=o} wb hiddenState) rwBuf
              cellResult = maybe (lstmCellVar mulIW mulRW bias cellState)
                                 (\bb => lstmCellVarBuf mulIW mulRW bb cellState) bBuf
              newCell = fst cellResult
              newHidden = snd cellResult
              updatedLayer = LstmLayer inputWeights recurrentWeights bias newHidden newCell iwBuf rwBuf bBuf
          in (updatedLayer, newHidden, Nothing)
  applyLstmGetBuf layer xs =
    let r = applyLayerVar layer xs
    in (fst r, snd r, Nothing)

  export
  applyLayerVar : {i, o : Nat} -> Layer i o Variable -> Vector i Variable -> (Layer i o Variable, Vector o Variable)
  applyLayerVar layer@(LinearLayer weights bias wBuf bBuf) xs =
    if i * o <= 4
      then applyLayer layer xs
      else case (wBuf, bBuf) of
        (Just wb, Just bb) => (layer, matrixVectorMultiplyVarBufBias {m=o, n=i} wb bb xs)
        (Just wb, Nothing) => (layer, matrixVectorMultiplyVarBuf {m=o, n=i} wb xs + bias)
        _ => (layer, matrixVectorMultiplyVar {m=o, n=i} weights xs + bias)
  applyLayerVar (RnnLayer inputWeights recurrentWeights bias previousOutput iwBuf rwBuf) xs =
    if i * o <= 4
      then applyLayer (RnnLayer inputWeights recurrentWeights bias previousOutput iwBuf rwBuf) xs
      else let
        mulIW : Vector o Variable
        mulIW = maybe (matrixVectorMultiplyVar inputWeights xs)
                      (\wb => matrixVectorMultiplyVarBuf {m=o, n=i} wb xs) iwBuf
        mulRW : Vector o Variable
        mulRW = maybe (matrixVectorMultiplyVar recurrentWeights previousOutput)
                      (\wb => matrixVectorMultiplyVarBuf {m=o, n=o} wb previousOutput) rwBuf
        output = mulIW + mulRW + bias
        updatedLayer = RnnLayer inputWeights recurrentWeights bias output iwBuf rwBuf
      in (updatedLayer, output)
  applyLayerVar {i} {o} (LstmLayer inputWeights recurrentWeights bias hiddenState cellState iwBuf rwBuf bBuf) xs =
    if i * o <= 4
      then applyLayer (LstmLayer inputWeights recurrentWeights bias hiddenState cellState iwBuf rwBuf bBuf) xs
      else case (iwBuf, rwBuf, bBuf) of
        -- Full buffer-passing: MatVec outputs feed directly into LstmCell (no Variables)
        (Just iwb, Just rwb, Just bb) =>
          let gateSize : Nat
              gateSize = 4 * o
              mulIWResult = matrixVectorMultiplyVarBufOut {m=gateSize, n=i} iwb xs
              mulRWResult = matrixVectorMultiplyVarBufOut {m=gateSize, n=o} rwb hiddenState
              cellResult = lstmCellVarFromBufs
                             (fst mulIWResult) (snd mulIWResult)
                             (fst mulRWResult) (snd mulRWResult) bb cellState
              newCell = fst cellResult
              newHidden = snd cellResult
              updatedLayer = LstmLayer inputWeights recurrentWeights bias newHidden newCell iwBuf rwBuf bBuf
          in (updatedLayer, newHidden)
        -- Fallback: materialize Variables
        _ =>
          let gateSize : Nat
              gateSize = 4 * o
              mulIW : Vector gateSize Variable
              mulIW = maybe (matrixVectorMultiplyVar {m=gateSize, n=i} inputWeights xs)
                            (\wb => matrixVectorMultiplyVarBuf {m=gateSize, n=i} wb xs) iwBuf
              mulRW : Vector gateSize Variable
              mulRW = maybe (matrixVectorMultiplyVar {m=gateSize, n=o} recurrentWeights hiddenState)
                            (\wb => matrixVectorMultiplyVarBuf {m=gateSize, n=o} wb hiddenState) rwBuf
              cellResult = maybe (lstmCellVar mulIW mulRW bias cellState)
                                 (\bb => lstmCellVarBuf mulIW mulRW bb cellState) bBuf
              newCell = fst cellResult
              newHidden = snd cellResult
              updatedLayer = LstmLayer inputWeights recurrentWeights bias newHidden newCell iwBuf rwBuf bBuf
          in (updatedLayer, newHidden)
  applyLayerVar layer@(ActivationLayer _ f) xs = (layer, map f xs)
  applyLayerVar layer@(NormalizationLayer "softmax" _) xs = (layer, softmaxVar xs)
  applyLayerVar layer@(NormalizationLayer "logSoftmax" _) xs = (layer, logSoftmaxVar xs)
  applyLayerVar layer@(NormalizationLayer _ f) xs = (layer, f xs)
  -- Buffer-aware NTM forward pass (persistent memory buffer)
  -- Uses case destructuring at each level to prevent Idris 2 re-evaluation of FFI side effects.
  applyLayerVar (NtmLayer {n} {m} {h} lstm readFc writeFc outputFc memory readAddr writeAddr readOutput (Just memBuf)) inp =
    case applyLstmGetBuf lstm (readOutput ++ inp) of
      (updatedLstm, hidden, lstmBufInfo) =>
        let cell = extractCellState updatedLstm
            -- Read FC: prefer buffer-passing from LSTM cell output
            rawReadParams = case (lstmBufInfo, getLinearBufs readFc) of
              (Just (buf, ccs), (Just wb, Just bb)) =>
                matrixVectorMultiplyVarBufBiasFromBuf {m=ReadParamWidth m, n=h} wb bb buf 0 ccs
              _ => snd (applyLayerVar readFc cell)
            readParams = map (clampVar (-20.0) 20.0) rawReadParams
            -- Write FC: prefer buffer-passing from LSTM cell output
            rawWriteParams = case (lstmBufInfo, getLinearBufs writeFc) of
              (Just (buf, ccs), (Just wb, Just bb)) =>
                matrixVectorMultiplyVarBufBiasFromBuf {m=WriteParamWidth m, n=h} wb bb buf 0 ccs
              _ => snd (applyLayerVar writeFc cell)
            writeParams = map (clampVar (-20.0) 20.0) rawWriteParams
            -- Read head (buffer-aware: C memcpy pack)
            rh = MkReadHead readAddr
        in case forwardReadHeadUnboundedVarBuf memBuf rh readParams of
          (readHead, newReadOutput) =>
            let newReadAddr' = readHead.addressingWeights
                -- Write head (buffer-aware: mutates memBuf in-place)
                wh = MkWriteHead (MkReadHead writeAddr)
            in case forwardWriteHeadInterpVarBuf memBuf wh writeParams of
              (writeHead, mb') =>
                let newWriteAddr' = writeHead.readHead.addressingWeights
                    -- Output FC: prefer hybrid buffer+vec from LSTM hidden + readOutput
                    output = case (lstmBufInfo, getLinearBufs outputFc) of
                      (Just (buf, ccs), (Just wb, Just bb)) =>
                        matrixVectorMultiplyVarBufBiasFromBufAndVec {m=o, n1=h, n2=m}
                          wb bb buf (cast h) (ccs + cast h) newReadOutput
                      _ => snd (applyLayerVar outputFc (hidden ++ newReadOutput))
                    -- memory unchanged (for applyDeltas); buffer mutated via mb'
                    newLayer = NtmLayer updatedLstm readFc writeFc outputFc
                                        memory newReadAddr' newWriteAddr' newReadOutput (Just mb')
                in (newLayer, output)
  -- Variable-based NTM forward pass (no buffer)
  applyLayerVar (NtmLayer {n} {m} {h} lstm readFc writeFc outputFc memory readAddr writeAddr readOutput Nothing) inp =
    case applyLayerVar lstm (readOutput ++ inp) of
      (updatedLstm, hidden) =>
        let cell = extractCellState updatedLstm
        in case applyLayerVar readFc cell of
          (_, rawReadParams) =>
            let readParams = map (clampVar (-20.0) 20.0) rawReadParams
            in case applyLayerVar writeFc cell of
              (_, rawWriteParams) =>
                let writeParams = map (clampVar (-20.0) 20.0) rawWriteParams
                    rh = MkReadHead readAddr
                in case forwardReadHeadUnboundedVar memory rh readParams of
                  (readHead, newReadOutput) =>
                    let wh = MkWriteHead (MkReadHead writeAddr)
                    in case forwardWriteHeadInterpVar memory wh writeParams of
                      (writeHead, newMemory) =>
                        let newReadAddr' = readHead.addressingWeights
                            newWriteAddr' = writeHead.readHead.addressingWeights
                            output = snd (applyLayerVar outputFc (hidden ++ newReadOutput))
                            newLayer = NtmLayer updatedLstm readFc writeFc outputFc
                                                newMemory newReadAddr' newWriteAddr' newReadOutput Nothing
                        in (newLayer, output)

  export
  forwardVar : {i, o : Nat} -> {hs : List Nat} -> Network i hs o Variable -> Vector i Variable -> (Network i hs o Variable, Vector o Variable)
  forwardVar (OutputLayer layer) x =
    let (updatedLayer, output) = applyLayerVar layer x
    in (OutputLayer updatedLayer, output)
  forwardVar {hs = h :: _} (layer ~> layers) x =
    let
      (updatedLayer, layerOutput) = applyLayerVar layer x
      (updatedNetwork, networkOutput) = forwardVar layers layerOutput
    in (updatedLayer ~> updatedNetwork, networkOutput)

forwardNextVar : {i, o : Nat} -> {hs : List Nat} -> (Network i hs o Variable, Vect n (Vector o Variable)) -> Vector i Variable -> (Network i hs o Variable, Vect (S n) (Vector o Variable))
forwardNextVar (nn, outputs) inp =
  let (updatedModel, newOutput) = forwardVar nn inp
  in (updatedModel, snoc outputs newOutput)

forwardManyVar : {i, o : Nat} -> {hs : List Nat} -> Network i hs o Variable -> Vect n (Vector i Variable) -> (Network i hs o Variable, Vect n (Vector o Variable))
forwardManyVar network xs = foldlD (\k => (Network i hs o Variable, Vect k (Vector o Variable))) forwardNextVar (network, []) xs

export
calculateLossVar : {i, o, n : Nat} -> {hs : List Nat} -> LossFunction Variable -> Network i hs o Variable -> Vect n (DataPoint i o Variable) -> Variable
calculateLossVar lossFn model dataPoints =
  let
    xs = map x dataPoints
    ys = map y dataPoints
    (updatedNetwork, predictions) = forwardManyVar model xs
    losses = zipWith lossFn predictions ys
  in mean $ VTensor $ map STensor losses

recurVar : {i, o : Nat} -> {hs : List Nat} -> (Network i hs o Variable, List (Vector o Variable)) -> Vector i Variable -> (Network i hs o Variable, List (Vector o Variable))
recurVar (m, os) inp =
  let (updatedModel, output) = forwardVar m inp
  in (updatedModel, snoc os output)

export
forwardRecurrentVar : {i, o : Nat} -> {hs : List Nat} -> Network i hs o Variable -> List (Vector i Variable) -> (Network i hs o Variable, List (Vector o Variable))
forwardRecurrentVar model = foldl recurVar (model, [])

export
calculateLossRecurrentVar : {i, o, n : Nat} -> {hs : List Nat} -> LossFunction Variable -> Network i hs o Variable -> Vect n (RecurrentDataPoint i o Variable) -> Variable
calculateLossRecurrentVar lossFn model dataPoints =
  let
    perSequence : RecurrentDataPoint i o Variable -> List Variable
    perSequence dp =
      let (_, preds) = forwardRecurrentVar model (xs dp)
      in zipWith lossFn preds (ys dp)
    losses = map perSequence dataPoints
  in mean . VTensor $ map (STensor . mean) losses


----------------------------------------------------------------------
-- Two-Phase Loss Computation (Variable)
----------------------------------------------------------------------

||| Two-phase loss: encoding phase (discard outputs), then output phase
||| (feed zeros, compute loss on collected outputs vs targets).
export
calculateLossTwoPhaseVar : {i, o, n : Nat} -> {hs : List Nat} ->
    LossFunction Variable -> Network i hs o Variable ->
    Vect n (TwoPhaseDataPoint i o Variable) ->
    Variable
calculateLossTwoPhaseVar lossFn model dataPoints =
  let
    perSequence : TwoPhaseDataPoint i o Variable -> List Variable
    perSequence dp =
      let zeroInput : Vector i Variable
          zeroInput = map (const (fromDouble 0.0)) zeros
          outputInputs = Data.List.replicate (length (targets dp)) zeroInput
          encResult = forwardRecurrentVar model (encodingInputs dp)
          outResult = forwardRecurrentVar (fst encResult) outputInputs
      in zipWith lossFn (snd outResult) (targets dp)
    losses = map perSequence dataPoints
  in mean . VTensor $ map (STensor . mean) losses

||| Two-phase loss with C-backed BCE: encoding phase (discard outputs),
||| then output phase (feed zeros, compute fused BCE loss).
||| Uses bceWithLogitsVar for a single tape entry per output vector.
export
calculateLossTwoPhaseVarBce : {i, o, n : Nat} -> {hs : List Nat} ->
    Network i hs o Variable ->
    Vect n (TwoPhaseDataPoint i o Variable) ->
    Variable
calculateLossTwoPhaseVarBce model dataPoints =
  let
    perSequence : TwoPhaseDataPoint i o Variable -> List Variable
    perSequence dp =
      let zeroInput : Vector i Variable
          zeroInput = map (const (fromDouble 0.0)) zeros
          outputInputs = Data.List.replicate (length (targets dp)) zeroInput
          encResult = forwardRecurrentVar model (encodingInputs dp)
          outResult = forwardRecurrentVar (fst encResult) outputInputs
      in zipWith bceWithLogitsVar (snd outResult) (targets dp)
    losses = map perSequence dataPoints
  in mean . VTensor $ map (STensor . mean) losses


----------------------------------------------------------------------
-- Evaluation Functions
----------------------------------------------------------------------

evaluateSingleDataPoint : (FromDouble ty, Floating ty, Fractional ty, Neg ty, Num ty, Ord ty) => {i, o : Nat} -> {hs : List Nat} -> Network i hs o ty -> DataPoint i o ty -> Vector o ty
evaluateSingleDataPoint model = snd . (forward model) . x

export
evaluate : (FromDouble ty, Floating ty, Fractional ty, Neg ty, Num ty, Ord ty) => {i, o : Nat} -> {hs : List Nat} -> Network i hs o ty -> Vect n (DataPoint i o ty) -> Vect n (Vector o ty)
evaluate model = map (evaluateSingleDataPoint model)

forwardNext : (FromDouble ty, Floating ty, Fractional ty, Neg ty, Num ty, Ord ty) => {i, o : Nat} -> {hs : List Nat} -> (Network i hs o ty, Vect n (Vector o ty)) -> Vector i ty -> (Network i hs o ty, Vect (S n) (Vector o ty))
forwardNext (nn, outputs) inp =
  let (updatedModel, newOutput) = forward nn inp
  in (updatedModel, snoc outputs newOutput)

forwardMany : (FromDouble ty, Floating ty, Fractional ty, Neg ty, Num ty, Ord ty) => {i, o : Nat} -> {hs : List Nat} -> Network i hs o ty -> Vect n (Vector i ty) -> (Network i hs o ty, Vect n (Vector o ty))
forwardMany network xs = foldlD (\k => (Network i hs o ty, Vect k (Vector o ty))) forwardNext (network, []) xs

export
calculateLoss : (FromDouble ty, Floating ty, Fractional ty, Neg ty, Num ty, Ord ty) => {i, o, n : Nat} -> {hs : List Nat} -> LossFunction ty -> Network i hs o ty -> Vect n (DataPoint i o ty) -> ty
calculateLoss lossFn model dataPoints =
  let
    xs = map x dataPoints
    ys = map y dataPoints
    (updatedNetwork, predictions) = forwardMany model xs
    losses = zipWith lossFn predictions ys
  in mean $ VTensor $ map STensor losses


----------------------------------------------------------------------
-- Recurrent Evaluation Functions
----------------------------------------------------------------------

recur : (FromDouble ty, Floating ty, Fractional ty, Neg ty, Num ty, Ord ty) => {i, o : Nat} -> {hs : List Nat} -> (Network i hs o ty, List (Vector o ty)) -> Vector i ty -> (Network i hs o ty, List (Vector o ty))
recur (m, os) i =
  let (updatedModel, output) = forward m i
  in (updatedModel, snoc os output)

export
forwardRecurrent : (FromDouble ty, Floating ty, Fractional ty, Neg ty, Num ty, Ord ty) => {i, o : Nat} -> {hs : List Nat} -> Network i hs o ty -> List (Vector i ty) -> (Network i hs o ty, List (Vector o ty))
forwardRecurrent model = foldl recur (model, [])

evaluateSingleRecurrentDataPoint : (FromDouble ty, Floating ty, Fractional ty, Neg ty, Num ty, Ord ty) => {i, o : Nat} -> {hs : List Nat} -> Network i hs o ty -> RecurrentDataPoint i o ty -> List (Vector o ty)
evaluateSingleRecurrentDataPoint model dataPoints = snd $ (forwardRecurrent model) dataPoints.xs

export
evaluateRecurrent : (FromDouble ty, Floating ty, Fractional ty, Neg ty, Num ty, Ord ty) => {i, o : Nat} -> {hs : List Nat} -> Network i hs o ty -> Vect n (RecurrentDataPoint i o ty) -> Vect n (List (Vector o ty))
evaluateRecurrent model dataPoints = map (evaluateSingleRecurrentDataPoint model) dataPoints

export
calculateLossRecurrent : (FromDouble ty, Floating ty, Fractional ty, Neg ty, Num ty, Ord ty) => {i, o, n : Nat} -> {hs : List Nat} -> LossFunction ty -> Network i hs o ty -> Vect n (RecurrentDataPoint i o ty) -> ty
calculateLossRecurrent lossFn model dataPoints =
  let
    perSequence : RecurrentDataPoint i o ty -> List ty
    perSequence dp =
      let (_, preds) = forwardRecurrent model (xs dp)
      in zipWith lossFn preds (ys dp)
    losses = map perSequence dataPoints
  in mean . VTensor $ map (STensor . mean) losses


----------------------------------------------------------------------
-- Two-Phase Evaluation (Generic)
----------------------------------------------------------------------

||| Two-phase forward: encoding phase then output phase with zeros.
export
forwardTwoPhase : (FromDouble ty, Floating ty, Fractional ty, Neg ty, Num ty, Ord ty) =>
    {i, o : Nat} -> {hs : List Nat} ->
    Network i hs o ty -> TwoPhaseDataPoint i o ty ->
    (Network i hs o ty, List (Vector o ty))
forwardTwoPhase model dp =
  let encResult = forwardRecurrent model (encodingInputs dp)
      zeroInput : Vector i ty
      zeroInput = zeros
      outputInputs = Data.List.replicate (length (targets dp)) zeroInput
  in forwardRecurrent (fst encResult) outputInputs


----------------------------------------------------------------------
-- Layer Constructors
----------------------------------------------------------------------

export
linearLayerWith : {i, o : Nat} -> (Num ty, FromDouble ty) => InitStrategy -> IO (Layer i o ty)
linearLayerWith initFn = do
  weights <- traverse (\_ => map fromDouble (initFn i o)) (the (Matrix o i ty) zeros)
  pure $ LinearLayer weights zeros Nothing Nothing

||| Create a linear layer with custom weight and bias init.
export
linearLayerWithBias : {i, o : Nat} -> (Num ty, FromDouble ty) =>
    InitStrategy -> (biasStd : Double) -> IO (Layer i o ty)
linearLayerWithBias initFn biasStd = do
  weights <- traverse (\_ => map fromDouble (initFn i o)) (the (Matrix o i ty) zeros)
  bias <- traverse (\_ => map fromDouble (normalSample >>= \s => pure (s * biasStd)))
                    (the (Vector o ty) zeros)
  pure $ LinearLayer weights bias Nothing Nothing

export
linearLayer : {i, o : Nat} -> (Num ty, FromDouble ty) => IO (Layer i o ty)
linearLayer = linearLayerWith (xavier uniform)

export
rnnLayerWith : {i, o : Nat} -> (Num ty, FromDouble ty) => InitStrategy -> IO (Layer i o ty)
rnnLayerWith initFn = do
  inputWeights <- traverse (\_ => map fromDouble (initFn i o)) (the (Matrix o i ty) zeros)
  recurrentWeights <- traverse (\_ => map fromDouble (initFn o o)) (the (Matrix o o ty) zeros)
  pure $ RnnLayer inputWeights recurrentWeights zeros zeros Nothing Nothing

export
rnnLayer : {i, o : Nat} -> (Num ty, FromDouble ty) => IO (Layer i o ty)
rnnLayer = rnnLayerWith (xavier uniform)

||| Set forget gate bias to 1.0. Gate layout: input, forget, cell, output.
||| Forget gate occupies indices [o..2*o) in the bias vector.
setForgetBias : (FromDouble ty, Num ty) => {o : Nat} -> Vector (4 * o) ty -> Vector (4 * o) ty
setForgetBias {o} (VTensor elems) =
  VTensor (go 0 elems)
  where
    go : Nat -> Vect n (Scalar ty) -> Vect n (Scalar ty)
    go _ [] = []
    go idx (x :: xs) =
      if idx >= o && idx < 2 * o
        then STensor (fromDouble 1.0) :: go (idx + 1) xs
        else x :: go (idx + 1) xs

export
lstmLayerWith : {i, o : Nat} -> (Num ty, FromDouble ty) => InitStrategy -> IO (Layer i o ty)
lstmLayerWith {i} {o} initFn = do
  inputWeights <- traverse (\_ => map fromDouble (initFn i (4 * o))) (the (Matrix (4 * o) i ty) zeros)
  recurrentWeights <- traverse (\_ => map fromDouble (initFn o (4 * o))) (the (Matrix (4 * o) o ty) zeros)
  let bias = the (Vector (4 * o) ty) zeros
  pure $ LstmLayer inputWeights recurrentWeights bias zeros zeros Nothing Nothing Nothing

export
lstmLayer : {i, o : Nat} -> (Num ty, FromDouble ty) => IO (Layer i o ty)
lstmLayer = lstmLayerWith (xavier uniform)

||| Create a PyTorch-aligned NTM layer.
||| n = memory slots, m = memory width, h = controller hidden size.
||| Head FCs use xavier(gain=1.4) + normal(0.01) bias (matching PyTorch).
||| Output FC uses kaiming uniform + normal(0.01) bias.
||| Memory init: sigmoid(xavier_random) ≈ values in [0,1], matching PyTorch's sigmoid(FC_bias).
||| Read output: kaiming uniform (matching PyTorch).
export
ntmLayer : {inputSize, outputSize, n, m, h : Nat} ->
           (Num ty, FromDouble ty) => IO (Layer inputSize outputSize ty)
ntmLayer = do
  lstm <- lstmLayer {i = m + inputSize, o = h}
  readFc <- linearLayerWithBias (xavierGain 1.4 uniform) 0.01 {i = h, o = ReadParamWidth m}
  writeFc <- linearLayerWithBias (xavierGain 1.4 uniform) 0.01 {i = h, o = WriteParamWidth m}
  outputFc <- linearLayerWithBias (he uniform) 0.01 {i = h + m, o = outputSize}
  -- Memory: sigmoid(random) ≈ values in [0,1], matching PyTorch's sigmoid(FC_bias)
  memInit <- traverse (\_ => map fromDouble (xavier uniform n m >>= \v => pure (1.0 / (1.0 + exp (negate v)))))
                      (the (Matrix n m ty) zeros)
  let readAddr = the (Vector n ty) zeros
  let writeAddr = the (Vector n ty) zeros
  -- Read output: kaiming uniform, matching PyTorch
  readOut <- traverse (\_ => map fromDouble (he uniform m 1)) (the (Vector m ty) zeros)
  pure $ NtmLayer lstm readFc writeFc outputFc memInit readAddr writeAddr readOut Nothing

export
sigmoidLayer : (FromDouble ty, Neg ty, Fractional ty, Floating ty) => Layer n n ty
sigmoidLayer = ActivationLayer "sigmoid" sigmoid

export
tanhLayer : (FromDouble ty, Neg ty, Fractional ty, Floating ty) => Layer n n ty
tanhLayer = ActivationLayer "tanh" Math.tanh

export
softmaxLayer : (Fractional ty, Floating ty) => Layer n n ty
softmaxLayer = NormalizationLayer "softmax" softmax

export
logSoftmaxLayer : (FromDouble ty, Cast ty Double, Neg ty, Floating ty, Fractional ty) => Layer n n ty
logSoftmaxLayer = NormalizationLayer "logSoftmax" logSoftmax


----------------------------------------------------------------------
-- Parameter Naming
----------------------------------------------------------------------

mutual
  export
  nameParams : {i, o : Nat} -> String -> (Layer i o Variable) -> (Layer i o Variable)
  nameParams {i} {o} prefx layer =
    let np = nameParam . (prefx ++ "_" ++)
    in case layer of
      (LinearLayer weights bias _ _) =>
        let
          namedWeights = zipWith (np "weight") enumerate weights
          namedBias = zipWith (np "bias") enumerate bias
        in if i * o <= 4
          then LinearLayer namedWeights namedBias Nothing Nothing
          else let (VTensor namedRows) = namedWeights
                   (VTensor biasElems) = namedBias
                   wBuf = prim__weightBufAlloc (cast (o * i))
                   wBuf' = initWeightBuf wBuf 0 namedRows
                   bBuf = prim__weightBufAlloc (cast o)
                   bBuf' = initWeightBufRow bBuf 0 biasElems
               in LinearLayer namedWeights namedBias (Just wBuf') (Just bBuf')
      (RnnLayer inputWeights recurrentWeights bias previousOutput _ _) =>
        let
          namedInputWeights = zipWith (np "inputWeight") enumerate inputWeights
          namedRecurrentWeights = zipWith (np "recurrentWeight") enumerate recurrentWeights
          namedBias = zipWith (np "bias") enumerate bias
        in if i * o <= 4
          then RnnLayer namedInputWeights namedRecurrentWeights namedBias previousOutput Nothing Nothing
          else let (VTensor iwRows) = namedInputWeights
                   (VTensor rwRows) = namedRecurrentWeights
                   iwBuf = prim__weightBufAlloc (cast (o * i))
                   iwBuf' = initWeightBuf iwBuf 0 iwRows
                   rwBuf = prim__weightBufAlloc (cast (o * o))
                   rwBuf' = initWeightBuf rwBuf 0 rwRows
               in RnnLayer namedInputWeights namedRecurrentWeights namedBias previousOutput (Just iwBuf') (Just rwBuf')
      (LstmLayer inputWeights recurrentWeights bias hiddenState cellState _ _ _) =>
        let
          gateSize : Nat
          gateSize = 4 * o
          namedInputWeights = zipWith (np "inputWeight") enumerate inputWeights
          namedRecurrentWeights = zipWith (np "recurrentWeight") enumerate recurrentWeights
          namedBias = zipWith (np "bias") enumerate bias
        in if i * o <= 4
          then LstmLayer namedInputWeights namedRecurrentWeights namedBias hiddenState cellState Nothing Nothing Nothing
          else let (VTensor iwRows) = namedInputWeights
                   (VTensor rwRows) = namedRecurrentWeights
                   (VTensor biasElems) = namedBias
                   iwBuf = prim__weightBufAlloc (cast (gateSize * i))
                   iwBuf' = initWeightBuf iwBuf 0 iwRows
                   rwBuf = prim__weightBufAlloc (cast (gateSize * o))
                   rwBuf' = initWeightBuf rwBuf 0 rwRows
                   bBuf = prim__weightBufAlloc (cast gateSize)
                   bBuf' = initWeightBufRow bBuf 0 biasElems
               in LstmLayer namedInputWeights namedRecurrentWeights namedBias hiddenState cellState (Just iwBuf') (Just rwBuf') (Just bBuf')
      (NtmLayer {n} {m} lstm readFc writeFc outputFc memory readAddr writeAddr readOutput _) =>
        let namedMemory = zipWith (np "mem") enumerate memory
            namedReadAddr = zipWith (np "rAddr") enumerate readAddr
            namedWriteAddr = zipWith (np "wAddr") enumerate writeAddr
            namedReadOut = zipWith (np "rOut") enumerate readOutput
            namedLstm = nameParams (prefx ++ "_lstm") lstm
            namedReadFc = nameParams (prefx ++ "_readFc") readFc
            namedWriteFc = nameParams (prefx ++ "_writeFc") writeFc
            namedOutputFc = nameParams (prefx ++ "_outputFc") outputFc
            (VTensor memRows) = namedMemory
            memBuf = prim__ntmMemBufAlloc (cast n) (cast m)
            memBuf' = initNtmMemBuf memBuf 0 memRows
        in NtmLayer namedLstm namedReadFc namedWriteFc namedOutputFc
                    namedMemory namedReadAddr namedWriteAddr namedReadOut (Just memBuf')
      _ => layer

  export
  nameNetworkParams : {i, o : Nat} -> {hs : List Nat} -> String -> Network i hs o Variable -> Network i hs o Variable
  nameNetworkParams prefx (OutputLayer layer) = OutputLayer (nameParams prefx layer)
  nameNetworkParams prefx (layer ~> rest) = nameParams prefx layer ~> nameNetworkParams prefx rest


----------------------------------------------------------------------
-- Automatic Parameter Naming
----------------------------------------------------------------------

layerPrefix : Layer i o ty -> String
layerPrefix (LinearLayer _ _ _ _) = "ll"
layerPrefix (RnnLayer _ _ _ _ _ _) = "rnn"
layerPrefix (LstmLayer _ _ _ _ _ _ _ _) = "lstm"
layerPrefix (NtmLayer _ _ _ _ _ _ _ _ _) = "ntm"
layerPrefix _ = ""

mutual
  autoNameLayer : String -> SortedMap String Nat -> {i, o : Nat}
              -> Layer i o Variable
              -> (SortedMap String Nat, Layer i o Variable)
  autoNameLayer scope counts layer =
    let pfx = layerPrefix layer
    in if pfx == "" then (counts, layer)
       else let n = fromMaybe 0 (lookup pfx counts)
                counts' = insert pfx (n + 1) counts
                fullName = scope ++ pfx ++ show n
            in case layer of
              (NtmLayer {n=nn} {m=mm} lstm readFc writeFc outputFc memory readAddr writeAddr readOutput _) =>
                let np = nameParam . (fullName ++ "_" ++)
                    namedMemory = zipWith (np "mem") enumerate memory
                    namedReadAddr = zipWith (np "rAddr") enumerate readAddr
                    namedWriteAddr = zipWith (np "wAddr") enumerate writeAddr
                    namedReadOut = zipWith (np "rOut") enumerate readOutput
                    (_, lstm') = autoNameLayer (fullName ++ "_") empty lstm
                    (_, readFc') = autoNameLayer (fullName ++ "_readFc_") empty readFc
                    (_, writeFc') = autoNameLayer (fullName ++ "_writeFc_") empty writeFc
                    (_, outputFc') = autoNameLayer (fullName ++ "_outputFc_") empty outputFc
                    (VTensor memRows) = namedMemory
                    memBuf = prim__ntmMemBufAlloc (cast nn) (cast mm)
                    memBuf' = initNtmMemBuf memBuf 0 memRows
                in (counts', NtmLayer lstm' readFc' writeFc' outputFc'
                             namedMemory namedReadAddr namedWriteAddr namedReadOut (Just memBuf'))
              _ => (counts', nameParams fullName layer)

  autoNameNetwork : String -> SortedMap String Nat
                 -> {i, o : Nat} -> {hs : List Nat}
                 -> Network i hs o Variable
                 -> (SortedMap String Nat, Network i hs o Variable)
  autoNameNetwork scope counts (OutputLayer layer) =
    let (counts', layer') = autoNameLayer scope counts layer
    in (counts', OutputLayer layer')
  autoNameNetwork scope counts (layer ~> rest) =
    let (counts', layer') = autoNameLayer scope counts layer
        (counts'', rest') = autoNameNetwork scope counts' rest
    in (counts'', layer' ~> rest')

||| Automatically name all parameters using type-based prefixes.
||| LinearLayer -> ll0, ll1, ...; RnnLayer -> rnn0, rnn1, ...; NtmLayer -> ntm0, ...
||| Each layer gets unique names, preventing gradient cross-contamination.
export
autoName : {i, o : Nat} -> {hs : List Nat}
        -> Network i hs o Variable -> Network i hs o Variable
autoName net = snd (autoNameNetwork "" empty net)


----------------------------------------------------------------------
-- Weight Buffer Sync (after applyDeltas)
----------------------------------------------------------------------

||| Project addressing weights onto the probability simplex after
||| gradient updates. Clamps to [epsilon, inf) and renormalizes to
||| sum to 1, preventing NaN from pow(negative, non-integer) in focus.
projectWeights : {n : Nat} -> Vector n Variable -> Vector n Variable
projectWeights (VTensor vs) =
  let clamp : Tensor [] Variable -> Tensor [] Variable
      clamp (STensor v) = STensor ({ value $= max 0.00000001 } v)
      clamped = map clamp vs
      s : Double
      s = foldl (\acc, (STensor v) => acc + v.value) 0.0 clamped
      normalize : Tensor [] Variable -> Tensor [] Variable
      normalize (STensor v) = STensor ({ value $= (/ s) } v)
  in VTensor (map normalize clamped)

mutual
  export
  syncLayerBuffers : {i, o : Nat} -> Layer i o Variable -> Layer i o Variable
  syncLayerBuffers (LinearLayer (VTensor wRows) (VTensor biasElems) (Just wb) (Just bb)) =
    let wb' = syncWeightBuf wb 0 wRows
        bb' = syncWeightBufRow bb 0 biasElems
    in LinearLayer (VTensor wRows) (VTensor biasElems) (Just wb') (Just bb')
  syncLayerBuffers (LinearLayer (VTensor wRows) bias (Just wb) Nothing) =
    let wb' = syncWeightBuf wb 0 wRows
    in LinearLayer (VTensor wRows) bias (Just wb') Nothing
  syncLayerBuffers (RnnLayer (VTensor iwRows) (VTensor rwRows) bias po (Just iwb) (Just rwb)) =
    let iwb' = syncWeightBuf iwb 0 iwRows
        rwb' = syncWeightBuf rwb 0 rwRows
    in RnnLayer (VTensor iwRows) (VTensor rwRows) bias po (Just iwb') (Just rwb')
  syncLayerBuffers (LstmLayer (VTensor iwRows) (VTensor rwRows) (VTensor biasElems) hs cs (Just iwb) (Just rwb) (Just bb)) =
    let iwb' = syncWeightBuf iwb 0 iwRows
        rwb' = syncWeightBuf rwb 0 rwRows
        bb' = syncWeightBufRow bb 0 biasElems
    in LstmLayer (VTensor iwRows) (VTensor rwRows) (VTensor biasElems) hs cs (Just iwb') (Just rwb') (Just bb')
  syncLayerBuffers (LstmLayer (VTensor iwRows) (VTensor rwRows) bias hs cs (Just iwb) (Just rwb) Nothing) =
    let iwb' = syncWeightBuf iwb 0 iwRows
        rwb' = syncWeightBuf rwb 0 rwRows
    in LstmLayer (VTensor iwRows) (VTensor rwRows) bias hs cs (Just iwb') (Just rwb') Nothing
  syncLayerBuffers (NtmLayer lstm readFc writeFc outputFc (VTensor memRows) ra wa ro (Just memBuf)) =
    let mb' = prim__ntmMemBufResetCache (syncNtmMemBuf memBuf 0 memRows)
    in NtmLayer (syncLayerBuffers lstm) (syncLayerBuffers readFc)
                (syncLayerBuffers writeFc) (syncLayerBuffers outputFc)
                (VTensor memRows) (projectWeights ra) (projectWeights wa) ro (Just mb')
  syncLayerBuffers (NtmLayer lstm readFc writeFc outputFc mem ra wa ro Nothing) =
    NtmLayer (syncLayerBuffers lstm) (syncLayerBuffers readFc)
             (syncLayerBuffers writeFc) (syncLayerBuffers outputFc)
             mem (projectWeights ra) (projectWeights wa) ro Nothing
  syncLayerBuffers l = l

  export
  syncNetworkBuffers : {i, o : Nat} -> {hs : List Nat} -> Network i hs o Variable -> Network i hs o Variable
  syncNetworkBuffers (OutputLayer layer) = OutputLayer (syncLayerBuffers layer)
  syncNetworkBuffers (layer ~> rest) = syncLayerBuffers layer ~> syncNetworkBuffers rest


----------------------------------------------------------------------
-- Bulk Delta Application (C-direct, no emap/sync)
----------------------------------------------------------------------

mutual
  ||| Apply optimizer deltas directly to C buffers (WeightBuf/NtmMemBuf),
  ||| bypassing emap + syncLayerBuffers. Resets cache generations and
  ||| projects addressing weights. Variable.value fields are NOT updated
  ||| (forward pass reads from buffers, not Variable records).
  export
  applyDeltasAndSyncLayer : {i, o : Nat} -> AnyPtr -> Layer i o Variable -> Layer i o Variable
  applyDeltasAndSyncLayer deltas (LinearLayer w b (Just wb) (Just bb)) =
    let wb' = prim__weightBufApplyDeltas wb deltas
        bb' = prim__weightBufApplyDeltas bb deltas
    in LinearLayer w b (Just wb') (Just bb')
  applyDeltasAndSyncLayer deltas (LinearLayer w b (Just wb) Nothing) =
    let wb' = prim__weightBufApplyDeltas wb deltas
    in LinearLayer w b (Just wb') Nothing
  applyDeltasAndSyncLayer deltas (LstmLayer iw rw b hs cs (Just iwb) (Just rwb) (Just bb)) =
    let iwb' = prim__weightBufApplyDeltas iwb deltas
        rwb' = prim__weightBufApplyDeltas rwb deltas
        bb' = prim__weightBufApplyDeltas bb deltas
    in LstmLayer iw rw b hs cs (Just iwb') (Just rwb') (Just bb')
  applyDeltasAndSyncLayer deltas (LstmLayer iw rw b hs cs (Just iwb) (Just rwb) Nothing) =
    let iwb' = prim__weightBufApplyDeltas iwb deltas
        rwb' = prim__weightBufApplyDeltas rwb deltas
    in LstmLayer iw rw b hs cs (Just iwb') (Just rwb') Nothing
  applyDeltasAndSyncLayer deltas (NtmLayer lstm readFc writeFc outputFc mem ra wa ro (Just memBuf)) =
    let mb' = prim__ntmMemBufApplyDeltas memBuf deltas
    in NtmLayer (applyDeltasAndSyncLayer deltas lstm) (applyDeltasAndSyncLayer deltas readFc)
                (applyDeltasAndSyncLayer deltas writeFc) (applyDeltasAndSyncLayer deltas outputFc)
                mem (projectWeights ra) (projectWeights wa) ro (Just mb')
  applyDeltasAndSyncLayer deltas (NtmLayer lstm readFc writeFc outputFc mem ra wa ro Nothing) =
    NtmLayer (applyDeltasAndSyncLayer deltas lstm) (applyDeltasAndSyncLayer deltas readFc)
             (applyDeltasAndSyncLayer deltas writeFc) (applyDeltasAndSyncLayer deltas outputFc)
             mem (projectWeights ra) (projectWeights wa) ro Nothing
  applyDeltasAndSyncLayer _ l = l

  export
  applyDeltasAndSyncNetwork : {i, o : Nat} -> {hs : List Nat} -> AnyPtr -> Network i hs o Variable -> Network i hs o Variable
  applyDeltasAndSyncNetwork deltas (OutputLayer layer) = OutputLayer (applyDeltasAndSyncLayer deltas layer)
  applyDeltasAndSyncNetwork deltas (layer ~> rest) = applyDeltasAndSyncLayer deltas layer ~> applyDeltasAndSyncNetwork deltas rest


----------------------------------------------------------------------
-- Read From Buffers (C buffer -> Variable.value)
----------------------------------------------------------------------

mutual
  ||| Read values from C buffers back into Variable.value fields.
  ||| Reverse of syncLayerBuffers. Call after dense training before
  ||| toDoubleNetwork to ensure Variable records reflect trained weights.
  export
  readFromBuffersLayer : {i, o : Nat} -> Layer i o Variable -> Layer i o Variable
  readFromBuffersLayer (LinearLayer (VTensor wRows) (VTensor biasElems) (Just wb) (Just bb)) =
    LinearLayer (VTensor (readWeightBuf wb 0 wRows)) (VTensor (readWeightBufRow bb 0 biasElems)) (Just wb) (Just bb)
  readFromBuffersLayer (LinearLayer (VTensor wRows) bias (Just wb) Nothing) =
    LinearLayer (VTensor (readWeightBuf wb 0 wRows)) bias (Just wb) Nothing
  readFromBuffersLayer (LstmLayer (VTensor iwRows) (VTensor rwRows) (VTensor biasElems) hs cs (Just iwb) (Just rwb) (Just bb)) =
    LstmLayer (VTensor (readWeightBuf iwb 0 iwRows)) (VTensor (readWeightBuf rwb 0 rwRows))
              (VTensor (readWeightBufRow bb 0 biasElems)) hs cs (Just iwb) (Just rwb) (Just bb)
  readFromBuffersLayer (LstmLayer (VTensor iwRows) (VTensor rwRows) bias hs cs (Just iwb) (Just rwb) Nothing) =
    LstmLayer (VTensor (readWeightBuf iwb 0 iwRows)) (VTensor (readWeightBuf rwb 0 rwRows))
              bias hs cs (Just iwb) (Just rwb) Nothing
  readFromBuffersLayer (NtmLayer lstm readFc writeFc outputFc (VTensor memRows) ra wa ro (Just memBuf)) =
    NtmLayer (readFromBuffersLayer lstm) (readFromBuffersLayer readFc)
             (readFromBuffersLayer writeFc) (readFromBuffersLayer outputFc)
             (VTensor (readNtmMemBuf memBuf 0 memRows)) ra wa ro (Just memBuf)
  readFromBuffersLayer (NtmLayer lstm readFc writeFc outputFc mem ra wa ro Nothing) =
    NtmLayer (readFromBuffersLayer lstm) (readFromBuffersLayer readFc)
             (readFromBuffersLayer writeFc) (readFromBuffersLayer outputFc)
             mem ra wa ro Nothing
  readFromBuffersLayer l = l

  export
  readFromBuffersNetwork : {i, o : Nat} -> {hs : List Nat} -> Network i hs o Variable -> Network i hs o Variable
  readFromBuffersNetwork (OutputLayer layer) = OutputLayer (readFromBuffersLayer layer)
  readFromBuffersNetwork (layer ~> rest) = readFromBuffersLayer layer ~> readFromBuffersNetwork rest
