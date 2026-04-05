module Layer.Ntm

import Data.Vect
import Data.Zippable
import System.Random

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
-- Addressing Weight Projection
----------------------------------------------------------------------

||| Project addressing weights onto the probability simplex after
||| gradient updates. Clamps to [epsilon, inf) and renormalizes.
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


----------------------------------------------------------------------
-- NTM State
----------------------------------------------------------------------

public export
record NtmState (n : Nat) (m : Nat) (h : Nat) (inputSize : Nat) (outputSize : Nat) (ty : Type) where
  constructor MkNtm
  lstm : LstmState (m + inputSize) h ty
  readFc : LinearState h (ReadParamWidth m) ty
  writeFc : LinearState h (WriteParamWidth m) ty
  outputFc : LinearState (h + m) outputSize ty
  memory : Matrix n m ty
  readAddr : Vector n ty
  writeAddr : Vector n ty
  readOutput : Vector m ty
  -- Consolidated tensors (Just after nameLayer)
  memTensor : Maybe AnyPtr      -- [n, m]
  readAddrTensor : Maybe AnyPtr -- [n]
  writeAddrTensor : Maybe AnyPtr -- [n]
  readOutTensor : Maybe AnyPtr  -- [m]


----------------------------------------------------------------------
-- Variable-specialized NTM Head Operations
----------------------------------------------------------------------

||| Variable-specialized read head with unbounded gamma (softplus).
forwardReadHeadUnboundedVar : {n, w : Nat} ->
    Matrix n w Variable -> ReadHead n Variable ->
    Vector ((w + ShiftKernelSize) + 3) Variable ->
    (ReadHead n Variable, Vector w Variable)
forwardReadHeadUnboundedVar memory rh inp =
  let (mainInput, params) = splitAt (w + ShiftKernelSize) inp
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

||| Variable-specialized write head with interpolation write (no erase).
forwardWriteHeadInterpVar : {n, w : Nat} ->
    Matrix n w Variable -> WriteHead n Variable ->
    Vector (((w + ShiftKernelSize) + 3) + w) Variable ->
    (WriteHead n Variable, Matrix n w Variable)
forwardWriteHeadInterpVar memory (MkWriteHead readHead) inp =
  let (readHeadInput, rawAdd) = Tensor.splitAt ((w + ShiftKernelSize) + 3) inp
      addVector = rawAdd
      (newReadHead, _) = forwardReadHeadUnboundedVar memory readHead readHeadInput
      newWriteHead = MkWriteHead newReadHead
      newMemoryMatrix = interpolationWriteVar newWriteHead.readHead.addressingWeights memory addVector
  in (newWriteHead, newMemoryMatrix)

----------------------------------------------------------------------
-- Debug Helpers
----------------------------------------------------------------------

showVecD : {n : Nat} -> Vector n Double -> String
showVecD (VTensor xs) = "[" ++ go xs ++ "]"
  where
    go : Vect k (Tensor [] Double) -> String
    go [] = ""
    go [STensor x] = show x
    go (STensor x :: rest) = show x ++ " " ++ go rest

showMatD : {r, c : Nat} -> Matrix r c Double -> String
showMatD (VTensor rows) = "[" ++ go rows ++ "]"
  where
    go : Vect k (Vector c Double) -> String
    go [] = ""
    go [row] = showVecD row
    go (row :: rest) = showVecD row ++ "\n " ++ go rest


----------------------------------------------------------------------
-- LayerLike Instance
----------------------------------------------------------------------

export
{n, m, h : Nat} -> LayerLike (NtmState n m h) where
  -- Generic forward pass (Double-based)
  applyGeneric (MkNtm lstm readFc writeFc outputFc memory readAddr writeAddr readOutput _ _ _ _) inp =
    let -- 1. Controller: LSTM(readOutput ++ input)
        (updLstm, hidden) = applyGeneric lstm (readOutput ++ inp)
        -- 2. Cell state for head FCs
        cell = extractCellState updLstm
        -- 3. Head params from cell state
        readParams = snd (applyGeneric readFc cell)
        writeParams = snd (applyGeneric writeFc cell)
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
        output = snd (applyGeneric outputFc (hidden ++ newReadOutput))
    in (MkNtm updLstm readFc writeFc outputFc
             newMemory newReadAddr' newWriteAddr' newReadOutput
             Nothing Nothing Nothing Nothing, output)

  -- Variable-based NTM forward pass: fused C ops when tensor handles available
  applyVar (MkNtm lstm readFc writeFc outputFc memory readAddr writeAddr readOutput (Just memT) (Just raT) (Just waT) (Just roT)) inp =
    case applyVar lstm (readOutput ++ inp) of
      (updLstm, hidden) =>
        let cell = extractCellState updLstm
        in case applyVar readFc cell of
          (_, readParams) =>
            case applyVar writeFc cell of
              (_, writeParams) =>
                -- Parse read head params (scalar Variables from FC output)
                let (mainInput, params) = splitAt (m + ShiftKernelSize) readParams
                    (keyVector, shiftVector) = splitAt m mainInput
                    (betaVec, params') = splitAt 1 params
                    (gVec, gammaVec) = splitAt 1 params'
                    beta = softplus (sum betaVec)
                    g = sigmoidVar (sum gVec)
                    gamma = 1 + softplus (sum gammaVec)
                    shiftKernel = softmaxVar shiftVector
                    -- Stack key and shift kernel (small: m and 3 elements)
                    (VTensor keyElems) = keyVector
                    keyT = vecStackTensor keyElems
                    (VTensor shiftElems) = shiftKernel
                    shiftT = vecStackTensor shiftElems
                    -- Fused read head: 1 C call for entire addressing pipeline
                    readPair = prim__ntmReadHead memT raT keyT beta.tensorPtr g.tensorPtr gamma.tensorPtr shiftT
                    newReadAddrT = prim__pairFirst readPair
                    newReadOutT = prim__pairSecond readPair
                    -- Parse write head params
                    (wReadHeadInput, rawAdd) = Tensor.splitAt ((m + ShiftKernelSize) + 3) writeParams
                    (wMainInput, wParams) = splitAt (m + ShiftKernelSize) wReadHeadInput
                    (wKeyVector, wShiftVector) = splitAt m wMainInput
                    (wBetaVec, wParams') = splitAt 1 wParams
                    (wGVec, wGammaVec) = splitAt 1 wParams'
                    wBeta = softplus (sum wBetaVec)
                    wG = sigmoidVar (sum wGVec)
                    wGamma = 1 + softplus (sum wGammaVec)
                    wShiftKernel = softmaxVar wShiftVector
                    (VTensor wKeyElems) = wKeyVector
                    wKeyT = vecStackTensor wKeyElems
                    (VTensor wShiftElems) = wShiftKernel
                    wShiftT = vecStackTensor wShiftElems
                    -- Fused write head addressing
                    writePair = prim__ntmReadHead memT waT wKeyT wBeta.tensorPtr wG.tensorPtr wGamma.tensorPtr wShiftT
                    newWriteAddrT = prim__pairFirst writePair
                    -- Interpolation write: memory' = memory + outer(writeAddr, addVector)
                    (VTensor addElems) = rawAdd
                    addT = vecStackTensor addElems
                    newMemT = prim__ntmInterpWrite memT newWriteAddrT addT
                    -- Output FC: hidden ++ readOutput
                    newReadOutput = VTensor (tensorToScalars newReadOutT 0 m)
                    output = snd (applyVar outputFc (hidden ++ newReadOutput))
                    -- Skip unpacking addressing weights — tensor handles carry the real
                    -- data, scalar Vects are stale placeholders (never read in tensor path)
                in (MkNtm updLstm readFc writeFc outputFc
                         memory readAddr writeAddr newReadOutput
                         (Just newMemT) (Just newReadAddrT) (Just newWriteAddrT) (Just newReadOutT), output)

  -- Variable-based NTM forward pass: fallback using individual Variable ops
  applyVar (MkNtm lstm readFc writeFc outputFc memory readAddr writeAddr readOutput _ _ _ _) inp =
    case applyVar lstm (readOutput ++ inp) of
      (updLstm, hidden) =>
        let cell = extractCellState updLstm
        in case applyVar readFc cell of
          (_, readParams) =>
            case applyVar writeFc cell of
              (_, writeParams) =>
                let rh = MkReadHead readAddr
                in case forwardReadHeadUnboundedVar memory rh readParams of
                  (readHead, newReadOutput) =>
                    let wh = MkWriteHead (MkReadHead writeAddr)
                    in case forwardWriteHeadInterpVar memory wh writeParams of
                      (writeHead, newMemory) =>
                        let newReadAddr' = readHead.addressingWeights
                            newWriteAddr' = writeHead.readHead.addressingWeights
                            output = snd (applyVar outputFc (hidden ++ newReadOutput))
                        in (MkNtm updLstm readFc writeFc outputFc
                                 newMemory newReadAddr' newWriteAddr' newReadOutput
                                 Nothing Nothing Nothing Nothing, output)

  emapLayer f (MkNtm lstm rfc wfc ofc mem ra wa ro _ _ _ _) =
    MkNtm (emapLayer f lstm) (emapLayer f rfc) (emapLayer f wfc) (emapLayer f ofc)
           (map f mem) (map f ra) (map f wa) (map f ro)
           Nothing Nothing Nothing Nothing

  showLayer {i} {o} _ =
    "Ntm<" ++ show i ++ ":" ++ show o
    ++ ", mem=" ++ show n ++ "x" ++ show m ++ ", h=" ++ show h ++ ">"

  nameLayer prefx (MkNtm lstm readFc writeFc outputFc memory readAddr writeAddr readOutput _ _ _ _) =
    let namedLstm = nameLayer (prefx ++ "_lstm0") lstm
        namedReadFc = nameLayer (prefx ++ "_readFc_ll0") readFc
        namedWriteFc = nameLayer (prefx ++ "_writeFc_ll0") writeFc
        namedOutputFc = nameLayer (prefx ++ "_outputFc_ll0") outputFc
    in if prim__backendSupportsTensorParams == 1
      then -- Tensor path: consolidated tensors for sub-layers + non-param state tensors
        let (VTensor memRows) = memory
            (VTensor raElems) = readAddr
            (VTensor waElems) = writeAddr
            (VTensor roElems) = readOutput
            nI = cast {to=Int} n
            mI = cast {to=Int} m
            -- NTM state: persistent but NOT params (non-learnable, reset per sequence)
            memT = prim__createState2d nI mI (let buf = prim__allocDoubles (nI * mI) in packMatrixValues buf 0 {n=m} memRows)
            raT = prim__createState1d nI (let buf = prim__allocDoubles nI in packScalarValues buf 0 raElems)
            waT = prim__createState1d nI (let buf = prim__allocDoubles nI in packScalarValues buf 0 waElems)
            roT = prim__createState1d mI (let buf = prim__allocDoubles mI in packScalarValues buf 0 roElems)
        in MkNtm namedLstm namedReadFc namedWriteFc namedOutputFc
                 memory readAddr writeAddr readOutput
                 (Just memT) (Just raT) (Just waT) (Just roT)
      else -- Scalar path (tape backend)
        let np = nameParam . (prefx ++ "_" ++)
        in MkNtm namedLstm namedReadFc namedWriteFc namedOutputFc
                 (zipWith (np "mem") enumerate memory)
                 (zipWith (np "rAddr") enumerate readAddr)
                 (zipWith (np "wAddr") enumerate writeAddr)
                 (zipWith (np "rOut") enumerate readOutput)
                 Nothing Nothing Nothing Nothing

  layerPrefix _ = "ntm"

  toDoubleLayer (MkNtm lstm rfc wfc ofc mem ra wa ro _ _ _ _) =
    MkNtm (toDoubleLayer lstm) (toDoubleLayer rfc) (toDoubleLayer wfc) (toDoubleLayer ofc)
           (map value mem) (map value ra) (map value wa) (map value ro)
           Nothing Nothing Nothing Nothing

  debugApply {i} {o} (MkNtm lstm readFc writeFc outputFc memory readAddr writeAddr readOutput mt rat wat rot) inp =
    let st = MkNtm lstm readFc writeFc outputFc memory readAddr writeAddr readOutput mt rat wat rot
        (updated, output) = applyGeneric st inp
        entry = MkDebugEntry ("Ntm<" ++ show i ++ ":" ++ show o
                ++ ", mem=" ++ show n ++ "x" ++ show m ++ ">")
          [ ("readAddr",   showVecD readAddr)
          , ("writeAddr",  showVecD writeAddr)
          , ("readOutput", showVecD readOutput)
          , ("memory",     showMatD memory)
          ]
    in (updated, output, entry)

  syncBuffers (MkNtm lstm readFc writeFc outputFc mem ra wa ro mt rat wat rot) =
    MkNtm (syncBuffers lstm) (syncBuffers readFc) (syncBuffers writeFc) (syncBuffers outputFc)
           mem (projectWeights ra) (projectWeights wa) ro
           mt rat wat rot

  applyDeltasAndSync deltas (MkNtm lstm readFc writeFc outputFc mem ra wa ro mt rat wat rot) =
    MkNtm (applyDeltasAndSync deltas lstm) (applyDeltasAndSync deltas readFc)
           (applyDeltasAndSync deltas writeFc) (applyDeltasAndSync deltas outputFc)
           mem (projectWeights ra) (projectWeights wa) ro
           mt rat wat rot

  readFromBuffers (MkNtm lstm readFc writeFc outputFc mem ra wa ro mt rat wat rot) =
    MkNtm (readFromBuffers lstm) (readFromBuffers readFc)
           (readFromBuffers writeFc) (readFromBuffers outputFc)
           mem ra wa ro
           mt rat wat rot

  getParamIds (MkNtm lstm readFc writeFc outputFc mem ra wa ro _ _ _ _) =
    getParamIds lstm ++ getParamIds readFc ++ getParamIds writeFc ++ getParamIds outputFc
      ++ tensorIds mem ++ tensorIds ra ++ tensorIds wa ++ tensorIds ro
    where
      tensorIds : {dims : Vect rank Nat} -> Tensor dims Variable -> List String
      tensorIds = mapMaybe paramId . toList


----------------------------------------------------------------------
-- Constructor
----------------------------------------------------------------------

||| Create a PyTorch-aligned NTM layer.
||| n = memory slots, m = memory width, h = controller hidden size.
export
ntmLayer : {inputSize, outputSize, n, m, h : Nat} ->
           (Num ty, FromDouble ty) => IO (AnyLayer inputSize outputSize ty)
ntmLayer = do
  lstm <- mkLstm {i = m + inputSize, o = h}
  readFc <- mkLinearWithBias (xavierGain 1.4 uniform) 0.01 {i = h, o = ReadParamWidth m}
  writeFc <- mkLinearWithBias (xavierGain 1.4 uniform) 0.01 {i = h, o = WriteParamWidth m}
  outputFc <- mkLinearWithBias (he uniform) 0.01 {i = h + m, o = outputSize}
  -- Memory: sigmoid(random) ≈ values in [0,1]
  memInit <- traverse (\_ => map fromDouble (xavier uniform n m >>= \v => pure (1.0 / (1.0 + exp (negate v)))))
                      (the (Matrix n m ty) zeros)
  let readAddr = the (Vector n ty) zeros
  let writeAddr = the (Vector n ty) zeros
  readOut <- traverse (\_ => map fromDouble (he uniform m 1)) (the (Vector m ty) zeros)
  pure $ MkAnyLayer (NtmState n m h) $ MkNtm lstm readFc writeFc outputFc memInit readAddr writeAddr readOut Nothing Nothing Nothing Nothing
