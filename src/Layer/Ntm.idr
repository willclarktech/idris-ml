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
  applyGeneric (MkNtm lstm readFc writeFc outputFc memory readAddr writeAddr readOutput) inp =
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
             newMemory newReadAddr' newWriteAddr' newReadOutput, output)

  -- Variable-based NTM forward pass
  applyVar (MkNtm lstm readFc writeFc outputFc memory readAddr writeAddr readOutput) inp =
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
                                 newMemory newReadAddr' newWriteAddr' newReadOutput, output)

  emapLayer f (MkNtm lstm rfc wfc ofc mem ra wa ro) =
    MkNtm (emapLayer f lstm) (emapLayer f rfc) (emapLayer f wfc) (emapLayer f ofc)
           (map f mem) (map f ra) (map f wa) (map f ro)

  showLayer {i} {o} _ =
    "Ntm<" ++ show i ++ ":" ++ show o
    ++ ", mem=" ++ show n ++ "x" ++ show m ++ ", h=" ++ show h ++ ">"

  nameLayer prefx (MkNtm lstm readFc writeFc outputFc memory readAddr writeAddr readOutput) =
    let np = nameParam . (prefx ++ "_" ++)
        namedMemory = zipWith (np "mem") enumerate memory
        namedReadAddr = zipWith (np "rAddr") enumerate readAddr
        namedWriteAddr = zipWith (np "wAddr") enumerate writeAddr
        namedReadOut = zipWith (np "rOut") enumerate readOutput
        -- Sub-layers: auto-name with scoped prefixes (always counter 0)
        namedLstm = nameLayer (prefx ++ "_lstm0") lstm
        namedReadFc = nameLayer (prefx ++ "_readFc_ll0") readFc
        namedWriteFc = nameLayer (prefx ++ "_writeFc_ll0") writeFc
        namedOutputFc = nameLayer (prefx ++ "_outputFc_ll0") outputFc
    in MkNtm namedLstm namedReadFc namedWriteFc namedOutputFc
             namedMemory namedReadAddr namedWriteAddr namedReadOut

  layerPrefix _ = "ntm"

  toDoubleLayer (MkNtm lstm rfc wfc ofc mem ra wa ro) =
    MkNtm (toDoubleLayer lstm) (toDoubleLayer rfc) (toDoubleLayer wfc) (toDoubleLayer ofc)
           (map value mem) (map value ra) (map value wa) (map value ro)

  debugApply {i} {o} (MkNtm lstm readFc writeFc outputFc memory readAddr writeAddr readOutput) inp =
    let st = MkNtm lstm readFc writeFc outputFc memory readAddr writeAddr readOutput
        (updated, output) = applyGeneric st inp
        entry = MkDebugEntry ("Ntm<" ++ show i ++ ":" ++ show o
                ++ ", mem=" ++ show n ++ "x" ++ show m ++ ">")
          [ ("readAddr",   showVecD readAddr)
          , ("writeAddr",  showVecD writeAddr)
          , ("readOutput", showVecD readOutput)
          , ("memory",     showMatD memory)
          ]
    in (updated, output, entry)

  syncBuffers (MkNtm lstm readFc writeFc outputFc mem ra wa ro) =
    MkNtm (syncBuffers lstm) (syncBuffers readFc) (syncBuffers writeFc) (syncBuffers outputFc)
           mem (projectWeights ra) (projectWeights wa) ro

  applyDeltasAndSync deltas (MkNtm lstm readFc writeFc outputFc mem ra wa ro) =
    MkNtm (applyDeltasAndSync deltas lstm) (applyDeltasAndSync deltas readFc)
           (applyDeltasAndSync deltas writeFc) (applyDeltasAndSync deltas outputFc)
           mem (projectWeights ra) (projectWeights wa) ro

  readFromBuffers (MkNtm lstm readFc writeFc outputFc mem ra wa ro) =
    MkNtm (readFromBuffers lstm) (readFromBuffers readFc)
           (readFromBuffers writeFc) (readFromBuffers outputFc)
           mem ra wa ro

  getParamIds (MkNtm lstm readFc writeFc outputFc mem ra wa ro) =
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
  pure $ MkAnyLayer (NtmState n m h) $ MkNtm lstm readFc writeFc outputFc memInit readAddr writeAddr readOut
