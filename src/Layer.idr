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

||| Read head output + input
public export
NtmInputWidth : Nat -> Nat
NtmInputWidth w = w + w

||| Key vector + shift vector (ShiftKernelSize) + params (beta, g, gamma)
public export
ReadHeadInputWidth : Nat -> Nat -> Nat
ReadHeadInputWidth _ w = (w + ShiftKernelSize) + 3

||| Read head input + erase vector + add vector
public export
WriteHeadInputWidth : Nat -> Nat -> Nat
WriteHeadInputWidth n w = ReadHeadInputWidth n w + w + w

||| Read head input + Write head input + output
public export
NtmOutputWidth : Nat -> Nat -> Nat
NtmOutputWidth n w = ReadHeadInputWidth n w + (WriteHeadInputWidth n w + w)


----------------------------------------------------------------------
-- Layer and Network Types (mutually recursive)
----------------------------------------------------------------------

mutual
  public export
  data Layer : (inputSize : Nat) -> (outputSize : Nat) -> Type -> Type where
    LinearLayer : (weights : Matrix outputSize inputSize ty) -> (bias : Vector outputSize ty) -> (wBuf : Maybe AnyPtr) -> Layer inputSize outputSize ty
    RnnLayer : (inputWeights : Matrix outputSize inputSize ty) -> (recurrentWeights : Matrix outputSize outputSize ty) -> (bias : Vector outputSize ty) -> (previousOutput : Vector outputSize ty) -> (iwBuf : Maybe AnyPtr) -> (rwBuf : Maybe AnyPtr) -> Layer inputSize outputSize ty
    ActivationLayer : (name : String) -> (f : ActivationFunction ty) -> Layer n n ty
    NormalizationLayer : (name : String) -> (f : NormalizationFunction ty) -> Layer n n ty
    LstmLayer : (inputWeights : Matrix (4 * outputSize) inputSize ty) ->
                (recurrentWeights : Matrix (4 * outputSize) outputSize ty) ->
                (bias : Vector (4 * outputSize) ty) ->
                (hiddenState : Vector outputSize ty) ->
                (cellState : Vector outputSize ty) ->
                (iwBuf : Maybe AnyPtr) -> (rwBuf : Maybe AnyPtr) ->
                Layer inputSize outputSize ty
    NtmLayer : {n : Nat} -> {hs : List Nat} ->
               (controller : Network (NtmInputWidth w) hs (NtmOutputWidth n w) ty) ->
               (memory : Matrix n w ty) ->
               (readHead : ReadHead n ty) ->
               (writeHead : WriteHead n ty) ->
               (readHeadOutput : Vector w ty) ->
               Layer w w ty

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
  show {inputSize} {outputSize} (LinearLayer _ _ _) = "Linear<" ++ show inputSize ++ ":" ++ show outputSize ++ ">"
  show {inputSize} {outputSize} (RnnLayer _ _ _ _ _ _) = "Rnn<" ++ show inputSize ++ ":" ++ show outputSize ++ ">"
  show {inputSize} {outputSize} (LstmLayer _ _ _ _ _ _ _) = "Lstm<" ++ show inputSize ++ ":" ++ show outputSize ++ ">"
  show (ActivationLayer name _) = "Activation<" ++ name ++ ">"
  show (NormalizationLayer name _) = "Normalization<" ++ name ++ ">"
  show {inputSize} (NtmLayer {n} _ _ _ _ _) = "Ntm<" ++ show inputSize ++ ", mem=" ++ show n ++ ">"

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
    emap f (LinearLayer w b wb) = LinearLayer (map f w) (map f b) wb
    emap f (RnnLayer iw rw b po iwb rwb) = RnnLayer (map f iw) (map f rw) (map f b) (map f po) iwb rwb
    emap f (LstmLayer iw rw b hs cs iwb rwb) = LstmLayer (map f iw) (map f rw) (map f b) (map f hs) (map f cs) iwb rwb
    emap f (NtmLayer controller mem rh wh ro) =
      NtmLayer (emap f controller) (map f mem) (map f rh) (map f wh) (map f ro)
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
-- Forward Pass
----------------------------------------------------------------------

mutual
  export
  applyLayer : (Floating ty, Fractional ty, Neg ty, Num ty, Ord ty) => {i, o : Nat} -> Layer i o ty -> Vector i ty -> (Layer i o ty, Vector o ty)
  applyLayer layer@(LinearLayer weights bias _) xs = (layer, matrixVectorMultiply {m=o, n=i} weights xs + bias)
  applyLayer (RnnLayer inputWeights recurrentWeights bias previousOutput iwb rwb) xs =
    let
      output = matrixVectorMultiply inputWeights xs + matrixVectorMultiply recurrentWeights previousOutput + bias
      updatedLayer = RnnLayer inputWeights recurrentWeights bias output iwb rwb
    in (updatedLayer, output)
  applyLayer layer@(ActivationLayer _ f) xs = (layer, map f xs)
  applyLayer layer@(NormalizationLayer _ f) xs = (layer, f xs)
  applyLayer {i} {o} (LstmLayer inputWeights recurrentWeights bias hiddenState cellState iwb rwb) xs =
    let
      combined = matrixVectorMultiply inputWeights xs + matrixVectorMultiply recurrentWeights hiddenState + bias
      gates = lstmSplitGates {o} combined
      iGate = fst gates
      fGate = fst (snd gates)
      gGate = fst (snd (snd gates))
      oGate = snd (snd (snd gates))
      newCell = map sig fGate * cellState + map sig iGate * map tanhBound gGate
      newHidden = map sig oGate * map tanhBound newCell
      updatedLayer = LstmLayer inputWeights recurrentWeights bias newHidden newCell iwb rwb
    in (updatedLayer, newHidden)
  applyLayer {i} (NtmLayer {n} {hs} controller memory readHead writeHead readHeadOutput) inp =
    let
      (newController, controllerOutput) = forward controller (readHeadOutput ++ inp)
      (readHeadInput, controllerOutput') = Tensor.splitAt (ReadHeadInputWidth n i) controllerOutput
      (writeHeadInput, networkOutput) = Tensor.splitAt (WriteHeadInputWidth n i) controllerOutput'
      (newReadHead, newReadHeadOutput) = forwardReadHead softmax memory readHead readHeadInput
      (newWriteHead, rawMemory) = forwardWriteHead softmax memory writeHead writeHeadInput
      newMemory = map tanhBound rawMemory
      newLayer = NtmLayer newController newMemory newReadHead newWriteHead newReadHeadOutput
    in (newLayer, networkOutput)

  export
  forward : (Floating ty, Fractional ty, Neg ty, Num ty, Ord ty) => {i, o : Nat} -> {hs : List Nat} -> Network i hs o ty -> Vector i ty -> (Network i hs o ty, Vector o ty)
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
    g = sig (sum gVec)
    gamma = 1 + 4 * sig (sum gammaVec)
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
    eraseVector = map sig rawErase
    addVector = map (\x => 2 * sig (2 * x) - 1) rawAdd
    (newReadHead, _) = forwardReadHeadVar memory readHead readHeadInput
    newWriteHead = MkWriteHead newReadHead
    newMemoryMatrix = writeOpVar newWriteHead.readHead.addressingWeights memory eraseVector addVector
  in (newWriteHead, newMemoryMatrix)


----------------------------------------------------------------------
-- Variable-specialized Forward Pass (C-backed matvec/dot)
----------------------------------------------------------------------

mutual
  export
  applyLayerVar : {i, o : Nat} -> Layer i o Variable -> Vector i Variable -> (Layer i o Variable, Vector o Variable)
  applyLayerVar layer@(LinearLayer weights bias wBuf) xs =
    if i * o <= 4
      then applyLayer layer xs
      else case wBuf of
        Just wb => (layer, matrixVectorMultiplyVarBuf {m=o, n=i} wb xs + bias)
        Nothing => (layer, matrixVectorMultiplyVar {m=o, n=i} weights xs + bias)
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
  applyLayerVar {i} {o} (LstmLayer inputWeights recurrentWeights bias hiddenState cellState iwBuf rwBuf) xs =
    if i * o <= 4
      then applyLayer (LstmLayer inputWeights recurrentWeights bias hiddenState cellState iwBuf rwBuf) xs
      else
        let gateSize : Nat
            gateSize = 4 * o
            mulIW : Vector gateSize Variable
            mulIW = maybe (matrixVectorMultiplyVar {m=gateSize, n=i} inputWeights xs)
                          (\wb => matrixVectorMultiplyVarBuf {m=gateSize, n=i} wb xs) iwBuf
            mulRW : Vector gateSize Variable
            mulRW = maybe (matrixVectorMultiplyVar {m=gateSize, n=o} recurrentWeights hiddenState)
                          (\wb => matrixVectorMultiplyVarBuf {m=gateSize, n=o} wb hiddenState) rwBuf
            combined = mulIW + mulRW + bias
            gates = lstmSplitGates {o} combined
            iGate = fst gates
            fGate = fst (snd gates)
            gGate = fst (snd (snd gates))
            oGate = snd (snd (snd gates))
            newCell = map sig fGate * cellState + map sig iGate * map tanhBound gGate
            newHidden = map sig oGate * map tanhBound newCell
            updatedLayer = LstmLayer inputWeights recurrentWeights bias newHidden newCell iwBuf rwBuf
        in (updatedLayer, newHidden)
  applyLayerVar layer@(ActivationLayer _ f) xs = (layer, map f xs)
  applyLayerVar layer@(NormalizationLayer "softmax" _) xs = (layer, softmaxVar xs)
  applyLayerVar layer@(NormalizationLayer "logSoftmax" _) xs = (layer, logSoftmaxVar xs)
  applyLayerVar layer@(NormalizationLayer _ f) xs = (layer, f xs)
  applyLayerVar {i} (NtmLayer {n} {hs} controller memory readHead writeHead readHeadOutput) inp =
    let
      (newController, rawControllerOutput) = forwardVar controller (readHeadOutput ++ inp)
      controllerOutput = map (clampVar (-20.0) 20.0) rawControllerOutput
      (readHeadInput, controllerOutput') = Tensor.splitAt (ReadHeadInputWidth n i) controllerOutput
      (writeHeadInput, networkOutput) = Tensor.splitAt (WriteHeadInputWidth n i) controllerOutput'
      (newReadHead, newReadHeadOutput) = forwardReadHeadVar memory readHead readHeadInput
      (newWriteHead, rawMemory) = forwardWriteHeadVar memory writeHead writeHeadInput
      newMemory = map tanhBound rawMemory
      newLayer = NtmLayer newController newMemory newReadHead newWriteHead newReadHeadOutput
    in (newLayer, networkOutput)

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


----------------------------------------------------------------------
-- Evaluation Functions
----------------------------------------------------------------------

evaluateSingleDataPoint : (Floating ty, Fractional ty, Neg ty, Num ty, Ord ty) => {i, o : Nat} -> {hs : List Nat} -> Network i hs o ty -> DataPoint i o ty -> Vector o ty
evaluateSingleDataPoint model = snd . (forward model) . x

export
evaluate : (Floating ty, Fractional ty, Neg ty, Num ty, Ord ty) => {i, o : Nat} -> {hs : List Nat} -> Network i hs o ty -> Vect n (DataPoint i o ty) -> Vect n (Vector o ty)
evaluate model = map (evaluateSingleDataPoint model)

forwardNext : (Floating ty, Fractional ty, Neg ty, Num ty, Ord ty) => {i, o : Nat} -> {hs : List Nat} -> (Network i hs o ty, Vect n (Vector o ty)) -> Vector i ty -> (Network i hs o ty, Vect (S n) (Vector o ty))
forwardNext (nn, outputs) inp =
  let (updatedModel, newOutput) = forward nn inp
  in (updatedModel, snoc outputs newOutput)

forwardMany : (Floating ty, Fractional ty, Neg ty, Num ty, Ord ty) => {i, o : Nat} -> {hs : List Nat} -> Network i hs o ty -> Vect n (Vector i ty) -> (Network i hs o ty, Vect n (Vector o ty))
forwardMany network xs = foldlD (\k => (Network i hs o ty, Vect k (Vector o ty))) forwardNext (network, []) xs

export
calculateLoss : (Floating ty, Fractional ty, Neg ty, Num ty, Ord ty) => {i, o, n : Nat} -> {hs : List Nat} -> LossFunction ty -> Network i hs o ty -> Vect n (DataPoint i o ty) -> ty
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

recur : (Floating ty, Fractional ty, Neg ty, Num ty, Ord ty) => {i, o : Nat} -> {hs : List Nat} -> (Network i hs o ty, List (Vector o ty)) -> Vector i ty -> (Network i hs o ty, List (Vector o ty))
recur (m, os) i =
  let (updatedModel, output) = forward m i
  in (updatedModel, snoc os output)

export
forwardRecurrent : (Floating ty, Fractional ty, Neg ty, Num ty, Ord ty) => {i, o : Nat} -> {hs : List Nat} -> Network i hs o ty -> List (Vector i ty) -> (Network i hs o ty, List (Vector o ty))
forwardRecurrent model = foldl recur (model, [])

evaluateSingleRecurrentDataPoint : (Floating ty, Fractional ty, Neg ty, Num ty, Ord ty) => {i, o : Nat} -> {hs : List Nat} -> Network i hs o ty -> RecurrentDataPoint i o ty -> List (Vector o ty)
evaluateSingleRecurrentDataPoint model dataPoints = snd $ (forwardRecurrent model) dataPoints.xs

export
evaluateRecurrent : (Floating ty, Fractional ty, Neg ty, Num ty, Ord ty) => {i, o : Nat} -> {hs : List Nat} -> Network i hs o ty -> Vect n (RecurrentDataPoint i o ty) -> Vect n (List (Vector o ty))
evaluateRecurrent model dataPoints = map (evaluateSingleRecurrentDataPoint model) dataPoints

export
calculateLossRecurrent : (Floating ty, Fractional ty, Neg ty, Num ty, Ord ty) => {i, o, n : Nat} -> {hs : List Nat} -> LossFunction ty -> Network i hs o ty -> Vect n (RecurrentDataPoint i o ty) -> ty
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
forwardTwoPhase : (Floating ty, Fractional ty, Neg ty, Num ty, Ord ty) =>
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
  pure $ LinearLayer weights zeros Nothing

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
  inputWeights <- traverse (\_ => map fromDouble (initFn i o)) (the (Matrix (4 * o) i ty) zeros)
  recurrentWeights <- traverse (\_ => map fromDouble (initFn o o)) (the (Matrix (4 * o) o ty) zeros)
  let bias = setForgetBias {o} (the (Vector (4 * o) ty) zeros)
  pure $ LstmLayer inputWeights recurrentWeights bias zeros zeros Nothing Nothing

export
lstmLayer : {i, o : Nat} -> (Num ty, FromDouble ty) => IO (Layer i o ty)
lstmLayer = lstmLayerWith (xavier uniform)

export
ntmLayer : {n, w : Nat} -> {hs : List Nat} -> (FromDouble ty, Num ty) =>
           Network (NtmInputWidth w) hs (NtmOutputWidth n w) ty -> IO (Layer w w ty)
ntmLayer controller = do
  let memory = the (Matrix n w ty) (pure (fromDouble 1.0e-6))
  let readHead = the (ReadHead n ty) initReadHead
  let writeHead = the (WriteHead n ty) initWriteHead
  let readHeadOutput = zeros
  pure $ NtmLayer controller memory readHead writeHead readHeadOutput

||| Extract the cell state from an LSTM layer (for NTM head FC input).
export
extractCellState : Layer i o ty -> Maybe (Vector o ty)
extractCellState (LstmLayer _ _ _ _ cell _ _) = Just cell
extractCellState _ = Nothing

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
      (LinearLayer weights bias _) =>
        let
          namedWeights = zipWith (np "weight") enumerate weights
          namedBias = zipWith (np "bias") enumerate bias
        in if i * o <= 4
          then LinearLayer namedWeights namedBias Nothing
          else let (VTensor namedRows) = namedWeights
                   wBuf = prim__weightBufAlloc (cast (o * i))
                   wBuf' = initWeightBuf wBuf 0 namedRows
               in LinearLayer namedWeights namedBias (Just wBuf')
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
      (LstmLayer inputWeights recurrentWeights bias hiddenState cellState _ _) =>
        let
          gateSize : Nat
          gateSize = 4 * o
          namedInputWeights = zipWith (np "inputWeight") enumerate inputWeights
          namedRecurrentWeights = zipWith (np "recurrentWeight") enumerate recurrentWeights
          namedBias = zipWith (np "bias") enumerate bias
        in if i * o <= 4
          then LstmLayer namedInputWeights namedRecurrentWeights namedBias hiddenState cellState Nothing Nothing
          else let (VTensor iwRows) = namedInputWeights
                   (VTensor rwRows) = namedRecurrentWeights
                   iwBuf = prim__weightBufAlloc (cast (gateSize * i))
                   iwBuf' = initWeightBuf iwBuf 0 iwRows
                   rwBuf = prim__weightBufAlloc (cast (gateSize * o))
                   rwBuf' = initWeightBuf rwBuf 0 rwRows
               in LstmLayer namedInputWeights namedRecurrentWeights namedBias hiddenState cellState (Just iwBuf') (Just rwBuf')
      (NtmLayer controller memory readHead writeHead readHeadOutput) =>
        let namedMemory = zipWith (np "mem") enumerate memory
            namedReadHead = { addressingWeights $= zipWith (np "rAddr") enumerate } readHead
            namedWriteHead = { readHead.addressingWeights $= zipWith (np "wAddr") enumerate } writeHead
            namedReadOut = zipWith (np "rOut") enumerate readHeadOutput
        in NtmLayer (nameNetworkParams (prefx ++ "_ctrl") controller)
                    namedMemory namedReadHead namedWriteHead namedReadOut
      _ => layer

  export
  nameNetworkParams : {i, o : Nat} -> {hs : List Nat} -> String -> Network i hs o Variable -> Network i hs o Variable
  nameNetworkParams prefx (OutputLayer layer) = OutputLayer (nameParams prefx layer)
  nameNetworkParams prefx (layer ~> rest) = nameParams prefx layer ~> nameNetworkParams prefx rest


----------------------------------------------------------------------
-- Automatic Parameter Naming
----------------------------------------------------------------------

layerPrefix : Layer i o ty -> String
layerPrefix (LinearLayer _ _ _) = "ll"
layerPrefix (RnnLayer _ _ _ _ _ _) = "rnn"
layerPrefix (LstmLayer _ _ _ _ _ _ _) = "lstm"
layerPrefix (NtmLayer _ _ _ _ _) = "ntm"
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
              (NtmLayer controller memory readHead writeHead readHeadOutput) =>
                let np = nameParam . (fullName ++ "_" ++)
                    namedMemory = zipWith (np "mem") enumerate memory
                    namedReadHead = { addressingWeights $= zipWith (np "rAddr") enumerate } readHead
                    namedWriteHead = { readHead.addressingWeights $= zipWith (np "wAddr") enumerate } writeHead
                    namedReadOut = zipWith (np "rOut") enumerate readHeadOutput
                    (_, controller') = autoNameNetwork (fullName ++ "_") empty controller
                in (counts', NtmLayer controller' namedMemory namedReadHead namedWriteHead namedReadOut)
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
  syncLayerBuffers (LinearLayer (VTensor wRows) bias (Just wb)) =
    let wb' = syncWeightBuf wb 0 wRows
    in LinearLayer (VTensor wRows) bias (Just wb')
  syncLayerBuffers (RnnLayer (VTensor iwRows) (VTensor rwRows) bias po (Just iwb) (Just rwb)) =
    let iwb' = syncWeightBuf iwb 0 iwRows
        rwb' = syncWeightBuf rwb 0 rwRows
    in RnnLayer (VTensor iwRows) (VTensor rwRows) bias po (Just iwb') (Just rwb')
  syncLayerBuffers (LstmLayer (VTensor iwRows) (VTensor rwRows) bias hs cs (Just iwb) (Just rwb)) =
    let iwb' = syncWeightBuf iwb 0 iwRows
        rwb' = syncWeightBuf rwb 0 rwRows
    in LstmLayer (VTensor iwRows) (VTensor rwRows) bias hs cs (Just iwb') (Just rwb')
  syncLayerBuffers (NtmLayer controller mem rh wh ro) =
    let rh' = { addressingWeights $= projectWeights } rh
        wh' = { readHead.addressingWeights $= projectWeights } wh
    in NtmLayer (syncNetworkBuffers controller) mem rh' wh' ro
  syncLayerBuffers l = l

  export
  syncNetworkBuffers : {i, o : Nat} -> {hs : List Nat} -> Network i hs o Variable -> Network i hs o Variable
  syncNetworkBuffers (OutputLayer layer) = OutputLayer (syncLayerBuffers layer)
  syncNetworkBuffers (layer ~> rest) = syncLayerBuffers layer ~> syncNetworkBuffers rest
