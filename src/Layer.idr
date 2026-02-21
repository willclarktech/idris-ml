module Layer

import Data.List
import Data.Vect
import Data.Zippable
import System.Random

import DataPoint
import Endofunctor
import Floating
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

||| Key vector + shift vector
public export
ReadHeadInputWidth : Nat -> Nat -> Nat
ReadHeadInputWidth n w = w + n

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
    LinearLayer : (weights : Matrix outputSize inputSize ty) -> (bias : Vector outputSize ty) -> Layer inputSize outputSize ty
    RnnLayer : (inputWeights : Matrix outputSize inputSize ty) -> (recurrentWeights : Matrix outputSize outputSize ty) -> (bias : Vector outputSize ty) -> (previousOutput : Vector outputSize ty) -> Layer inputSize outputSize ty
    ActivationLayer : (name : String) -> (f : ActivationFunction ty) -> Layer n n ty
    NormalizationLayer : (name : String) -> (f : NormalizationFunction ty) -> Layer n n ty
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
  show {inputSize} {outputSize} (LinearLayer _ _) = "Linear<" ++ show inputSize ++ ":" ++ show outputSize ++ ">"
  show {inputSize} {outputSize} (RnnLayer _ _ _ _) = "Rnn<" ++ show inputSize ++ ":" ++ show outputSize ++ ">"
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
    emap f (LinearLayer w b) = LinearLayer (map f w) (map f b)
    emap f (RnnLayer iw rw b po) = RnnLayer (map f iw) (map f rw) (map f b) (map f po)
    emap f (NtmLayer controller mem rh wh ro) =
      NtmLayer (emap f controller) (map f mem) (map f rh) (map f wh) (map f ro)
    emap _ l = l

  public export
  implementation Endofunctor (Network i hs o) where
    emap f (OutputLayer layer) = OutputLayer (emap f layer)
    emap f (layer ~> layers) = emap f layer ~> emap f layers


----------------------------------------------------------------------
-- Forward Pass
----------------------------------------------------------------------

mutual
  export
  applyLayer : (Floating ty, Fractional ty, Neg ty, Num ty, Ord ty) => {i, o : Nat} -> Layer i o ty -> Vector i ty -> (Layer i o ty, Vector o ty)
  applyLayer layer@(LinearLayer weights bias) xs = (layer, matrixVectorMultiply {m=o, n=i} weights xs + bias)
  applyLayer (RnnLayer inputWeights recurrentWeights bias previousOutput) xs =
    let
      output = matrixVectorMultiply inputWeights xs + matrixVectorMultiply recurrentWeights previousOutput + bias
      updatedLayer = RnnLayer inputWeights recurrentWeights bias output
    in (updatedLayer, output)
  applyLayer layer@(ActivationLayer _ f) xs = (layer, map f xs)
  applyLayer layer@(NormalizationLayer _ f) xs = (layer, f xs)
  applyLayer {i} (NtmLayer {n} {hs} controller memory readHead writeHead readHeadOutput) inp =
    let
      (newController, controllerOutput) = forward controller (readHeadOutput ++ inp)
      (readHeadInput, controllerOutput') = Tensor.splitAt (ReadHeadInputWidth n i) controllerOutput
      (writeHeadInput, networkOutput) = Tensor.splitAt (WriteHeadInputWidth n i) controllerOutput'
      (newReadHead, newReadHeadOutput) = forwardReadHead memory readHead readHeadInput
      (newWriteHead, newMemory) = forwardWriteHead memory writeHead writeHeadInput
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

forwardNextRecurrent : (Floating ty, Fractional ty, Neg ty, Num ty, Ord ty) => {i, o : Nat} -> {hs : List Nat} -> (Network i hs o ty, Vect n (List (Vector o ty))) -> List (Vector i ty) -> (Network i hs o ty, Vect (1 + n) (List (Vector o ty)))
forwardNextRecurrent (nn, outputs) inps =
  let (updatedModel, newOutput) = forwardRecurrent nn inps
  in (updatedModel, snoc outputs newOutput)

forwardManyRecurrent : (Floating ty, Fractional ty, Neg ty, Num ty, Ord ty) => {i, o : Nat} -> {hs : List Nat} -> Network i hs o ty -> Vect n (List (Vector i ty)) -> (Network i hs o ty, Vect n (List (Vector o ty)))
forwardManyRecurrent network xs = foldlD (\k => (Network i hs o ty, Vect k (List (Vector o ty)))) forwardNextRecurrent (network, []) xs

export
calculateLossRecurrent : (Floating ty, Fractional ty, Neg ty, Num ty, Ord ty) => {i, o, n : Nat} -> {hs : List Nat} -> LossFunction ty -> Network i hs o ty -> Vect n (RecurrentDataPoint i o ty) -> ty
calculateLossRecurrent lossFn model dataPoints =
  let
    xs = map xs dataPoints
    ys = map ys dataPoints
    (updatedNetwork, predictions) = forwardManyRecurrent model xs
    losses = zipWith (zipWith lossFn) predictions ys
  in mean . VTensor $ map (STensor . mean) losses


----------------------------------------------------------------------
-- Layer Constructors
----------------------------------------------------------------------

export
linearLayer : {i, o : Nat} -> (Random ty, FromDouble ty, Neg ty) => IO (Layer i o ty)
linearLayer = do
  weights <- randomRIO (-1.0, 1.0)
  bias <- randomRIO (-1.0, 1.0)
  pure $ LinearLayer weights bias

export
rnnLayer : {i, o : Nat} -> (Random ty, FromDouble ty, Neg ty) => IO (Layer i o ty)
rnnLayer = do
  inputWeights <- randomRIO (-1.0, 1.0)
  recurrentWeights <- randomRIO (-1.0, 1.0)
  bias <- randomRIO (-1.0, 1.0)
  pure $ RnnLayer inputWeights recurrentWeights bias zeros

export
ntmLayer : {n, w : Nat} -> {hs : List Nat} -> (Random ty, FromDouble ty, Neg ty, Num ty) =>
           Network (NtmInputWidth w) hs (NtmOutputWidth n w) ty -> IO (Layer w w ty)
ntmLayer controller = do
  memory <- randomRIO (-0.1, 0.1)
  let blending = 0.5
  let sharpening = 1.5
  let readHead = initReadHead blending sharpening
  let writeHead = initWriteHead blending sharpening
  let readHeadOutput = zeros
  pure $ NtmLayer controller memory readHead writeHead readHeadOutput

export
sigmoidLayer : (FromDouble ty, Neg ty, Fractional ty, Floating ty) => Layer n n ty
sigmoidLayer = ActivationLayer "sigmoid" sigmoid

export
softmaxLayer : (Fractional ty, Floating ty) => Layer n n ty
softmaxLayer = NormalizationLayer "softmax" softmax


----------------------------------------------------------------------
-- Parameter Naming
----------------------------------------------------------------------

mutual
  export
  nameParams : {i, o : Nat} -> String -> (Layer i o Variable) -> (Layer i o Variable)
  nameParams prefx layer =
    let np = nameParam . (prefx ++ "_" ++)
    in case layer of
      (LinearLayer weights bias) =>
        let
          namedWeights = zipWith (np "weight") enumerate weights
          namedBias = zipWith (np "bias") enumerate bias
        in LinearLayer namedWeights namedBias
      (RnnLayer inputWeights recurrentWeights bias previousOutput) =>
        let
          namedInputWeights = zipWith (np "inputWeight") enumerate inputWeights
          namedRecurrentWeights = zipWith (np "recurrentWeight") enumerate recurrentWeights
          namedBias = zipWith (np "bias") enumerate bias
        in RnnLayer namedInputWeights namedRecurrentWeights namedBias previousOutput
      (NtmLayer controller memory readHead writeHead readHeadOutput) =>
        NtmLayer (nameNetworkParams (prefx ++ "_ctrl") controller) memory readHead writeHead readHeadOutput
      _ => layer

  export
  nameNetworkParams : {i, o : Nat} -> {hs : List Nat} -> String -> Network i hs o Variable -> Network i hs o Variable
  nameNetworkParams prefx (OutputLayer layer) = OutputLayer (nameParams prefx layer)
  nameNetworkParams prefx (layer ~> rest) = nameParams prefx layer ~> nameNetworkParams prefx rest
