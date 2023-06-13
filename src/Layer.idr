module Layer

import Data.Vect
import System.Random

import Endofunctor
import Math
import Tensor
import Variable


public export
data Layer : (inputSize : Nat) -> (outputSize : Nat) -> Type -> Type where
  LinearLayer : (weights : Matrix outputSize inputSize ty) -> (bias : Vector outputSize ty) -> Layer inputSize outputSize ty
  RnnLayer : (inputWeights : Matrix outputSize inputSize ty) -> (recurrentWeights : Matrix outputSize outputSize ty) -> (bias : Vector outputSize ty) -> (previousOutput : Vector outputSize ty) -> Layer inputSize outputSize ty
  ActivationLayer : (name : String) -> (f : ActivationFunction ty) -> Layer n n ty
  NormalizationLayer : (name : String) -> (f : NormalizationFunction ty) -> Layer n n ty

public export
implementation {inputSize : Nat} -> {outputSize : Nat} -> Show a => Show (Layer inputSize outputSize a) where
  show {inputSize} {outputSize} (LinearLayer _ _) = "Linear<" ++ show inputSize ++ "->" ++ show outputSize ++ ">"
  show {inputSize} {outputSize} (RnnLayer _ _ _ _) = "Rnn<" ++ show inputSize ++ "->" ++ show outputSize ++ ">"
  show (ActivationLayer name _) = "Activation<" ++ name ++ ">"
  show (NormalizationLayer name _) = "Normalization<" ++ name ++ ">"

public export
implementation Endofunctor (Layer i o) where
  emap f (LinearLayer w b) = LinearLayer (map f w) (map f b)
  emap f (RnnLayer iw rw b po) = RnnLayer (map f iw) (map f rw) (map f b) (map f po)
  emap _ l = l

export
applyLayer : Num ty => {i, o : Nat} -> Layer i o ty -> Vector i ty -> (Layer i o ty, Vector o ty)
applyLayer layer@(LinearLayer weights bias) xs = (layer, matrixVectorMultiply {m=o, n=i} weights xs + bias)
applyLayer (RnnLayer inputWeights recurrentWeights bias previousOutput) xs =
  let
    output = matrixVectorMultiply inputWeights xs + matrixVectorMultiply recurrentWeights previousOutput + bias
    updatedLayer = RnnLayer inputWeights recurrentWeights bias output
  in (updatedLayer, output)
applyLayer layer@(ActivationLayer _ f) xs = (layer, map f xs)
applyLayer layer@(NormalizationLayer _ f) xs = (layer, f xs)

linearLayer : {i, o : Nat} -> (Random ty, FromDouble ty, Neg ty) => IO (Layer i o ty)
linearLayer = do
  weights <- randomRIO (-1.0, 1.0)
  bias <- randomRIO (-1.0, 1.0)
  pure $ LinearLayer weights bias

export
linearLayerWithNamedParams : {i, o : Nat} -> IO (Layer i o Variable)
linearLayerWithNamedParams = do
  layer <- linearLayer
  case layer of
    (LinearLayer weights bias) => do
      let namedWeights = zipWith (nameParam "weight") enumerate weights
      let namedBias = zipWith (nameParam "bias") enumerate bias
      pure $ LinearLayer namedWeights namedBias
    _ => pure layer

export
sigmoidLayer : (FromDouble ty, Neg ty, Fractional ty, Floating ty) => Layer n n ty
sigmoidLayer = ActivationLayer "sigmoid" sigmoid

export
softmaxLayer : (Fractional ty, Floating ty) => Layer n n ty
softmaxLayer = NormalizationLayer "softmax" softmax
