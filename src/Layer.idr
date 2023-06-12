module Layer

import Data.Vect

import Endofunctor
import Math
import Tensor


public export
data Layer : (inputSize : Nat) -> (outputSize : Nat) -> Type -> Type where
  LinearLayer : (matrix : Matrix outputSize inputSize ty) -> (vector : Vector outputSize ty) -> Layer inputSize outputSize ty
  ActivationLayer : (name : String) -> (f : ActivationFunction ty) -> Layer n n ty
  NormalizationLayer : (name : String) -> (f : NormalizationFunction ty) -> Layer n n ty

public export
implementation {inputSize : Nat} -> {outputSize : Nat} -> Show a => Show (Layer inputSize outputSize a) where
  show {inputSize} {outputSize} (LinearLayer _ _) = "Linear<" ++ show outputSize ++ "x" ++ show inputSize ++ ">"
  show (ActivationLayer name _) = "Activation<" ++ name ++ ">"
  show (NormalizationLayer name _) = "Normalization<" ++ name ++ ">"

public export
implementation Endofunctor (Layer i o) where
  emap f (LinearLayer w b) = LinearLayer (map f w) (map f b)
  emap _ l = l

export
applyLayer : Num ty => {i, o : Nat} -> Layer i o ty -> Vector i ty -> Vector o ty
applyLayer (LinearLayer weights bias) xs = matrixVectorMultiply {m=o, n=i} weights xs + bias
applyLayer (ActivationLayer _ f) xs = map f xs
applyLayer (NormalizationLayer _ f) xs = f xs

export
sigmoidLayer : (FromDouble ty, Neg ty, Fractional ty, Floating ty) => Layer n n ty
sigmoidLayer = ActivationLayer "sigmoid" sigmoid

export
softmaxLayer : (Fractional ty, Floating ty) => Layer n n ty
softmaxLayer = NormalizationLayer "softmax" softmax
