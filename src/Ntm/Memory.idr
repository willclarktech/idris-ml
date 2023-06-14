module Ntm.Memory

import Data.Vect

import Floating
import Math
import Tensor
import Util


public export
record ReadHead n ty where
  constructor MkReadHead
  blending : ty
  sharpening : ty
  addressingWeights : Vector n ty

export
readHead : (Num ty) => {n: Nat} -> ty -> ty -> ReadHead n ty
readHead blending sharpening = MkReadHead blending sharpening zeros

getContentAddress : (Floating ty, Fractional ty) => {n, w : Nat} -> Matrix n w ty -> Vector w ty -> Vector n ty
getContentAddress (VTensor memory) keyVector = softmax $ VTensor $ map (STensor . (cosineSimilarity keyVector)) memory

interpolate : (Neg ty, Num ty) => ty -> Vector n ty -> Vector n ty -> Vector n ty
interpolate g = zipWith (\c, l => (c * g) + (l * (1 - g)))

-- TODO: Make a simpler version of this
cycleForward : {n : Nat} -> (i : Fin n) -> Vect n ty -> Vect n ty
cycleForward {n = Z} _ _ = []
cycleForward {n = (S k)} i xs =
  let indices = map (i+) (Util.allFins (S k))
  in permute xs indices

shift : (Floating ty, Fractional ty) => {n : Nat} -> Vector n ty -> Vector n ty -> Vector n ty
shift addressingWeights shiftVector =
  let
      (VTensor probabilities) = softmax shiftVector
      shifted = map (\i => cycleForward i probabilities) (allFins n)
   in VTensor $ map (STensor . (dotProduct addressingWeights)) (map VTensor shifted)

||| n is the number of entries
||| w is the width of each entry
forwardReadHead : (Floating ty, Fractional ty, Neg ty) => {n, w : Nat} -> Matrix n w ty -> ReadHead n ty -> Vector (w + n) ty -> (ReadHead n ty, Vector n ty)
forwardReadHead memory rh inp =
  let
    (keyVector, shiftVector) = splitAt w inp
    contentWeights = getContentAddress memory keyVector
    interpolated = interpolate rh.blending rh.addressingWeights contentWeights
    shifted = shift interpolated shiftVector
    -- focused
  in ?res

public export
record WriteHead n ty where
  constructor MkWriteHead
  readHead : ReadHead n ty

export
writeHead : (Num ty) => {n: Nat} -> ty -> ty -> WriteHead n ty
writeHead blending sharpening =
  let rh = readHead blending sharpening
  in MkWriteHead rh
