||| Throughout this file:
||| n is the number of entries in the memory matrix
||| w is the width of each entry
module Memory

import Data.Vect

import Floating
import Math
import Tensor


public export
record ReadHead n ty where
  constructor MkReadHead
  blending : ty
  sharpening : ty
  addressingWeights : Vector n ty

export
initReadHead : (Num ty) => {n: Nat} -> ty -> ty -> ReadHead n ty
initReadHead blending sharpening = MkReadHead blending sharpening zeros

getContentAddress : (Floating ty, Fractional ty, Ord ty) => {n, w : Nat} -> Matrix n w ty -> Vector w ty -> Vector n ty
getContentAddress (VTensor memory) keyVector = softmax $ VTensor $ map (STensor . (cosineSimilarity keyVector)) memory

interpolate : (Neg ty, Num ty) => ty -> Vector n ty -> Vector n ty -> Vector n ty
interpolate g = zipWith (\c, l => (c * g) + (l * (1 - g)))

-- TODO: Make a simpler version of this
cycleForward : {n : Nat} -> (i : Fin n) -> Vect n ty -> Vect n ty
cycleForward {n = Z} _ _ = []
cycleForward {n = (S k)} i xs =
  let indices = map (i+) (Data.Vect.allFins (S k))
  in Data.Vect.permute xs indices

shift : (Floating ty, Fractional ty) => {n : Nat} -> Vector n ty -> Vector n ty -> Vector n ty
shift addressingWeights shiftVector =
  let
      (VTensor probabilities) = softmax shiftVector
      shifted = map (\i => cycleForward i probabilities) (Data.Vect.allFins n)
   in VTensor $ map (STensor . (dotProduct addressingWeights)) (map VTensor shifted)

focus : (Floating ty, Fractional ty, Num ty) => {n : Nat} -> ty -> Vector n ty -> Vector n ty
focus gamma addressingWeights =
  let
    raised = map (^ gamma) addressingWeights
    sigma = sum raised
  in map (/ sigma) raised

readOp : (Num ty) => {n, w : Nat} -> ReadHead n ty -> Matrix n w ty -> Vector w ty
readOp rh (VTensor memoryRows) =
  let
    (VTensor addressingWeights) = rh.addressingWeights
    weightedRows = zipWith (\(STensor weight), row => map (*weight) row) addressingWeights memoryRows
  in sum weightedRows

||| Input is key vector (w) + shift vector (n)
export
forwardReadHead : (Floating ty, Fractional ty, Neg ty, Ord ty) => {n, w : Nat} -> Matrix n w ty -> ReadHead n ty -> Vector (w + n) ty -> (ReadHead n ty, Vector w ty)
forwardReadHead memory rh inp =
  let
    (keyVector, shiftVector) = splitAt w inp
    contentWeights = getContentAddress memory keyVector
    interpolated = interpolate rh.blending rh.addressingWeights contentWeights
    shifted = shift interpolated shiftVector
    focused = focus rh.sharpening shifted
    newReadHead = { addressingWeights := focused } rh
    output = readOp newReadHead memory
  in (newReadHead, output)

public export
record WriteHead n ty where
  constructor MkWriteHead
  readHead : ReadHead n ty

export
initWriteHead : (Num ty) => {n: Nat} -> ty -> ty -> WriteHead n ty
initWriteHead blending sharpening =
  let rh = initReadHead blending sharpening
  in MkWriteHead rh

eraseMemory : (Neg ty, Num ty) => {n, w : Nat} -> Matrix n w ty -> Vector n ty -> Vector w ty -> Matrix n w ty
eraseMemory memory (VTensor addressVector) eraseVector =
  let complements = complement $ VTensor $ map (\(STensor weight) => map (* weight) eraseVector) addressVector
  in memory * complements

addMemory : (Num ty) => {n, w : Nat} -> Matrix n w ty -> Vector n ty -> Vector w ty -> Matrix n w ty
addMemory memory (VTensor addressVector) addVector =
  let weightedAddVectors = VTensor $ map (\(STensor weight) => map (* weight) addVector) addressVector
  in memory + weightedAddVectors

writeOp : (Neg ty) => {n, w : Nat} -> WriteHead n ty -> Matrix n w ty -> Vector w ty -> Vector w ty -> Matrix n w ty
writeOp (MkWriteHead rh) memory eraseVector addVector =
  let
    erased = eraseMemory memory rh.addressingWeights eraseVector
    newMemory = addMemory erased rh.addressingWeights addVector
  in newMemory

||| Input is Read head input (w + n) + erase vector (w) + add vector (w)
export
forwardWriteHead : (Floating ty, Fractional ty, Neg ty, Ord ty) => {n, w : Nat} -> Matrix n w ty -> WriteHead n ty -> Vector (w + n + w + w) ty -> (WriteHead n ty, Matrix n w ty)
forwardWriteHead memory (MkWriteHead readHead) inp =
  let
    inp' = rewrite plusAssociative (w + n) w w in inp
    (readHeadInput, remainingInput) = Tensor.splitAt (w + n) inp'
    (eraseVector, addVector) = splitAt w remainingInput
    (newReadHead, _) = forwardReadHead memory readHead readHeadInput
    newWriteHead = MkWriteHead newReadHead
    newMemoryMatrix = writeOp newWriteHead memory eraseVector addVector
  in (newWriteHead, newMemoryMatrix)
