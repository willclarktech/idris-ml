||| Throughout this file:
||| n is the number of entries in the memory matrix
||| w is the width of each entry
module Memory

import Data.Fin
import Data.Vect

import Floating
import Math
import Tensor


----------------------------------------------------------------------
-- Shift Kernel Size
----------------------------------------------------------------------

||| Decouples shift kernel size from number of memory slots.
||| 3 elements: {shift_left, stay, shift_right}
public export
ShiftKernelSize : Nat
ShiftKernelSize = 3


----------------------------------------------------------------------
-- Read Head
----------------------------------------------------------------------

public export
record ReadHead n ty where
  constructor MkReadHead
  addressingWeights : Vector n ty

export
initReadHead : (Num ty) => {n : Nat} -> ReadHead n ty
initReadHead {n = Z} = MkReadHead (VTensor [])
initReadHead {n = S k} = MkReadHead $ VTensor $ replaceAt FZ (STensor 1) (replicate (S k) (STensor 0))

public export
Functor (ReadHead n) where
  map f (MkReadHead addressingWeights) = MkReadHead (map f addressingWeights)

sig : (Num ty, Neg ty, Fractional ty, Floating ty) => ty -> ty
sig x = 1 / (1 + exp (-x))

softplus : (Num ty, Floating ty) => ty -> ty
softplus x = log (1 + exp x)

getContentAddress : (Floating ty, Fractional ty, Ord ty) => {n, w : Nat} -> ty -> Matrix n w ty -> Vector w ty -> Vector n ty
getContentAddress beta (VTensor memory) keyVector = softmax $ map (* beta) $ VTensor $ map (STensor . (cosineSimilarity keyVector)) memory

interpolate : (Neg ty, Num ty) => ty -> Vector n ty -> Vector n ty -> Vector n ty
interpolate g = zipWith (\c, l => (c * g) + (l * (1 - g)))

-- TODO: Make a simpler version of this
export
cycleForward : {n : Nat} -> (i : Fin n) -> Vect n ty -> Vect n ty
cycleForward {n = Z} _ _ = []
cycleForward {n = (S k)} i xs =
  let indices = map (i+) (Data.Vect.allFins (S k))
  in Data.Vect.permute xs indices

shift : (Floating ty, Fractional ty) => {n : Nat} -> Vector n ty -> Vector ShiftKernelSize ty -> Vector n ty
shift {n = Z} aw _ = aw
shift {n = S Z} aw _ = aw
shift {n = S (S k)} aw kernel =
  let (VTensor [STensor sl, STensor ss, STensor sr]) = softmax kernel
      (VTensor ws) = aw
      fwdV = VTensor (cycleForward 1 ws)
      bwdV = VTensor (cycleForward Fin.last ws)
  in map (sl *) fwdV + map (ss *) aw + map (sr *) bwdV

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

||| Input is key vector (w) + shift vector (ShiftKernelSize) + params (3: beta, g, gamma)
export
forwardReadHead : (Floating ty, Fractional ty, Neg ty, Ord ty) => {n, w : Nat} -> Matrix n w ty -> ReadHead n ty -> Vector ((w + ShiftKernelSize) + 3) ty -> (ReadHead n ty, Vector w ty)
forwardReadHead memory rh inp =
  let
    (mainInput, params) = splitAt (w + ShiftKernelSize) inp
    (keyVector, shiftVector) = splitAt w mainInput
    (betaVec, params') = splitAt 1 params
    (gVec, gammaVec) = splitAt 1 params'
    beta = softplus (sum betaVec)
    g = sig (sum gVec)
    gamma = 1 + 4 * sig (sum gammaVec)
    contentWeights = getContentAddress beta memory keyVector
    interpolated = interpolate g rh.addressingWeights contentWeights
    shifted = shift interpolated shiftVector
    focused = focus gamma shifted
    newReadHead = { addressingWeights := focused } rh
    output = readOp newReadHead memory
  in (newReadHead, output)

----------------------------------------------------------------------
-- Write Head
----------------------------------------------------------------------

public export
record WriteHead n ty where
  constructor MkWriteHead
  readHead : ReadHead n ty

export
initWriteHead : (Num ty) => {n: Nat} -> WriteHead n ty
initWriteHead = MkWriteHead initReadHead

public export
Functor (WriteHead n) where
  map f (MkWriteHead rh) = MkWriteHead (map f rh)

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

||| Input is Read head input ((w + ShiftKernelSize) + 3) + erase vector (w) + add vector (w)
export
forwardWriteHead : (Floating ty, Fractional ty, Neg ty, Ord ty) => {n, w : Nat} -> Matrix n w ty -> WriteHead n ty -> Vector ((w + ShiftKernelSize) + 3 + w + w) ty -> (WriteHead n ty, Matrix n w ty)
forwardWriteHead memory (MkWriteHead readHead) inp =
  let
    inp' = rewrite plusAssociative ((w + ShiftKernelSize) + 3) w w in inp
    (readHeadInput, remainingInput) = Tensor.splitAt ((w + ShiftKernelSize) + 3) inp'
    (rawErase, rawAdd) = splitAt w remainingInput
    eraseVector = map sig rawErase
    addVector = map (\x => 2 * sig (2 * x) - 1) rawAdd
    (newReadHead, _) = forwardReadHead memory readHead readHeadInput
    newWriteHead = MkWriteHead newReadHead
    newMemoryMatrix = writeOp newWriteHead memory eraseVector addVector
  in (newWriteHead, newMemoryMatrix)
