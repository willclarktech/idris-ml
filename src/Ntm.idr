||| Throughout this file:
||| n is the number of entries in the memory matrix
||| w is the width of each entry
module Ntm

import Data.Vect
import System.Random

import Floating
import Layer
import Memory
import Network
import Tensor
import Util
import Variable


||| Read head output + input
export
InputWidth : Nat -> Nat
InputWidth w = w + w

||| Key vector + shift vector
export
ReadHeadInputWidth : Nat -> Nat -> Nat
ReadHeadInputWidth n w = w + n

||| Read head input + erase vector + add vector
export
WriteHeadInputWidth : Nat -> Nat -> Nat
WriteHeadInputWidth n w = ReadHeadInputWidth n w + w + w

||| Read head input + Write head input + output
export
OutputWidth : Nat -> Nat -> Nat
OutputWidth n w = ReadHeadInputWidth n w + (WriteHeadInputWidth n w + w)

public export
record Ntm n w hs ty where
  constructor MkNtm
  controller : Network (InputWidth w) hs (OutputWidth n w) ty
  memory : Matrix n w ty
  readHead : ReadHead n ty
  writeHead : WriteHead n ty
  readHeadOutput : Vector w ty

public export
implementation {n, w : Nat} -> Show ty => Show (Ntm n w hs ty) where
  show ntm = "NTM<" ++ show n ++ "," ++ show w ++ ">" ++ show ntm.memory

export
initNtm : (FromDouble ty, Num ty) => {n, w : Nat} -> {hs : List Nat} -> Network (InputWidth w) hs (OutputWidth n w) ty -> Matrix n w ty -> Ntm n w hs ty
initNtm controller memory =
  let
    blending = 0.5
    sharpening = 1.5
    readHead = initReadHead blending sharpening
    writeHead = initWriteHead blending sharpening
    output = zeros
  in MkNtm controller memory readHead writeHead output

export
forwardNtm : (Floating ty, Fractional ty, Neg ty, Num ty) => {n, w : Nat} -> {hs : List Nat} -> Ntm n w hs ty -> Vector w ty -> (Ntm n w hs ty, Vector w ty)
forwardNtm ntm inp =
  let
    (newController, controllerOutput) = forward ntm.controller (ntm.readHeadOutput ++ inp)
    (readHeadInput, controllerOutput') = Tensor.splitAt (ReadHeadInputWidth n w) controllerOutput
    (writeHeadInput, networkOutput) = Tensor.splitAt (WriteHeadInputWidth n w) controllerOutput'
    (newReadHead, newReadHeadOutput) = forwardReadHead ntm.memory ntm.readHead readHeadInput
    (newWriteHead, newMemory) = forwardWriteHead ntm.memory ntm.writeHead writeHeadInput
    newNtm = MkNtm newController newMemory newReadHead newWriteHead newReadHeadOutput
  in (newNtm, networkOutput)
