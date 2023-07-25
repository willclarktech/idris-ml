module Tensor

import Data.Vect
import Data.Fin
import System.Random

import Floating
import Util


public export
data Tensor : Vect rank Nat -> Type -> Type where
  STensor : ty -> Tensor [] ty
  VTensor : Vect dim (Tensor dims ty) -> Tensor (dim :: dims) ty

Scalar = Tensor []

Vector elems = Tensor [elems]

Matrix rows columns = Tensor [rows, columns]

export
shapeOf : {dims : Vect rank Nat} -> Tensor dims ty -> Vect rank Nat
shapeOf {dims = ds} _ = ds

export
length : {dim : Nat} -> Vector dim ty -> Nat
length {dim} _ = dim

export
fromList : (xs : List ty) -> Vector (length xs) ty
fromList xs = VTensor $ map STensor $ Data.Vect.fromList xs

public export
implementation Show ty => Show (Tensor dims ty) where
  show (STensor x) = show x
  show (VTensor v) = show v

public export
implementation Eq ty => Eq (Tensor dims ty) where
  (STensor x) == (STensor y) = x == y
  (VTensor v1) == (VTensor v2) = v1 == v2

public export
implementation Functor (Tensor dims) where
  map f (STensor x) = STensor (f x)
  map f (VTensor xs) = VTensor (map (map f) xs)

export
replicate : {dims : Vect rank Nat} -> ty -> Tensor dims ty
replicate {dims = []} x = STensor x
replicate {dims = dim :: dims} x = VTensor $ replicate dim (replicate x)

public export
implementation {dims : Vect rank Nat} -> Applicative (Tensor dims) where
  pure = replicate
  (STensor f) <*> (STensor x) = STensor (f x)
  (VTensor fs) <*> (VTensor xs) = VTensor (zipWith (<*>) fs xs)

export
zeros : Num ty => {dims : Vect rank Nat} -> Tensor dims ty
zeros = pure 0

export
ones : Num ty => {dims : Vect rank Nat} -> Tensor dims ty
ones = pure 1

implementation {n : Nat} -> Random ty => Random (Vect n ty) where
  randomIO {n = Z} = pure []
  randomIO {n = S k} = do
    x <- randomIO
    xs <- randomIO
    pure $ x :: xs
  randomRIO {n = Z} _ = pure []
  randomRIO {n = S k} (lo::los, hi::his) = do
    x <- randomRIO (lo, hi)
    xs <- randomRIO (los, his)
    pure $ x :: xs

public export
implementation {dims : Vect rank Nat} -> Random ty => Random (Tensor dims ty) where
  randomIO {dims = []} = map pure randomIO
  randomIO {dims = Z :: ds} = pure $ VTensor []
  randomIO {dims = (S k) :: ds} = do
    x <- randomIO
    xs <- randomIO
    pure $ VTensor (x :: xs)
  randomRIO {dims = []} (STensor lo, STensor hi) = map pure (randomRIO (lo, hi))
  randomRIO {dims = Z :: ds} _ = pure $ VTensor []
  randomRIO {dims = (S k) :: ds} (VTensor (lo :: los), VTensor (hi :: his)) = do
    x <- randomRIO (lo, hi)
    xs <- randomRIO (los, his)
    pure $ VTensor (x :: xs)

export
indices : {dims : Vect rank Nat} -> (startIndex : Nat) -> Tensor dims Nat
indices {dims = []} startIndex = STensor startIndex
indices {dims = (d :: ds)} startIndex = VTensor $ map (\i => indices (startIndex + ((finToNat i) * (product ds)))) (allFins d)

export
enumerate : {dims : Vect rank Nat} -> Tensor dims Nat
enumerate = indices 0

export
generate : {dims : Vect rank Nat} -> (Nat -> ty) -> Tensor dims ty
generate f = map f (indices 0)

public export
implementation {dims : Vect rank Nat} -> Foldable (Tensor dims) where
  foldr f acc (STensor x) = f x acc
  foldr _ acc (VTensor []) = acc
  foldr f acc (VTensor (x :: xs)) = foldr f (foldr f acc x) (VTensor xs)

  null {dims = dims} _ = any (== 0) dims

public export
implementation Zippable (Tensor dims) where
  zipWith f (STensor x) (STensor y) = STensor (f x y)
  zipWith f (VTensor xs) (VTensor ys) = VTensor $ zipWith (zipWith f) xs ys

  unzipWith f (STensor x) =
    let (l, r) = f x
    in (STensor l, STensor r)
  unzipWith f (VTensor xs) =
    let (ls, rs) = unzipWith (unzipWith f) xs
    in (VTensor ls, VTensor rs)

  zipWith3 f (STensor x) (STensor y) (STensor z) = STensor (f x y z)
  zipWith3 f (VTensor xs) (VTensor ys) (VTensor zs) = VTensor $ zipWith3 (zipWith3 f) xs ys zs

  unzipWith3 f (STensor x) =
    let (l, m, r) = f x
    in (STensor l, STensor m, STensor r)
  unzipWith3 f (VTensor xs) =
    let (ls, ms, rs) = unzipWith3 (unzipWith3 f) xs
    in (VTensor ls, VTensor ms, VTensor rs)

||| Note that multiplication is elementwise
public export
implementation {dims : Vect rank Nat} -> Num ty => Num (Tensor dims ty) where
  fromInteger = pure . fromInteger
  (*) = zipWith (*)
  (+) = zipWith (+)

public export
implementation {dims : Vect rank Nat} -> FromDouble ty => FromDouble (Tensor dims ty) where
  fromDouble = pure . fromDouble

public export
implementation {dims : Vect rank Nat} -> Neg ty => Neg (Tensor dims ty) where
  (-) = zipWith (-)
  negate = map negate

public export
implementation {dims : Vect rank Nat} -> Abs ty => Abs (Tensor dims ty) where
  abs = map abs

public export
implementation {dims : Vect rank Nat} -> Fractional ty => Fractional (Tensor dims ty) where
  (/) = zipWith (/)

public export
implementation {dims : Vect rank Nat} -> Integral ty => Integral (Tensor dims ty) where
  div = zipWith div
  mod = zipWith mod

public export
implementation {dims : Vect rank Nat} -> Floating ty => Floating (Tensor dims ty) where
  exp = map exp
  log = map log
  pow = zipWith pow
  sqrt = map sqrt

export
complement : (Neg ty) => Tensor dims ty -> Tensor dims ty
complement = map (1-)

export
head : Tensor (1 + dim :: dims) ty -> Tensor dims ty
head (VTensor (x :: xs)) = x

export
tail : Tensor (1 + dim :: dims) ty -> Tensor (dim :: dims) ty
tail (VTensor (x :: xs)) = VTensor xs

export
index : Fin dim -> Tensor (dim :: dims) ty -> Tensor dims ty
index i (VTensor xs) = Data.Vect.index i xs

export
transpose : {columns : Nat} -> Matrix rows columns ty -> Matrix columns rows ty
transpose (VTensor []) = VTensor $ replicate columns $ VTensor []
transpose (VTensor vec) = VTensor $ map (\i => VTensor (map (index i) vec)) range

export
unsqueeze : {rank : Nat} -> {dims : Vect rank Nat} -> (dim : Fin (S rank)) -> Tensor dims ty -> Tensor (insertAt dim 1 dims) ty
unsqueeze FZ x = VTensor [x]
unsqueeze (FS y) (VTensor xs) = VTensor $ map (unsqueeze y) xs

export
(++) : Tensor (a :: dims) ty -> Tensor (b :: dims) ty -> Tensor ((a + b) :: dims) ty
(VTensor xs) ++ (VTensor ys) = VTensor $ xs ++ ys

export
splitAt : (n : Nat) -> (xs : Tensor ((n + m) :: dims) ty) -> (Tensor (n :: dims) ty, Tensor (m :: dims) ty)
splitAt Z xs = (VTensor [], xs)
splitAt (S k) (VTensor (x :: xs)) with (splitAt k {m} xs)
  splitAt (S k) (VTensor (x :: xs)) | (tk, dr) = (VTensor (x :: tk), VTensor dr)

-- TODO: Do concatentation properly: "All tensors must either have the same shape (except in the concatenating dimension) or be empty."

export
concat : Vect n (Tensor (dim :: dims) ty) -> Tensor ((n * dim) :: dims) ty
concat [] = VTensor []
concat (x :: xs) = x ++ concat xs

export
concat' : Tensor (n :: dim :: dims) ty -> Tensor ((n * dim) :: dims) ty
concat' (VTensor xs) = concat xs

export
concatAlong : {rank : Nat} -> {dims: Vect rank Nat} -> (fRank : Fin rank) -> Tensor dims ty -> Tensor dims ty -> Tensor (replaceAt fRank (2 * index fRank dims) dims) ty
concatAlong {dims = d :: ds} FZ x y = concat [x, y]
concatAlong {dims = (d :: ds)} (FS z) (VTensor x) (VTensor y) = VTensor $ map (uncurry (concatAlong z)) (zip x y)

multFoldAssociative : (d: Nat) -> (x : Nat) -> (xs : Vect n Nat) -> foldl (*) (d * x) xs = d * (foldl (*) x xs)
multFoldAssociative d x [] = Refl
multFoldAssociative d x (y :: ys) =
  rewrite sym (multAssociative d x y) in
  rewrite multFoldAssociative d (x * y) ys in
    Refl

productCons : (d : Nat) -> (ds : Vect n Nat) -> product (d :: ds) = d * product ds
productCons d [] =
  rewrite plusZeroRightNeutral d in
  rewrite multOneRightNeutral d in
    Refl
productCons d (x :: xs) =
  rewrite plusZeroRightNeutral d in
  rewrite plusZeroRightNeutral x in
  rewrite multFoldAssociative d x xs in
    Refl

export
flatten : {dims : Vect rank Nat} -> Tensor dims ty -> Tensor [product dims] ty
flatten {dims = []} (STensor x) = VTensor [STensor x]
flatten {dims = (d :: ds)} (VTensor xs) =
  let mapped = map (flatten {dims = ds}) xs
  in rewrite productCons d ds in concat mapped

-- Adapted from kSplits in idris2 main but not v0.6.0
chunks : {chunkSize : Nat} -> (nChunks : Nat) -> Vect (nChunks * chunkSize) ty -> Vect nChunks (Vect chunkSize ty)
chunks 0 xs = []
chunks {chunkSize} (S n) xs =
  let (hs, ts) = splitAt chunkSize xs
  in hs :: chunks {chunkSize} n ts

export
unflatten : {dims : Vect n Nat} -> Tensor [product dims] ty -> Tensor dims ty
unflatten {dims = []} (VTensor [STensor x]) = STensor x
unflatten {dims = (d :: ds)} (VTensor xs) =
  let
    xs' = rewrite sym (productCons d ds) in xs
    cs = chunks {chunkSize = product ds} d xs'
    tensorChunks = map ((unflatten {dims = ds}) . VTensor) cs
  in VTensor tensorChunks

export
reshape : {dims1 : Vect m Nat} -> {dims2 : Vect n Nat} -> {auto p : product dims2 = product dims1} -> Tensor dims1 ty -> Tensor dims2 ty
reshape {dims1} {dims2} t =
  let flattened = rewrite p in flatten t
  in unflatten flattened
