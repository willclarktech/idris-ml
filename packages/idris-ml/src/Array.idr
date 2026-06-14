module Array

import Data.Fin
import Data.Vect

import Compat.Random
import Floating

----------------------------------------------------------------------
-- Core Type and Aliases
----------------------------------------------------------------------

public export
data Array : Vect rank Nat -> Type -> Type where
  SArray : ty -> Array [] ty
  VArray : Vect dim (Array dims ty) -> Array (dim :: dims) ty

public export
0 Scalar : Type -> Type
Scalar = Array []

public export
0 Vector : Nat -> Type -> Type
Vector elems = Array [elems]

public export
0 Matrix : Nat -> Nat -> Type -> Type
Matrix rows columns = Array [rows, columns]

export
shapeOf : {dims : Vect rank Nat} -> Array dims ty -> Vect rank Nat
shapeOf {dims = ds} _ = ds

export
length : {dim : Nat} -> Vector dim ty -> Nat
length {dim} _ = dim

export
fromList : (xs : List ty) -> Vector (length xs) ty
fromList xs = VArray $ map SArray $ Data.Vect.fromList xs

----------------------------------------------------------------------
-- Instances
----------------------------------------------------------------------

public export
implementation Show ty => Show (Array dims ty) where
  show (SArray x) = show x
  show (VArray v) = show v

public export
implementation Eq ty => Eq (Array dims ty) where
  (SArray x) == (SArray y) = x == y
  (VArray v1) == (VArray v2) = v1 == v2

public export
implementation Ord ty => Ord (Array [] ty) where
  (SArray x) > (SArray y) = x > y
  (SArray x) >= (SArray y) = x >= y
  (SArray x) < (SArray y) = x < y
  (SArray x) <= (SArray y) = x <= y

public export
implementation Functor (Array dims) where
  map f (SArray x) = SArray (f x)
  map f (VArray xs) = VArray (map (map f) xs)

export
replicate : {dims : Vect rank Nat} -> ty -> Array dims ty
replicate {dims = []} x = SArray x
replicate {dims = dim :: dims} x = VArray $ replicate dim (replicate x)

public export
implementation {dims : Vect rank Nat} -> Applicative (Array dims) where
  pure = replicate
  (SArray f) <*> (SArray x) = SArray (f x)
  (VArray fs) <*> (VArray xs) = VArray (zipWith (<*>) fs xs)

export
zeros : Num ty => {dims : Vect rank Nat} -> Array dims ty
zeros = pure 0

export
ones : Num ty => {dims : Vect rank Nat} -> Array dims ty
ones = pure 1

----------------------------------------------------------------------
-- Random
----------------------------------------------------------------------

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
implementation {dims : Vect rank Nat} -> Random ty => Random (Array dims ty) where
  randomIO {dims = []} = map pure randomIO
  randomIO {dims = Z :: ds} = pure $ VArray []
  randomIO {dims = (S k) :: ds} = do
    x <- randomIO
    xs <- randomIO
    pure $ VArray (x :: xs)
  randomRIO {dims = []} (SArray lo, SArray hi) = map pure (randomRIO (lo, hi))
  randomRIO {dims = Z :: ds} _ = pure $ VArray []
  randomRIO {dims = (S k) :: ds} (VArray (lo :: los), VArray (hi :: his)) = do
    x <- randomRIO (lo, hi)
    xs <- randomRIO (los, his)
    pure $ VArray (x :: xs)

export
indices : {dims : Vect rank Nat} -> (startIndex : Nat) -> Array dims Nat
indices {dims = []} startIndex = SArray startIndex
indices {dims = (d :: ds)} startIndex = VArray $ map (\i => indices (startIndex + ((finToNat i) * (product ds)))) (Data.Vect.allFins d)

export
enumerate : {dims : Vect rank Nat} -> Array dims Nat
enumerate = indices 0

export
generate : {dims : Vect rank Nat} -> (Nat -> ty) -> Array dims ty
generate f = map f (indices 0)

----------------------------------------------------------------------
-- Foldable and Zippable
----------------------------------------------------------------------

public export
implementation {dims : Vect rank Nat} -> Foldable (Array dims) where
  foldr f acc (SArray x) = f x acc
  foldr _ acc (VArray []) = acc
  foldr f acc (VArray (x :: xs)) = foldr f (foldr f acc x) (VArray xs)

  null {dims = dims} _ = any (== 0) dims

public export
implementation {dims : Vect rank Nat} -> Traversable (Array dims) where
  traverse f (SArray x) = map SArray (f x)
  traverse f (VArray xs) = map VArray (traverse (traverse f) xs)

public export
implementation Zippable (Array dims) where
  zipWith f (SArray x) (SArray y) = SArray (f x y)
  zipWith f (VArray xs) (VArray ys) = VArray $ zipWith (zipWith f) xs ys

  unzipWith f (SArray x) =
    let (l, r) = f x
    in (SArray l, SArray r)
  unzipWith f (VArray xs) =
    let (ls, rs) = unzipWith (unzipWith f) xs
    in (VArray ls, VArray rs)

  zipWith3 f (SArray x) (SArray y) (SArray z) = SArray (f x y z)
  zipWith3 f (VArray xs) (VArray ys) (VArray zs) = VArray $ zipWith3 (zipWith3 f) xs ys zs

  unzipWith3 f (SArray x) =
    let (l, m, r) = f x
    in (SArray l, SArray m, SArray r)
  unzipWith3 f (VArray xs) =
    let (ls, ms, rs) = unzipWith3 (unzipWith3 f) xs
    in (VArray ls, VArray ms, VArray rs)

----------------------------------------------------------------------
-- Arithmetic
----------------------------------------------------------------------

||| Note that multiplication is elementwise
public export
implementation {dims : Vect rank Nat} -> Num ty => Num (Array dims ty) where
  fromInteger = pure . fromInteger
  (*) = zipWith (*)
  (+) = zipWith (+)

public export
implementation {dims : Vect rank Nat} -> FromDouble ty => FromDouble (Array dims ty) where
  fromDouble = pure . fromDouble

public export
implementation {dims : Vect rank Nat} -> Neg ty => Neg (Array dims ty) where
  (-) = zipWith (-)
  negate = map negate

public export
implementation {dims : Vect rank Nat} -> Abs ty => Abs (Array dims ty) where
  abs = map abs

public export
implementation {dims : Vect rank Nat} -> Fractional ty => Fractional (Array dims ty) where
  (/) = zipWith (/)

public export
implementation {dims : Vect rank Nat} -> Integral ty => Integral (Array dims ty) where
  div = zipWith div
  mod = zipWith mod

public export
implementation {dims : Vect rank Nat} -> Floating ty => Floating (Array dims ty) where
  exp = map exp
  log = map log
  pow = zipWith pow
  sqrt = map sqrt

----------------------------------------------------------------------
-- Structural Operations
----------------------------------------------------------------------

export
complement : (Neg ty) => Array dims ty -> Array dims ty
complement = map (1-)

export
head : Array (1 + dim :: dims) ty -> Array dims ty
head (VArray (x :: xs)) = x

export
tail : Array (1 + dim :: dims) ty -> Array (dim :: dims) ty
tail (VArray (x :: xs)) = VArray xs

export
index : Fin dim -> Array (dim :: dims) ty -> Array dims ty
index i (VArray xs) = Data.Vect.index i xs

export
transpose : {columns : Nat} -> Matrix rows columns ty -> Matrix columns rows ty
transpose (VArray []) = VArray $ replicate columns $ VArray []
transpose (VArray vec) = VArray $ map (\i => VArray (map (index i) vec)) range

export
unsqueeze : {rank : Nat} -> {dims : Vect rank Nat} -> (dim : Fin (S rank)) -> Array dims ty -> Array (insertAt dim 1 dims) ty
unsqueeze FZ x = VArray [x]
unsqueeze (FS y) (VArray xs) = VArray $ map (unsqueeze y) xs

export
(++) : Array (a :: dims) ty -> Array (b :: dims) ty -> Array ((a + b) :: dims) ty
(VArray xs) ++ (VArray ys) = VArray $ xs ++ ys

export
splitAt : (n : Nat) -> (xs : Array ((n + m) :: dims) ty) -> (Array (n :: dims) ty, Array (m :: dims) ty)
splitAt Z xs = (VArray [], xs)
splitAt (S k) (VArray (x :: xs)) with (splitAt k {m} xs)
  splitAt (S k) (VArray (x :: xs)) | (tk, dr) = (VArray (x :: tk), VArray dr)

-- TODO: Do concatentation properly: "All tensors must either have the same shape (except in the concatenating dimension) or be empty."

export
concat : Vect n (Array (dim :: dims) ty) -> Array ((n * dim) :: dims) ty
concat [] = VArray []
concat (x :: xs) = x ++ concat xs

export
concat' : Array (n :: dim :: dims) ty -> Array ((n * dim) :: dims) ty
concat' (VArray xs) = concat xs

export
concatAlong : {rank : Nat} -> {dims: Vect rank Nat} -> (fRank : Fin rank) -> Array dims ty -> Array dims ty -> Array (replaceAt fRank (2 * index fRank dims) dims) ty
concatAlong {dims = d :: ds} FZ x y = concat [x, y]
concatAlong {dims = (d :: ds)} (FS z) (VArray x) (VArray y) = VArray $ map (uncurry (concatAlong z)) (zip x y)

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
flatten : {dims : Vect rank Nat} -> Array dims ty -> Array [product dims] ty
flatten {dims = []} (SArray x) = VArray [SArray x]
flatten {dims = (d :: ds)} (VArray xs) =
  let mapped = map (flatten {dims = ds}) xs
  in rewrite productCons d ds in concat mapped

-- Adapted from kSplits in idris2 main but not v0.6.0
chunks : {chunkSize : Nat} -> (nChunks : Nat) -> Vect (nChunks * chunkSize) ty -> Vect nChunks (Vect chunkSize ty)
chunks 0 xs = []
chunks {chunkSize} (S n) xs =
  let (hs, ts) = splitAt chunkSize xs
  in hs :: chunks {chunkSize} n ts

export
unflatten : {dims : Vect n Nat} -> Array [product dims] ty -> Array dims ty
unflatten {dims = []} (VArray [SArray x]) = SArray x
unflatten {dims = (d :: ds)} (VArray xs) =
  let
    xs' = rewrite sym (productCons d ds) in xs
    cs = chunks {chunkSize = product ds} d xs'
    tensorChunks = map ((unflatten {dims = ds}) . VArray) cs
  in VArray tensorChunks

export
reshape : {dims1 : Vect m Nat} -> {dims2 : Vect n Nat} -> {auto p : product dims2 = product dims1} -> Array dims1 ty -> Array dims2 ty
reshape {dims1} {dims2} t =
  let flattened = rewrite p in flatten t
  in unflatten flattened
