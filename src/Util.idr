||| Things I wish were provided by the base library
module Util

import Data.Vect


export
implementation Cast Bool Integer where
  cast True = 1
  cast False = 0

export
implementation Cast Bool Double where
  cast = fromInteger . cast

export
signum : Double -> Double
signum x = case compare x 0 of
  GT => 1.0
  EQ => 0.0
  LT => -1.0

export
mean : (Num ty, Fractional ty) => List ty -> ty
mean xs =
  let tot = fromInteger $ natToInteger $ length xs
  in sum xs / tot

-- Copied from https://github.com/idris-lang/Idris2/blob/26c5c4db03f361443b0581d3d0878adcbf42832a/libs/base/Data/Vect.idr#L120-L126
export
allFins : (n : Nat) -> Vect n (Fin n)
allFins 0 = []
allFins (S k) = FZ :: map FS (allFins k)

-- Copied from https://github.com/idris-lang/Idris2/blob/c6e476ed1a7811f05bc2174db45f5e50fa73ec24/libs/base/Data/Vect.idr#L915-L917
export
permute : (v : Vect len a) -> (p : Vect len (Fin len)) -> Vect len a
permute v p = (`index` v) <$> p

-- Copied from https://github.com/idris-lang/Idris2/pull/2707/files#diff-ff81a71a1254f20ad8ec34869deb9ada6f744fefee2e584c03a3c32367ddb8f7R395-R405
export
foldlD : (0 accTy : Nat -> Type) ->
  (f : forall k. accTy k -> a -> accTy (S k)) ->
  (acc : accTy Z) ->
  (xs : Vect n a) ->
  accTy n
foldlD _ _ acc [] = acc
foldlD accTy f acc (x :: xs) = foldlD (accTy . S) f (acc `f` x) xs
