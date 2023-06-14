module Util

import Data.Vect


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

-- Copied from https://github.com/idris-lang/Idris2/pull/2707/files#diff-ff81a71a1254f20ad8ec34869deb9ada6f744fefee2e584c03a3c32367ddb8f7R395-R405
export
foldlD : (0 accTy : Nat -> Type) ->
  (f : forall k. accTy k -> a -> accTy (S k)) ->
  (acc : accTy Z) ->
  (xs : Vect n a) ->
  accTy n
foldlD _ _ acc [] = acc
foldlD accTy f acc (x :: xs) = foldlD (accTy . S) f (acc `f` x) xs
