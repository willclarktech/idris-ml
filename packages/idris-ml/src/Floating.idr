module Floating

public export
interface Floating ty where
  exp  : ty -> ty
  log  : ty -> ty
  pow  : ty -> ty -> ty
  sqrt : ty -> ty

public export
implementation Floating Double where
  exp  = prim__doubleExp
  log  = prim__doubleLog
  pow  = prim__doublePow
  sqrt = prim__doubleSqrt

export infixr 9 ^
export
(^) : Floating ty => ty -> ty -> ty
(^) = Floating.pow
