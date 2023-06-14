module Floating


public export
interface Floating ty where
  exp : ty -> ty
  log : ty -> ty
  pow : ty -> ty -> ty

infixr 9 ^
export
(^) : Floating ty => ty -> ty -> ty
(^) = Floating.pow
