module Endofunctor


public export
interface Endofunctor e where
  emap : (ty -> ty) -> e ty -> e ty

public export
implementation Functor f => Endofunctor f where
  emap = Prelude.map
