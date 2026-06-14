||| Parameter-name derivation. `Init` is a small state monad over `IO`
||| carrying a scope path + per-(scope,kind) counters; it controls only the
||| *string* a layer passes to the C parameter registry (`param` /
||| `primParamRegister`) — the registry itself is unchanged.
|||
||| A layer smart constructor runs inside `Init`, asks for its own module
||| name with `freshChild "linear"` (auto-numbered within the current
||| scope), and registers its leaves as `<name>.weight` / `<name>.bias`.
||| `scoped "actor"` nests a namespace; `named "embed"` pins the next
||| child's name verbatim (the checkpoint-pinning escape hatch). This
||| replaces flat string prefixes (`"actor_ll0"`) + the substring matching
||| they required.
module Nn.Init

import Data.List

%default total

||| Mutable derivation state threaded through an `Init` computation.
public export
record InitState where
  constructor MkInitState
  ||| Current scope segments, outermost-first (joined with ".").
  path        : List String
  ||| Next index per "<scope>.<kind>" key, for positional auto-numbering.
  counters    : List (String, Nat)
  ||| If set, the next `freshChild` uses this name verbatim (set by `named`).
  pendingName : Maybe String

||| A name-deriving computation over `IO`.
public export
record Init a where
  constructor MkInit
  unInit : InitState -> IO (a, InitState)

public export
Functor Init where
  map f (MkInit g) = MkInit $ \s => do
    (a, s') <- g s
    pure (f a, s')

public export
Applicative Init where
  pure x = MkInit $ \s => pure (x, s)
  (MkInit f) <*> (MkInit x) = MkInit $ \s => do
    (g, s1) <- f s
    (a, s2) <- x s1
    pure (g a, s2)

public export
Monad Init where
  (MkInit m) >>= k = MkInit $ \s => do
    (a, s1) <- m s
    unInit (k a) s1

public export
HasIO Init where
  liftIO io = MkInit $ \s => do
    a <- io
    pure (a, s)

-- Join the scope path + a leaf into a dotted name (PyTorch state-dict style).
qualify : List String -> String -> String
qualify [] leaf = leaf
qualify path leaf = concat (intersperse "." path) ++ "." ++ leaf

setCounter : String -> Nat -> List (String, Nat) -> List (String, Nat)
setCounter key v xs = (key, v) :: filter ((/= key) . fst) xs

||| Run a namespaced sub-computation: `body` sees an extra path segment;
||| the segment is popped afterwards (counters accumulated inside persist).
export
scoped : String -> Init a -> Init a
scoped seg (MkInit body) = MkInit $ \s => do
  (a, s') <- body ({ path := s.path ++ [seg] } s)
  pure (a, { path := s.path } s')

||| Pin the *next* `freshChild`'s name verbatim (no positional number) —
||| the escape hatch for matching a foreign checkpoint's exact key.
export
named : String -> Init a -> Init a
named nm (MkInit body) = MkInit $ \s => do
  (a, s') <- body ({ pendingName := Just nm } s)
  pure (a, { pendingName := s.pendingName } s')

||| Derive a fresh child module name within the current scope: a pinned
||| name from a wrapping `named`, else `<scope>.<kind>_<n>` with `n`
||| auto-incrementing per (scope, kind).
export
freshChild : String -> Init String
freshChild kind = MkInit $ \s => case s.pendingName of
  Just nm => pure (qualify s.path nm, { pendingName := Nothing } s)
  Nothing => do
    let key = qualify s.path kind
    let n   = maybe 0 id (lookup key s.counters)
    let nm  = qualify s.path (kind ++ "_" ++ show n)
    pure (nm, { counters := setCounter key (S n) s.counters } s)

||| Run a derivation from the empty scope.
export
runInit : Init a -> IO a
runInit (MkInit f) = do
  (a, _) <- f (MkInitState [] [] Nothing)
  pure a
