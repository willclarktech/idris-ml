-- | Portable random number generation (Chez + RefC compatible).
-- |
-- | Drop-in replacement for System.Random from contrib, using C FFI
-- | instead of Scheme-only FFI. All functions have the same names
-- | and signatures.

module Compat.Random

import Data.Fin
import Data.Vect
import Data.List

-- C standard library random functions
%foreign "C:srand,libc"
prim__srand : Int -> PrimIO ()

%foreign "C:rand,libc"
prim__rand : PrimIO Int

||| Seed the random number generator.
export
srand : HasIO io => Bits64 -> io ()
srand s = liftIO $ primIO (prim__srand (cast s))

||| Generate a random Int.
export
randomInt32 : HasIO io => io Int
randomInt32 = liftIO $ primIO prim__rand

public export
interface Random a where
  randomIO : HasIO io => io a
  randomRIO : HasIO io => (a, a) -> io a

export
Random Double where
  randomIO = do
    r <- randomInt32
    pure (cast (abs r) / 2147483647.0)
  randomRIO (lo, hi) = do
    r <- randomIO {a = Double}
    pure (lo + r * (hi - lo))

export
Random Int32 where
  randomIO = cast <$> randomInt32
  randomRIO (lo, hi) = do
    r <- randomInt32
    let range = cast {to=Int} (cast {to=Integer} hi - cast {to=Integer} lo + 1)
    pure (cast (cast {to=Int} lo + (r `mod` range)))

-- Note: Vect Random instance is defined in Tensor.idr (avoids orphan overlap)
