-- | Portable random number generation (Chez + RefC compatible).
-- |
-- | Drop-in replacement for System.Random from contrib, using C FFI
-- | instead of Scheme-only FFI. All functions have the same names
-- | and signatures.

module Ml.Compat.Random

import Data.Fin
import Data.List
import Data.Vect

-- Backed by the repo's own SplitMix64 (`shared_utils.c`), not libc
-- `srand`/`rand`: the C standard leaves `rand`'s algorithm to the
-- implementation, so glibc and the libc macOS ships produce different streams
-- from the same seed and a run's parameter init would not reproduce across
-- the two CI legs. The C side's dropout mask seed draws from the same
-- generator, so one `srand` pins the whole run.
%foreign "C:idrisml_srand,libidrisml"
prim__srand : Bits64 -> PrimIO ()

%foreign "C:idrisml_rand,libidrisml"
prim__rand : PrimIO Int

||| Seed the random number generator.
export
srand : HasIO io => Bits64 -> io ()
srand s = liftIO $ primIO (prim__srand s)

||| Generate a random non-negative Int.
export
randomInt32 : HasIO io => io Int
randomInt32 = liftIO $ primIO prim__rand

public export
interface Random a where
  randomIO  : HasIO io => io a
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
  randomIO           = cast <$> randomInt32
  randomRIO (lo, hi) = do
    r <- randomInt32
    let range = cast {to=Int} (cast {to=Integer} hi - cast {to=Integer} lo + 1)
    pure (cast (cast {to=Int} lo + (r `mod` range)))

-- Note: Vect Random instance is defined in Array.idr (avoids orphan overlap)
