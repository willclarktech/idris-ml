||| Small shared data-generation helpers for the examples. Once the legacy
||| `DataPoint`/`RecurrentDataPoint`/`TwoPhaseDataPoint` surface was retired,
||| every task-specific producer here became dead (the migrated examples
||| generate their own `Dataset`/`DataStream`/tuple-shaped data inline). The
||| one helper still shared across examples is `randomInt`.
module Generate

import Compat.Random

||| Random integer in [lo, hi] inclusive.
export
randomInt : (lo, hi : Nat) -> IO Nat
randomInt lo hi = do
  n <- randomRIO (cast {to=Int32} (natToInteger lo), cast {to=Int32} (natToInteger hi))
  pure (fromInteger (cast {to=Integer} n))
