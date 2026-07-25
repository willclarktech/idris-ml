module Test.Source

import Random.Source
import Test.Harness

%default total

export
tests : List (IO Bool)
tests =
  [ let (a, _) = Source.take (Seeded 42) 8
        (b, _) = Source.take (Seeded 42) 8
    in check "seeded: same seed, same draws" (a == b)

  , let (xs, _) = Source.take (Seeded 42) 256
    in check "seeded: every draw in [0, 1)" (all (\d => d >= 0.0 && d < 1.0) xs)

  , let (xs, _) = Source.take (Seeded 123) 2000
        m       = sum xs / cast (length xs)
    in check "seeded: mean near 0.5" (abs (m - 0.5) < 0.03)

  , let recorded = [0.1, 0.2, 0.3]
        (xs, _)  = Source.take (Recorded recorded) 3
    in check "recorded: replays in order" (xs == recorded)

  , -- The property the whole replay story rests on: record a generator's own
    -- output, play it back, and the consumer cannot tell the difference.
    let (recorded, _) = Source.take (Seeded 2026) 32
        (replayed, _) = Source.take (Recorded recorded) 32
    in check "round-trip: recorded seeded draws replay identically"
             (replayed == recorded)

  , check "recorded: exhausted reports empty"
          (exhausted (Recorded []) && not (exhausted (Recorded [0.5])))

  , check "seeded: never exhausted" (not (exhausted (Seeded 0)))

  , -- Running out must not silently resume generating: that would turn a
    -- replay into a fresh run partway through and report the two as one.
    let (xs, s) = Source.take (Recorded [0.25]) 3
    in check "recorded: past the end yields 0.0, not fresh draws"
             (xs == [0.25, 0.0, 0.0] && exhausted s)
  ]
