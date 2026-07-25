# idris-random

Pseudorandom generation for Idris 2, with zero dependencies beyond `base`. Two generators, the
distributions built on them, and a source abstraction that lets a run be replayed from recorded
draws.

Everything is pure and threaded: each draw returns its value beside the advanced state, so callers
stay total and reproducible without carrying an effect.

```idris
import Random.Dist
import Random.Source

-- Ordinary use: a seeded generator.
let (x, s1) = uniform (Seeded 42) (-0.05) 0.05
    (z, s2) = normal s1
```

## Two generators, on purpose

`Random.SplitMix` implements SplitMix64 (Steele, Lea, Flood 2014): one word of state, a fixed
increment and a finalizing mix. It is cheap, and adequate wherever the consumer is not itself
statistically demanding — seeding another generator, a handful of draws per call, per-element
masks.

`Random.Xoshiro` implements xoshiro256 (Blackman, Vigna 2018): four words, a long period and good
equidistribution. Reach for it when the *stream* has to hold up rather than merely being cheap —
shuffling, sampling without replacement, anything where a whole permutation depends on generator
quality. Its four state words are seeded from SplitMix64, as the reference implementation
recommends, so a caller supplying a small or low-entropy seed still starts from well-spread state.

Both output scramblers ship, over one shared state update: `nextStarStar` returns
`rotl(s1 * 5, 7) * 9` and `nextPlusPlus` returns `rotl(s0 + s3, 23) + s0`. That one line is the
entire difference between xoshiro256** and xoshiro256++, which makes them trivial to conflate in
prose and impossible to substitute in practice — the streams diverge from the first draw. `next`
is the `**` scrambler, which is what `shuffle` uses.

`Random.Xoshiro.shuffle` is Fisher-Yates walking from the last index down, swapping each element
with a uniformly chosen index at or below it. The swap *order* is part of the contract, not an
implementation detail: a variant walking upward would be an equally valid shuffle and an entirely
different permutation.

## Replay

`Random.Source` is either a generator or a recording:

```idris
data Source
  = Seeded SplitMix.Seed
  | Recorded (List Double)
```

Code written against `Source` works unchanged under either arm, which makes deterministic replay a
property of the *caller* rather than something the consuming code has to know about. That is worth
having for reproducing a reported bug, stepping two implementations through the same trajectory,
or re-running an experiment on a machine whose libc differs.

Two details that matter in practice:

- A `Recorded` source that runs out returns `0.0` rather than falling back to generating. Silently
  resuming would turn a replay into a fresh run partway through and report the two as one. Check
  `exhausted` if that must be an error.
- `Random.Dist.uniformInverse` recovers the draw that would have produced a given value on
  `[lo, hi)`. `uniform` is affine, so this is exact — which lets a recording be built from observed
  *values* when they came from somewhere else entirely, such as another implementation's generator.

## Modules

| module | contents |
|---|---|
| `Random.SplitMix` | `Seed`, `next`, `expand` |
| `Random.Xoshiro` | `Gen`, `seed`, `next`, `boundedNat`, `shuffle` |
| `Random.Source` | `Source`, `nextDouble`, `exhausted`, `take` |
| `Random.Dist` | `uniform`, `uniformInverse`, `boundedNat`, `normal`, `normalWith`, `categorical` |

## Tests

```bash
make test-unit-random
```

Pure suite: both generators' reproducibility and distinctness, `Source`'s two arms and the
round-trip between them (record a seeded stream, replay it, get the same draws), and the
distributions' bounds, moments and boundary cases.

The C-versus-Idris differential tests — asserting these implementations agree bit for bit with the
copies in the idris-ml backends — live in the idris-ml suite instead, since they need the dylib and
this package must not depend on it.
