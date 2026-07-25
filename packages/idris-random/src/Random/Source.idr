||| `Source` — where a run's raw draws come from: a generator, or a recording.
|||
||| Deterministic replay is a first-class use of a PRNG, not a testing
||| afterthought. Reproducing a reported bug, stepping two implementations
||| through the same trajectory, or re-running an experiment on a machine whose
||| libc differs all want the same thing: substitute the exact numbers a
||| previous run drew, without the consuming code knowing.
|||
||| A `Source` is threaded like a seed — every draw returns the advanced source
||| beside its value — so consumers stay pure. Code written against `Source`
||| works unchanged under either arm.
module Random.Source

-- `import public`: `Seeded` takes a `SplitMix.Seed`, so a consumer that has
-- only this module in scope cannot even write `Seeded 42` — the literal has no
-- `Num` to resolve against until the alias is transparent.
import public Random.SplitMix

%default total

||| Where draws come from.
|||
||| @Seeded    a SplitMix64 generator; the ordinary case
||| @Recorded  raw uniforms in [0, 1) to hand back in order
public export
data Source
  = Seeded SplitMix.Seed
  | Recorded (List Double)

public export
Eq Source where
  (Seeded a)   == (Seeded b)   = a == b
  (Recorded a) == (Recorded b) = a == b
  _            == _            = False

public export
Show Source where
  show (Seeded s)   = "Seeded " ++ show s
  show (Recorded d) = "Recorded " ++ show (length d) ++ " draw(s)"

||| A uniform in [0, 1), and the advanced source.
|||
||| Named `next` rather than `nextDouble` because this module re-exports
||| `Random.SplitMix`, which has a `nextDouble` of its own over a bare `Seed`;
||| two of that name in scope would be ambiguous at every call site.
|||
||| A `Recorded` source that has run out returns 0.0 rather than falling back
||| to sampling. Silently resuming generation is the worse failure: it would
||| turn a replay into a fresh run partway through and report the two as one.
||| Callers that must not tolerate it should check `exhausted` first.
export
next : Source -> (Double, Source)
next (Seeded s)           = let (d, s') = SplitMix.nextDouble s in (d, Seeded s')
next (Recorded [])        = (0.0, Recorded [])
next (Recorded (x :: xs)) = (x, Recorded xs)

||| Whether a `Recorded` source has no draws left. Always `False` for `Seeded`,
||| which never runs out.
export
exhausted : Source -> Bool
exhausted (Seeded _)    = False
exhausted (Recorded []) = True
exhausted (Recorded _)  = False

||| Take `n` draws, returning them in order beside the advanced source.
export
take : Source -> (n : Nat) -> (List Double, Source)
take s Z     = ([], s)
take s (S k) =
  let (d, s')   = Source.next s
      (ds, s'') = Source.take s' k
  in (d :: ds, s'')
