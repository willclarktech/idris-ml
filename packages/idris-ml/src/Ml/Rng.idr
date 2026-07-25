||| `Rng` — a program's source of stochastic choices, passed explicitly.
|||
||| Randomness elsewhere in this library is ambient: `Ml.Compat.Random` draws
||| from a process-global generator that `srand` seeds once. That suits
||| parameter init, which is reproduced by loading a checkpoint rather than by
||| re-drawing, but it leaves no way to *supply* the draws a run should make.
||| An `Rng` threaded as an ordinary argument gives that, in the shape
||| `Gym.Rng` already uses for environment resets.
|||
||| Two things it buys a user: a run can be replayed exactly from recorded
||| draws (bug reproduction, and comparing two implementations step for step),
||| and a caller can substitute a distribution without the callee knowing.
|||
||| `choice` sits beside the raw draws deliberately. A recorded categorical
||| decision cannot be replayed through `uniform`, because which outcome a
||| given draw selects is sampler-specific — this library's inverse CDF and
||| PyTorch's `multinomial` disagree. Overriding the decision itself needs no
||| agreement between the two.
module Ml.Rng

import Data.IORef
import Data.List
import Data.String
import System.File

import Random.Source

import Ml.Compat.Random
import Ml.Sampler

%default total

||| Where a run's stochastic choices come from. Each field is an action, so
||| reading it twice draws twice.
public export
record Rng where
  constructor MkRng
  ||| Uniform on [0, 1).
  uniform : IO Double
  ||| Standard normal, N(0, 1).
  normal : IO Double
  ||| An index sampled from a categorical, given its probabilities.
  choice : List Double -> IO Nat
  ||| Uniform integer on [lo, hi], both ends inclusive. A recorded run
  ||| replays these as decisions, on the same channel as `choice` — both are
  ||| discrete outcomes, and sharing the channel keeps a recording's
  ||| consumption order whole.
  natRange : (lo, hi : Nat) -> IO Nat

||| The ordinary run: draws from the process-global generator, so a program
||| that threads `liveRng` behaves exactly as one calling `Ml.Compat.Random`
||| and `Ml.Sampler` directly. `natRange` draws exactly as the examples'
||| shared `Generate.randomInt` always has (one `Int32` bounded draw).
export
liveRng : IO Rng
liveRng = pure $ MkRng
  (randomRIO (0.0, 1.0))
  normalSample
  (\ps => (\u => categoricalSample ps u) <$> randomRIO (0.0, 1.0))
  (\lo, hi => do
     n <- randomRIO (cast {to = Int32} (natToInteger lo), cast {to = Int32} (natToInteger hi))
     pure (fromInteger (cast {to = Integer} n)))

||| Replay recorded draws in order. Exhausting any channel is a bug in
||| whatever produced the recording, so it fails loudly rather than silently
||| falling back to sampling — a replay that quietly stops replaying would
||| compare two different runs and report them as one.
export
replayRng : (uniforms : List Double) -> (normals : List Double) ->
            (choices : List Nat) -> IO Rng
replayRng uniforms normals choices = do
  uRef <- newIORef uniforms
  nRef <- newIORef normals
  cRef <- newIORef choices
  pure $ MkRng (popDouble uRef "uniform") (popDouble nRef "normal")
               (\_ => popNat cRef) (\_, _ => popNat cRef)
  where
    exhausted : String -> a
    exhausted channel =
      assert_total $ idris_crash ("Ml.Rng.replayRng: ran out of recorded " ++ channel ++ " draws")

    popDouble : IORef (List Double) -> String -> IO Double
    popDouble ref channel = do
      xs <- readIORef ref
      case xs of
        []        => pure (exhausted channel)
        (v :: vs) => do writeIORef ref vs; pure v

    popNat : IORef (List Nat) -> IO Nat
    popNat ref = do
      xs <- readIORef ref
      case xs of
        []        => pure (exhausted "choice")
        (v :: vs) => do writeIORef ref vs; pure v

||| A run's complete stochastic input: the `Rng` for the program's own draws
||| and a `Source` for the environment's (env resets draw from the `Source`
||| threaded through `Gym.Env.reset`, which is pure and cannot share the
||| `Rng`'s IO channels).
public export
record Replay where
  constructor MkReplay
  rng       : Rng
  envSource : Source

||| The live counterpart of `loadReplay`: a fresh `Rng` on the process-global
||| generator and a `Seeded` env source drawn from it.
export
liveReplay : IO Replay
liveReplay = do
  envSeed <- randomInt32
  rng     <- liveRng
  pure (MkReplay rng (Seeded (cast envSeed)))

||| Load recorded draws from a replay file — one draw per line, in
||| consumption order per channel:
|||
|||     choice 1        -- a categorical decision (`Rng.choice`)
|||     uniform 0.5     -- a U[0,1) draw (`Rng.uniform`)
|||     normal -1.5     -- an N(0,1) draw (`Rng.normal`)
|||     env 0.25        -- a uniform for the environment `Source`
|||
||| Blank lines and `#` comment lines are ignored. Channels interleave
||| freely; only the order within a channel matters. The `Rng` channels fail
||| loudly when exhausted (`replayRng`); the env `Source` is `Recorded`, so
||| it draws 0.0 past the end of the recording — a replayed run is only
||| meaningful over the span the recording covers.
|||
||| Any unreadable file or unparseable line is a hard error: a replay that
||| silently dropped draws would compare two different runs and report them
||| as one.
covering
export
loadReplay : (path : String) -> IO Replay
loadReplay path = do
  Right text <- readFile path
    | Left err => pure (bad ("cannot read " ++ path ++ ": " ++ show err))
  let (us, ns, cs, es) = walk (lines text)
  rng <- replayRng us ns cs
  pure (MkReplay rng (Recorded es))
  where
    bad : String -> a
    bad msg = assert_total $ idris_crash ("Ml.Rng.loadReplay: " ++ msg)

    parseD : String -> Double
    parseD s = case parseDouble s of
                 Just d  => d
                 Nothing => bad ("not a double in " ++ path ++ ": " ++ s)

    parseN : String -> Nat
    parseN s = case parsePositive s of
                 Just n  => n
                 Nothing => bad ("not a nat in " ++ path ++ ": " ++ s)

    walk : List String -> (List Double, List Double, List Nat, List Double)
    walk []          = ([], [], [], [])
    walk (l :: rest) =
      let (us, ns, cs, es) = walk rest
      in case words l of
           []             => (us, ns, cs, es)
           ("#" :: _)     => (us, ns, cs, es)
           ["uniform", v] => (parseD v :: us, ns, cs, es)
           ["normal",  v] => (us, parseD v :: ns, cs, es)
           ["choice",  v] => (us, ns, parseN v :: cs, es)
           ["env",     v] => (us, ns, cs, parseD v :: es)
           (w :: _)       =>
             if isPrefixOf "#" w
               then (us, ns, cs, es)
               else bad ("unrecognized line in " ++ path ++ ": " ++ l)
