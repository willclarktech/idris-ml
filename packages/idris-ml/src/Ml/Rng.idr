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

||| The ordinary run: draws from the process-global generator, so a program
||| that threads `liveRng` behaves exactly as one calling `Ml.Compat.Random`
||| and `Ml.Sampler` directly.
export
liveRng : IO Rng
liveRng = pure $ MkRng
  (randomRIO (0.0, 1.0))
  normalSample
  (\ps => (\u => categoricalSample ps u) <$> randomRIO (0.0, 1.0))

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
               (\_ => popNat cRef)
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
