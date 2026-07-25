||| Uniform-sample replay buffer for off-policy RL (DQN, SAC).
|||
||| A fixed-capacity ring buffer backed by `Data.IOArray`. Push overwrites
||| the oldest entry once full. `sampleN` draws `n` transitions uniformly
||| at random (with replacement) through the caller's `Ml.Rng.Rng`, so a
||| recorded run replays the drawn indices as decisions.
|||
||| Actions are stored as `Vect actDim Double`: for discrete envs, wrap the
||| action index as a 1-element vector; for continuous envs, store the raw
||| action directly.
module Ml.RL.ReplayBuffer

import Data.IOArray
import Data.IORef
import Data.Vect

import Ml.Rng

||| One recorded transition.
public export
record Transition (obsDim : Nat) (actDim : Nat) where
  constructor MkTransition
  obs     : Vect obsDim Double
  action  : Vect actDim Double
  reward  : Double
  nextObs : Vect obsDim Double
  done    : Bool

||| A uniform-sample ring buffer with capacity `capacity`.
public export
record ReplayBuffer (obsDim : Nat) (actDim : Nat) where
  constructor MkReplayBuffer
  capacity : Int
  cursor   : IORef Int
  size     : IORef Int
  storage  : IOArray (Transition obsDim actDim)

||| Create an empty buffer. obsDim/actDim are passed explicitly so they
||| can be inferred at instantiation (e.g. when mkBuffer is consumed into
||| a record typed `ReplayBuffer 4 1`).
export
mkBuffer : {obsDim, actDim : Nat} -> (capacity : Nat) ->
           IO (ReplayBuffer obsDim actDim)
mkBuffer capacity = do
  let cap = the Int (cast capacity)
  cur <- newIORef (the Int 0)
  sz  <- newIORef (the Int 0)
  arr <- newArray cap
  pure (MkReplayBuffer cap cur sz arr)

||| Number of transitions currently stored (0..capacity).
export
bufferSize : ReplayBuffer obsDim actDim -> IO Nat
bufferSize buf = do
  n <- readIORef buf.size
  pure (integerToNat (cast n))

||| Push a transition. When full, overwrites the oldest entry.
export
push : ReplayBuffer obsDim actDim -> Transition obsDim actDim -> IO ()
push buf t = do
  cur <- readIORef buf.cursor
  _   <- writeArray buf.storage cur t
  writeIORef buf.cursor ((cur + 1) `mod` buf.capacity)
  sz  <- readIORef buf.size
  if sz < buf.capacity
    then writeIORef buf.size (sz + 1)
    else pure ()

||| Sample `n` transitions uniformly at random (with replacement), each
||| index drawn through `rng.natRange` — a recorded run replays them as
||| decisions on the choice channel. Returns `Nothing` if the buffer is
||| empty.
export
sampleN : Rng -> (n : Nat) -> ReplayBuffer obsDim actDim ->
          IO (Maybe (Vect n (Transition obsDim actDim)))
sampleN _   Z     _   = pure (Just [])
sampleN rng (S k) buf = do
  sz <- readIORef buf.size
  if sz <= 0
    then pure Nothing
    else do
      idxN <- rng.natRange 0 (cast (sz - 1))
      mt   <- readArray buf.storage (cast (natToInteger idxN))
      rest <- sampleN rng k buf
      case (mt, rest) of
        (Just t, Just rs) => pure (Just (t :: rs))
        _                 => pure Nothing
