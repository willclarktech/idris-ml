||| Uniform-sample replay buffer for off-policy RL (DQN, SAC).
|||
||| A fixed-capacity ring buffer backed by `Data.IOArray`. Push overwrites
||| the oldest entry once full. `sampleN` draws `n` transitions uniformly
||| at random (with replacement) using `Compat.Random.randomRIO` — so this
||| module honours the zero-`unsafePerformIO` invariant via explicit IO.
|||
||| Actions are stored as `Vect actDim Double`: for discrete envs, wrap the
||| action index as a 1-element vector; for continuous envs, store the raw
||| action directly.
module Ml.RL.ReplayBuffer

import Data.IOArray
import Data.IORef
import Data.Vect

import Ml.Compat.Random

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

-- Draw one uniform random index in [0, n).
randomIdx : Int -> IO Int
randomIdx n = do
  r <- randomRIO (the Double 0.0, 1.0)
  let scaled : Int
      scaled = cast (r * cast n)
  -- clamp to [0, n-1] defensively against r==1.0
  pure (if scaled >= n then n - 1 else scaled)

||| Sample `n` transitions uniformly at random (with replacement).
||| Returns `Nothing` if the buffer is empty.
export
sampleN : (n : Nat) -> ReplayBuffer obsDim actDim ->
          IO (Maybe (Vect n (Transition obsDim actDim)))
sampleN Z _       = pure (Just [])
sampleN (S k) buf = do
  sz <- readIORef buf.size
  if sz <= 0
    then pure Nothing
    else do
      idx  <- randomIdx sz
      mt   <- readArray buf.storage idx
      rest <- sampleN k buf
      case (mt, rest) of
        (Just t, Just rs) => pure (Just (t :: rs))
        _                 => pure Nothing
