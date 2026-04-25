module Test.RL.ReplayBuffer

import Data.Vect
import Data.List

import Harness
import Compat.Random
import RL.ReplayBuffer


sampleTransition : Double -> Transition 2 1
sampleTransition k =
  MkTransition [k, k + 1.0] [k * 0.1] k [k + 10.0, k + 11.0] False


export
tests : List (IO Bool)
tests =
  [ do buf <- mkBuffer {obsDim=2, actDim=1} 10
       sz  <- bufferSize buf
       check "buffer starts empty" (sz == 0)

  , do buf <- mkBuffer {obsDim=2, actDim=1} 10
       push buf (sampleTransition 1.0)
       sz  <- bufferSize buf
       check "push increments size" (sz == 1)

  , do buf <- mkBuffer {obsDim=2, actDim=1} 10
       traverse_ (\k => push buf (sampleTransition (cast {to=Double} k))) [1..5]
       sz <- bufferSize buf
       check "size reflects multiple pushes" (sz == 5)

  , do buf <- mkBuffer {obsDim=2, actDim=1} 3
       traverse_ (\k => push buf (sampleTransition (cast {to=Double} k))) [1..10]
       sz <- bufferSize buf
       check "size clamps at capacity" (sz == 3)

  , do buf <- mkBuffer {obsDim=2, actDim=1} 10
       push buf (sampleTransition 7.0)
       mRes <- sampleN 1 buf
       case mRes of
         Just [t] => check "sample one returns the pushed transition" (t.reward == 7.0)
         _        => check "sample one returned unexpected shape" False

  , do buf <- mkBuffer {obsDim=2, actDim=1} 10
       mRes <- sampleN 1 buf
       check "sample on empty returns Nothing" (isNothing mRes)

  , do srand 123
       buf <- mkBuffer {obsDim=2, actDim=1} 100
       traverse_ (\k => push buf (sampleTransition (cast {to=Double} k))) [1..100]
       mRes <- sampleN 20 buf
       case mRes of
         Just ts =>
           -- All sampled rewards should fall inside [1.0, 100.0] (we pushed values 1..100)
           let allInRange = all (\t => t.reward >= 1.0 && t.reward <= 100.0) (toList ts)
           in check "sampled rewards are from pushed range" allInRange
         Nothing => check "sampleN failed on nonempty buffer" False

  , do srand 42
       buf <- mkBuffer {obsDim=2, actDim=1} 50
       traverse_ (\k => push buf (sampleTransition (cast {to=Double} k))) [1..50]
       mRes <- sampleN 500 buf
       case mRes of
         Just ts =>
           -- Uniform sampling should hit variety. Count distinct rewards; expect >= 15/50
           -- in 500 draws (extremely loose bound; collision probability is basically 0).
           let rewards = map (\t => t.reward) (toList ts)
               distinct = length (nub rewards)
           in check "uniform sample spread" (distinct >= 15)
         Nothing => check "sampleN failed" False

  , do buf <- mkBuffer {obsDim=2, actDim=1} 3
       traverse_ (\k => push buf (sampleTransition (cast {to=Double} k))) [1..3]
       -- capacity-sized ring: push 4th -> slot 0 overwritten with value 4
       push buf (sampleTransition 4.0)
       sz <- bufferSize buf
       -- Cursor is now at slot 1; sampling with enough draws should see value 4
       mRes <- sampleN 100 buf
       case mRes of
         Just ts =>
           let rewards = map (\t => t.reward) (toList ts)
           in check "ring overwrite visible in samples"
                (sz == 3 && elem 4.0 rewards && not (elem 1.0 rewards))
         Nothing => check "sampleN failed" False
  ]
