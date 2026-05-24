module Test.Vector

import Data.Vect
import Test.Harness
import Gym.Env
import Gym.Vector
import Gym.ClassicControl.CartPole


-- Explicit type witnesses keep the Env instance resolution unambiguous
-- (the interface has three parameters — state/action/obs — but resetAll
-- and friends only reveal state in their signatures).

cpVec : (n : Nat) -> VecEnv n CPState
cpVec n = resetAll {state=CPState} {action=Nat} {obs=Vect 4 Double} {n}


export
tests : List (IO Bool)
tests =
  [ check "resetAll creates n envs" $
      let v = cpVec 4
      in case v.envs of
           (first :: _) => first.cpX == 0.0

  , check "stepAll reward vector length" $
      let v = cpVec 3
      in case stepAll {state=CPState} {action=Nat} {obs=Vect 4 Double} v [1, 0, 1] of
           (_, results) => length results == 3

  , check "stepAutoReset produces n outputs" $
      let v = cpVec 2
      in case stepAutoReset {state=CPState} {action=Nat}
                            {obs=Vect 4 Double} v [1, 1] of
           (_, rewards, obs, outcomes) =>
             length rewards == 2 && length obs == 2 && length outcomes == 2

  , check "stepAutoReset resets terminated envs" $
      let v : VecEnv 1 CPState
          v = MkVecEnv [MkCP 3.0 0.0 0.0 0.0]
      in case stepAutoReset {state=CPState} {action=Nat}
                            {obs=Vect 4 Double} v [1] of
           (v', _, _, outcomes) =>
             case v'.envs of
               (s' :: _) =>
                 s'.cpX == 0.0 &&
                 case outcomes of
                   (o :: _) => o == Terminated
  ]
