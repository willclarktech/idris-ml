||| `Env.reset` draws through a `Random.Source`, so an environment can be
||| started either from a seed or from recorded draws.
|||
||| Two things need pinning. The `Seeded` arm must produce byte-for-byte what
||| it produced when `reset` took a bare `Seed` — the widening is meant to be
||| invisible to every existing caller, and a silent shift would move every
||| recorded convergence number. The `Recorded` arm must map draws through each
||| environment's own start distribution, which is what makes a trajectory
||| replayable.
module Test.Reset

import Data.List
import Data.Vect

import Gym.ClassicControl.Acrobot
import Gym.ClassicControl.CartPole
import Gym.ClassicControl.MountainCar
import Gym.ClassicControl.MountainCarCont
import Gym.ClassicControl.Pendulum
import Gym.Env
import Gym.Rng
import Test.Harness

-- Captured from the tree at the commit before `reset` widened from `Seed` to
-- `Source`, by observing each env's reset state at seed 42. Not derived from
-- the new code: that would make the pin circular.
cartpoleAt42 : List Double
cartpoleAt42 =
  [0.02415648787718233, -0.034008960712307995, -0.022139886974486135, -0.015580928347636247]

acrobotAt42 : List Double
acrobotAt42 =
  [0.9988331551786536, 0.048294183043672737, 0.9976876728752317, -0.06796548677677916,
   -0.04427977394897227, -0.031161856695272494]

pendulumAt42 : List Double
pendulumAt42 = [0.052974621217253236, 0.9985958589473964, -0.6801792142461598]

mountainCarAt42 : List Double
mountainCarAt42 = [-0.45168702424563534, 0.0]

closeAll : String -> List Double -> List Double -> IO Bool
closeAll name expected actual =
  if length expected == length actual && all (\d => abs d < 1.0e-15) (zipWith (-) expected actual)
    then check name True
    else do putStrLn ("  FAIL: " ++ name)
            putStrLn ("    expected " ++ show expected)
            putStrLn ("    actual   " ++ show actual)
            pure False

export
tests : List (IO Bool)
tests =
  [ let (s, _) = reset {state=CPState} {action=Nat} {obs=Vect 4 Double} (Seeded 42)
    in closeAll "CartPole seeded reset unchanged" cartpoleAt42
                (toList (observe {state=CPState} {action=Nat} {obs=Vect 4 Double} s))

  , let (s, _) = reset {state=AState} {action=Nat} {obs=Vect 6 Double} (Seeded 42)
    in closeAll "Acrobot seeded reset unchanged" acrobotAt42
                (toList (observe {state=AState} {action=Nat} {obs=Vect 6 Double} s))

  , let (s, _) = reset {state=PState} {action=Double} {obs=Vect 3 Double} (Seeded 42)
    in closeAll "Pendulum seeded reset unchanged" pendulumAt42
                (toList (observe {state=PState} {action=Double} {obs=Vect 3 Double} s))

  , let (s, _) = reset {state=MCState} {action=Nat} {obs=Vect 2 Double} (Seeded 42)
    in closeAll "MountainCar seeded reset unchanged" mountainCarAt42
                (toList (observe {state=MCState} {action=Nat} {obs=Vect 2 Double} s))

  , let (s, _) = reset {state=MCCState} {action=Double} {obs=Vect 2 Double} (Seeded 42)
    in closeAll "MountainCarCont seeded reset unchanged" mountainCarAt42
                (toList (observe {state=MCCState} {action=Double} {obs=Vect 2 Double} s))

  , -- CartPole draws four uniforms on U(-0.05, 0.05), in state order. Feeding
    -- the extremes and the midpoint pins both the mapping and the order.
    let (s, _) = reset {state=CPState} {action=Nat} {obs=Vect 4 Double}
                       (Recorded [0.0, 1.0, 0.5, 0.0])
    in closeAll "CartPole replays recorded draws" [-0.05, 0.05, 0.0, -0.05]
                (toList (observe {state=CPState} {action=Nat} {obs=Vect 4 Double} s))

  , -- MountainCar's position is U(-0.6, -0.4) and its velocity is always 0,
    -- so only one draw is consumed.
    let (s, _) = reset {state=MCState} {action=Nat} {obs=Vect 2 Double} (Recorded [0.5])
    in closeAll "MountainCar replays recorded draws" [-0.5, 0.0]
                (toList (observe {state=MCState} {action=Nat} {obs=Vect 2 Double} s))

  , -- A recording is consumed in order across successive resets, so a whole
    -- episode sequence can be replayed from one list.
    let (s1, src) = reset {state=MCState} {action=Nat} {obs=Vect 2 Double}
                          (Recorded [0.0, 1.0])
        (s2, _) = reset {state=MCState} {action=Nat} {obs=Vect 2 Double} src
        p1      = index 0 (observe {state=MCState} {action=Nat} {obs=Vect 2 Double} s1)
        p2      = index 0 (observe {state=MCState} {action=Nat} {obs=Vect 2 Double} s2)
    in closeAll "successive resets advance the recording" [-0.6, -0.4] [p1, p2]
  ]
