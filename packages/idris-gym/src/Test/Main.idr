module Test.Main

import Test.Harness

import Test.ClassicControl.Acrobot
import Test.ClassicControl.CartPole
import Test.ClassicControl.MountainCar
import Test.ClassicControl.MountainCarCont
import Test.ClassicControl.Pendulum
import Test.Reset
import Test.Rng
import Test.Space
import Test.ToyText.Blackjack
import Test.ToyText.CliffWalking
import Test.ToyText.FrozenLake
import Test.ToyText.Taxi
import Test.Vector
import Test.Wrapper

main : IO ()
main = runAll
  [ ("Rng",                            Test.Rng.tests)
  , ("Reset",                          Test.Reset.tests)
  , ("Space",                          Test.Space.tests)
  , ("Wrapper",                        Test.Wrapper.tests)
  , ("Vector",                         Test.Vector.tests)
  , ("ClassicControl.CartPole",        Test.ClassicControl.CartPole.tests)
  , ("ClassicControl.MountainCar",     Test.ClassicControl.MountainCar.tests)
  , ("ClassicControl.MountainCarCont", Test.ClassicControl.MountainCarCont.tests)
  , ("ClassicControl.Pendulum",        Test.ClassicControl.Pendulum.tests)
  , ("ClassicControl.Acrobot",         Test.ClassicControl.Acrobot.tests)
  , ("ToyText.CliffWalking",           Test.ToyText.CliffWalking.tests)
  , ("ToyText.Taxi",                   Test.ToyText.Taxi.tests)
  , ("ToyText.FrozenLake",             Test.ToyText.FrozenLake.tests)
  , ("ToyText.Blackjack",              Test.ToyText.Blackjack.tests)
  ]
