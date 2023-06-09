module Main

import Data.Vect
import Tensor
import Variable


v1 : Variable
v1 = 1.23

v2 : Variable
v2 = 4.56

v3 : Variable
v3 = -2.22

main : IO ()
main = do
  let v4 = v1 * v2 + v3
  printLn $ v4
  let v5 = backwardVariable 1.0 v4
  printLn $ v5
