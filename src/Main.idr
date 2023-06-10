module Main

import Data.Vect
import Math
import Tensor
import Variable


v1 : Vector 5 Double
v1 = VTensor [1, 2, 3, 4, 5]

m1 : Matrix 2 5 Double
m1 = VTensor [
  VTensor [2, 2, 2, 2, 2],
  VTensor [1, 1, 1, 1, 1]
]

main : IO ()
main = do
  let res1 = dotProduct v1 v1
  printLn res1
  let res2 = matrixVectorMultiply m1 v1
  printLn $ meanSquaredError res2 (res2 + 2)
