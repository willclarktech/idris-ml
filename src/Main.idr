module Main

import Data.Vect
import Tensor


e1 : Vector 0 Int
e1 = VTensor []

e2 : Matrix 2 0 Int
e2 = VTensor [VTensor [], VTensor []]

s : Scalar Int
s = STensor 123

v : Vector 3 Nat
v = VTensor (map (STensor . finToNat) range)

m : Matrix 2 3 Nat
m = VTensor [v, v]

t : Tensor [4, 2, 3] Nat
t = VTensor [m, m, m, m]

r0 : Scalar Int
r0 = pure 2

r1 : Vector 3 Int
r1 = pure 2

r2 : Matrix 2 3 Int
r2 = pure 2

r3 : Tensor [2,3,4] Int
r3 = pure 2

z4 : Tensor [1,2,3,4] Int
z4 = zeros

rs4 : Tensor [3, 8] Int
rs4 = reshape z4

o2 : Matrix 5 2 Int
o2 = ones

main : IO ()
main = do
  printLn $ concatAlong 0 t t
  printLn $ shapeOf $ concatAlong 0 t t
  printLn $ concatAlong 1 t t
  printLn $ shapeOf $ concatAlong 1 t t
  printLn $ concatAlong 2 t t
  printLn $ shapeOf $ concatAlong 2 t t
  printLn $ unflatten {dims = [1, 3, 1, 1]} v
  printLn $ (rs4, shapeOf rs4)
