module Test.TensorExpr

import Data.Vect

import Test.Harness
import Array
import Executor
import Tensor
import Test.Config


-- Small fixed tensors for the expression-op suite. ioRerun pins FFI
-- ordering (pure-typed bulkToTensor* reorders across sibling
-- let-bindings otherwise; see gotchas).

row2 : Double -> Double -> Vector 2 Double
row2 a b = VArray [SArray a, SArray b]

mk32 : (Double, Double) -> (Double, Double) -> (Double, Double) ->
       IO (Tensor [3, 2] TestExecutor TestDType WithGrad)
mk32 (a, b) (c, d) (e, f) = ioRerun (\_ =>
  MkTensor (bulkToTensor2d {ex=TestExecutor} {dt=TestDType} [row2 a b, row2 c d, row2 e f]) Nothing)

mkIdx : Vect 3 Double -> IO (Tensor [3] TestExecutor TestDType NoGrad)
mkIdx xs = ioRerun (\_ =>
  MkTensor (bulkToTensor {ex=TestExecutor} {dt=TestDType} (VArray (map SArray xs))) Nothing)

read3 : Tensor [3] TestExecutor TestDType g -> (Double, Double, Double)
read3 t = ( primItem1d {ex=TestExecutor} t.tensorPtr 0
          , primItem1d {ex=TestExecutor} t.tensorPtr 1
          , primItem1d {ex=TestExecutor} t.tensorPtr 2 )

gatherRowsPicks : IO Bool
gatherRowsPicks = do
  q <- mk32 (1, 2) (3, 4) (5, 6)
  ix <- mkIdx [1, 0, 1]
  r <- tgatherRows q ix
  let (x, y, z) = read3 r
  check "tgatherRows [[1,2],[3,4],[5,6]] @ [1,0,1] = (2,3,6)"
        (x == 2 && y == 3 && z == 6)

maxRowsValues : IO Bool
maxRowsValues = do
  q <- mk32 (1, 5) (7, 2) (3, 4)
  r <- tmaxRows q
  let (x, y, z) = read3 r
  check "tmaxRows [[1,5],[7,2],[3,4]] = (5,7,4)" (x == 5 && y == 7 && z == 4)


export
tests : List (IO Bool)
tests = [gatherRowsPicks, maxRowsValues]
