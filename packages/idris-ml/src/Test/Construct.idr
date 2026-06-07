-- Test.Construct — the tensor/param × InitSpec construction facade.
--
-- Registry assertions use set-membership (primParamName walk), never
-- "newly-added slot counts": param_register dedups by name, replacing
-- entries in place, so counts lie when names collide across suites.
module Test.Construct

import Data.List
import Data.Vect

import Test.Harness
import Array
import Executor
import Tensor
import Test.Config

read4 : Tensor [4] TestExecutor TestDType g -> List Double
read4 t = map (primItem1d {ex=TestExecutor} t.tensorPtr) [0, 1, 2, 3]

hasParam : String -> IO Bool
hasParam name = do
  n <- primIO (primParamCount {ex=TestExecutor})
  go 0 n
  where
    go : Int -> Int -> IO Bool
    go i n = if i >= n then pure False else do
      nm <- primIO (primParamName {ex=TestExecutor} i)
      if nm == name then pure True else go (i + 1) n

constFills : IO Bool
constFills = do
  t <- tensor {ex=TestExecutor} {dt=TestDType} {dims=[4]} (Const 7.0)
  check ("tensor (Const 7.0) fills [4] (got " ++ show (read4 t) ++ ")")
        (all (== 7.0) (read4 t))

read22 : Tensor [2, 2] TestExecutor TestDType g -> List Double
read22 t = [ primItem2d {ex=TestExecutor} t.tensorPtr i j
           | (i, j) <- the (List (Int, Int)) [(0, 0), (0, 1), (1, 0), (1, 1)] ]

zerosFillRank2 : IO Bool
zerosFillRank2 = do
  t <- tensor {ex=TestExecutor} {dt=TestDType} {dims=[2, 2]} Zeros
  check ("tensor Zeros fills [2,2] (got " ++ show (read22 t) ++ ")")
        (all (== 0.0) (read22 t))

fromVectMatchesBulk : IO Bool
fromVectMatchesBulk = do
  t <- tensor {ex=TestExecutor} {dt=TestDType} {dims=[4]} (FromVect [1.5, -2.0, 0.25, 9.0])
  o <- ioRerun (\_ => MkTensor
        (bulkToTensor {ex=TestExecutor} {dt=TestDType}
          (VArray [SArray 1.5, SArray (-2.0), SArray 0.25, SArray 9.0])) Nothing)
  let ok = read4 t == read4 (the (Tensor [4] TestExecutor TestDType NoGrad) o)
  check "tensor (FromVect ...) == bulkToTensor oracle" ok

paramRegistersAndFills : IO Bool
paramRegistersAndFills = do
  p <- param {ex=TestExecutor} {dt=TestDType} {dims=[4]} "cn_const_w" (Const 3.0)
  found <- hasParam "cn_const_w"
  check ("param (Const 3.0) registers + fills (got " ++ show (read4 p)
         ++ ", registered " ++ show found ++ ")")
        (found && all (== 3.0) (read4 p))

paramNormalVaries : IO Bool
paramNormalVaries = do
  p <- param {ex=TestExecutor} {dt=TestDType} {dims=[4]} "cn_norm_w" (Normal 0.0 1.0)
  found <- hasParam "cn_norm_w"
  let vs = read4 p
  check ("param (Normal 0 1) registers + not-all-equal (got " ++ show vs ++ ")")
        (found && not (all (== head' vs) vs))
  where
    head' : List Double -> Double
    head' (x :: _) = x
    head' [] = 0.0

uniformWithinBounds : IO Bool
uniformWithinBounds = do
  t <- tensor {ex=TestExecutor} {dt=TestDType} {dims=[4]} (Uniform 2.0 3.0)
  let vs = read4 t
  check ("tensor (Uniform 2 3) in bounds (got " ++ show vs ++ ")")
        (all (\v => v >= 2.0 && v <= 3.0) vs)

scalarParam : IO Bool
scalarParam = do
  p <- param {ex=TestExecutor} {dt=TestDType} {dims=[]} "cn_scalar_w" (Const 0.5)
  found <- hasParam "cn_scalar_w"
  check ("rank-0 param (Const 0.5) (got " ++ show (tensorItem p) ++ ")")
        (found && tensorItem p == 0.5)

fromRowsBatch : IO Bool
fromRowsBatch = do
  t <- tensor {ex=TestExecutor} {dt=TestDType} {dims=[2, 2]}
         (fromRows [[1.0, 2.0], [3.0, 4.0]])
  check ("fromRows row-major (got " ++ show (read22 t) ++ ")")
        (read22 t == [1.0, 2.0, 3.0, 4.0])

export
tests : List (IO Bool)
tests = [ constFills, zerosFillRank2, fromVectMatchesBulk
        , paramRegistersAndFills, paramNormalVaries
        , uniformWithinBounds, scalarParam, fromRowsBatch ]
