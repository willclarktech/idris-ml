module Test.Nn.Seq

import Control.Linear.LIO
import Data.Linear.Notation
import Data.Vect

import Executor
import Nn.Module
import Nn.Seq
import Tensor
import Test.Config
import Test.Harness

-- Trivial identity layer (i = o, no params) — exercises Seq's Nil/(::)
-- composition + Params concatenation without a real layer port.
data Id : Nat -> Nat -> (0 _ : Executor) -> (0 _ : DType) -> (0 _ : GradMode) -> Type where
  MkId : Id n n ex dt g

Params Id where
  params MkId   = []
  reflect MkId  = MkBang [] # MkId
  castGrad MkId = MkId
  discard MkId  = pure ()

Module Id where
  forward MkId x = pure1 (MkBang x # MkId)

-- Concrete-typed identity value — mirrors a real layer smart constructor
-- (`linear : ... -> IO (Linear i o ex dt)`) whose concrete return type pins
-- the element's `l` for `(::)` without higher-order unification on a bare
-- polymorphic constructor.
idLayer : Id 3 3 TestExecutor TestDType NoGrad
idLayer = MkId

read6 : Tensor [2, 3] TestExecutor TestDType g -> List Double
read6 t = [ primItem2d {ex=TestExecutor} t.tensorPtr i j
          | (i, j) <- the (List (Int, Int)) [(0,0),(0,1),(0,2),(1,0),(1,1),(1,2)] ]

-- A two-layer identity Seq threads the activation unchanged.
seqForwards : IO Bool
seqForwards = do
  t <- tensor {ex=TestExecutor} {dt=TestDType} {dims=[2, 3]} (Const 4.0)
  let net = the (Seq 3 3 TestExecutor TestDType NoGrad) (idLayer :: idLayer :: Nil)
  out <- Control.Linear.LIO.run (do
           (MkBang o # m') <- forwardSeq {b=2} net t
           discard m'
           pure o)
  check ("forwardSeq threads through a 2-layer chain (got " ++ show (read6 out) ++ ")")
        (read6 out == [4.0, 4.0, 4.0, 4.0, 4.0, 4.0])

-- The chain operator (~~>) desugars to (::).
seqViaChainOp : IO Bool
seqViaChainOp = do
  t <- tensor {ex=TestExecutor} {dt=TestDType} {dims=[2, 3]} (Const 4.0)
  let net = the (Seq 3 3 TestExecutor TestDType NoGrad) (idLayer ~~> idLayer ~~> Nil)
  out <- Control.Linear.LIO.run (do
           (MkBang o # m') <- forwardSeq {b=2} net t
           discard m'
           pure o)
  check "(~~>) builds the same chain as (::)"
        (read6 out == [4.0, 4.0, 4.0, 4.0, 4.0, 4.0])

-- Params (Seq) concatenates element params (param-free chain → empty).
seqParamsCompose : IO Bool
seqParamsCompose =
  check "Params (Seq) concatenates element params"
        (length (params (the (Seq 3 3 TestExecutor TestDType NoGrad) (idLayer :: idLayer :: Nil))) == 0)

export
tests : List (IO Bool)
tests = [seqForwards, seqViaChainOp, seqParamsCompose]
