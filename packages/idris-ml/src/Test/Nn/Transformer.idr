module Test.Nn.Transformer

import Data.List
import Data.Vect

import Test.Harness
import Executor
import Tensor
import Nn.Init
import Nn.Module
import Nn.Seq
import Nn.Transformer
import Test.Config

-- dModel=4, numHeads=2, headDim=2 (dModel = numHeads*headDim), seqLen=3.
read12 : Tensor [3, 4] TestExecutor TestDType g -> List Double
read12 t = [ primItem2d {ex=TestExecutor} t.tensorPtr i j
           | i <- the (List Int) [0,1,2], j <- the (List Int) [0,1,2,3] ]

allFinite : List Double -> Bool
allFinite = all (\v => v == v)  -- NaN != NaN

-- A single block preserves [seqLen, dModel] shape and produces finite output.
blockForwardShape : IO Bool
blockForwardShape = do
  blk <- runInit (transformerBlock {ex=TestExecutor} {dt=TestDType} {dModel=4} {numHeads=2} {headDim=2})
  x   <- tensor {ex=TestExecutor} {dt=TestDType} {dims=[3, 4]} (Const 0.5)
  out <- forward {b=3} blk x
  let vs = read12 out
  check ("TransformerBlock preserves [3,4] + finite (got " ++ show (length vs) ++ " vals)")
        (length vs == 12 && allFinite vs)

-- The payoff: blocks stack via Seq like any other Module.
blocksStackInSeq : IO Bool
blocksStackInSeq = do
  (b1, b2) <- runInit $ do
    a <- scopedChild "block" (transformerBlock {ex=TestExecutor} {dt=TestDType} {dModel=4} {numHeads=2} {headDim=2})
    b <- scopedChild "block" (transformerBlock {ex=TestExecutor} {dt=TestDType} {dModel=4} {numHeads=2} {headDim=2})
    pure (a, b)
  let net = the (Seq 4 4 TestExecutor TestDType) (b1 :: b2 :: Nil)
  x   <- tensor {ex=TestExecutor} {dt=TestDType} {dims=[3, 4]} (Const 0.5)
  out <- forwardSeq {b=3} net x
  check ("two TransformerBlocks stack in a Seq + finite (got "
         ++ show (length (read12 out)) ++ " vals)")
        (allFinite (read12 out))

paramsCompose : IO Bool
paramsCompose = do
  blk <- runInit (transformerBlock {ex=TestExecutor} {dt=TestDType} {dModel=4} {numHeads=2} {headDim=2})
  -- attention 4×numHeads(2)=8 + norm1/norm2 (2 each = 4) + ff1/ff2 (2) = 14.
  check ("Params (TransformerBlock) composes attn+norms+ff (got " ++ show (length (params blk)) ++ ")")
        (length (params blk) == 14)

export
tests : List (IO Bool)
tests = [blockForwardShape, blocksStackInSeq, paramsCompose]
