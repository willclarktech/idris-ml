module Test.Nn.Transformer

import Control.Linear.LIO
import Data.Linear.Notation
import Data.List
import Data.Vect

import Ml.Executor
import Ml.Nn.Init
import Ml.Nn.Module
import Ml.Nn.Seq
import Ml.Nn.Transformer
import Ml.Tensor
import Test.Harness

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
  blk <- runInit (transformerBlock {ex=TestExecutor} {dt=TestDType} {g=WithGrad} {dModel=4} {numHeads=2} {headDim=2})
  x   <- tensor {ex=TestExecutor} {dt=TestDType} {dims=[3, 4]} (Const 0.5)
  out <- Control.Linear.LIO.run (do
           (MkBang o # m') <- forward {b=3} blk (retypeGrad x)
           discard m'
           pure o)
  let vs = read12 out
  check ("TransformerBlock preserves [3,4] + finite (got " ++ show (length vs) ++ " vals)")
        (length vs == 12 && allFinite vs)

-- The payoff: blocks stack via Seq like any other Module.
blocksStackInSeq : IO Bool
blocksStackInSeq = do
  (b1, b2) <- runInit $ do
    a <- scopedChild "block" (transformerBlock {ex=TestExecutor} {dt=TestDType} {g=WithGrad} {dModel=4} {numHeads=2} {headDim=2})
    b <- scopedChild "block" (transformerBlock {ex=TestExecutor} {dt=TestDType} {g=WithGrad} {dModel=4} {numHeads=2} {headDim=2})
    pure (a, b)
  let net = the (Seq 4 4 TestExecutor TestDType WithGrad) (b1 :: b2 :: Nil)
  x   <- tensor {ex=TestExecutor} {dt=TestDType} {dims=[3, 4]} (Const 0.5)
  out <- Control.Linear.LIO.run (do
           (MkBang o # m') <- forwardSeq {b=3} net (retypeGrad x)
           discard m'
           pure o)
  check ("two TransformerBlocks stack in a Seq + finite (got "
         ++ show (length (read12 out)) ++ " vals)")
        (allFinite (read12 out))

paramsCompose : IO Bool
paramsCompose = do
  blk <- runInit (transformerBlock {ex=TestExecutor} {dt=TestDType} {g=WithGrad} {dModel=4} {numHeads=2} {headDim=2})
  -- attention 4×numHeads(2)=8 + norm1/norm2 (2 each = 4) + ff1/ff2 (2) = 14.
  check ("Params (TransformerBlock) composes attn+norms+ff (got " ++ show (length (params blk)) ++ ")")
        (length (params blk) == 14)

-- Init contract: the attention projections and the block's feed-forward
-- weights take the shared dense bound `U(±1/√fan_in)`, matching
-- `init_linear_` on the reference's `MultiHeadTransformer` (whose q/k/v/out
-- and ff layers are all `nn.Linear`). Until 2026-07-31 the Idris side drew
-- normals whose *std* equalled that bound, so it ran 1.73× wide.
-- `attentionParams` orders the heads query, key, value, out_proj, so index 0
-- is the first query head, `[headDim, dModel]`.
attentionInitInRange : IO Bool
attentionInitInRange = do
  blk <- runInit $ scoped "ti"
           (transformerBlock {ex=TestExecutor} {dt=TestDType} {g=WithGrad}
                             {dModel=32} {numHeads=2} {headDim=16})
  let bound = 1.0 / sqrt 32.0
      ws    = case params blk of
                (p :: _) => [ primItem1d {ex=TestExecutor} p.paramPtr k
                            | k <- map (cast {to=Int}) [the Nat 0 .. 511] ]
                []       => []
  check ("attention query weight ~ U(±1/√fan_in) (bound " ++ show bound
         ++ ", max " ++ show (foldl (\a, w => max a (abs w)) 0.0 ws) ++ ")")
        (all (\w => abs w <= bound) ws && any (\w => w /= 0.0) ws)

export
tests : List (IO Bool)
tests = [blockForwardShape, blocksStackInSeq, paramsCompose, attentionInitInRange]
