module Test.Nn.Embedding

import Data.List
import Data.Vect

import Executor
import Nn.Embedding
import Nn.Init
import Nn.Module
import Tensor
import Test.Config
import Test.Harness

-- weight [3,2] = rows [1,2],[3,4],[5,6]; tokens [0,2] → flattened
-- [row0, row2] = [1,2,5,6].
lookupForward : IO Bool
lookupForward = do
  w <- param {ex=TestExecutor} {dt=TestDType} {dims=[3, 2]} "et.w"
            (FromVect [1.0, 2.0, 3.0, 4.0, 5.0, 6.0])
  let emb = the (Embedding 3 2 TestExecutor TestDType WithGrad) (MkEmbedding w)
  toks <- tensor {ex=TestExecutor} {dt=TestDType} {dims=[2]} (FromVect [0.0, 2.0])
  out  <- embeddingForward {vocab=3} emb (retypeGrad toks)
  let vs = [ primItem1d {ex=TestExecutor} out.tensorPtr i | i <- the (List Int) [0,1,2,3] ]
  check ("Embedding lookup flattens rows (got " ++ show vs ++ ")")
        (vs == [1.0, 2.0, 5.0, 6.0])

paramExposed : IO Bool
paramExposed = do
  w <- param {ex=TestExecutor} {dt=TestDType} {dims=[3, 2]} "ep.w" (Const 0.0)
  let emb = the (Embedding 3 2 TestExecutor TestDType WithGrad) (MkEmbedding w)
  check "Params (Embedding) exposes weight"
        (mapMaybe paramName (params emb) == ["ep.w"])

smartCtorName : IO Bool
smartCtorName = do
  _ <- runInit $ scoped "tok" (embedding {ex=TestExecutor} {dt=TestDType} {g=WithGrad} {vocab=5} {embedDim=4})
  cnt <- getParamCount {ex=TestExecutor}
  names <- traverse (\i => getParamName {ex=TestExecutor} i) [0 .. cnt - 1]
  check "embedding registers tok.embedding_0.weight"
        ("tok.embedding_0.weight" `elem` names)

export
tests : List (IO Bool)
tests = [lookupForward, paramExposed, smartCtorName]
