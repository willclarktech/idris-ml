||| `Embedding` — a learnable lookup table. Params-only (NOT a `Module`):
||| its forward maps a `[seqLen]` tensor of token ids (doubles) to a
||| flattened `[seqLen * embedDim]` tensor, which is not the
||| `Tensor [b,i] -> Tensor [b,o]` shape `Module` demands. The layer's two
||| Nat indices (vocab, embedDim) DO match `Params`'s kind, so it gets
||| `Params` (and the generic `freeze`/`unfreeze`) for free; composition
||| happens at the example level via `embeddingForward`.
module Nn.Embedding

import Control.Linear.LIO
import Data.Linear
import Data.Vect

import Executor
import Nn.Init
import Nn.Module
import Tensor

%default total

||| A `vocab × embedDim` lookup table. Params are `WithGrad` by construction.
public export
record Embedding (vocab : Nat) (embedDim : Nat) (0 ex : Executor) (0 dt : DType) (0 g : GradMode) where
  constructor MkEmbedding
  weightT : TMat vocab embedDim ex dt g

||| The lookup table `w` is the single ω param field.
public export
Params Embedding where
  params (MkEmbedding w)   = [toParam w]
  reflect (MkEmbedding w)  = MkBang [toParam w] # MkEmbedding w
  castGrad (MkEmbedding w) = MkEmbedding (retypeGrad w)
  discard (MkEmbedding _)  = pure ()

||| Lookup forward: `tokens : [seqLen]` (ids as doubles) → flattened
||| `[seqLen * embedDim]` embedding vectors. (Standalone, not a `Module`.)
export
embeddingForward : {0 ex : Executor} -> Backend ex dt => {seqLen, embedDim, vocab : Nat} ->
                   Embedding vocab embedDim ex dt g -> TVec seqLen ex dt g ->
                   IO (TVec (seqLen * embedDim) ex dt g)
embeddingForward {seqLen} {embedDim} (MkEmbedding w) tokens = ioRerun (\_ =>
  MkTensor (primEmbedding {ex} w.tensorPtr tokens.tensorPtr
                          (cast {to=Int} seqLen) (cast {to=Int} embedDim)) Nothing)

||| Construct an `Embedding vocab embedDim` inside an `Init` derivation;
||| registers `<scope>.embedding_<n>.weight`, weights ~ N(0, 0.02) (HF
||| default for token/position embeddings).
export
embedding : KnownGrad g => {0 ex : Executor} -> Backend ex dt => {vocab, embedDim : Nat} ->
            Init (Embedding vocab embedDim ex dt g)
embedding = do
  name <- freshChild "embedding"
  w <- liftIO $ tparam2dNormal {ex} {dt} {o=vocab} {i=embedDim} (name ++ ".weight") 0.0 0.02
  case sgrad {g} of
    SWithGrad => pure (MkEmbedding w)
    SNoGrad   => do w' <- liftIO (weakenGrad w)
                    pure (MkEmbedding w')
