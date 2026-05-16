module Layer.Transformer

import Data.Vect
import Decidable.Equality

import Compat.Random
import Device
import Init
import Layer.Core
import Layer.LayerNorm
import Layer.Linear
import Sampler
import Tensor


----------------------------------------------------------------------
-- Transformer — typed-surface multi-block transformer (Path C)
----------------------------------------------------------------------
--
-- Pre-LN architecture, learned token embedding, sinusoidal PE,
-- multi-head causal self-attention. Mirrors V1 `Layer/Transformer.idr`'s
-- single-sequence `applyVarTensor` path; batched forward is a TODO
-- (V1's `transformerForwardBatch` would translate similarly).
--
-- Type parameters: seqLen, dModel, numHeads, headDim, numBlocks,
-- vocabSize. The constructor takes an `auto prf : dModel = numHeads
-- * headDim` to ensure heads tile dModel exactly.
--
-- Input: `TVec seqLen d` of token indices (encoded as doubles).
-- Output: `TVec (seqLen * vocabSize) d` of per-position logits.


----------------------------------------------------------------------
-- BlockState
----------------------------------------------------------------------

public export
record BlockState (dModel : Nat) (numHeads : Nat) (headDim : Nat)
                    (0 d : Device) where
  constructor MkBlock
  queryWs   : Vect numHeads (LinearState dModel headDim d)
  keyWs     : Vect numHeads (LinearState dModel headDim d)
  valueWs   : Vect numHeads (LinearState dModel headDim d)
  outProjWs : Vect numHeads (LinearState headDim dModel d)
  norm1     : LayerNormState dModel dModel d
  norm2     : LayerNormState dModel dModel d
  ff1       : LinearState dModel (4 * dModel) d
  ff2       : LinearState (4 * dModel) dModel d


----------------------------------------------------------------------
-- TransformerState
----------------------------------------------------------------------

public export
data TransformerState :
  (seqLen : Nat) -> (dModel : Nat) -> (numHeads : Nat) ->
  (headDim : Nat) -> (numBlocks : Nat) -> (vocabSize : Nat) ->
  Nat -> Nat -> (0 _ : Device) -> Type
  where
  MkTransformer :
    {0 prf : dModel = numHeads * headDim} ->
    TMat vocabSize dModel d ->                        -- token embedding
    Vect numBlocks (BlockState dModel numHeads headDim d) ->
    LayerNormState dModel dModel d ->                -- final norm
    LinearState dModel vocabSize d ->                -- output projection
    TransformerState seqLen dModel numHeads headDim numBlocks vocabSize
                       seqLen (seqLen * vocabSize) d


----------------------------------------------------------------------
-- Sinusoidal Positional Encoding (matches V1)
----------------------------------------------------------------------

posEncVal : Nat -> Nat -> Nat -> Double
posEncVal dModel pos dim =
  let p = cast {to=Double} pos
      i = cast {to=Double} (div dim 2)
      dm = cast {to=Double} dModel
      angle = p / pow 10000.0 (2.0 * i / dm)
  in if modNatNZ dim 2 ItIsSucc == 0 then sin angle else cos angle

writePE : (dModel : Nat) -> AnyPtr -> Int -> Int -> Int -> Int -> AnyPtr
writePE dModel buf pos dim sLen dMod =
  if pos >= sLen then buf
  else if dim >= dMod then writePE dModel buf (pos + 1) 0 sLen dMod
  else let val = posEncVal dModel (cast pos) (cast dim)
           buf' = prim__setDouble buf (pos * dMod + dim) val
       in writePE dModel buf' pos (dim + 1) sLen dMod


----------------------------------------------------------------------
-- Per-block forward (single sequence: [seqLen, dModel] tensor handle)
----------------------------------------------------------------------

%default partial

-- Recursive head loop — accumulates per-head projections over numHeads.
runHeadAttn : {dModel, headDim : Nat} ->
              Vect k (LinearState dModel headDim d) ->
              Vect k (LinearState dModel headDim d) ->
              Vect k (LinearState dModel headDim d) ->
              Vect k (LinearState headDim dModel d) ->
              AnyPtr -> Int -> Int -> Maybe AnyPtr -> AnyPtr
runHeadAttn [] [] [] [] _ _ _ (Just acc) = acc
runHeadAttn [] [] [] [] normed _ _ Nothing = normed
runHeadAttn (q :: qs) (k :: ks) (v :: vs) (op :: ops) normed sI hdI acc =
  let qW = q.weightT.tensorPtr
      kW = k.weightT.tensorPtr
      vW = v.weightT.tensorPtr
      opW = op.weightT.tensorPtr
      qi = prim__mm normed (prim__transpose2d qW)
      ki = prim__mm normed (prim__transpose2d kW)
      vi = prim__mm normed (prim__transpose2d vW)
      scale = 1.0 / sqrt (cast {to=Double} hdI)
      scores = prim__mulScalar (prim__mm qi (prim__transpose2d ki)) scale
      mask = prim__causalMask sI
      masked = prim__maskedFill scores mask (-1.0e20)
      attn = prim__softmax2d masked
      headOut = prim__mm attn vi
      proj = prim__mm headOut (prim__transpose2d opW)
      acc' = case acc of
        Nothing => proj
        Just prev => prim__add prev proj
  in runHeadAttn qs ks vs ops normed sI hdI (Just acc')

-- Forward one block on `[seqLen, dModel]` tensor handle.
blockForward : {dModel, numHeads, headDim : Nat} ->
                 BlockState dModel numHeads headDim d ->
                 AnyPtr -> Int -> Int -> AnyPtr
blockForward (MkBlock qs ks vs ops
                          (MkLayerNorm n1g n1b)
                          (MkLayerNorm n2g n2b)
                          ff1 ff2) h sI hdI =
  let f1W = ff1.weightT.tensorPtr
      f2W = ff2.weightT.tensorPtr
      normed1 = prim__layerNorm2d h n1g.tensorPtr n1b.tensorPtr 1.0e-5
      attnOut = runHeadAttn qs ks vs ops normed1 sI hdI Nothing
      h1 = prim__add attnOut h
      normed2 = prim__layerNorm2d h1 n2g.tensorPtr n2b.tensorPtr 1.0e-5
      ffHidden = prim__clampMin (prim__mm normed2 (prim__transpose2d f1W)) 0.0
      ffOut = prim__mm ffHidden (prim__transpose2d f2W)
  in prim__add ffOut h1

-- Fold over blocks.
foldBlocks : {dModel, numHeads, headDim : Nat} ->
               Vect k (BlockState dModel numHeads headDim d) ->
               AnyPtr -> Int -> Int -> AnyPtr
foldBlocks [] h _ _ = h
foldBlocks (b :: bs) h sI hdI =
  foldBlocks bs (blockForward b h sI hdI) sI hdI


----------------------------------------------------------------------
-- Forward
----------------------------------------------------------------------

export
applyTransformer : {seqLen, dModel, numHeads, headDim, numBlocks, vocabSize : Nat} ->
                     TransformerState seqLen dModel numHeads headDim numBlocks
                                       vocabSize seqLen (seqLen * vocabSize) d ->
                     TVec seqLen d ->
                     TVec (seqLen * vocabSize) d
applyTransformer {seqLen} {dModel} {headDim} {vocabSize}
                   (MkTransformer embedW blocks (MkLayerNorm nfg nfb) vocabProj) tokens =
  let sI = cast {to=Int} seqLen
      dI = cast {to=Int} dModel
      vI = cast {to=Int} vocabSize
      hdI = cast {to=Int} headDim
      embFlat = prim__embedding embedW.tensorPtr tokens.tensorPtr sI dI
      embedded = prim__reshape2d embFlat sI dI
      peBuf = prim__allocDoubles (sI * dI)
      peBuf' = writePE dModel peBuf 0 0 sI dI
      peT = prim__createState2d sI dI peBuf'
      h0 = prim__add embedded peT
      hN = foldBlocks blocks h0 sI hdI
      normedFinal' = prim__layerNorm2d hN nfg.tensorPtr nfb.tensorPtr 1.0e-5
      vpW = vocabProj.weightT.tensorPtr
      outT = prim__mm normedFinal' (prim__transpose2d vpW)
      outFlatPtr = prim__narrow outT 0 0 (sI * vI)
  in MkTensor outFlatPtr Nothing


----------------------------------------------------------------------
-- Batched per-block forward (mirrors V1 `batchBlockForward`)
----------------------------------------------------------------------
--
-- Operates on a flat [B*seqLen, dModel] handle. LayerNorm + FFN are
-- shape-agnostic in the leading dim; attention reshapes to [B, seqLen,
-- dModel] for fused 3D ops then reshapes back.

-- Per-head batched accumulator: project Q/K/V via `bmm`, fused
-- `prim__crossAttention` (Q·K^T·scale → mask → softmax → ·V), then
-- output projection via `bmm`. Sums per-head contributions.
batchedHeadLoop : {dModel, headDim : Nat} ->
                    Vect k (LinearState dModel headDim d) ->
                    Vect k (LinearState dModel headDim d) ->
                    Vect k (LinearState dModel headDim d) ->
                    Vect k (LinearState headDim dModel d) ->
                    AnyPtr -> AnyPtr -> Double -> Maybe AnyPtr -> AnyPtr
batchedHeadLoop [] [] [] [] _ _ _ (Just acc) = acc
batchedHeadLoop [] [] [] [] normed _ _ Nothing = normed
batchedHeadLoop (q :: qs) (k :: ks) (v :: vs) (op :: ops) normed mask sc acc =
  let qW = q.weightT.tensorPtr
      kW = k.weightT.tensorPtr
      vW = v.weightT.tensorPtr
      opW = op.weightT.tensorPtr
      qi = prim__bmm normed (prim__transpose2d qW)
      ki = prim__bmm normed (prim__transpose2d kW)
      vi = prim__bmm normed (prim__transpose2d vW)
      headOut = prim__crossAttention qi ki vi mask sc
      proj = prim__bmm headOut (prim__transpose2d opW)
      acc' = case acc of
        Nothing => proj
        Just prev => prim__add prev proj
  in batchedHeadLoop qs ks vs ops normed mask sc (Just acc')

batchBlockForward : {dModel, numHeads, headDim : Nat} ->
                      BlockState dModel numHeads headDim d ->
                      AnyPtr -> Int -> Int -> Int -> AnyPtr
batchBlockForward (MkBlock qs ks vs ops
                                (MkLayerNorm n1g n1b)
                                (MkLayerNorm n2g n2b)
                                ff1 ff2) h bsI sI dI =
  let f1W = ff1.weightT.tensorPtr
      f2W = ff2.weightT.tensorPtr
      batchSize = bsI `div` sI
      normed1 = prim__layerNorm2d h n1g.tensorPtr n1b.tensorPtr 1.0e-5
      normed3d = prim__reshape3d normed1 batchSize sI dI
      mask3d = prim__expandMask (prim__causalMask sI) batchSize
      scale = 1.0 / sqrt (cast {to=Double} (dI `div` cast {to=Int} numHeads))
      attnOut3d = batchedHeadLoop qs ks vs ops normed3d mask3d scale Nothing
      attnOut = prim__reshape2d attnOut3d bsI dI
      h1 = prim__add attnOut h
      normed2 = prim__layerNorm2d h1 n2g.tensorPtr n2b.tensorPtr 1.0e-5
      f1Wt = prim__transpose2d f1W
      f2Wt = prim__transpose2d f2W
      ffOut = prim__mm (prim__clampMin (prim__mm normed2 f1Wt) 0.0) f2Wt
  in prim__add ffOut h1

foldBlocksBatched : {dModel, numHeads, headDim : Nat} ->
                      Vect k (BlockState dModel numHeads headDim d) ->
                      AnyPtr -> Int -> Int -> Int -> AnyPtr
foldBlocksBatched [] h _ _ _ = h
foldBlocksBatched (b :: bs) h bsI sI dI =
  foldBlocksBatched bs (batchBlockForward b h bsI sI dI) bsI sI dI

-- Write positional encoding for B*seqLen rows (PE repeated per sample).
writePEBatch : (dModel : Nat) -> AnyPtr -> Int -> Int -> Int -> Int -> Int -> AnyPtr
writePEBatch dModel buf pos dim bsLen dMod sLen =
  if pos >= bsLen then buf
  else if dim >= dMod then writePEBatch dModel buf (pos + 1) 0 bsLen dMod sLen
  else let origPos = pos `mod` sLen
           val = posEncVal dModel (cast origPos) (cast dim)
           buf' = prim__setDouble buf (pos * dMod + dim) val
       in writePEBatch dModel buf' pos (dim + 1) bsLen dMod sLen

||| Batched transformer forward: `Tensor [b, seqLen] d` (token indices) →
||| `Tensor [b, seqLen * vocabSize] d` (per-position logits flattened).
||| Mirrors V1's `transformerForwardBatch` but on Tensor inputs and a
||| single batched output instead of List AnyPtr.
export
applyTransformerBatch :
  {seqLen, dModel, numHeads, headDim, numBlocks, vocabSize : Nat} ->
  {b : Nat} ->
  TransformerState seqLen dModel numHeads headDim numBlocks vocabSize
                     seqLen (seqLen * vocabSize) d ->
  Tensor [b, seqLen] d WithGrad ->
  Tensor [b, seqLen * vocabSize] d WithGrad
applyTransformerBatch {seqLen} {dModel} {headDim} {vocabSize} {b}
                        (MkTransformer embedW blocks (MkLayerNorm nfg nfb) vocabProj)
                        tokens =
  let bsI = cast {to=Int} (b * seqLen)
      sI = cast {to=Int} seqLen
      dI = cast {to=Int} dModel
      vI = cast {to=Int} vocabSize
      flatTokens = prim__reshape1d tokens.tensorPtr bsI
      embFlat = prim__embedding embedW.tensorPtr flatTokens bsI dI
      embedded = prim__reshape2d embFlat bsI dI
      peBuf = prim__allocDoubles (bsI * dI)
      peBuf' = writePEBatch dModel peBuf 0 0 bsI dI sI
      peT = prim__createState2d bsI dI peBuf'
      h0 = prim__add embedded peT
      hN = foldBlocksBatched blocks h0 bsI sI dI
      normedFinal' = prim__layerNorm2d hN nfg.tensorPtr nfb.tensorPtr 1.0e-5
      vpW = vocabProj.weightT.tensorPtr
      outBatch = prim__mm normedFinal' (prim__transpose2d vpW)
      -- outBatch : [b * seqLen, vocabSize]. Reshape to [b, seqLen * vocabSize].
      outReshaped = prim__reshape2d outBatch (cast {to=Int} b) (sI * vI)
  in MkTensor outReshaped Nothing


----------------------------------------------------------------------
-- Constructors
----------------------------------------------------------------------

-- Build a Vect of n Linear layers with sequential paramId suffixes.
mkLinearVec : {i, o : Nat} -> (n : Nat) -> String -> IO (Vect n (LinearState i o CPU))
mkLinearVec Z _ = pure []
mkLinearVec (S k) pfx = do
  l <- linearLayer {i} {o} (pfx ++ show k)
  rest <- mkLinearVec k pfx
  pure (l :: rest)

-- Build one transformer block.
mkBlock : {dModel, numHeads, headDim : Nat} ->
            (paramPrefix : String) ->
            IO (BlockState dModel numHeads headDim CPU)
mkBlock pfx = do
  qs <- mkLinearVec {i = dModel} {o = headDim} numHeads (pfx ++ "_q")
  ks <- mkLinearVec {i = dModel} {o = headDim} numHeads (pfx ++ "_k")
  vs <- mkLinearVec {i = dModel} {o = headDim} numHeads (pfx ++ "_v")
  ops <- mkLinearVec {i = headDim} {o = dModel} numHeads (pfx ++ "_o")
  n1 <- layerNormLayer {n = dModel} (pfx ++ "_n1")
  n2 <- layerNormLayer {n = dModel} (pfx ++ "_n2")
  f1 <- linearLayer {i = dModel} {o = 4 * dModel} (pfx ++ "_ff1")
  f2 <- linearLayer {i = 4 * dModel} {o = dModel} (pfx ++ "_ff2")
  pure $ MkBlock qs ks vs ops n1 n2 f1 f2

mkBlocks : {dModel, numHeads, headDim : Nat} ->
             (k : Nat) -> (paramPrefix : String) ->
             IO (Vect k (BlockState dModel numHeads headDim CPU))
mkBlocks Z _ = pure []
mkBlocks (S k) paramPrefix = do
  blk <- mkBlock paramPrefix
  rest <- mkBlocks k (paramPrefix ++ "_n")
  pure (blk :: rest)

||| Build a Transformer with Xavier-uniform embedding init, He-init
||| linears (via Linear's default), and standard LayerNorm init.
||| All params register as C params under their respective prefixes.
export
transformerLayer :
  {seqLen, dModel, numHeads, headDim, numBlocks, vocabSize : Nat} ->
  {auto prf : dModel = numHeads * headDim} ->
  (paramPrefix : String) ->
  IO (TransformerState seqLen dModel numHeads headDim numBlocks vocabSize
                         seqLen (seqLen * vocabSize) CPU)
transformerLayer {prf} paramPrefix = do
  let n = vocabSize * dModel
  embedVals <- traverse (\_ => xavier uniform vocabSize dModel) (Vect.replicate n ())
  let nI = cast {to=Int} n
      vI = cast {to=Int} vocabSize
      dI = cast {to=Int} dModel
      embBuf = prim__allocDoubles nI
      embBuf' = packDoubles embBuf 0 embedVals
      embName = paramPrefix ++ "_embed"
      embPtr = prim__paramRegister embName (prim__createParam2d vI dI embBuf')
      embTV : TMat vocabSize dModel CPU
      embTV = MkTensor embPtr (Just embName)
  blks <- mkBlocks numBlocks (paramPrefix ++ "_b")
  nf <- layerNormLayer {n = dModel} (paramPrefix ++ "_nf")
  vp <- linearLayer {i = dModel} {o = vocabSize} (paramPrefix ++ "_vp")
  pure $ MkTransformer {prf} embTV blks nf vp


----------------------------------------------------------------------
-- LayerLike instance
----------------------------------------------------------------------

public export
{seqLen, dModel, numHeads, headDim, numBlocks, vocabSize : Nat} ->
  LayerLike (TransformerState seqLen dModel numHeads headDim numBlocks vocabSize) where
  applyVar st@(MkTransformer _ _ _ _) input = (st, applyTransformer st input)
  applyVarBatch st@(MkTransformer _ _ _ _) input =
    (st, applyTransformerBatch st input)
  layerPrefix _ = "tfm"

export
transformerLayerAny :
  {seqLen, dModel, numHeads, headDim, numBlocks, vocabSize : Nat} ->
  {auto prf : dModel = numHeads * headDim} ->
  (paramPrefix : String) ->
  IO (AnyLayer seqLen (seqLen * vocabSize) CPU)
transformerLayerAny {prf} pid =
  map (MkAnyLayer (TransformerState seqLen dModel numHeads headDim numBlocks vocabSize))
      (transformerLayer {prf} pid)
