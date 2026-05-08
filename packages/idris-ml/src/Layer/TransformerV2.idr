module Layer.TransformerV2

import Data.Vect
import Decidable.Equality

import Compat.Random
import Device
import Init
import Layer.CoreV2
import Layer.LayerNormV2
import Layer.LinearV2
import Sampler
import Variable


----------------------------------------------------------------------
-- TransformerV2 — typed-surface multi-block transformer (Path C)
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
-- BlockStateV2
----------------------------------------------------------------------

public export
record BlockStateV2 (dModel : Nat) (numHeads : Nat) (headDim : Nat)
                    (0 d : Device) where
  constructor MkBlockV2
  queryWs   : Vect numHeads (LinearStateV2 dModel headDim d)
  keyWs     : Vect numHeads (LinearStateV2 dModel headDim d)
  valueWs   : Vect numHeads (LinearStateV2 dModel headDim d)
  outProjWs : Vect numHeads (LinearStateV2 headDim dModel d)
  norm1     : LayerNormStateV2 dModel dModel d
  norm2     : LayerNormStateV2 dModel dModel d
  ff1       : LinearStateV2 dModel (4 * dModel) d
  ff2       : LinearStateV2 (4 * dModel) dModel d


----------------------------------------------------------------------
-- TransformerStateV2
----------------------------------------------------------------------

public export
data TransformerStateV2 :
  (seqLen : Nat) -> (dModel : Nat) -> (numHeads : Nat) ->
  (headDim : Nat) -> (numBlocks : Nat) -> (vocabSize : Nat) ->
  Nat -> Nat -> (0 _ : Device) -> Type
  where
  MkTransformerV2 :
    {0 prf : dModel = numHeads * headDim} ->
    TMat vocabSize dModel d ->                        -- token embedding
    Vect numBlocks (BlockStateV2 dModel numHeads headDim d) ->
    LayerNormStateV2 dModel dModel d ->                -- final norm
    LinearStateV2 dModel vocabSize d ->                -- output projection
    TransformerStateV2 seqLen dModel numHeads headDim numBlocks vocabSize
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
              Vect k (LinearStateV2 dModel headDim d) ->
              Vect k (LinearStateV2 dModel headDim d) ->
              Vect k (LinearStateV2 dModel headDim d) ->
              Vect k (LinearStateV2 headDim dModel d) ->
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
blockForwardV2 : {dModel, numHeads, headDim : Nat} ->
                 BlockStateV2 dModel numHeads headDim d ->
                 AnyPtr -> Int -> Int -> AnyPtr
blockForwardV2 (MkBlockV2 qs ks vs ops
                          (MkLayerNormV2 n1g n1b)
                          (MkLayerNormV2 n2g n2b)
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
foldBlocksV2 : {dModel, numHeads, headDim : Nat} ->
               Vect k (BlockStateV2 dModel numHeads headDim d) ->
               AnyPtr -> Int -> Int -> AnyPtr
foldBlocksV2 [] h _ _ = h
foldBlocksV2 (b :: bs) h sI hdI =
  foldBlocksV2 bs (blockForwardV2 b h sI hdI) sI hdI


----------------------------------------------------------------------
-- Forward
----------------------------------------------------------------------

export
applyTransformerV2 : {seqLen, dModel, numHeads, headDim, numBlocks, vocabSize : Nat} ->
                     TransformerStateV2 seqLen dModel numHeads headDim numBlocks
                                       vocabSize seqLen (seqLen * vocabSize) d ->
                     TVec seqLen d ->
                     TVec (seqLen * vocabSize) d
applyTransformerV2 {seqLen} {dModel} {headDim} {vocabSize}
                   (MkTransformerV2 embedW blocks (MkLayerNormV2 nfg nfb) vocabProj) tokens =
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
      hN = foldBlocksV2 blocks h0 sI hdI
      normedFinal' = prim__layerNorm2d hN nfg.tensorPtr nfb.tensorPtr 1.0e-5
      vpW = vocabProj.weightT.tensorPtr
      outT = prim__mm normedFinal' (prim__transpose2d vpW)
      outFlatPtr = prim__narrow outT 0 0 (sI * vI)
  in MkTVar outFlatPtr Nothing


----------------------------------------------------------------------
-- Constructors
----------------------------------------------------------------------

-- Build a Vect of n LinearV2 layers with sequential paramId suffixes.
mkLinearVecV2 : {i, o : Nat} -> (n : Nat) -> String -> IO (Vect n (LinearStateV2 i o CPU))
mkLinearVecV2 Z _ = pure []
mkLinearVecV2 (S k) pfx = do
  l <- linearLayerV2 {i} {o} (pfx ++ show k)
  rest <- mkLinearVecV2 k pfx
  pure (l :: rest)

-- Build one transformer block.
mkBlockV2 : {dModel, numHeads, headDim : Nat} ->
            (paramPrefix : String) ->
            IO (BlockStateV2 dModel numHeads headDim CPU)
mkBlockV2 pfx = do
  qs <- mkLinearVecV2 {i = dModel} {o = headDim} numHeads (pfx ++ "_q")
  ks <- mkLinearVecV2 {i = dModel} {o = headDim} numHeads (pfx ++ "_k")
  vs <- mkLinearVecV2 {i = dModel} {o = headDim} numHeads (pfx ++ "_v")
  ops <- mkLinearVecV2 {i = headDim} {o = dModel} numHeads (pfx ++ "_o")
  n1 <- layerNormLayerV2 {n = dModel} (pfx ++ "_n1")
  n2 <- layerNormLayerV2 {n = dModel} (pfx ++ "_n2")
  f1 <- linearLayerV2 {i = dModel} {o = 4 * dModel} (pfx ++ "_ff1")
  f2 <- linearLayerV2 {i = 4 * dModel} {o = dModel} (pfx ++ "_ff2")
  pure $ MkBlockV2 qs ks vs ops n1 n2 f1 f2

mkBlocksV2 : {dModel, numHeads, headDim : Nat} ->
             (k : Nat) -> (paramPrefix : String) ->
             IO (Vect k (BlockStateV2 dModel numHeads headDim CPU))
mkBlocksV2 Z _ = pure []
mkBlocksV2 (S k) paramPrefix = do
  blk <- mkBlockV2 paramPrefix
  rest <- mkBlocksV2 k (paramPrefix ++ "_n")
  pure (blk :: rest)

-- Pack a Vect of Doubles into a buffer.
packDoubles : AnyPtr -> Int -> Vect k Double -> AnyPtr
packDoubles buf _ [] = buf
packDoubles buf off (x :: rest) =
  packDoubles (prim__setDouble buf off x) (off + 1) rest

||| Build a TransformerV2 with Xavier-uniform embedding init, He-init
||| linears (via LinearV2's default), and standard LayerNorm init.
||| All params register as C params under their respective prefixes.
export
transformerLayerV2 :
  {seqLen, dModel, numHeads, headDim, numBlocks, vocabSize : Nat} ->
  {auto prf : dModel = numHeads * headDim} ->
  (paramPrefix : String) ->
  IO (TransformerStateV2 seqLen dModel numHeads headDim numBlocks vocabSize
                         seqLen (seqLen * vocabSize) CPU)
transformerLayerV2 {prf} paramPrefix = do
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
      embTV = MkTVar embPtr (Just embName)
  blks <- mkBlocksV2 numBlocks (paramPrefix ++ "_b")
  nf <- layerNormLayerV2 {n = dModel} (paramPrefix ++ "_nf")
  vp <- linearLayerV2 {i = dModel} {o = vocabSize} (paramPrefix ++ "_vp")
  pure $ MkTransformerV2 {prf} embTV blks nf vp


----------------------------------------------------------------------
-- LayerLikeV2 instance
----------------------------------------------------------------------

public export
{seqLen, dModel, numHeads, headDim, numBlocks, vocabSize : Nat} ->
  LayerLikeV2 (TransformerStateV2 seqLen dModel numHeads headDim numBlocks vocabSize) where
  applyTVar st@(MkTransformerV2 _ _ _ _) input = (st, applyTransformerV2 st input)
  layerPrefixV2 _ = "tfmV2"

export
transformerLayerV2Any :
  {seqLen, dModel, numHeads, headDim, numBlocks, vocabSize : Nat} ->
  {auto prf : dModel = numHeads * headDim} ->
  (paramPrefix : String) ->
  IO (AnyLayerV2 seqLen (seqLen * vocabSize) CPU)
transformerLayerV2Any {prf} pid =
  map (MkAnyLayerV2 (TransformerStateV2 seqLen dModel numHeads headDim numBlocks vocabSize))
      (transformerLayerV2 {prf} pid)
