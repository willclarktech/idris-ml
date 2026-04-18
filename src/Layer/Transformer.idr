-- | Transformer with Multi-Block Stacking, Pre-LN, Learned Embeddings
-- |
-- | Standard transformer: N stacked blocks (each: layer norm, multi-head
-- | causal self-attention, residual, layer norm, FFN, residual), plus
-- | learned token embedding, sinusoidal PE, final layer norm, output proj.
-- |
-- | Input: seqLen (token indices as doubles)
-- | Output: seqLen * vocabSize (per-position logits)

module Layer.Transformer

import Data.Vect
import Decidable.Equality

import Device
import Endofunctor
import Floating
import Init
import Layer.Core
import Layer.LayerNorm
import Layer.Linear
import Math
import Tensor
import Util
import Variable


----------------------------------------------------------------------
-- Block State
----------------------------------------------------------------------

||| One transformer block: attention + FFN with layer norms.
public export
record BlockState (dModel : Nat) (numHeads : Nat) (headDim : Nat) (ty : Type) where
  constructor MkBlock
  queryWs   : Vect numHeads (LinearState dModel headDim ty)
  keyWs     : Vect numHeads (LinearState dModel headDim ty)
  valueWs   : Vect numHeads (LinearState dModel headDim ty)
  outProjWs : Vect numHeads (LinearState headDim dModel ty)
  norm1     : LayerNormState dModel ty
  norm2     : LayerNormState dModel ty
  ff1       : LinearState dModel (4 * dModel) ty
  ff2       : LinearState (4 * dModel) dModel ty

export
mkBlock : {dModel, numHeads, headDim : Nat} -> (Num ty, FromDouble ty) =>
          IO (BlockState dModel numHeads headDim ty)
mkBlock = do
  qs  <- mkLinears numHeads {i=dModel, o=headDim}
  ks  <- mkLinears numHeads {i=dModel, o=headDim}
  vs  <- mkLinears numHeads {i=dModel, o=headDim}
  ops <- mkLinears numHeads {i=headDim, o=dModel}
  n1  <- mkLayerNorm {dim=dModel}
  n2  <- mkLayerNorm {dim=dModel}
  f1  <- mkLinear {i=dModel, o=4*dModel}
  f2  <- mkLinear {i=4*dModel, o=dModel}
  pure $ MkBlock qs ks vs ops n1 n2 f1 f2
  where
    mkLinears : (k : Nat) -> {i, o : Nat} -> IO (Vect k (LinearState i o ty))
    mkLinears Z = pure []
    mkLinears (S k) = [| mkLinear :: mkLinears k |]

export
emapBlock : {dModel, numHeads, headDim : Nat} -> (ty -> ty) -> BlockState dModel numHeads headDim ty -> BlockState dModel numHeads headDim ty
emapBlock f (MkBlock qs ks vs ops n1 n2 f1 f2) =
  MkBlock (map (emapLayer f) qs) (map (emapLayer f) ks)
          (map (emapLayer f) vs) (map (emapLayer f) ops)
          (emapLayerNorm f n1) (emapLayerNorm f n2)
          (emapLayer f f1) (emapLayer f f2)

nameHeads : {d : Device} -> {a, b : Nat} -> String -> Nat -> Vect k (LinearState a b (Variable d)) -> Vect k (LinearState a b (Variable d))
nameHeads _ _ [] = []
nameHeads pfx idx (h :: hs) = nameLayer (pfx ++ show idx) h :: nameHeads pfx (S idx) hs

export
nameBlock : {d : Device} -> {dModel, numHeads, headDim : Nat} -> String -> BlockState dModel numHeads headDim (Variable d)
          -> BlockState dModel numHeads headDim (Variable d)
nameBlock pfx (MkBlock qs ks vs ops n1 n2 f1 f2) =
  MkBlock (nameHeads (pfx ++ "_q") 0 qs) (nameHeads (pfx ++ "_k") 0 ks)
          (nameHeads (pfx ++ "_v") 0 vs) (nameHeads (pfx ++ "_o") 0 ops)
          (nameLayerNorm (pfx ++ "_n1") n1) (nameLayerNorm (pfx ++ "_n2") n2)
          (nameLayer (pfx ++ "_ff1") f1) (nameLayer (pfx ++ "_ff2") f2)

export
toDoubleBlock : {d : Device} -> {dModel, numHeads, headDim : Nat} -> BlockState dModel numHeads headDim (Variable d)
              -> BlockState dModel numHeads headDim Double
toDoubleBlock (MkBlock qs ks vs ops n1 n2 f1 f2) =
  MkBlock (map toDoubleLayer qs) (map toDoubleLayer ks)
          (map toDoubleLayer vs) (map toDoubleLayer ops)
          (toDoubleLayerNorm n1) (toDoubleLayerNorm n2)
          (toDoubleLayer f1) (toDoubleLayer f2)

export
getBlockParamIds : {d : Device} -> {dModel, numHeads, headDim : Nat} -> BlockState dModel numHeads headDim (Variable d) -> List String
getBlockParamIds (MkBlock qs ks vs ops n1 n2 f1 f2) =
  concatMap getParamIds (toList qs) ++ concatMap getParamIds (toList ks) ++
  concatMap getParamIds (toList vs) ++ concatMap getParamIds (toList ops) ++
  getLayerNormParamIds n1 ++ getLayerNormParamIds n2 ++
  getParamIds f1 ++ getParamIds f2

nameBlocks : {d : Device} -> {dModel, numHeads, headDim : Nat} -> String -> Nat ->
             Vect k (BlockState dModel numHeads headDim (Variable d)) ->
             Vect k (BlockState dModel numHeads headDim (Variable d))
nameBlocks _ _ [] = []
nameBlocks pfx idx (b :: bs) = nameBlock (pfx ++ show idx) b :: nameBlocks pfx (S idx) bs


----------------------------------------------------------------------
-- Transformer State
----------------------------------------------------------------------

||| Multi-block Transformer.
||| Type parameters: seqLen, dModel, numHeads, headDim, numBlocks, vocabSize.
public export
record TransformerState
    (seqLen : Nat) (dModel : Nat) (numHeads : Nat) (headDim : Nat)
    (numBlocks : Nat) (vocabSize : Nat)
    (inputSize : Nat) (outputSize : Nat) (ty : Type) where
  constructor MkTransformer
  0 headDimPrf  : dModel = numHeads * headDim
  0 inputPrf    : inputSize = seqLen
  0 outputPrf   : outputSize = seqLen * vocabSize
  tokenEmbedWeight : Vector (vocabSize * dModel) ty
  tokenEmbedTensor : Maybe AnyPtr    -- [vocabSize, dModel] param tensor
  blocks        : Vect numBlocks (BlockState dModel numHeads headDim ty)
  normFinal     : LayerNormState dModel ty
  vocabProj     : LinearState dModel vocabSize ty


----------------------------------------------------------------------
-- Sinusoidal Positional Encoding
----------------------------------------------------------------------

posEncVal : Nat -> Nat -> Nat -> Double
posEncVal dModel pos dim =
  let p = cast {to=Double} pos
      i = cast {to=Double} (div dim 2)
      dm = cast {to=Double} dModel
      angle = p / pow 10000.0 (2.0 * i / dm)
  in if modNatNZ dim 2 ItIsSucc == 0 then sin angle else cos angle


----------------------------------------------------------------------
-- Pure Idris block forward (for applyGeneric)
----------------------------------------------------------------------

blockForwardGeneric : (FromDouble ty, Floating ty, Fractional ty, Neg ty, Ord ty, Num ty) =>
                      {seqLen, dModel, headDim : Nat} ->
                      BlockState dModel numHeads headDim ty ->
                      Matrix seqLen dModel ty -> Matrix seqLen dModel ty
blockForwardGeneric blk h =
  let -- Pre-LN Multi-Head Attention
      normed1 = layerNormMatrix h (gamma (norm1 blk)) (beta (norm1 blk)) (fromDouble 1.0e-5)
      attnOut = headLoopGeneric (queryWs blk) (keyWs blk) (valueWs blk) (outProjWs blk) normed1
      h1 = attnOut + h
      -- Pre-LN Feedforward
      normed2 = layerNormMatrix h1 (gamma (norm2 blk)) (beta (norm2 blk)) (fromDouble 1.0e-5)
      ffHidden = clampMinTensor (fromDouble 0.0) (matrixMultiply normed2 (transpose (weights (ff1 blk))))
      ffOut = matrixMultiply ffHidden (transpose (weights (ff2 blk)))
  in ffOut + h1
  where
    headLoopGeneric : Vect k (LinearState dModel headDim ty) ->
                      Vect k (LinearState dModel headDim ty) ->
                      Vect k (LinearState dModel headDim ty) ->
                      Vect k (LinearState headDim dModel ty) ->
                      Matrix seqLen dModel ty -> Matrix seqLen dModel ty
    headLoopGeneric [] [] [] [] normed = scaleMatrix (fromDouble 0.0) normed
    headLoopGeneric (q :: qs) (k :: ks) (v :: vs) (op :: ops) normed =
      let qi = matrixMultiply normed (transpose (weights q))
          ki = matrixMultiply normed (transpose (weights k))
          vi = matrixMultiply normed (transpose (weights v))
          scores = scaleMatrix (fromDouble (1.0 / sqrt (cast (natToInteger headDim))))
                     (matrixMultiply qi (transpose ki))
          masked = causalMaskMatrix scores
          attn = softmaxMatrix masked
          headOut = matrixMultiply attn vi
          proj = matrixMultiply headOut (transpose (weights op))
      in proj + headLoopGeneric qs ks vs ops normed


----------------------------------------------------------------------
-- C tensor block forward (for applyVarTensor)
----------------------------------------------------------------------

%default partial

||| Run multi-head attention on a single [seqLen, dModel] input.
runHeadAttention : {d : Device} -> {headDim : Nat} ->
                   Vect k (LinearState dModel headDim (Variable d)) ->
                   Vect k (LinearState dModel headDim (Variable d)) ->
                   Vect k (LinearState dModel headDim (Variable d)) ->
                   Vect k (LinearState headDim dModel (Variable d)) ->
                   AnyPtr -> Int -> Int -> Maybe AnyPtr -> AnyPtr
runHeadAttention [] [] [] [] _ _ _ (Just acc) = acc
runHeadAttention [] [] [] [] normed sI hdI Nothing = normed
runHeadAttention (q :: qs) (k :: ks) (v :: vs) (op :: ops) normed sI hdI acc =
  case (extractWeightTensor q, extractWeightTensor k, extractWeightTensor v, extractWeightTensor op) of
    (Just qW, Just kW, Just vW, Just opW) =>
      let qi = prim__mm normed (prim__transpose2d qW)
          ki = prim__mm normed (prim__transpose2d kW)
          vi = prim__mm normed (prim__transpose2d vW)
          scores = prim__mulScalar (prim__mm qi (prim__transpose2d ki))
                     (1.0 / sqrt (cast {to=Double} headDim))
          mask = prim__causalMask sI
          masked = prim__maskedFill scores mask (-1.0e20)
          attn = prim__softmax2d masked
          headOut = prim__mm attn vi
          proj = prim__mm headOut (prim__transpose2d opW)
          acc' = case acc of
            Nothing => proj
            Just prev => tensorAdd prev proj
      in runHeadAttention qs ks vs ops normed sI hdI (Just acc')
    _ => idris_crash "Transformer: head weight not initialized"

||| Forward one block on a [seqLen, dModel] tensor.
blockForwardTensor : {d : Device} -> {headDim : Nat} -> BlockState dModel numHeads headDim (Variable d)
                   -> AnyPtr -> Int -> Int -> AnyPtr
blockForwardTensor blk h sI hdI =
  let Just n1g = extractGammaTensor (norm1 blk)   | Nothing => idris_crash "block: n1g"
      Just n1b = extractBetaTensor (norm1 blk)    | Nothing => idris_crash "block: n1b"
      Just n2g = extractGammaTensor (norm2 blk)   | Nothing => idris_crash "block: n2g"
      Just n2b = extractBetaTensor (norm2 blk)    | Nothing => idris_crash "block: n2b"
      Just f1W = extractWeightTensor (ff1 blk)    | Nothing => idris_crash "block: ff1"
      Just f2W = extractWeightTensor (ff2 blk)    | Nothing => idris_crash "block: ff2"
  in let normed1 = prim__layerNorm2d h n1g n1b 1.0e-5
         attnOut = runHeadAttention (queryWs blk) (keyWs blk) (valueWs blk) (outProjWs blk)
                     normed1 sI hdI Nothing
         h1 = tensorAdd attnOut h
         normed2 = prim__layerNorm2d h1 n2g n2b 1.0e-5
         ffHidden = prim__clampMin (prim__mm normed2 (prim__transpose2d f1W)) 0.0
         ffOut = prim__mm ffHidden (prim__transpose2d f2W)
     in tensorAdd ffOut h1

||| Fold over blocks, threading [seqLen, dModel] tensor.
foldBlocks : {d : Device} -> {headDim : Nat} -> Vect k (BlockState dModel numHeads headDim (Variable d))
           -> AnyPtr -> Int -> Int -> AnyPtr
foldBlocks [] h _ _ = h
foldBlocks (b :: bs) h sI hdI = foldBlocks bs (blockForwardTensor b h sI hdI) sI hdI


----------------------------------------------------------------------
-- LayerLike Instance
----------------------------------------------------------------------

export
{seqLen : Nat} -> {dModel : Nat} -> {numHeads : Nat} -> {headDim : Nat} ->
{numBlocks : Nat} -> {vocabSize : Nat} ->
LayerLike (TransformerState seqLen dModel numHeads headDim numBlocks vocabSize) where

  applyGeneric _ _ = idris_crash "Transformer: use tensor path (embedding requires C backend)"

  applyVar {d} {i} {o} st xs =
    let (VTensor xElems) = xs
        inputFlat = vecStackTensor xElems
        (st', outT) = applyVarTensor st inputFlat
        output = VTensor (tensorToScalars outT 0 o)
    in (st', output)

  applyVarTensor {d} {i} {o} st inputFlat =
    case st.tokenEmbedTensor of
      Just embedT =>
        let Just vpW = extractWeightTensor (vocabProj st) | Nothing => idris_crash "Transformer: vocabProj"
            Just nfg = extractGammaTensor (normFinal st)  | Nothing => idris_crash "Transformer: nfg"
            Just nfb = extractBetaTensor (normFinal st)   | Nothing => idris_crash "Transformer: nfb"
        in
        let sI = cast {to=Int} seqLen
            dI = cast {to=Int} dModel
            vI = cast {to=Int} vocabSize
            hdI = cast {to=Int} headDim
            -- Embedding lookup: [seqLen] indices -> [seqLen * dModel] flat -> [seqLen, dModel]
            embFlat = prim__embedding embedT inputFlat sI dI
            embedded = prim__reshape2d embFlat sI dI
            peBuf = prim__allocDoubles (sI * dI)
            peBuf' = writePE peBuf 0 0 sI dI
            peT = prim__createState2d sI dI peBuf'
            h0 = tensorAdd embedded peT
            hN = foldBlocks (blocks st) h0 sI hdI
            normedFinal = prim__layerNorm2d hN nfg nfb 1.0e-5
            outT = prim__mm normedFinal (prim__transpose2d vpW)
        in (st, prim__narrow outT 0 0 (sI * vI))
      Nothing => idris_crash "Transformer: tokenEmbedTensor not initialized"
    where
      writePE : AnyPtr -> Int -> Int -> Int -> Int -> AnyPtr
      writePE buf pos dim sLen dMod =
        if pos >= sLen then buf
        else if dim >= dMod then writePE buf (pos + 1) 0 sLen dMod
        else let val = posEncVal dModel (cast pos) (cast dim)
                 buf' = prim__setDouble buf (pos * dMod + dim) val
             in writePE buf' pos (dim + 1) sLen dMod

  emapLayer f (MkTransformer hdp ip op tew tet blks nf vp) =
    MkTransformer hdp ip op (map f tew) tet (map (emapBlock f) blks)
                  (emapLayerNorm f nf) (emapLayer f vp)

  showLayer _ = "Transformer<" ++ show seqLen ++ "x" ++ show dModel
    ++ " h=" ++ show numHeads ++ " blocks=" ++ show numBlocks
    ++ " v=" ++ show vocabSize ++ ">"

  nameLayer {d} {i} {o} prefx (MkTransformer hdp ip op tew _ blks nf vp) =
    let vI = cast {to=Int} vocabSize
        dI = cast {to=Int} dModel
        nI = cast {to=Int} (vocabSize * dModel)
        buf = prim__allocDoubles nI
        (VTensor elems) = tew
        buf' = packScalarValues buf 0 elems
        embedT = prim__paramRegister (prefx ++ "_embed_weight")
                   (prim__createParam2d vI dI buf')
    in MkTransformer hdp ip op tew (Just embedT)
      (nameBlocks (prefx ++ "_b") 0 blks)
      (nameLayerNorm (prefx ++ "_nf") nf)
      (nameLayer (prefx ++ "_vp") vp)

  layerPrefix _ = "tfm"

  toDoubleLayer {d} (MkTransformer hdp ip op tew _ blks nf vp) =
    MkTransformer hdp ip op (map value tew) Nothing (map toDoubleBlock blks)
                  (toDoubleLayerNorm nf) (toDoubleLayer vp)

  debugApply st inp =
    let (updated, out) = applyGeneric st inp
    in (updated, out, MkDebugEntry (showLayer @{%search} st) [])

  getParamIds {d} (MkTransformer _ _ _ _ _ blks nf vp) =
    concatMap getBlockParamIds (toList blks) ++
    getLayerNormParamIds nf ++ getParamIds vp


----------------------------------------------------------------------
-- Constructors
----------------------------------------------------------------------

export
mkTransformer : {seqLen, dModel, numHeads, headDim, numBlocks, vocabSize : Nat} ->
                {auto prf : dModel = numHeads * headDim} ->
                (Num ty, FromDouble ty) =>
                IO (TransformerState seqLen dModel numHeads headDim numBlocks vocabSize
                                     seqLen (seqLen * vocabSize) ty)
mkTransformer {prf} = do
  -- Embedding weights: small random init (same as nn.Embedding)
  embedW <- traverse (\_ => map fromDouble (pure 0.0))
                     (the (Vector (vocabSize * dModel) ty) zeros)
  blks <- mkBlocks numBlocks
  nf  <- mkLayerNorm {dim=dModel}
  vp  <- mkLinear {i=dModel, o=vocabSize}
  pure $ MkTransformer prf Refl Refl embedW Nothing blks nf vp
  where
    mkBlocks : (k : Nat) -> IO (Vect k (BlockState dModel numHeads headDim ty))
    mkBlocks Z = pure []
    mkBlocks (S k) = [| mkBlock :: mkBlocks k |]

export
transformerLayer : {seqLen, dModel, numHeads, headDim, numBlocks, vocabSize : Nat} ->
                   {auto prf : dModel = numHeads * headDim} ->
                   (Num ty, FromDouble ty) =>
                   IO (AnyLayer seqLen (seqLen * vocabSize) ty)
transformerLayer = map (MkAnyLayer (TransformerState seqLen dModel numHeads headDim numBlocks vocabSize)) mkTransformer


----------------------------------------------------------------------
-- Batched Forward (for epochNativeTensorBatch)
----------------------------------------------------------------------

||| Forward one block on batched data.
||| Projections/FF/norms batched as [B*seqLen, dim].
||| Batched attention: all sequences processed in parallel via 3D ops.
batchBlockForward : {d : Device} -> {seqLen, dModel, numHeads, headDim : Nat} ->
                    BlockState dModel numHeads headDim (Variable d) ->
                    AnyPtr -> Int -> Int -> Int -> Int -> AnyPtr
batchBlockForward blk h bsI sI dI hdI =
  let Just n1g = extractGammaTensor (norm1 blk)   | Nothing => idris_crash "bblock: n1g"
      Just n1b = extractBetaTensor (norm1 blk)    | Nothing => idris_crash "bblock: n1b"
      Just n2g = extractGammaTensor (norm2 blk)   | Nothing => idris_crash "bblock: n2g"
      Just n2b = extractBetaTensor (norm2 blk)    | Nothing => idris_crash "bblock: n2b"
      Just f1W = extractWeightTensor (ff1 blk)    | Nothing => idris_crash "bblock: ff1"
      Just f2W = extractWeightTensor (ff2 blk)    | Nothing => idris_crash "bblock: ff2"
  in let batchSize = bsI `div` sI
         -- Pre-LN: [B*seqLen, dModel]
         normed1 = prim__layerNorm2d h n1g n1b 1.0e-5
         -- Reshape to 3D: [B, seqLen, dModel]
         normed3d = prim__reshape3d normed1 batchSize sI dI
         -- Causal mask: [seqLen, seqLen] → [B, seqLen, seqLen]
         mask3d = prim__expandMask (prim__causalMask sI) batchSize
         -- Batched multi-head attention (per-head loop, batched over B)
         scale = 1.0 / sqrt (cast {to=Double} headDim)
         attnOut3d = batchedHeadLoop (queryWs blk) (keyWs blk) (valueWs blk) (outProjWs blk)
                       normed3d mask3d batchSize sI dI hdI scale Nothing
         -- Reshape back to [B*seqLen, dModel]
         attnOut = prim__reshape2d attnOut3d bsI dI
         -- Residual 1
         h1 = tensorAdd attnOut h
         -- Pre-LN Feedforward: [B*seqLen, dModel]
         normed2 = prim__layerNorm2d h1 n2g n2b 1.0e-5
         ffHidden = prim__clampMin (prim__mm normed2 (prim__transpose2d f1W)) 0.0
         ffOut = prim__mm ffHidden (prim__transpose2d f2W)
     in tensorAdd ffOut h1
  where
    batchedHeadLoop : Vect nh (LinearState dModel headDim (Variable d)) ->
                      Vect nh (LinearState dModel headDim (Variable d)) ->
                      Vect nh (LinearState dModel headDim (Variable d)) ->
                      Vect nh (LinearState headDim dModel (Variable d)) ->
                      AnyPtr -> AnyPtr -> Int -> Int -> Int -> Int -> Double ->
                      Maybe AnyPtr -> AnyPtr
    batchedHeadLoop [] [] [] [] _ _ _ _ _ _ _ (Just acc) = acc
    batchedHeadLoop [] [] [] [] normed _ _ _ _ _ _ Nothing = normed
    batchedHeadLoop (q :: qs) (k :: ks) (v :: vs) (op :: ops) normed mask bsz sLen dMd hdDim sc acc =
      case (extractWeightTensor q, extractWeightTensor k, extractWeightTensor v, extractWeightTensor op) of
        (Just qW, Just kW, Just vW, Just opW) =>
          -- Q/K/V projections: [B,sLen,dModel] × [dModel,hdDim] → [B,sLen,hdDim]
          let qi = prim__bmm normed (prim__transpose2d qW)
              ki = prim__bmm normed (prim__transpose2d kW)
              vi = prim__bmm normed (prim__transpose2d vW)
              -- Scaled dot-product: [B,sLen,hdDim] × [B,hdDim,sLen] → [B,sLen,sLen]
              kiT = prim__transposeLast2 ki
              scores = prim__mulScalar (prim__bmm3x3 qi kiT) sc
              -- Causal mask + softmax: [B,sLen,sLen]
              masked = prim__maskedFill scores mask (-1.0e20)
              attn = prim__softmax3d masked
              -- Attention @ V: [B,sLen,sLen] × [B,sLen,hdDim] → [B,sLen,hdDim]
              headOut = prim__bmm3x3 attn vi
              -- Output projection: [B,sLen,hdDim] × [hdDim,dModel] → [B,sLen,dModel]
              proj = prim__bmm headOut (prim__transpose2d opW)
              acc' = case acc of
                Nothing => proj
                Just prev => tensorAdd prev proj
          in batchedHeadLoop qs ks vs ops normed mask bsz sLen dMd hdDim sc (Just acc')
        _ => idris_crash "Transformer: head weight not initialized"

||| Forward B sequences through the transformer in a single call.
export
transformerForwardBatch :
  {d : Device} -> {seqLen, dModel, numHeads, headDim, numBlocks, vocabSize : Nat} ->
  TransformerState seqLen dModel numHeads headDim numBlocks vocabSize
                   seqLen (seqLen * vocabSize) (Variable d) ->
  List AnyPtr -> Int -> List AnyPtr
transformerForwardBatch st inputs batchSize =
  case st.tokenEmbedTensor of
    Just embedT =>
      let Just vpW = extractWeightTensor (vocabProj st) | Nothing => idris_crash "Transformer: vocabProj"
          Just nfg = extractGammaTensor (normFinal st)  | Nothing => idris_crash "Transformer: nfg"
          Just nfb = extractBetaTensor (normFinal st)   | Nothing => idris_crash "Transformer: nfb"
      in
      let sI = cast {to=Int} seqLen
          dI = cast {to=Int} dModel
          vI = cast {to=Int} vocabSize
          hdI = cast {to=Int} headDim
          bsI = batchSize * sI

          catted = catAll inputs
          -- Embedding lookup: [B*seqLen] indices -> [B*seqLen * dModel] -> [B*seqLen, dModel]
          embFlat = prim__embedding embedT catted bsI dI
          embedded = prim__reshape2d embFlat bsI dI

          peBuf = prim__allocDoubles (bsI * dI)
          peBuf' = writePEBatch peBuf 0 0 bsI dI sI
          peT = prim__createState2d bsI dI peBuf'
          h0 = tensorAdd embedded peT

          -- Fold through blocks (batched)
          hN = foldl (\h, blk => batchBlockForward {seqLen} {dModel} {numHeads} {headDim} blk h bsI sI dI hdI) h0 (blocks st)

          normedFinal = prim__layerNorm2d hN nfg nfb 1.0e-5
          outBatch = prim__mm normedFinal (prim__transpose2d vpW)
          outFlat = prim__narrow outBatch 0 0 (bsI * vI)
      in splitOutputs outFlat 0 batchSize (sI * vI)
    Nothing => idris_crash "Transformer: tokenEmbedTensor not initialized"
  where
    catAll : List AnyPtr -> AnyPtr
    catAll [] = idris_crash "catAll: empty"
    catAll [x] = x
    catAll (x :: y :: rest) = catAll (prim__cat2 x y :: rest)

    writePEBatch : AnyPtr -> Int -> Int -> Int -> Int -> Int -> AnyPtr
    writePEBatch buf pos dim bsLen dMod sLen =
      if pos >= bsLen then buf
      else if dim >= dMod then assert_total $ writePEBatch buf (pos + 1) 0 bsLen dMod sLen
      else let origPos = pos `mod` sLen
               val = posEncVal dModel (cast origPos) (cast dim)
               buf' = prim__setDouble buf (pos * dMod + dim) val
           in assert_total $ writePEBatch buf' pos (dim + 1) bsLen dMod sLen

    splitOutputs : AnyPtr -> Int -> Int -> Int -> List AnyPtr
    splitOutputs flat idx bTotal chunkSize =
      if idx >= bTotal then []
      else let chunk = prim__narrow flat 0 (idx * chunkSize) chunkSize
           in assert_total $ chunk :: splitOutputs flat (idx + 1) bTotal chunkSize
