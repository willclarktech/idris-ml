-- | Multi-Head Transformer with Pre-LN, Learned Embeddings, Sinusoidal PE
-- |
-- | Standard transformer block: learned token embedding, sinusoidal
-- | positional encoding, multi-head causal self-attention (per-head
-- | weights, sum-not-concat), layer normalization, feedforward.
-- |
-- | Input: seqLen * vocabSize (one-hot tokens)
-- | Output: seqLen * vocabSize (per-position logits)

module Layer.MultiHeadTransformer

import Data.Vect
import Decidable.Equality

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
-- State
----------------------------------------------------------------------

||| Multi-head Transformer block.
||| Type parameters: seqLen, dModel, numHeads, headDim, vocabSize.
||| Proof: dModel = numHeads * headDim (compile-time checked).
public export
record MHTransformerState
    (seqLen : Nat) (dModel : Nat) (numHeads : Nat) (headDim : Nat)
    (vocabSize : Nat) (inputSize : Nat) (outputSize : Nat) (ty : Type) where
  constructor MkMHTransformer
  -- Type safety proofs (erased at runtime)
  0 headDimPrf  : dModel = numHeads * headDim
  0 inputPrf    : inputSize = seqLen * vocabSize
  0 outputPrf   : outputSize = seqLen * vocabSize
  -- Learned token embedding [dModel, vocabSize]
  tokenEmbed    : LinearState vocabSize dModel ty
  -- Per-head Q/K/V projections [headDim, dModel]
  queryWs       : Vect numHeads (LinearState dModel headDim ty)
  keyWs         : Vect numHeads (LinearState dModel headDim ty)
  valueWs       : Vect numHeads (LinearState dModel headDim ty)
  -- Per-head output projections [dModel, headDim]
  outProjWs     : Vect numHeads (LinearState headDim dModel ty)
  -- Layer norms
  norm1         : LayerNormState dModel ty
  norm2         : LayerNormState dModel ty
  normFinal     : LayerNormState dModel ty
  -- FFN
  ff1           : LinearState dModel (4 * dModel) ty
  ff2           : LinearState (4 * dModel) dModel ty
  -- Output head
  vocabProj     : LinearState dModel vocabSize ty


----------------------------------------------------------------------
-- Sinusoidal Positional Encoding
----------------------------------------------------------------------

||| Sinusoidal positional encoding value.
posEncVal : Nat -> Nat -> Nat -> Double
posEncVal dModel pos dim =
  let p = cast {to=Double} pos
      i = cast {to=Double} (div dim 2)
      dm = cast {to=Double} dModel
      angle = p / pow 10000.0 (2.0 * i / dm)
  in if modNatNZ dim 2 ItIsSucc == 0 then sin angle else cos angle


----------------------------------------------------------------------
-- Helpers
----------------------------------------------------------------------

||| Per-head attention loop for pure Idris path.
headLoopGeneric : (FromDouble ty, Floating ty, Fractional ty, Neg ty, Ord ty, Num ty) =>
                  {seqLen, dModel, headDim : Nat} ->
                  Vect k (LinearState dModel headDim ty) ->
                  Vect k (LinearState dModel headDim ty) ->
                  Vect k (LinearState dModel headDim ty) ->
                  Vect k (LinearState headDim dModel ty) ->
                  Matrix seqLen dModel ty -> Matrix seqLen dModel ty
headLoopGeneric [] [] [] [] normed = scaleMatrix (fromDouble 0.0) normed  -- zeros
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
-- LayerLike Instance
----------------------------------------------------------------------

-- applyVar uses idris_crash for unreachable fallbacks
%default partial
export
{seqLen : Nat} -> {dModel : Nat} -> {numHeads : Nat} -> {headDim : Nat} ->
{vocabSize : Nat} -> LayerLike (MHTransformerState seqLen dModel numHeads headDim vocabSize) where

  -- Pure Idris path (Double-level evaluation)
  applyGeneric {i} {o} st xs =
    let -- Reshape input: [seqLen * vocabSize] -> [seqLen, vocabSize]
        input = reshapeToMatrix {m=seqLen, n=vocabSize} (rewrite sym st.inputPrf in xs)
        -- Token embedding: [seqLen, vocabSize] @ [vocabSize, dModel]^T -> [seqLen, dModel]
        embedW = weights (tokenEmbed st)
        embedded = matrixMultiply input (transpose embedW)  -- [seqLen, dModel]
        -- Add sinusoidal positional encoding
        posEnc = VTensor $ map (\pi =>
          VTensor $ map (\di => STensor (fromDouble (posEncVal dModel (finToNat pi) (finToNat di))))
                        (Data.Vect.Fin.range {len=dModel}))
                  (Data.Vect.Fin.range {len=seqLen})
        h0 = embedded + posEnc

        -- Pre-LN Multi-Head Attention
        normed1 = layerNormMatrix h0 (gamma (norm1 st)) (beta (norm1 st)) (fromDouble 1.0e-5)
        -- Per-head attention
        attnOut = headLoopGeneric (queryWs st) (keyWs st) (valueWs st) (outProjWs st) normed1
        -- Residual 1
        h1 = attnOut + h0

        -- Pre-LN Feedforward
        normed2 = layerNormMatrix h1 (gamma (norm2 st)) (beta (norm2 st)) (fromDouble 1.0e-5)
        ffHidden = clampMinTensor (fromDouble 0.0) (matrixMultiply normed2 (transpose (weights (ff1 st))))
        ffOut = matrixMultiply ffHidden (transpose (weights (ff2 st)))
        -- Residual 2
        h2 = ffOut + h1

        -- Final LayerNorm + output projection
        normedFinal = layerNormMatrix h2 (gamma (normFinal st)) (beta (normFinal st)) (fromDouble 1.0e-5)
        output = matrixMultiply normedFinal (transpose (weights (vocabProj st)))

    in let result = flattenMatrix output
       in (st, replace {p = \x => Vector x ty} (sym st.outputPrf) result)

  applyVar {i} {o} st xs =
    case extractWeightTensor (tokenEmbed st) of
      Just embedW =>
        let Just f1W = extractWeightTensor (ff1 st)   | Nothing => idris_crash "MHTransformer: ff1 not initialized"
            Just f2W = extractWeightTensor (ff2 st)   | Nothing => idris_crash "MHTransformer: ff2 not initialized"
            Just vpW = extractWeightTensor (vocabProj st) | Nothing => idris_crash "MHTransformer: vocabProj not initialized"
            Just n1g = extractGammaTensor (norm1 st)   | Nothing => idris_crash "MHTransformer: norm1 gamma not initialized"
            Just n1b = extractBetaTensor (norm1 st)    | Nothing => idris_crash "MHTransformer: norm1 beta not initialized"
            Just n2g = extractGammaTensor (norm2 st)   | Nothing => idris_crash "MHTransformer: norm2 gamma not initialized"
            Just n2b = extractBetaTensor (norm2 st)    | Nothing => idris_crash "MHTransformer: norm2 beta not initialized"
            Just nfg = extractGammaTensor (normFinal st) | Nothing => idris_crash "MHTransformer: normFinal gamma not initialized"
            Just nfb = extractBetaTensor (normFinal st)  | Nothing => idris_crash "MHTransformer: normFinal beta not initialized"
        in
        let sI = cast {to=Int} seqLen
            dI = cast {to=Int} dModel
            vI = cast {to=Int} vocabSize
            hdI = cast {to=Int} headDim

            -- Pack input: Vector (seqLen*vocabSize) -> [seqLen, vocabSize]
            (VTensor xElems) = xs
            inputFlat = vecStackTensor xElems
            inputMat = prim__reshape2d inputFlat sI vI

            -- Token embedding: input @ embedW^T -> [seqLen, dModel]
            embedded = prim__mm inputMat (prim__transpose2d embedW)

            -- Add sinusoidal positional encoding
            peBuf = prim__allocDoubles (sI * dI)
            peBuf' = writePE peBuf 0 0 sI dI
            peT = prim__createState2d sI dI peBuf'
            h0 = tensorAdd embedded peT

            -- Pre-LN Multi-Head Attention
            normed1 = prim__layerNorm2d h0 n1g n1b 1.0e-5

            -- Per-head attention, summed
            attnOut = headLoop (queryWs st) (keyWs st) (valueWs st) (outProjWs st) normed1 sI hdI Nothing

            -- Residual 1
            h1 = tensorAdd attnOut h0

            -- Pre-LN Feedforward
            normed2 = prim__layerNorm2d h1 n2g n2b 1.0e-5
            ffHidden = prim__clampMin (prim__mm normed2 (prim__transpose2d f1W)) 0.0
            ffOut = prim__mm ffHidden (prim__transpose2d f2W)

            -- Residual 2
            h2 = tensorAdd ffOut h1

            -- Final LayerNorm + output projection
            normedFinal = prim__layerNorm2d h2 nfg nfb 1.0e-5
            outT = prim__mm normedFinal (prim__transpose2d vpW)

            -- Flatten to 1D
            flat1d = prim__narrow outT 0 0 (sI * vI)
            output = VTensor (tensorToScalars flat1d 0 o)
        in (st, output)
      Nothing => idris_crash "MHTransformer: tokenEmbed not initialized"
    where
      writePE : AnyPtr -> Int -> Int -> Int -> Int -> AnyPtr
      writePE buf pos dim sLen dMod =
        if pos >= sLen then buf
        else if dim >= dMod then writePE buf (pos + 1) 0 sLen dMod
        else let val = posEncVal dModel (cast pos) (cast dim)
                 buf' = prim__setDouble buf (pos * dMod + dim) val
             in writePE buf' pos (dim + 1) sLen dMod

      headLoop : Vect k (LinearState dModel headDim Variable) ->
                 Vect k (LinearState dModel headDim Variable) ->
                 Vect k (LinearState dModel headDim Variable) ->
                 Vect k (LinearState headDim dModel Variable) ->
                 AnyPtr -> Int -> Int -> Maybe AnyPtr -> AnyPtr
      headLoop [] [] [] [] _ _ _ (Just acc) = acc
      headLoop [] [] [] [] normed sI hdI Nothing = normed  -- shouldn't happen
      headLoop (q :: qs) (k :: ks) (v :: vs) (op :: ops) normed sI hdI acc =
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
            in headLoop qs ks vs ops normed sI hdI (Just acc')
          _ => idris_crash "MHTransformer: head weight not initialized"

  emapLayer f (MkMHTransformer hdp ip op te qs ks vs ops n1 n2 nf f1 f2 vp) =
    MkMHTransformer hdp ip op
      (emapLayer f te)
      (map (emapLayer f) qs) (map (emapLayer f) ks)
      (map (emapLayer f) vs) (map (emapLayer f) ops)
      (emapLayerNorm f n1) (emapLayerNorm f n2) (emapLayerNorm f nf)
      (emapLayer f f1) (emapLayer f f2)
      (emapLayer f vp)

  showLayer _ = "MHTransformer<" ++ show seqLen ++ "x" ++ show dModel
    ++ " h=" ++ show numHeads ++ " v=" ++ show vocabSize ++ ">"

  nameLayer prefx (MkMHTransformer hdp ip op te qs ks vs ops n1 n2 nf f1 f2 vp) =
    MkMHTransformer hdp ip op
      (nameLayer (prefx ++ "_embed") te)
      (nameHeads (prefx ++ "_q") 0 qs) (nameHeads (prefx ++ "_k") 0 ks)
      (nameHeads (prefx ++ "_v") 0 vs) (nameHeads (prefx ++ "_o") 0 ops)
      (nameLayerNorm (prefx ++ "_n1") n1)
      (nameLayerNorm (prefx ++ "_n2") n2)
      (nameLayerNorm (prefx ++ "_nf") nf)
      (nameLayer (prefx ++ "_ff1") f1) (nameLayer (prefx ++ "_ff2") f2)
      (nameLayer (prefx ++ "_vp") vp)
    where
      nameHeads : {a, b : Nat} -> String -> Nat -> Vect k (LinearState a b Variable) -> Vect k (LinearState a b Variable)
      nameHeads _ _ [] = []
      nameHeads pfx idx (h :: hs) = nameLayer (pfx ++ show idx) h :: nameHeads pfx (S idx) hs

  layerPrefix _ = "mht"

  toDoubleLayer (MkMHTransformer hdp ip op te qs ks vs ops n1 n2 nf f1 f2 vp) =
    MkMHTransformer hdp ip op
      (toDoubleLayer te)
      (map toDoubleLayer qs) (map toDoubleLayer ks)
      (map toDoubleLayer vs) (map toDoubleLayer ops)
      (toDoubleLayerNorm n1) (toDoubleLayerNorm n2) (toDoubleLayerNorm nf)
      (toDoubleLayer f1) (toDoubleLayer f2)
      (toDoubleLayer vp)

  debugApply st inp =
    let (updated, out) = applyGeneric st inp
    in (updated, out, MkDebugEntry (showLayer @{%search} st) [])

  getParamIds (MkMHTransformer _ _ _ te qs ks vs ops n1 n2 nf f1 f2 vp) =
    getParamIds te ++
    concatMap getParamIds (toList qs) ++ concatMap getParamIds (toList ks) ++
    concatMap getParamIds (toList vs) ++ concatMap getParamIds (toList ops) ++
    getLayerNormParamIds n1 ++ getLayerNormParamIds n2 ++ getLayerNormParamIds nf ++
    getParamIds f1 ++ getParamIds f2 ++ getParamIds vp


----------------------------------------------------------------------
-- Constructors
----------------------------------------------------------------------

export
mkMHTransformer : {seqLen, dModel, numHeads, headDim, vocabSize : Nat} ->
                  {auto prf : dModel = numHeads * headDim} ->
                  (Num ty, FromDouble ty) =>
                  IO (MHTransformerState seqLen dModel numHeads headDim vocabSize
                                        (seqLen * vocabSize) (seqLen * vocabSize) ty)
mkMHTransformer {prf} = do
  te  <- mkLinear {i=vocabSize, o=dModel}
  qs  <- mkLinears numHeads {i=dModel, o=headDim}
  ks  <- mkLinears numHeads {i=dModel, o=headDim}
  vs  <- mkLinears numHeads {i=dModel, o=headDim}
  ops <- mkLinears numHeads {i=headDim, o=dModel}
  n1  <- mkLayerNorm {dim=dModel}
  n2  <- mkLayerNorm {dim=dModel}
  nf  <- mkLayerNorm {dim=dModel}
  f1  <- mkLinear {i=dModel, o=4*dModel}
  f2  <- mkLinear {i=4*dModel, o=dModel}
  vp  <- mkLinear {i=dModel, o=vocabSize}
  pure $ MkMHTransformer prf Refl Refl te qs ks vs ops n1 n2 nf f1 f2 vp
  where
    mkLinears : (k : Nat) -> {i, o : Nat} -> IO (Vect k (LinearState i o ty))
    mkLinears Z = pure []
    mkLinears (S k) = [| mkLinear :: mkLinears k |]

export
mhTransformerLayer : {seqLen, dModel, numHeads, headDim, vocabSize : Nat} ->
                     {auto prf : dModel = numHeads * headDim} ->
                     (Num ty, FromDouble ty) =>
                     IO (AnyLayer (seqLen * vocabSize) (seqLen * vocabSize) ty)
mhTransformerLayer = map (MkAnyLayer (MHTransformerState seqLen dModel numHeads headDim vocabSize)) mkMHTransformer
