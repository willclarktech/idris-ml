||| `Attention` — multi-head causal self-attention, a composable building
||| block for transformer models. Bias-free by construction (the projections
||| are pure matmuls — the legacy carried unused `Linear` biases here, a
||| wrinkle this drops): each head holds raw weight tensors for Q/K/V and an
||| output projection. `attentionForward` maps `[seqLen, dModel] → [seqLen,
||| dModel]` for any `seqLen` (the causal mask is built per call from the
||| input length). Used inside `Nn.Transformer`'s block; `Params` exposes
||| the per-head weights so it composes.
module Nn.Attention

import Data.Vect

import Executor
import Tensor
import Nn.Init
import Nn.Module

%default total

||| Multi-head attention: `numHeads` heads, each projecting `dModel → headDim`
||| (Q/K/V) and `headDim → dModel` (output). All weights `WithGrad`.
public export
record Attention (dModel : Nat) (numHeads : Nat) (headDim : Nat) (0 ex : Executor) (0 dt : DType) where
  constructor MkAttention
  queryWs   : Vect numHeads (TMat headDim dModel ex dt WithGrad)
  keyWs     : Vect numHeads (TMat headDim dModel ex dt WithGrad)
  valueWs   : Vect numHeads (TMat headDim dModel ex dt WithGrad)
  outProjWs : Vect numHeads (TMat dModel headDim ex dt WithGrad)

||| The per-head weights as a flat param list. Attention's three Nat
||| params (dModel/numHeads/headDim) don't fit `Params`' 2-Nat (i,o) kind,
||| so this is a plain function the enclosing block splices into its own
||| `Params`.
export
attentionParams : {0 dModel, numHeads, headDim : Nat} -> {0 ex : Executor} -> {0 dt : DType} ->
                  Attention dModel numHeads headDim ex dt -> List SomeParam
attentionParams (MkAttention qs ks vs ops) =
  map toParam (toList qs) ++ map toParam (toList ks)
    ++ map toParam (toList vs) ++ map toParam (toList ops)

-- Strict upper triangle = 1.0 (future positions to mask); calloc zeroes
-- the rest. (Int-indexed recursion → partial.)
partial
writeCausalMask : AnyPtr -> Int -> Int -> Int -> AnyPtr
writeCausalMask buf i j n =
  if i >= n then buf
  else if j >= n then writeCausalMask buf (i + 1) (i + 2) n
  else writeCausalMask (prim__setDouble buf (i * n + j) 1.0) i (j + 1) n

partial
buildCausalMask : {0 ex : Executor} -> Backend ex dt => (seqLen : Nat) -> AnyPtr
buildCausalMask seqLen =
  let sI = cast {to=Int} seqLen
      buf = writeCausalMask (prim__allocDoubles (sI * sI)) 0 1 sI
  in dtCreateState2d {ex} {t=dt} sI sI buf (deviceStreamTag {ex})

-- Per-head scaled-dot-product attention, accumulated over heads.
runHeads : {0 ex : Executor} -> UserExecutorTraining ex => {k : Nat} ->
           Vect k (TMat hd dM ex dt WithGrad) -> Vect k (TMat hd dM ex dt WithGrad) ->
           Vect k (TMat hd dM ex dt WithGrad) -> Vect k (TMat dM hd ex dt WithGrad) ->
           (normed, mask : AnyPtr) -> (scale : Double) -> Maybe AnyPtr -> AnyPtr
runHeads [] [] [] [] normed _ _ Nothing = normed
runHeads [] [] [] [] _ _ _ (Just acc) = acc
runHeads (q :: qs) (k :: ks) (v :: vs) (op :: ops) normed mask scale acc =
  let qi = primMm {ex} normed (primTranspose2d {ex} q.tensorPtr)
      ki = primMm {ex} normed (primTranspose2d {ex} k.tensorPtr)
      vi = primMm {ex} normed (primTranspose2d {ex} v.tensorPtr)
      scores = primMulScalar {ex} (primMm {ex} qi (primTranspose2d {ex} ki)) scale
      attn = primSoftmax2d {ex} (primMaskedFill {ex} scores mask (-1.0e20))
      proj = primMm {ex} (primMm {ex} attn vi) (primTranspose2d {ex} op.tensorPtr)
      acc' = maybe proj (\prev => primAdd {ex} prev proj) acc
  in runHeads qs ks vs ops normed mask scale (Just acc')

||| Causal self-attention forward on a `[seqLen, dModel]` sequence.
export partial
attentionForward : {0 ex : Executor} -> Backend ex dt => {0 g : GradMode} ->
                   {dModel, numHeads, headDim, seqLen : Nat} ->
                   Attention dModel numHeads headDim ex dt ->
                   Tensor [seqLen, dModel] ex dt g -> IO (Tensor [seqLen, dModel] ex dt g)
attentionForward {headDim} (MkAttention qs ks vs ops) input = ioRerun (\_ =>
  let scale = 1.0 / sqrt (cast {to=Double} headDim)
      out = runHeads {ex} qs ks vs ops input.tensorPtr (buildCausalMask {ex} {dt} seqLen) scale Nothing
  in MkTensor out Nothing)

-- Build `numHeads` registered weight tensors named `<kind>_<j>.weight`.
mkHeads : {0 ex : Executor} -> Backend ex dt => {a, b : Nat} ->
          String -> (count : Nat) -> Double -> Init (Vect count (TMat a b ex dt WithGrad))
mkHeads _ Z _ = pure []
mkHeads kind (S c) std = do
  name <- freshChild kind
  w <- liftIO $ tparam2dNormal {ex} {dt} {o=a} {i=b} (name ++ ".weight") 0.0 std
  rest <- mkHeads kind c std
  pure (w :: rest)

||| Construct multi-head attention inside an `Init` derivation. Per-head
||| Q/K/V/output projections ~ N(0, 1/√fan_in); registers
||| `<scope>.{query,key,value,out_proj}_<j>.weight`.
export
attention : {0 ex : Executor} -> Backend ex dt => {dModel, numHeads, headDim : Nat} ->
            Init (Attention dModel numHeads headDim ex dt)
attention = do
  let projStd = 1.0 / sqrt (cast {to=Double} dModel)
      outStd  = 1.0 / sqrt (cast {to=Double} headDim)
  qs <- mkHeads {a=headDim} {b=dModel} "query"    numHeads projStd
  ks <- mkHeads {a=headDim} {b=dModel} "key"      numHeads projStd
  vs <- mkHeads {a=headDim} {b=dModel} "value"    numHeads projStd
  ops <- mkHeads {a=dModel} {b=headDim} "out_proj" numHeads outStd
  pure (MkAttention qs ks vs ops)
