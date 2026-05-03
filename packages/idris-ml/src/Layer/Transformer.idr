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
                    (0 d : Device) (0 dt : DType) (0 g : GradMode) where
  constructor MkBlock
  queryWs   : Vect numHeads (LinearState dModel headDim d dt g)
  keyWs     : Vect numHeads (LinearState dModel headDim d dt g)
  valueWs   : Vect numHeads (LinearState dModel headDim d dt g)
  outProjWs : Vect numHeads (LinearState headDim dModel d dt g)
  norm1     : LayerNormState dModel dModel d dt g
  norm2     : LayerNormState dModel dModel d dt g
  ff1       : LinearState dModel (4 * dModel) d dt g
  ff2       : LinearState (4 * dModel) dModel d dt g


----------------------------------------------------------------------
-- TransformerState
----------------------------------------------------------------------

public export
data TransformerState :
  (seqLen : Nat) -> (dModel : Nat) -> (numHeads : Nat) ->
  (headDim : Nat) -> (numBlocks : Nat) -> (vocabSize : Nat) ->
  Nat -> Nat -> (0 _ : Device) -> (0 _ : DType) -> (0 _ : GradMode) -> Type
  where
  MkTransformer :
    {0 prf : dModel = numHeads * headDim} ->
    TMat vocabSize dModel d dt g ->                        -- token embedding
    Vect numBlocks (BlockState dModel numHeads headDim d dt g) ->
    LayerNormState dModel dModel d dt g ->                -- final norm
    LinearState dModel vocabSize d dt g ->                -- output projection
    -- Cached positional encoding `[seqLen, dModel]` — sinusoidal, shape-only,
    -- non-learnable. Built once at construction; forward passes reuse it
    -- instead of recomputing via `writePE` (which was the 1B-Nat-ops/epoch
    -- bottleneck per the 2026-05-14 profile diagnostic in perf-changes.md).
    TMat seqLen dModel d dt g ->
    -- Cached causal mask `[seqLen, seqLen]` — depends only on `seqLen` (fixed
    -- at construction). Reused across blocks and forwards; for the batched
    -- path `applyTransformerBatch` calls `primExpandMask {d}` once per batch
    -- rather than once per block.
    TMat seqLen seqLen d dt g ->
    TransformerState seqLen dModel numHeads headDim numBlocks vocabSize
                       seqLen (seqLen * vocabSize) d dt g


----------------------------------------------------------------------
-- Sinusoidal Positional Encoding (matches V1)
----------------------------------------------------------------------

||| Cast `Nat` args to `Int` internally before doing `div`/`mod`. The stdlib
||| `Data.Nat.divNat` / `modNatNZ` compile to recursive Peano walks
||| (`Data.Nat.lte` / `divC-39` / `modC-39`) — even with `Nat` stored as
||| `Integer` at runtime, the pattern-match-on-`S k`-defined functions
||| still recursively decrement. Profile showed 3.9B such operations per
||| epoch on GptLarge just from this call site
||| (`docs/develop/perf-changes.md` 2026-05-14 entry). Plain `Int div`/`Int
||| mod` compile to single CPU instructions. We keep the `Nat` interface
||| (call sites pass shape-derived `Nat` values directly) and pay the
||| cast once.
posEncVal : Nat -> Nat -> Nat -> Double
posEncVal dModel pos dim =
  let dimI = the Int (cast dim)
      p = cast {to=Double} pos
      i = cast {to=Double} (dimI `div` 2)
      dm = cast {to=Double} dModel
      angle = p / pow 10000.0 (2.0 * i / dm)
  in if (dimI `mod` 2) == 0 then sin angle else cos angle

writePE : (dModel : Nat) -> AnyPtr -> Int -> Int -> Int -> Int -> AnyPtr
writePE dModel buf pos dim sLen dMod =
  if pos >= sLen then buf
  else if dim >= dMod then writePE dModel buf (pos + 1) 0 sLen dMod
  else let val = posEncVal dModel (cast pos) (cast dim)
           buf' = prim__setDouble buf (pos * dMod + dim) val
       in writePE dModel buf' pos (dim + 1) sLen dMod

-- Fill the strict upper triangle of an n×n buffer with 1.0; rely on
-- prim__allocDoubles (calloc-backed) for the zero baseline.
writeCausalMask : AnyPtr -> Int -> Int -> Int -> AnyPtr
writeCausalMask buf i j n =
  if i >= n then buf
  else if j >= n then writeCausalMask buf (i + 1) (i + 2) n
  else let buf' = prim__setDouble buf (i * n + j) 1.0
       in writeCausalMask buf' i (j + 1) n


----------------------------------------------------------------------
-- Per-block forward (single sequence: [seqLen, dModel] tensor handle)
----------------------------------------------------------------------

%default partial

-- Recursive head loop — accumulates per-head projections over numHeads.
runHeadAttn : {0 d : Device} -> UserDeviceTape d => {dModel, headDim : Nat} ->
              Vect k (LinearState dModel headDim d dt g) ->
              Vect k (LinearState dModel headDim d dt g) ->
              Vect k (LinearState dModel headDim d dt g) ->
              Vect k (LinearState headDim dModel d dt g) ->
              AnyPtr -> AnyPtr -> Int -> Int -> Maybe AnyPtr -> AnyPtr
runHeadAttn [] [] [] [] _ _ _ _ (Just acc) = acc
runHeadAttn [] [] [] [] normed _ _ _ Nothing = normed
runHeadAttn (q :: qs) (k :: ks) (v :: vs) (op :: ops) normed mask sI hdI acc =
  let qW = q.weightT.tensorPtr
      kW = k.weightT.tensorPtr
      vW = v.weightT.tensorPtr
      opW = op.weightT.tensorPtr
      qi = primMm {d} normed (primTranspose2d {d} qW)
      ki = primMm {d} normed (primTranspose2d {d} kW)
      vi = primMm {d} normed (primTranspose2d {d} vW)
      scale = 1.0 / sqrt (cast {to=Double} hdI)
      scores = primMulScalar {d} (primMm {d} qi (primTranspose2d {d} ki)) scale
      masked = primMaskedFill {d} scores mask (-1.0e20)
      attn = primSoftmax2d {d} masked
      headOut = primMm {d} attn vi
      proj = primMm {d} headOut (primTranspose2d {d} opW)
      acc' = case acc of
        Nothing => proj
        Just prev => primAdd {d} prev proj
  in runHeadAttn qs ks vs ops normed mask sI hdI (Just acc')

-- Forward one block on `[seqLen, dModel]` tensor handle. The caller passes
-- the cached causal mask AnyPtr (shared across blocks; built once on
-- `TransformerState`).
blockForward : {0 d : Device} -> UserDeviceTape d => {dModel, numHeads, headDim : Nat} ->
                 BlockState dModel numHeads headDim d dt g ->
                 AnyPtr -> AnyPtr -> Int -> Int -> AnyPtr
blockForward (MkBlock qs ks vs ops
                          (MkLayerNorm n1g n1b)
                          (MkLayerNorm n2g n2b)
                          ff1 ff2) h mask sI hdI =
  let f1W = ff1.weightT.tensorPtr
      f2W = ff2.weightT.tensorPtr
      normed1 = primLayerNorm2d {d} h n1g.tensorPtr n1b.tensorPtr 1.0e-5
      attnOut = runHeadAttn qs ks vs ops normed1 mask sI hdI Nothing
      h1 = primAdd {d} attnOut h
      normed2 = primLayerNorm2d {d} h1 n2g.tensorPtr n2b.tensorPtr 1.0e-5
      ffHidden = primClampMin {d} (primMm {d} normed2 (primTranspose2d {d} f1W)) 0.0
      ffOut = primMm {d} ffHidden (primTranspose2d {d} f2W)
  in primAdd {d} ffOut h1

-- Fold over blocks.
foldBlocks : {0 d : Device} -> UserDeviceTape d => {dModel, numHeads, headDim : Nat} ->
               Vect k (BlockState dModel numHeads headDim d dt g) ->
               AnyPtr -> AnyPtr -> Int -> Int -> AnyPtr
foldBlocks [] h _ _ _ = h
foldBlocks (b :: bs) h mask sI hdI =
  foldBlocks bs (blockForward b h mask sI hdI) mask sI hdI


----------------------------------------------------------------------
-- Forward
----------------------------------------------------------------------

export
applyTransformer : {0 d : Device} -> UserDeviceTape d => UserDeviceCore d => RuntimeDType dt => Linked d => Compatible d dt => {seqLen, dModel, numHeads, headDim, numBlocks, vocabSize : Nat} ->
                     TransformerState seqLen dModel numHeads headDim numBlocks
                                       vocabSize seqLen (seqLen * vocabSize) d dt g ->
                     TVec seqLen d dt g ->
                     TVec (seqLen * vocabSize) d dt g
applyTransformer {seqLen} {dModel} {headDim} {vocabSize}
                   (MkTransformer embedW blocks (MkLayerNorm nfg nfb) vocabProj peCached
                                  maskCached) tokens =
  let sI = cast {to=Int} seqLen
      dI = cast {to=Int} dModel
      vI = cast {to=Int} vocabSize
      hdI = cast {to=Int} headDim
      embFlat = primEmbedding {d} embedW.tensorPtr tokens.tensorPtr sI dI
      embedded = primReshape2d {d} embFlat sI dI
      h0 = primAdd {d} embedded peCached.tensorPtr
      hN = foldBlocks blocks h0 maskCached.tensorPtr sI hdI
      normedFinal' = primLayerNorm2d {d} hN nfg.tensorPtr nfb.tensorPtr 1.0e-5
      vpW = vocabProj.weightT.tensorPtr
      outT = primMm {d} normedFinal' (primTranspose2d {d} vpW)
      outFlatPtr = primNarrow {d} outT 0 0 (sI * vI)
  in MkTensor outFlatPtr Nothing


----------------------------------------------------------------------
-- Batched per-block forward (mirrors V1 `batchBlockForward`)
----------------------------------------------------------------------
--
-- Operates on a flat [B*seqLen, dModel] handle. LayerNorm + FFN are
-- shape-agnostic in the leading dim; attention reshapes to [B, seqLen,
-- dModel] for fused 3D ops then reshapes back.

-- Per-head batched accumulator: project Q/K/V via `bmm`, fused
-- `primCrossAttention {d}` (Q·K^T·scale → mask → softmax → ·V), then
-- output projection via `bmm`. Sums per-head contributions.
batchedHeadLoop : {0 d : Device} -> UserDeviceTape d => {dModel, headDim : Nat} ->
                    Vect k (LinearState dModel headDim d dt g) ->
                    Vect k (LinearState dModel headDim d dt g) ->
                    Vect k (LinearState dModel headDim d dt g) ->
                    Vect k (LinearState headDim dModel d dt g) ->
                    AnyPtr -> AnyPtr -> Double -> Maybe AnyPtr -> AnyPtr
batchedHeadLoop [] [] [] [] _ _ _ (Just acc) = acc
batchedHeadLoop [] [] [] [] normed _ _ Nothing = normed
batchedHeadLoop (q :: qs) (k :: ks) (v :: vs) (op :: ops) normed mask sc acc =
  let qW = q.weightT.tensorPtr
      kW = k.weightT.tensorPtr
      vW = v.weightT.tensorPtr
      opW = op.weightT.tensorPtr
      qi = primBmm {d} normed (primTranspose2d {d} qW)
      ki = primBmm {d} normed (primTranspose2d {d} kW)
      vi = primBmm {d} normed (primTranspose2d {d} vW)
      headOut = primCrossAttention {d} qi ki vi mask sc
      proj = primBmm {d} headOut (primTranspose2d {d} opW)
      acc' = case acc of
        Nothing => proj
        Just prev => primAdd {d} prev proj
  in batchedHeadLoop qs ks vs ops normed mask sc (Just acc')

-- Batched per-block forward. The caller passes the 3D causal mask AnyPtr
-- (built once per batch by `applyTransformerBatch` via `primExpandMask {d}`
-- on the cached 2D mask).
batchBlockForward : {0 d : Device} -> UserDeviceTape d => {dModel, numHeads, headDim : Nat} ->
                      BlockState dModel numHeads headDim d dt g ->
                      AnyPtr -> AnyPtr -> Int -> Int -> Int -> AnyPtr
batchBlockForward (MkBlock qs ks vs ops
                                (MkLayerNorm n1g n1b)
                                (MkLayerNorm n2g n2b)
                                ff1 ff2) h mask3d bsI sI dI =
  let f1W = ff1.weightT.tensorPtr
      f2W = ff2.weightT.tensorPtr
      batchSize = bsI `div` sI
      normed1 = primLayerNorm2d {d} h n1g.tensorPtr n1b.tensorPtr 1.0e-5
      normed3d = primReshape3d {d} normed1 batchSize sI dI
      scale = 1.0 / sqrt (cast {to=Double} (dI `div` cast {to=Int} numHeads))
      attnOut3d = batchedHeadLoop qs ks vs ops normed3d mask3d scale Nothing
      attnOut = primReshape2d {d} attnOut3d bsI dI
      h1 = primAdd {d} attnOut h
      normed2 = primLayerNorm2d {d} h1 n2g.tensorPtr n2b.tensorPtr 1.0e-5
      f1Wt = primTranspose2d {d} f1W
      f2Wt = primTranspose2d {d} f2W
      ffOut = primMm {d} (primClampMin {d} (primMm {d} normed2 f1Wt) 0.0) f2Wt
  in primAdd {d} ffOut h1

foldBlocksBatched : {0 d : Device} -> UserDeviceTape d => {dModel, numHeads, headDim : Nat} ->
                      Vect k (BlockState dModel numHeads headDim d dt g) ->
                      AnyPtr -> AnyPtr -> Int -> Int -> Int -> AnyPtr
foldBlocksBatched [] h _ _ _ _ = h
foldBlocksBatched (b :: bs) h mask3d bsI sI dI =
  foldBlocksBatched bs (batchBlockForward b h mask3d bsI sI dI) mask3d bsI sI dI

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
  {0 d : Device} -> UserDeviceTape d =>
  {seqLen, dModel, numHeads, headDim, numBlocks, vocabSize : Nat} ->
  {b : Nat} ->
  TransformerState seqLen dModel numHeads headDim numBlocks vocabSize
                     seqLen (seqLen * vocabSize) d dt g ->
  Tensor [b, seqLen] d dt g ->
  Tensor [b, seqLen * vocabSize] d dt g
applyTransformerBatch {seqLen} {dModel} {headDim} {vocabSize} {b}
                        (MkTransformer embedW blocks (MkLayerNorm nfg nfb) vocabProj peCached
                                       maskCached)
                        tokens =
  let bI = cast {to=Int} b
      bsI = cast {to=Int} (b * seqLen)
      sI = cast {to=Int} seqLen
      dI = cast {to=Int} dModel
      vI = cast {to=Int} vocabSize
      flatTokens = primReshape1d {d} tokens.tensorPtr bsI
      embFlat = primEmbedding {d} embedW.tensorPtr flatTokens bsI dI
      embedded = primReshape2d {d} embFlat bsI dI
      -- Tile cached PE [seqLen, dModel] vertically `b` times to get
      -- [b*seqLen, dModel], then add directly to the flat embedded. One
      -- fused op per backend (`mx::tile` / `at::tile` / manual memcpy)
      -- replaces the earlier reshape3d → add → reshape2d dance, which
      -- regressed mlx perf on small-model shapes — see `perf-changes.md`
      -- 2026-05-15 "tile_2d" entry.
      peTiled = primTile2d {d} peCached.tensorPtr bI 1
      h0 = primAdd {d} embedded peTiled
      -- Expand the cached 2D mask once per batch (depends on `b`, which can
      -- vary between train/eval) and thread the 3D handle through every
      -- block.
      mask3d = primExpandMask {d} maskCached.tensorPtr bI
      hN = foldBlocksBatched blocks h0 mask3d bsI sI dI
      normedFinal' = primLayerNorm2d {d} hN nfg.tensorPtr nfb.tensorPtr 1.0e-5
      vpW = vocabProj.weightT.tensorPtr
      outBatch = primMm {d} normedFinal' (primTranspose2d {d} vpW)
      -- outBatch : [b * seqLen, vocabSize]. Reshape to [b, seqLen * vocabSize].
      outReshaped = primReshape2d {d} outBatch (cast {to=Int} b) (sI * vI)
  in MkTensor outReshaped Nothing


----------------------------------------------------------------------
-- Constructors
----------------------------------------------------------------------

-- Build a Vect of n Linear layers with sequential paramId suffixes.
mkLinearVec : UserDeviceTape d => RuntimeDType dt => Linked d => Compatible d dt => {i, o : Nat} -> (n : Nat) -> String -> IO (Vect n (LinearState i o d dt WithGrad))
mkLinearVec Z _ = pure []
mkLinearVec (S k) pfx = do
  l <- linearLayer {i} {o} (pfx ++ show k)
  rest <- mkLinearVec k pfx
  pure (l :: rest)

-- Build one transformer block.
mkBlock : UserDeviceTape d => RuntimeDType dt => Linked d => Compatible d dt => {dModel, numHeads, headDim : Nat} ->
            (paramPrefix : String) ->
            IO (BlockState dModel numHeads headDim d dt WithGrad)
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

mkBlocks : UserDeviceTape d => RuntimeDType dt => Linked d => Compatible d dt => {dModel, numHeads, headDim : Nat} ->
             (k : Nat) -> (paramPrefix : String) ->
             IO (Vect k (BlockState dModel numHeads headDim d dt WithGrad))
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
  UserDeviceTape d => RuntimeDType dt => Linked d => Compatible d dt =>
  {seqLen, dModel, numHeads, headDim, numBlocks, vocabSize : Nat} ->
  {auto prf : dModel = numHeads * headDim} ->
  (paramPrefix : String) ->
  IO (TransformerState seqLen dModel numHeads headDim numBlocks vocabSize
                         seqLen (seqLen * vocabSize) d dt WithGrad)
transformerLayer {prf} paramPrefix = do
  let n = vocabSize * dModel
  embedVals <- traverse (\_ => xavier uniform vocabSize dModel) (Vect.replicate n ())
  let nI = cast {to=Int} n
      vI = cast {to=Int} vocabSize
      dI = cast {to=Int} dModel
      embBuf = prim__allocDoubles nI
      embBuf' = packDoubles embBuf 0 embedVals
      embName = paramPrefix ++ "_embed"
      embPtr = primParamRegister {d} embName (dtCreateParam2d {d} {t=dt} vI dI embBuf' (deviceStreamTag {d}))
      embTV : TMat vocabSize dModel d dt WithGrad
      embTV = MkTensor embPtr (Just embName)
  blks <- mkBlocks numBlocks (paramPrefix ++ "_b")
  nf <- layerNormLayer {n = dModel} (paramPrefix ++ "_nf")
  vp <- linearLayer {i = dModel} {o = vocabSize} (paramPrefix ++ "_vp")
  -- Build sinusoidal positional encoding once. Forward passes reuse this
  -- cached tensor instead of running writePE every step (which was the
  -- 1M-posEncVal-calls/epoch bottleneck per the 2026-05-14 profile).
  let sI = cast {to=Int} seqLen
      peBuf = prim__allocDoubles (sI * dI)
      peBuf' = writePE dModel peBuf 0 0 sI dI
      peTV : TMat seqLen dModel d dt WithGrad
      peTV = MkTensor (dtCreateState2d {d} {t=dt} sI dI peBuf' (deviceStreamTag {d})) Nothing
      -- Build causal mask once via the same persistent-state path as PE
      -- (routing through `dtCreateState2d {t=dt}    (deviceStreamTag {d})`). `primCausalMask {d}` itself
      -- returns an arena/intermediate tensor whose memory gets clobbered by
      -- `tape_reset` and `free_intermediates` between training steps, so
      -- caching its result would dangle after the first optimizer step.
      maskBufRaw = prim__allocDoubles (sI * sI)
      maskBuf = writeCausalMask maskBufRaw 0 1 sI
      maskTV : TMat seqLen seqLen d dt WithGrad
      maskTV = MkTensor (dtCreateState2d {d} {t=dt} sI sI maskBuf (deviceStreamTag {d})) Nothing
  pure $ MkTransformer {prf} embTV blks nf vp peTV maskTV


----------------------------------------------------------------------
-- Freeze / unfreeze helpers for nested state
----------------------------------------------------------------------

-- Vect of linear states: walk linearly via manual recursion (traverse
-- can't be used because freezeLayer / unfreezeLayer are linear in
-- their argument, not unrestricted).
freezeLinearVec : {i, o : Nat} -> {0 d : Device} -> UserDeviceTape d => {0 g : GradMode} ->
                    Vect k (LinearState i o d dt g) ->
                    IO (Vect k (LinearState i o d dt NoGrad))
freezeLinearVec [] = pure []
freezeLinearVec (l :: ls) = do
  l' <- freezeLayer l
  ls' <- freezeLinearVec ls
  pure (l' :: ls')

unfreezeLinearVec : {i, o : Nat} -> {0 d : Device} -> UserDeviceTape d =>
                      Vect k (LinearState i o d dt NoGrad) ->
                      IO (Vect k (LinearState i o d dt WithGrad))
unfreezeLinearVec [] = pure []
unfreezeLinearVec (l :: ls) = do
  l' <- unfreezeLayer l
  ls' <- unfreezeLinearVec ls
  pure (l' :: ls')

freezeBlock : {dModel, numHeads, headDim : Nat} -> {0 d : Device} -> UserDeviceTape d => {0 g : GradMode} ->
                BlockState dModel numHeads headDim d dt g ->
                IO (BlockState dModel numHeads headDim d dt NoGrad)
freezeBlock (MkBlock qs ks vs ops n1 n2 ff1 ff2) = do
  qs'  <- freezeLinearVec qs
  ks'  <- freezeLinearVec ks
  vs'  <- freezeLinearVec vs
  ops' <- freezeLinearVec ops
  n1'  <- freezeLayer n1
  n2'  <- freezeLayer n2
  ff1' <- freezeLayer ff1
  ff2' <- freezeLayer ff2
  pure (MkBlock qs' ks' vs' ops' n1' n2' ff1' ff2')

unfreezeBlock : {dModel, numHeads, headDim : Nat} -> {0 d : Device} -> UserDeviceTape d =>
                  BlockState dModel numHeads headDim d dt NoGrad ->
                  IO (BlockState dModel numHeads headDim d dt WithGrad)
unfreezeBlock (MkBlock qs ks vs ops n1 n2 ff1 ff2) = do
  qs'  <- unfreezeLinearVec qs
  ks'  <- unfreezeLinearVec ks
  vs'  <- unfreezeLinearVec vs
  ops' <- unfreezeLinearVec ops
  n1'  <- unfreezeLayer n1
  n2'  <- unfreezeLayer n2
  ff1' <- unfreezeLayer ff1
  ff2' <- unfreezeLayer ff2
  pure (MkBlock qs' ks' vs' ops' n1' n2' ff1' ff2')

freezeBlockVec : {dModel, numHeads, headDim : Nat} -> {0 d : Device} -> UserDeviceTape d => {0 g : GradMode} ->
                   Vect k (BlockState dModel numHeads headDim d dt g) ->
                   IO (Vect k (BlockState dModel numHeads headDim d dt NoGrad))
freezeBlockVec [] = pure []
freezeBlockVec (b :: bs) = do
  b' <- freezeBlock b
  bs' <- freezeBlockVec bs
  pure (b' :: bs')

unfreezeBlockVec : {dModel, numHeads, headDim : Nat} -> {0 d : Device} -> UserDeviceTape d =>
                     Vect k (BlockState dModel numHeads headDim d dt NoGrad) ->
                     IO (Vect k (BlockState dModel numHeads headDim d dt WithGrad))
unfreezeBlockVec [] = pure []
unfreezeBlockVec (b :: bs) = do
  b' <- unfreezeBlock b
  bs' <- unfreezeBlockVec bs
  pure (b' :: bs')


----------------------------------------------------------------------
-- LayerLike instance
----------------------------------------------------------------------

public export
{seqLen, dModel, numHeads, headDim, numBlocks, vocabSize : Nat} ->
  LayerLike (TransformerState seqLen dModel numHeads headDim numBlocks vocabSize) where
  applyVar st@(MkTransformer _ _ _ _ _ _) input = ioRerun (\_ => (st, applyTransformer st input))
  applyVarBatch st@(MkTransformer _ _ _ _ _ _) input = ioRerun (\_ =>
    (st, applyTransformerBatch st input))
  layerPrefix _ = "tfm"

  freezeLayer (MkTransformer {prf} embedW blocks finalNorm vocabProj peCached maskCached) = do
    embedW'    <- weakenGrad embedW
    blocks'    <- freezeBlockVec blocks
    finalNorm' <- freezeLayer finalNorm
    vocabProj' <- freezeLayer vocabProj
    pure (MkTransformer {prf} embedW' blocks' finalNorm' vocabProj' (retypeGrad peCached) (retypeGrad maskCached))

  unfreezeLayer (MkTransformer {prf} embedW blocks finalNorm vocabProj peCached maskCached) = do
    primIO (primSetRequiresGrad {d} embedW.tensorPtr 1)
    blocks'    <- unfreezeBlockVec blocks
    finalNorm' <- unfreezeLayer finalNorm
    vocabProj' <- unfreezeLayer vocabProj
    pure (MkTransformer {prf} (retypeGrad embedW) blocks' finalNorm' vocabProj' (retypeGrad peCached) (retypeGrad maskCached))

export
transformerLayerAny :
  UserDeviceTape d => RuntimeDType dt => Linked d => Compatible d dt =>
  {seqLen, dModel, numHeads, headDim, numBlocks, vocabSize : Nat} ->
  {auto prf : dModel = numHeads * headDim} ->
  (paramPrefix : String) ->
  IO (AnyLayer seqLen (seqLen * vocabSize) d dt WithGrad)
transformerLayerAny {prf} pid =
  map (MkAnyLayer (TransformerState seqLen dModel numHeads headDim numBlocks vocabSize))
      (transformerLayer {prf} pid)
