||| BERT encoder + pooler, HF-aligned.
|||
||| Target: `google/bert_uncased_L-2_H-128_A-2` (and any BERT-family
||| HuggingFace checkpoint sharing the same `state_dict()` naming
||| convention).
|||
||| This module follows the rules in `CONVENTIONS.md`:
|||   - Param names are literal HF-on-disk strings
|||     (`bert.encoder.layer.0.attention.self.query.weight`, …),
|||     not idris-ml's `_weights` plural / underscore convention.
|||   - Storage shapes match HF on disk: BERT does NOT fuse QKV;
|||     `attention.self.{query,key,value}.weight` is three separate
|||     `[hidden, hidden]` tensors per encoder layer.
|||
||| Forward pass is implemented in a follow-up commit; this file lands
||| the param-name catalogue + the typed state records + the
||| `hfBertModel` constructor.
module HfBert

import Data.Vect

import Compat.Random
import Device
import Init
import Sampler
import Tensor


----------------------------------------------------------------------
-- Config
----------------------------------------------------------------------

||| HF BERT architecture knobs. Field names map to HF's `BertConfig`
||| spelling 1:1 (`hidden_size` → `hidden`, `num_hidden_layers` →
||| `numLayers`, etc.).
public export
record BertConfig where
  constructor MkBertConfig
  vocabSize     : Nat
  hidden        : Nat
  numLayers     : Nat
  numHeads      : Nat
  intermediate  : Nat
  maxPosition   : Nat
  typeVocabSize : Nat  -- HF default: 2 (sentence-A / sentence-B)

||| `google/bert_uncased_L-2_H-128_A-2` — the proof-of-concept target
||| anchored by `scripts/save_oracle.py`. 2 encoder layers, hidden
||| 128, 2 attention heads (head dim 64), FFN hidden 512.
public export
bertTinyConfig : BertConfig
bertTinyConfig = MkBertConfig
  { vocabSize     = 30522
  , hidden        = 128
  , numLayers     = 2
  , numHeads      = 2
  , intermediate  = 512
  , maxPosition   = 512
  , typeVocabSize = 2
  }


----------------------------------------------------------------------
-- Param-name catalogue (pure Idris — single source of truth)
----------------------------------------------------------------------

embeddingsPrefix : String -> String
embeddingsPrefix pfx = pfx ++ ".embeddings"

encoderLayerPrefix : String -> Nat -> String
encoderLayerPrefix pfx i = pfx ++ ".encoder.layer." ++ show i

poolerPrefix : String -> String
poolerPrefix pfx = pfx ++ ".pooler"

embeddingsParamNames : (pfx : String) -> List String
embeddingsParamNames pfx =
  let p = embeddingsPrefix pfx in
  [ p ++ ".word_embeddings.weight"
  , p ++ ".position_embeddings.weight"
  , p ++ ".token_type_embeddings.weight"
  , p ++ ".LayerNorm.weight"
  , p ++ ".LayerNorm.bias"
  ]

encoderLayerParamNames : (pfx : String) -> (i : Nat) -> List String
encoderLayerParamNames pfx i =
  let p = encoderLayerPrefix pfx i in
  [ p ++ ".attention.self.query.weight"
  , p ++ ".attention.self.query.bias"
  , p ++ ".attention.self.key.weight"
  , p ++ ".attention.self.key.bias"
  , p ++ ".attention.self.value.weight"
  , p ++ ".attention.self.value.bias"
  , p ++ ".attention.output.dense.weight"
  , p ++ ".attention.output.dense.bias"
  , p ++ ".attention.output.LayerNorm.weight"
  , p ++ ".attention.output.LayerNorm.bias"
  , p ++ ".intermediate.dense.weight"
  , p ++ ".intermediate.dense.bias"
  , p ++ ".output.dense.weight"
  , p ++ ".output.dense.bias"
  , p ++ ".output.LayerNorm.weight"
  , p ++ ".output.LayerNorm.bias"
  ]

poolerParamNames : (pfx : String) -> List String
poolerParamNames pfx =
  let p = poolerPrefix pfx in
  [ p ++ ".dense.weight"
  , p ++ ".dense.bias"
  ]

-- MLM head naming: HF stores the masked-language-modeling head under
-- the literal `cls.predictions.*` subtree, sibling to `bert.*`. The
-- `decoder.weight` is TIED to `bert.embeddings.word_embeddings.weight`
-- (so it's NOT on disk); only the bias is stored separately. The
-- transform block (dense + LayerNorm) is fully present.
--
-- The `clsPrefix` parameter exists so unit tests can register under a
-- distinct prefix (`clstest`) to avoid C-side param-registry collisions
-- with prior tests; real callers pass `"cls"` to match HF exactly.
export
mlmHeadParamNames : (clsPrefix : String) -> List String
mlmHeadParamNames pfx =
  let p = pfx ++ ".predictions" in
  [ p ++ ".transform.dense.weight"
  , p ++ ".transform.dense.bias"
  , p ++ ".transform.LayerNorm.weight"
  , p ++ ".transform.LayerNorm.bias"
  , p ++ ".bias"
  ]

encoderParamNames : (pfx : String) -> (numLayers : Nat) -> List String
encoderParamNames _   Z     = []
encoderParamNames pfx (S k) =
  encoderParamNames pfx k ++ encoderLayerParamNames pfx k

||| The complete BERT param catalogue for `cfg`, in the order
||| `hfBertModel` registers them. Equality against this list is the
||| unit-test gate that catches naming drift before it reaches the
||| C loader.
|||
||| Total length is `5 + 16*numLayers + 2`. For `bertTinyConfig`
||| (numLayers = 2) this is 39.
export
bertParamNames : (cfg : BertConfig) -> (paramPrefix : String) -> List String
bertParamNames cfg pfx =
  embeddingsParamNames pfx
    ++ encoderParamNames pfx cfg.numLayers
    ++ poolerParamNames pfx

||| Full catalogue for `BertForMaskedLM` — the encoder + pooler from
||| `bertParamNames` plus the 5 MLM-head params (`cls.predictions.*`).
||| For `bertTinyConfig` this is 39 + 5 = 44.
export
bertForMaskedLmParamNames : (cfg : BertConfig)
                         -> (bertPrefix : String)
                         -> (clsPrefix : String)
                         -> List String
bertForMaskedLmParamNames cfg bertPfx clsPfx =
  bertParamNames cfg bertPfx ++ mlmHeadParamNames clsPfx


----------------------------------------------------------------------
-- Param registration helpers (HF-native suffixes)
----------------------------------------------------------------------
--
-- Linear's existing `linearLayer` hardcodes `_weights` / `_biases`;
-- LayerNorm's `layerNormLayer` hardcodes `_gamma` / `_beta`;
-- Embedding's `embeddingLayer` hardcodes `_weight` (singular w/
-- underscore). HF uses `.weight` / `.bias` (and capital `LayerNorm`
-- in the path). We bypass those and call `tparam2d` / `tparam1d`
-- directly with the HF-literal names.
--
-- BERT init in HF transformers: weights ~ Normal(0, 0.02), biases
-- zero, LayerNorm weight=1, bias=0. We follow it here so an
-- un-loaded model still produces sensible numerics; loadModel
-- overwrites everything anyway.

-- Pack a Vect of Doubles into a buffer (mirrors Layer.Linear.packDoubles
-- — re-exporting that would couple to Linear's module; duplicate is
-- cheaper than an export chain).
packDs : AnyPtr -> Int -> Vect k Double -> AnyPtr
packDs buf _ []          = buf
packDs buf off (x :: xs) = packDs (prim__setDouble buf off x) (off + 1) xs

zeroBuf : AnyPtr -> Int -> Int -> AnyPtr
zeroBuf buf _ 0 = buf
zeroBuf buf off n =
  zeroBuf (prim__setDouble buf off 0.0) (off + 1) (n - 1)

fillConst : AnyPtr -> Int -> Int -> Double -> AnyPtr
fillConst buf _ 0 _ = buf
fillConst buf off n v =
  fillConst (prim__setDouble buf off v) (off + 1) (n - 1) v


-- HF-named Linear: registers `<pfx>.weight` and `<pfx>.bias`. The
-- record holds the typed handles for use at forward time. The pfx
-- is the *Linear*'s prefix (e.g. `bert.encoder.layer.0.attention.self.query`),
-- NOT the parent block.
public export
record BertLinearWb (i, o : Nat) (0 d : Device) (0 dt : DType) (0 g : GradMode) where
  constructor MkBertLinear
  weight : Tensor [o, i] d dt g
  bias   : Tensor [o] d dt g

makeBertLinear : UserDeviceTraining d => RuntimeDType dt => Linked d => Compatible d dt
              => {i, o : Nat}
              -> (paramPrefix : String)
              -> IO (BertLinearWb i o d dt WithGrad)
makeBertLinear pfx = do
  -- Fused C-side create + in-place init (added 2026-05-28). Weight:
  -- normal(0, 0.02) matching HF's default Linear init; bias: zero.
  -- Replaces the per-element `traverse normalSample` + `packDs` chain
  -- that dominated state construction on 1B-param models (see
  -- docs/develop/perf-changes.md).
  w <- tparam2dNormal {o} {i} (pfx ++ ".weight") 0.0 0.02
  b <- tparam1dConst  {n=o}   (pfx ++ ".bias")   0.0
  pure (MkBertLinear w b)


-- HF-named Embedding: registers `<pfx>.weight` (`[vocab, dim]`).
public export
record BertEmbedding (vocab, dim : Nat) (0 d : Device) (0 dt : DType) (0 g : GradMode) where
  constructor MkBertEmbedding
  weight : Tensor [vocab, dim] d dt g

makeBertEmbedding : UserDeviceTraining d => RuntimeDType dt => Linked d => Compatible d dt
                 => {vocab, dim : Nat}
                 -> (paramPrefix : String)
                 -> IO (BertEmbedding vocab dim d dt WithGrad)
makeBertEmbedding pfx = do
  -- Fused C-side create + normal(0, 0.02) init. See makeBertLinear
  -- for the bottleneck this replaces.
  w <- tparam2dNormal {o=vocab} {i=dim} (pfx ++ ".weight") 0.0 0.02
  pure (MkBertEmbedding w)


-- HF-named LayerNorm: registers `<pfx>.weight` (γ, init 1.0) and
-- `<pfx>.bias` (β, init 0.0). HF capitalises `LayerNorm` in the path
-- so callers pass e.g. `bert.embeddings.LayerNorm`.
public export
record BertLN (n : Nat) (0 d : Device) (0 dt : DType) (0 g : GradMode) where
  constructor MkBertLN
  gamma : Tensor [n] d dt g
  beta  : Tensor [n] d dt g

makeBertLN : UserDeviceTraining d => RuntimeDType dt => Linked d => Compatible d dt
          => {n : Nat}
          -> (paramPrefix : String)
          -> IO (BertLN n d dt WithGrad)
makeBertLN pfx = do
  -- Fused C-side const fill. γ = 1.0 (HF LayerNorm weight default),
  -- β = 0.0 (bias default). Replaces the host-side fillConst/zeroBuf
  -- loops + per-element prim__setDouble FFI.
  g <- tparam1dConst {n} (pfx ++ ".weight") 1.0
  b <- tparam1dConst {n} (pfx ++ ".bias")   0.0
  pure (MkBertLN g b)


----------------------------------------------------------------------
-- BERT state records
----------------------------------------------------------------------

public export
record BertEmbeddingsState
        (vocab, hidden, maxPos, typeVocab : Nat)
        (0 d : Device) (0 dt : DType) (0 g : GradMode) where
  constructor MkBertEmbeddings
  wordEmb     : BertEmbedding vocab hidden d dt g
  posEmb      : BertEmbedding maxPos hidden d dt g
  typeEmb     : BertEmbedding typeVocab hidden d dt g
  layerNorm   : BertLN hidden d dt g

public export
record BertSelfAttentionState
        (hidden : Nat) (0 d : Device) (0 dt : DType) (0 g : GradMode) where
  constructor MkBertSelfAttn
  query : BertLinearWb hidden hidden d dt g
  key   : BertLinearWb hidden hidden d dt g
  value : BertLinearWb hidden hidden d dt g

public export
record BertSelfOutputState
        (hidden : Nat) (0 d : Device) (0 dt : DType) (0 g : GradMode) where
  constructor MkBertSelfOut
  dense     : BertLinearWb hidden hidden d dt g
  layerNorm : BertLN hidden d dt g

public export
record BertIntermediateState
        (hidden, intermediate : Nat)
        (0 d : Device) (0 dt : DType) (0 g : GradMode) where
  constructor MkBertIntermediate
  dense : BertLinearWb hidden intermediate d dt g

public export
record BertOutputState
        (hidden, intermediate : Nat)
        (0 d : Device) (0 dt : DType) (0 g : GradMode) where
  constructor MkBertOut
  dense     : BertLinearWb intermediate hidden d dt g
  layerNorm : BertLN hidden d dt g

public export
record BertLayerState
        (hidden, intermediate : Nat)
        (0 d : Device) (0 dt : DType) (0 g : GradMode) where
  constructor MkBertLayer
  selfAttn   : BertSelfAttentionState hidden d dt g
  selfOut    : BertSelfOutputState hidden d dt g
  intermed   : BertIntermediateState hidden intermediate d dt g
  output     : BertOutputState hidden intermediate d dt g

public export
record BertPoolerState
        (hidden : Nat) (0 d : Device) (0 dt : DType) (0 g : GradMode) where
  constructor MkBertPooler
  dense : BertLinearWb hidden hidden d dt g

public export
record BertModelState
        (vocab, hidden, numLayers, intermediate, maxPos, typeVocab : Nat)
        (0 d : Device) (0 dt : DType) (0 g : GradMode) where
  constructor MkBertModel
  embeddings : BertEmbeddingsState vocab hidden maxPos typeVocab d dt g
  layers     : Vect numLayers (BertLayerState hidden intermediate d dt g)
  pooler     : BertPoolerState hidden d dt g


----------------------------------------------------------------------
-- Constructors
----------------------------------------------------------------------

makeEmbeddings : UserDeviceTraining d => RuntimeDType dt => Linked d => Compatible d dt
              => {vocab, hidden, maxPos, typeVocab : Nat}
              -> (paramPrefix : String)
              -> IO (BertEmbeddingsState vocab hidden maxPos typeVocab d dt WithGrad)
makeEmbeddings pfx = do
  let p = embeddingsPrefix pfx
  we <- makeBertEmbedding {vocab} {dim=hidden} (p ++ ".word_embeddings")
  pe <- makeBertEmbedding {vocab=maxPos} {dim=hidden} (p ++ ".position_embeddings")
  te <- makeBertEmbedding {vocab=typeVocab} {dim=hidden} (p ++ ".token_type_embeddings")
  ln <- makeBertLN {n=hidden} (p ++ ".LayerNorm")
  pure (MkBertEmbeddings we pe te ln)

makeSelfAttn : UserDeviceTraining d => RuntimeDType dt => Linked d => Compatible d dt
            => {hidden : Nat}
            -> (paramPrefix : String)
            -> IO (BertSelfAttentionState hidden d dt WithGrad)
makeSelfAttn pfx = do
  let p = pfx ++ ".attention.self"
  q <- makeBertLinear {i=hidden} {o=hidden} (p ++ ".query")
  k <- makeBertLinear {i=hidden} {o=hidden} (p ++ ".key")
  v <- makeBertLinear {i=hidden} {o=hidden} (p ++ ".value")
  pure (MkBertSelfAttn q k v)

makeSelfOut : UserDeviceTraining d => RuntimeDType dt => Linked d => Compatible d dt
           => {hidden : Nat}
           -> (paramPrefix : String)
           -> IO (BertSelfOutputState hidden d dt WithGrad)
makeSelfOut pfx = do
  let p = pfx ++ ".attention.output"
  dn <- makeBertLinear {i=hidden} {o=hidden} (p ++ ".dense")
  ln <- makeBertLN {n=hidden} (p ++ ".LayerNorm")
  pure (MkBertSelfOut dn ln)

makeIntermed : UserDeviceTraining d => RuntimeDType dt => Linked d => Compatible d dt
            => {hidden, intermediate : Nat}
            -> (paramPrefix : String)
            -> IO (BertIntermediateState hidden intermediate d dt WithGrad)
makeIntermed pfx = do
  dn <- makeBertLinear {i=hidden} {o=intermediate} (pfx ++ ".intermediate.dense")
  pure (MkBertIntermediate dn)

makeOutput : UserDeviceTraining d => RuntimeDType dt => Linked d => Compatible d dt
          => {hidden, intermediate : Nat}
          -> (paramPrefix : String)
          -> IO (BertOutputState hidden intermediate d dt WithGrad)
makeOutput pfx = do
  let p = pfx ++ ".output"
  dn <- makeBertLinear {i=intermediate} {o=hidden} (p ++ ".dense")
  ln <- makeBertLN {n=hidden} (p ++ ".LayerNorm")
  pure (MkBertOut dn ln)

makeLayer : UserDeviceTraining d => RuntimeDType dt => Linked d => Compatible d dt
         => {hidden, intermediate : Nat}
         -> (layerIdx : Nat)
         -> (paramPrefix : String)
         -> IO (BertLayerState hidden intermediate d dt WithGrad)
makeLayer i pfx = do
  let p = encoderLayerPrefix pfx i
  sa <- makeSelfAttn  {hidden} p
  so <- makeSelfOut   {hidden} p
  im <- makeIntermed  {hidden} {intermediate} p
  ou <- makeOutput    {hidden} {intermediate} p
  pure (MkBertLayer sa so im ou)

-- Build N layers in ascending index order (0, 1, …, N-1). Registers
-- params in the order the catalogue lists them.
makeLayers : UserDeviceTraining d => RuntimeDType dt => Linked d => Compatible d dt
          => {hidden, intermediate : Nat}
          -> (count : Nat)
          -> (paramPrefix : String)
          -> IO (Vect count (BertLayerState hidden intermediate d dt WithGrad))
makeLayers count pfx = go Z count
  where
    go : (idx : Nat) -> (remaining : Nat)
       -> IO (Vect remaining (BertLayerState hidden intermediate d dt WithGrad))
    go _   Z     = pure []
    go idx (S k) = do
      l  <- makeLayer {hidden} {intermediate} idx pfx
      ls <- go (S idx) k
      pure (l :: ls)

makePooler : UserDeviceTraining d => RuntimeDType dt => Linked d => Compatible d dt
          => {hidden : Nat}
          -> (paramPrefix : String)
          -> IO (BertPoolerState hidden d dt WithGrad)
makePooler pfx = do
  dn <- makeBertLinear {i=hidden} {o=hidden} (poolerPrefix pfx ++ ".dense")
  pure (MkBertPooler dn)


||| Build a fresh BERT model with HF-native param names registered
||| under the C-side param registry. Params are initialised to
||| reasonable defaults (Normal(0, 0.02) weights, zero biases,
||| LayerNorm γ=1 β=0) so a forward pass through the un-loaded model
||| is well-defined. `loadModel` then overwrites every weight from
||| disk — the init only matters when running without a checkpoint.
|||
||| The `paramPrefix` is the literal HF root, typically `"bert"`. All
||| 39 (for numLayers=2) param names appear in the C registry in
||| exactly the order `bertParamNames cfg paramPrefix` returns.
export
hfBertModel : UserDeviceTraining d => RuntimeDType dt => Linked d => Compatible d dt
           => {vocab, hidden, numLayers, numHeads, intermediate, maxPos, typeVocab : Nat}
           -> (paramPrefix : String)
           -> IO (BertModelState vocab hidden numLayers intermediate maxPos typeVocab d dt WithGrad)
hfBertModel pfx = do
  emb    <- makeEmbeddings {vocab} {hidden} {maxPos} {typeVocab} pfx
  layers <- makeLayers     {hidden} {intermediate} numLayers pfx
  pool   <- makePooler     {hidden} pfx
  pure (MkBertModel emb layers pool)


----------------------------------------------------------------------
-- Forward pass
----------------------------------------------------------------------
--
-- Math mirrors HF's BertModel forward (no decoder, no LM head):
--
--   x = embeddings(input_ids, position_ids, token_type_ids)  -- [seq, hidden]
--   for layer in encoder.layer:
--     attn_out = self_attention(x)
--     x1 = LayerNorm(x + attn.output.dense(attn_out))         -- post-attn
--     ffn = output.dense(GELU(intermediate.dense(x1)))
--     x  = LayerNorm(x1 + ffn)                                -- post-FFN
--   pooled = tanh(pooler.dense(x[CLS]))                       -- [hidden]
--
-- Inner attention uses primitives directly (primMm / primTranspose2d /
-- primSoftmax2d / primNarrow / primConcat2dAxis1) for the per-head
-- loop — same pattern as Layer/Transformer.idr's runHeadAttn. The
-- typed surface re-emerges at the BertLayer boundary.


-- ε for LayerNorm. HF BERT defaults to 1e-12.
bertLnEps : Double
bertLnEps = 1.0e-12

-- Embedding lookup returning [seqLen, dim]. Wraps primEmbedding +
-- primReshape2d in the same way Layer/Transformer.idr does its token
-- lookup.
applyEmbedLookup2d : {0 d : Device} -> UserDeviceTraining d
                  => {seqLen, vocab, dim : Nat}
                  -> BertEmbedding vocab dim d dt g
                  -> Tensor [seqLen] d dt g
                  -> IO (Tensor [seqLen, dim] d dt g)
applyEmbedLookup2d {seqLen} {dim} (MkBertEmbedding w) tokens = ioRerun (\_ =>
  let sI = cast {to=Int} seqLen
      dI = cast {to=Int} dim
      flat = primEmbedding {d} w.tensorPtr tokens.tensorPtr sI dI
      twoD = primReshape2d {d} flat sI dI
  in MkTensor twoD Nothing)

-- 2D LayerNorm: applies γ and β along the last dim of a [seq, hidden]
-- tensor. Wraps primLayerNorm2d.
applyLN2d : {0 d : Device} -> UserDeviceTraining d
         => BertLN hidden d dt g
         -> Tensor [seqLen, hidden] d dt g
         -> IO (Tensor [seqLen, hidden] d dt g)
applyLN2d (MkBertLN g b) input = ioRerun (\_ =>
  MkTensor (primLayerNorm2d {d} input.tensorPtr g.tensorPtr b.tensorPtr bertLnEps)
           Nothing)

-- Apply a BertLinearWb to a batched input [seq, i] -> [seq, o]. Uses
-- the typed tlinear2d which handles bias broadcast.
applyBertLinear2d : {0 d : Device} -> UserDeviceTraining d
                 => BertLinearWb i o d dt g
                 -> Tensor [seqLen, i] d dt g
                 -> IO (Tensor [seqLen, o] d dt g)
applyBertLinear2d (MkBertLinear w b) x = tlinear2d w x b


-- Per-head attention math. Returns AnyPtr to a [seqLen, headDim]
-- block; the caller's job is to either concat it with siblings or
-- (for the single-head case) wrap directly into a Tensor.
oneHeadCtx : {0 d : Device} -> UserDeviceTraining d
          => (qFull, kFull, vFull : AnyPtr)
          -> (startI, headDimI : Int) -> (scale : Double)
          -> AnyPtr
oneHeadCtx qFull kFull vFull startI headDimI scale =
  let qh     = primNarrow {d} qFull 1 startI headDimI
      kh     = primNarrow {d} kFull 1 startI headDimI
      vh     = primNarrow {d} vFull 1 startI headDimI
      kT     = primTranspose2d {d} kh
      scores = primMulScalar {d} (primMm {d} qh kT) scale
      attn   = primSoftmax2d {d} scores
  in primMm {d} attn vh

-- Build the multi-head context by concatenating per-head blocks
-- along axis=1. Head 0 is the starting accumulator; heads 1..N-1 are
-- folded in. `remaining` counts heads still to process; `startI`
-- is the column offset for the *next* head.
buildHeads : {0 d : Device} -> UserDeviceTraining d
          => (qFull, kFull, vFull : AnyPtr)
          -> (headDimI : Int) -> (scale : Double)
          -> (remaining : Nat) -> (startI : Int) -> (acc : AnyPtr)
          -> AnyPtr
buildHeads _ _ _ _ _ Z _ acc = acc
buildHeads qFull kFull vFull headDimI scale (S k) startI acc =
  let nextCtx = oneHeadCtx {d} qFull kFull vFull startI headDimI scale
      newAcc  = primConcat2dAxis1 {d} acc nextCtx
  in buildHeads {d} qFull kFull vFull headDimI scale k (startI + headDimI) newAcc

-- Full multi-head self-attention. Computes Q/K/V via the three fused
-- linears, then splits + recombines per-head. Output is the
-- attention.output.dense applied to the concatenated context — i.e.
-- HF's BertSelfAttention + BertSelfOutput's dense (residual + LN
-- come in the caller).
--
-- numHeads is matched at the type level so the single-head case
-- (S Z) can avoid `primNarrow` entirely — kept as an optimisation
-- (one less shape op + handle wrap per attention block). Multi-head
-- (S (S _)) goes through the axis-1 narrow path, which all three
-- backends handle correctly post the
-- `linear_shape_narrow::axis1_correctness_rank2` fix
-- (torch + mlx narrow kernels previously ignored the `dim` arg
-- and silently flattened to axis-0; tape was always right).
applySelfAttn : {0 d : Device} -> UserDeviceTraining d
             => {seqLen, hidden, numHeads, headDim : Nat}
             -> {auto prf : hidden = numHeads * headDim}
             -> BertSelfAttentionState hidden d dt g
             -> Tensor [seqLen, hidden] d dt g
             -> IO (Tensor [seqLen, hidden] d dt g)
applySelfAttn {numHeads = Z} _ input = pure input
applySelfAttn {numHeads = S Z} {headDim} sa input = do
  -- Single-head: q/k/v are already the full attention tensors;
  -- no narrow needed. Drop to primitives only for the matmul +
  -- softmax chain.
  q  <- applyBertLinear2d sa.query input  -- [seq, hidden]
  k' <- applyBertLinear2d sa.key   input
  v  <- applyBertLinear2d sa.value input
  ioRerun (\_ =>
    let scale  = 1.0 / sqrt (cast {to=Double} headDim)
        kT     = primTranspose2d {d} k'.tensorPtr
        scores = primMulScalar {d} (primMm {d} q.tensorPtr kT) scale
        attn   = primSoftmax2d {d} scores
        ctx    = primMm {d} attn v.tensorPtr
    in MkTensor ctx Nothing)
applySelfAttn {numHeads = S (S k)} {headDim} sa input = do
  -- Multi-head: per-head narrow → matmul → softmax → matmul, then
  -- concat. Crashes on tape today (see docstring above); torch/mlx
  -- expected to work.
  q  <- applyBertLinear2d sa.query input
  k' <- applyBertLinear2d sa.key   input
  v  <- applyBertLinear2d sa.value input
  let headDimI = cast {to=Int} headDim
      scale    = 1.0 / sqrt (cast {to=Double} headDim)
      qP       = q.tensorPtr
      kP       = k'.tensorPtr
      vP       = v.tensorPtr
      head0    = oneHeadCtx {d} qP kP vP 0 headDimI scale
      ctxPtr   = buildHeads {d} qP kP vP headDimI scale (S k) headDimI head0
  pure (MkTensor ctxPtr Nothing)

-- One BERT layer: self-attention + residual + LayerNorm + FFN
-- (intermediate + GELU + output dense) + residual + LayerNorm.
applyLayer : {0 d : Device} -> UserDeviceCore d => UserDeviceTraining d
          => {seqLen, hidden, intermediate, numHeads, headDim : Nat}
          -> {auto prf : hidden = numHeads * headDim}
          -> BertLayerState hidden intermediate d dt g
          -> Tensor [seqLen, hidden] d dt g
          -> IO (Tensor [seqLen, hidden] d dt g)
applyLayer (MkBertLayer sa so im out) input = do
  attnCtx  <- applySelfAttn {numHeads} {headDim} sa input
  attnDen  <- applyBertLinear2d so.dense attnCtx
  postAttn <- tadd input attnDen
  postLN1  <- applyLN2d so.layerNorm postAttn
  ffnHid   <- applyBertLinear2d im.dense postLN1
  ffnAct   <- tgelu ffnHid
  ffnOut   <- applyBertLinear2d out.dense ffnAct
  postFfn  <- tadd postLN1 ffnOut
  applyLN2d out.layerNorm postFfn

-- Fold over the encoder layers.
applyEncoder : {0 d : Device} -> UserDeviceCore d => UserDeviceTraining d
            => {seqLen, hidden, intermediate, numHeads, headDim, numLayers : Nat}
            -> {auto prf : hidden = numHeads * headDim}
            -> Vect numLayers (BertLayerState hidden intermediate d dt g)
            -> Tensor [seqLen, hidden] d dt g
            -> IO (Tensor [seqLen, hidden] d dt g)
applyEncoder []        h = pure h
applyEncoder (l :: ls) h = do
  h' <- applyLayer {numHeads} {headDim} l h
  applyEncoder {numHeads} {headDim} ls h'

-- Pooler: take the [CLS] (row 0), apply dense + tanh.
applyPooler : {0 d : Device} -> UserDeviceCore d => UserDeviceTraining d
           => {seqLen, hidden : Nat}
           -> BertPoolerState hidden d dt g
           -> Tensor [seqLen, hidden] d dt g
           -> IO (Tensor [hidden] d dt g)
applyPooler (MkBertPooler dn) input = do
  -- Extract row 0 — the [CLS] token's contextualised hidden state.
  cls    <- trowSelect input 0  -- [hidden]
  dense  <- tlinear dn.weight cls dn.bias
  ttanh dense

-- Embeddings forward: sum word + position + token-type, LayerNorm.
applyEmbeddings : {0 d : Device} -> UserDeviceCore d => UserDeviceTraining d
               => {seqLen, vocab, hidden, maxPos, typeVocab : Nat}
               -> BertEmbeddingsState vocab hidden maxPos typeVocab d dt g
               -> (inputIds     : Tensor [seqLen] d dt g)
               -> (positionIds  : Tensor [seqLen] d dt g)
               -> (tokenTypeIds : Tensor [seqLen] d dt g)
               -> IO (Tensor [seqLen, hidden] d dt g)
applyEmbeddings (MkBertEmbeddings we pe te ln) inputIds positionIds tokenTypeIds = do
  wordE   <- applyEmbedLookup2d we inputIds
  posE    <- applyEmbedLookup2d pe positionIds
  typeE   <- applyEmbedLookup2d te tokenTypeIds
  sum1    <- tadd wordE posE
  sum2    <- tadd sum1 typeE
  applyLN2d ln sum2

||| Full BERT forward: input IDs → pooled [CLS] output. The caller
||| supplies the three ID sequences explicitly (no tokenizer or
||| arange in v1; see Row 7's LLM-class example for tokenizer
||| integration).
|||
||| numHeads / headDim are implicit Nats with the
||| `hidden = numHeads * headDim` proof required at the call site.
export
hfBertForward : {0 d : Device} -> UserDeviceCore d => UserDeviceTraining d
             => {seqLen, vocab, hidden, numLayers, numHeads, headDim,
                 intermediate, maxPos, typeVocab : Nat}
             -> {auto prf : hidden = numHeads * headDim}
             -> BertModelState vocab hidden numLayers intermediate maxPos typeVocab d dt g
             -> (inputIds     : Tensor [seqLen] d dt g)
             -> (positionIds  : Tensor [seqLen] d dt g)
             -> (tokenTypeIds : Tensor [seqLen] d dt g)
             -> IO (Tensor [hidden] d dt g)
hfBertForward (MkBertModel emb layers pool) inputIds positionIds tokenTypeIds = do
  hEmb <- applyEmbeddings emb inputIds positionIds tokenTypeIds
  hEnc <- applyEncoder {numHeads} {headDim} layers hEmb
  applyPooler pool hEnc


----------------------------------------------------------------------
-- MLM head (BertForMaskedLM)
----------------------------------------------------------------------
--
-- HF's BertForMaskedLM = BertModel + a small head:
--
--   transform.dense  : Linear[hidden, hidden]
--   transform.act    : GELU                        (hidden_act in config)
--   transform.LN     : LayerNorm[hidden]
--   decoder          : Linear[hidden, vocab], weight TIED to
--                      bert.embeddings.word_embeddings.weight (so
--                      `decoder.weight` is NOT on disk)
--   bias             : [vocab]                     (decoder bias, named
--                      `cls.predictions.bias` on disk)
--
-- We model that as a triple: the transform Linear, the transform LN,
-- and a standalone bias Tensor. At forward time the tied decoder is
-- synthesised by reusing the embeddings' word_embeddings tensor as a
-- BertLinearWb [vocab, hidden] handle.

public export
record BertMlmHeadState
        (vocab, hidden : Nat) (0 d : Device) (0 dt : DType) (0 g : GradMode) where
  constructor MkBertMlmHead
  transformDense : BertLinearWb hidden hidden d dt g
  transformLn    : BertLN hidden d dt g
  bias           : Tensor [vocab] d dt g

||| Register the 5 MLM-head params under `<clsPrefix>.predictions.*`.
||| Real callers pass `"cls"` to match HF; tests pass a distinct prefix
||| to avoid C-side param-registry collisions.
makeMlmHead : UserDeviceTraining d => RuntimeDType dt => Linked d => Compatible d dt
           => {vocab, hidden : Nat}
           -> (clsPrefix : String)
           -> IO (BertMlmHeadState vocab hidden d dt WithGrad)
makeMlmHead clsPfx = do
  let p = clsPfx ++ ".predictions"
  td <- makeBertLinear {i=hidden} {o=hidden} (p ++ ".transform.dense")
  tn <- makeBertLN     {n=hidden}            (p ++ ".transform.LayerNorm")
  -- Standalone decoder bias. The decoder *weight* is tied to the word
  -- embedding and is not registered separately — only the bias is.
  let vI    = cast {to=Int} vocab
      bBuf  = prim__allocDoubles vI
      bBuf' = zeroBuf bBuf 0 vI
  bias <- tparam1d {n=vocab} (p ++ ".bias") bBuf'
  pure (MkBertMlmHead td tn bias)


public export
record BertForMaskedLmState
        (vocab, hidden, numLayers, intermediate, maxPos, typeVocab : Nat)
        (0 d : Device) (0 dt : DType) (0 g : GradMode) where
  constructor MkBertForMaskedLm
  base    : BertModelState vocab hidden numLayers intermediate maxPos typeVocab d dt g
  mlmHead : BertMlmHeadState vocab hidden d dt g

||| Build a fresh BertForMaskedLM. Combines `hfBertModel` (the
||| encoder + pooler under `<paramPrefix>.*`) with the MLM head under
||| literal `cls.predictions.*`. 44 params total for `bertTinyConfig`
||| (39 base + 5 head). Loading a HF safetensors with `loadModel` fills
||| every name; `bert.pooler.*` is filled too (HF's BertForMaskedLM
||| ships pooler weights, just doesn't use them at MLM time).
|||
||| The MLM head's `cls.predictions.*` prefix is fixed by HF
||| convention; a second `BertForMaskedLmState` in the same process
||| would re-register those names and collide. That's a non-issue for
||| the v1 demo (one model per process); a future row can parameterise
||| the prefix if multi-model workflows arrive.
export
hfBertForMaskedLm : UserDeviceTraining d => RuntimeDType dt => Linked d => Compatible d dt
                 => {vocab, hidden, numLayers, numHeads, intermediate, maxPos, typeVocab : Nat}
                 -> (paramPrefix : String)
                 -> IO (BertForMaskedLmState vocab hidden numLayers intermediate maxPos typeVocab d dt WithGrad)
hfBertForMaskedLm pfx = do
  base <- hfBertModel {vocab} {hidden} {numLayers} {numHeads}
                      {intermediate} {maxPos} {typeVocab} pfx
  mlm  <- makeMlmHead {vocab} {hidden} "cls"
  pure (MkBertForMaskedLm base mlm)


-- Apply the MLM head to encoder output [seq, hidden] producing logits
-- [seq, vocab]. The tied decoder is reconstituted as a BertLinearWb
-- whose `weight` is the embedding tensor and whose `bias` is the head's
-- standalone bias.
applyMlmHead : {0 d : Device} -> UserDeviceCore d => UserDeviceTraining d
            => {seqLen, vocab, hidden : Nat}
            -> (head : BertMlmHeadState vocab hidden d dt g)
            -> (wordEmb : Tensor [vocab, hidden] d dt g)
            -> (encoderOut : Tensor [seqLen, hidden] d dt g)
            -> IO (Tensor [seqLen, vocab] d dt g)
applyMlmHead (MkBertMlmHead td tn b) wordEmb x = do
  h1 <- applyBertLinear2d td x       -- [seq, hidden]
  h2 <- tgelu h1
  h3 <- applyLN2d tn h2
  -- Tied decoder: word_emb is shape [vocab, hidden] — exactly the
  -- BertLinearWb hidden vocab weight shape.
  let decoder = MkBertLinear {i=hidden} {o=vocab} wordEmb b
  applyBertLinear2d decoder h3       -- [seq, vocab]

||| Full MLM forward: input IDs → per-token vocab logits. Caller
||| extracts the row at any `[MASK]` position and takes top-K to get
||| candidate fill-ins.
export
hfBertMlmForward : {0 d : Device} -> UserDeviceCore d => UserDeviceTraining d
                => {seqLen, vocab, hidden, numLayers, numHeads, headDim,
                    intermediate, maxPos, typeVocab : Nat}
                -> {auto prf : hidden = numHeads * headDim}
                -> BertForMaskedLmState vocab hidden numLayers intermediate maxPos typeVocab d dt g
                -> (inputIds     : Tensor [seqLen] d dt g)
                -> (positionIds  : Tensor [seqLen] d dt g)
                -> (tokenTypeIds : Tensor [seqLen] d dt g)
                -> IO (Tensor [seqLen, vocab] d dt g)
hfBertMlmForward (MkBertForMaskedLm (MkBertModel emb layers _) head) i p t = do
  hEmb <- applyEmbeddings emb i p t
  hEnc <- applyEncoder {numHeads} {headDim} layers hEmb
  applyMlmHead head emb.wordEmb.weight hEnc
