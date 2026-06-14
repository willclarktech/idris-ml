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
module Transformers.Bert

import Control.Linear.LIO
import Data.Linear.Notation
import Data.Vect

import Backend
import Checkpoint
import Compat.Random
import Executor
import GradMode
import Init
import Nn.Embedding
import Nn.LayerNorm
import Nn.Linear
import Nn.Module
import Sampler
import Tensor
import Transformers.Common
import Transformers.Config

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
encoderParamNames pfx numLayers = forBlocks numLayers (encoderLayerParamNames pfx)

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
zeroBuf buf _ 0   = buf
zeroBuf buf off n =
  zeroBuf (prim__setDouble buf off 0.0) (off + 1) (n - 1)

fillConst : AnyPtr -> Int -> Int -> Double -> AnyPtr
fillConst buf _ 0 _   = buf
fillConst buf off n v =
  fillConst (prim__setDouble buf off v) (off + 1) (n - 1) v

-- HF-named Linear: registers `<pfx>.weight` and `<pfx>.bias`. The
-- record holds the typed handles for use at forward time. The pfx
-- is the *Linear*'s prefix (e.g. `bert.encoder.layer.0.attention.self.query`),
-- NOT the parent block.
makeBertLinear : UserExecutorTraining ex => RuntimeDType dt => Linked ex => Compatible ex dt => KnownGrad g
              => {i, o : Nat}
              -> (paramPrefix : String)
              -> IO (Linear i o ex dt g)
makeBertLinear pfx = do
  -- Fused C-side create + in-place init. Weight: normal(0, 0.02)
  -- matching HF's default Linear init; bias: zero. Replaces the per-
  -- element `traverse normalSample` + `packDs` chain that dominated
  -- state construction on 1B-param models (see
  -- docs/develop/perf-changes.md).
  w <- tparam2dNormal {ex} {dt} {o} {i} (pfx ++ ".weight") 0.0 0.02
  b <- tparam1dConst {ex} {dt} {n=o}   (pfx ++ ".bias")   0.0
  -- Build the requested grad-mode directly: WithGrad keeps the freshly
  -- registered (requires_grad=1) params; NoGrad weakens them in place
  -- (requires_grad=0 + retype) so the inference model is genuinely
  -- tape-free with no post-construction `eval` flip.
  case sgrad {g} of
    SWithGrad => pure (MkLinear w b)
    SNoGrad   => do
      w' <- weakenGrad w
      b' <- weakenGrad b
      pure (MkLinear w' b')

-- HF-named Embedding: registers `<pfx>.weight` (`[vocab, dim]`).
makeBertEmbedding : UserExecutorTraining ex => RuntimeDType dt => Linked ex => Compatible ex dt => KnownGrad g
                 => {vocab, dim : Nat}
                 -> (paramPrefix : String)
                 -> IO (Embedding vocab dim ex dt g)
makeBertEmbedding pfx = do
  -- Fused C-side create + normal(0, 0.02) init. See makeBertLinear
  -- for the bottleneck this replaces.
  w <- tparam2dNormal {ex} {dt} {o=vocab} {i=dim} (pfx ++ ".weight") 0.0 0.02
  case sgrad {g} of
    SWithGrad => pure (MkEmbedding w)
    SNoGrad   => do
      w' <- weakenGrad w
      pure (MkEmbedding w')

-- HF-named LayerNorm: registers `<pfx>.weight` (γ, init 1.0) and
-- `<pfx>.bias` (β, init 0.0). HF capitalises `LayerNorm` in the path
-- so callers pass e.g. `bert.embeddings.LayerNorm`.
makeBertLN : UserExecutorTraining ex => RuntimeDType dt => Linked ex => Compatible ex dt => KnownGrad g
          => {n : Nat}
          -> (paramPrefix : String)
          -> IO (LayerNorm n n ex dt g)
makeBertLN pfx = do
  -- Fused C-side const fill. γ = 1.0 (HF LayerNorm weight default),
  -- β = 0.0 (bias default). Replaces the host-side fillConst/zeroBuf
  -- loops + per-element prim__setDouble FFI.
  gw <- tparam1dConst {ex} {dt} {n} (pfx ++ ".weight") 1.0
  b  <- tparam1dConst {ex} {dt} {n} (pfx ++ ".bias")   0.0
  case sgrad {g} of
    SWithGrad => pure (MkLayerNorm gw b)
    SNoGrad   => do
      gw' <- weakenGrad gw
      b'  <- weakenGrad b
      pure (MkLayerNorm gw' b')

----------------------------------------------------------------------
-- BERT state records
----------------------------------------------------------------------

public export
record BertEmbeddingsState
        (vocab, hidden, maxPos, typeVocab : Nat)
        (0 ex : Executor) (0 dt : DType) (0 g : GradMode) where
  constructor MkBertEmbeddings
  wordEmb   : Embedding vocab hidden ex dt g
  posEmb    : Embedding maxPos hidden ex dt g
  typeEmb   : Embedding typeVocab hidden ex dt g
  layerNorm : LayerNorm hidden hidden ex dt g

public export
record BertSelfAttentionState
        (hidden : Nat) (0 ex : Executor) (0 dt : DType) (0 g : GradMode) where
  constructor MkBertSelfAttn
  query : Linear hidden hidden ex dt g
  key   : Linear hidden hidden ex dt g
  value : Linear hidden hidden ex dt g

public export
record BertSelfOutputState
        (hidden : Nat) (0 ex : Executor) (0 dt : DType) (0 g : GradMode) where
  constructor MkBertSelfOut
  dense     : Linear hidden hidden ex dt g
  layerNorm : LayerNorm hidden hidden ex dt g

public export
record BertIntermediateState
        (hidden, intermediate : Nat)
        (0 ex : Executor) (0 dt : DType) (0 g : GradMode) where
  constructor MkBertIntermediate
  dense : Linear hidden intermediate ex dt g

public export
record BertOutputState
        (hidden, intermediate : Nat)
        (0 ex : Executor) (0 dt : DType) (0 g : GradMode) where
  constructor MkBertOut
  dense     : Linear intermediate hidden ex dt g
  layerNorm : LayerNorm hidden hidden ex dt g

public export
record BertLayerState
        (hidden, intermediate : Nat)
        (0 ex : Executor) (0 dt : DType) (0 g : GradMode) where
  constructor MkBertLayer
  selfAttn : BertSelfAttentionState hidden ex dt g
  selfOut  : BertSelfOutputState hidden ex dt g
  intermed : BertIntermediateState hidden intermediate ex dt g
  output   : BertOutputState hidden intermediate ex dt g

public export
record BertPoolerState
        (hidden : Nat) (0 ex : Executor) (0 dt : DType) (0 g : GradMode) where
  constructor MkBertPooler
  dense : Linear hidden hidden ex dt g

public export
record BertModelState
        (vocab, hidden, numLayers, intermediate, maxPos, typeVocab : Nat)
        (0 ex : Executor) (0 dt : DType) (0 g : GradMode) where
  constructor MkBertModel
  embeddings : BertEmbeddingsState vocab hidden maxPos typeVocab ex dt g
  layers     : Vect numLayers (BertLayerState hidden intermediate ex dt g)
  pooler     : BertPoolerState hidden ex dt g

----------------------------------------------------------------------
-- Constructors
----------------------------------------------------------------------

makeEmbeddings : UserExecutorTraining ex => RuntimeDType dt => Linked ex => Compatible ex dt => KnownGrad g
              => {vocab, hidden, maxPos, typeVocab : Nat}
              -> (paramPrefix : String)
              -> IO (BertEmbeddingsState vocab hidden maxPos typeVocab ex dt g)
makeEmbeddings pfx = do
  let p = embeddingsPrefix pfx
  we <- makeBertEmbedding {vocab} {dim=hidden} (p ++ ".word_embeddings")
  pe <- makeBertEmbedding {vocab=maxPos} {dim=hidden} (p ++ ".position_embeddings")
  te <- makeBertEmbedding {vocab=typeVocab} {dim=hidden} (p ++ ".token_type_embeddings")
  ln <- makeBertLN {n=hidden} (p ++ ".LayerNorm")
  pure (MkBertEmbeddings we pe te ln)

makeSelfAttn : UserExecutorTraining ex => RuntimeDType dt => Linked ex => Compatible ex dt => KnownGrad g
            => {hidden : Nat}
            -> (paramPrefix : String)
            -> IO (BertSelfAttentionState hidden ex dt g)
makeSelfAttn pfx = do
  let p = pfx ++ ".attention.self"
  q <- makeBertLinear {i=hidden} {o=hidden} (p ++ ".query")
  k <- makeBertLinear {i=hidden} {o=hidden} (p ++ ".key")
  v <- makeBertLinear {i=hidden} {o=hidden} (p ++ ".value")
  pure (MkBertSelfAttn q k v)

makeSelfOut : UserExecutorTraining ex => RuntimeDType dt => Linked ex => Compatible ex dt => KnownGrad g
           => {hidden : Nat}
           -> (paramPrefix : String)
           -> IO (BertSelfOutputState hidden ex dt g)
makeSelfOut pfx = do
  let p = pfx ++ ".attention.output"
  dn <- makeBertLinear {i=hidden} {o=hidden} (p ++ ".dense")
  ln <- makeBertLN {n=hidden} (p ++ ".LayerNorm")
  pure (MkBertSelfOut dn ln)

makeIntermed : UserExecutorTraining ex => RuntimeDType dt => Linked ex => Compatible ex dt => KnownGrad g
            => {hidden, intermediate : Nat}
            -> (paramPrefix : String)
            -> IO (BertIntermediateState hidden intermediate ex dt g)
makeIntermed pfx = do
  dn <- makeBertLinear {i=hidden} {o=intermediate} (pfx ++ ".intermediate.dense")
  pure (MkBertIntermediate dn)

makeOutput : UserExecutorTraining ex => RuntimeDType dt => Linked ex => Compatible ex dt => KnownGrad g
          => {hidden, intermediate : Nat}
          -> (paramPrefix : String)
          -> IO (BertOutputState hidden intermediate ex dt g)
makeOutput pfx = do
  let p = pfx ++ ".output"
  dn <- makeBertLinear {i=intermediate} {o=hidden} (p ++ ".dense")
  ln <- makeBertLN {n=hidden} (p ++ ".LayerNorm")
  pure (MkBertOut dn ln)

makeLayer : UserExecutorTraining ex => RuntimeDType dt => Linked ex => Compatible ex dt => KnownGrad g
         => {hidden, intermediate : Nat}
         -> (layerIdx : Nat)
         -> (paramPrefix : String)
         -> IO (BertLayerState hidden intermediate ex dt g)
makeLayer i pfx = do
  let p = encoderLayerPrefix pfx i
  sa <- makeSelfAttn  {hidden} p
  so <- makeSelfOut   {hidden} p
  im <- makeIntermed  {hidden} {intermediate} p
  ou <- makeOutput    {hidden} {intermediate} p
  pure (MkBertLayer sa so im ou)

-- Recursive helper for makeLayers. Top-level (not a `where`-bound
-- helper) so its implicit binders don't shadow BertConfig's record
-- projectors — the `where`-helper form trips Idris's lowercase-name
-- shadowing warning that can't be silenced without breaking
-- unification at the recursive call.
makeLayersGo : UserExecutorTraining ex => RuntimeDType dt => Linked ex => Compatible ex dt => KnownGrad g
            => {hidden, intermediate : Nat}
            -> (paramPrefix : String)
            -> (idx : Nat) -> (remaining : Nat)
            -> IO (Vect remaining (BertLayerState hidden intermediate ex dt g))
makeLayersGo _   _   Z     = pure []
makeLayersGo pfx idx (S k) = do
  l  <- makeLayer {hidden} {intermediate} idx pfx
  ls <- makeLayersGo pfx (S idx) k
  pure (l :: ls)

-- Build N layers in ascending index order (0, 1, …, N-1). Registers
-- params in the order the catalogue lists them.
makeLayers : UserExecutorTraining ex => RuntimeDType dt => Linked ex => Compatible ex dt => KnownGrad g
          => {hidden, intermediate : Nat}
          -> (count : Nat)
          -> (paramPrefix : String)
          -> IO (Vect count (BertLayerState hidden intermediate ex dt g))
makeLayers count pfx = makeLayersGo pfx Z count

makePooler : UserExecutorTraining ex => RuntimeDType dt => Linked ex => Compatible ex dt => KnownGrad g
          => {hidden : Nat}
          -> (paramPrefix : String)
          -> IO (BertPoolerState hidden ex dt g)
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
hfBertModel : UserExecutorTraining ex => RuntimeDType dt => Linked ex => Compatible ex dt => KnownGrad g
           => {vocab, hidden, numLayers, numHeads, intermediate, maxPos, typeVocab : Nat}
           -> (paramPrefix : String)
           -> IO (BertModelState vocab hidden numLayers intermediate maxPos typeVocab ex dt g)
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

-- Embedding lookup returning [seqLen, dim]. Wraps primEmbedding2d
-- directly so we get the natural 2D output in one op.
applyEmbedLookup2d : {0 ex : Executor} -> UserExecutorTraining ex
                  => {seqLen, vocab, dim : Nat}
                  -> Embedding vocab dim ex dt g
                  -> Tensor [seqLen] ex dt g
                  -> IO (Tensor [seqLen, dim] ex dt g)
applyEmbedLookup2d {seqLen} {dim} (MkEmbedding w) tokens = ioRerun (\_ =>
  let sI = cast {to=Int} seqLen
      dI  = cast {to=Int} dim
      out = primEmbedding2d {ex} w.tensorPtr tokens.tensorPtr sI dI
  in MkTensor out Nothing)

-- 2D LayerNorm: applies γ and β along the last dim of a [seq, hidden]
-- tensor. Wraps primLayerNorm2d.
export
applyLN2d : {0 ex : Executor} -> UserExecutorTraining ex
         => {seqLen, hidden : Nat}
         -> LayerNorm hidden hidden ex dt g
         -> Tensor [seqLen, hidden] ex dt g
         -> IO (Tensor [seqLen, hidden] ex dt g)
applyLN2d (MkLayerNorm g b) input = ioRerun (\_ =>
  MkTensor (primLayerNorm2d {ex} input.tensorPtr g.tensorPtr b.tensorPtr bertLnEps)
           Nothing)

-- Apply a Linear to a batched input [seq, i] -> [seq, o]. Uses
-- the typed tlinear2d which handles bias broadcast.
applyBertLinear2d : {0 ex : Executor} -> UserExecutorTraining ex
                 => Linear i o ex dt g
                 -> Tensor [seqLen, i] ex dt g
                 -> IO (Tensor [seqLen, o] ex dt g)
applyBertLinear2d (MkLinear w b) x = tlinear2d w x b

-- Per-head attention math. Returns AnyPtr to a [seqLen, headDim]
-- block; the caller's job is to either concat it with siblings or
-- (for the single-head case) wrap directly into a Tensor.
--
-- `mask` is the optional [seqLen, seqLen] attention mask handle:
-- when `Just`, `primMaskedFill scores mask (-1.0e20)` is applied
-- between matmul and softmax. Convention matches HfGpt2's
-- `causalMask`: a non-zero entry means "mask out this position".
export
oneHeadCtx : {0 ex : Executor} -> UserExecutorTraining ex
          => (qFull, kFull, vFull : AnyPtr)
          -> (mask : Maybe AnyPtr)
          -> (startI, headDimI : Int) -> (scale : Double)
          -> AnyPtr
oneHeadCtx qFull kFull vFull mask startI headDimI scale =
  let qh     = primNarrow {ex} qFull 1 startI headDimI
      kh      = primNarrow {ex} kFull 1 startI headDimI
      vh      = primNarrow {ex} vFull 1 startI headDimI
      kT      = primTranspose2d {ex} kh
      scores  = primMulScalar {ex} (primMm {ex} qh kT) scale
      sMasked = case mask of
        Nothing => scores
        Just m  => primMaskedFill {ex} scores m (-1.0e20)
      attn   = primSoftmax2d {ex} sMasked
  in primMm {ex} attn vh

-- Build the multi-head context by concatenating per-head blocks
-- along axis=1. Head 0 is the starting accumulator; heads 1..N-1 are
-- folded in. `remaining` counts heads still to process; `startI`
-- is the column offset for the *next* head.
export
buildHeads : {0 ex : Executor} -> UserExecutorTraining ex
          => (qFull, kFull, vFull : AnyPtr)
          -> (mask : Maybe AnyPtr)
          -> (headDimI : Int) -> (scale : Double)
          -> (remaining : Nat) -> (startI : Int) -> (acc : AnyPtr)
          -> AnyPtr
buildHeads _ _ _ _ _ _ Z _ acc                                    = acc
buildHeads qFull kFull vFull mask headDimI scale (S k) startI acc =
  let nextCtx = oneHeadCtx {ex} qFull kFull vFull mask startI headDimI scale
      newAcc  = primConcat2dAxis1 {ex} acc nextCtx
  in buildHeads {ex} qFull kFull vFull mask headDimI scale k (startI + headDimI) newAcc

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
applySelfAttn : {0 ex : Executor} -> UserExecutorTraining ex
             => {seqLen, hidden, numHeads, headDim : Nat}
             -> {auto prf : hidden = numHeads * headDim}
             -> BertSelfAttentionState hidden ex dt g
             -> (mask : Maybe AnyPtr)
             -> Tensor [seqLen, hidden] ex dt g
             -> IO (Tensor [seqLen, hidden] ex dt g)
applySelfAttn {numHeads = Z} _ _ input                 = pure input
applySelfAttn {numHeads = S Z} {headDim} sa mask input = do
  -- Single-head: q/k/v are already the full attention tensors;
  -- no narrow needed. Drop to primitives only for the matmul +
  -- (optional masked-fill +) softmax chain.
  q  <- applyBertLinear2d sa.query input  -- [seq, hidden]
  k' <- applyBertLinear2d sa.key   input
  v  <- applyBertLinear2d sa.value input
  ioRerun (\_ =>
    let scale  = 1.0 / sqrt (cast {to=Double} headDim)
        kT      = primTranspose2d {ex} k'.tensorPtr
        scores  = primMulScalar {ex} (primMm {ex} q.tensorPtr kT) scale
        sMasked = case mask of
          Nothing => scores
          Just m  => primMaskedFill {ex} scores m (-1.0e20)
        attn = primSoftmax2d {ex} sMasked
        ctx  = primMm {ex} attn v.tensorPtr
    in MkTensor ctx Nothing)
applySelfAttn {numHeads = S (S k)} {headDim} sa mask input = do
  -- Multi-head: per-head narrow → matmul → (mask) → softmax → matmul,
  -- then concat. Same `mask` (over positions, not features) applies
  -- to every head.
  q  <- applyBertLinear2d sa.query input
  k' <- applyBertLinear2d sa.key   input
  v  <- applyBertLinear2d sa.value input
  let headDimI = cast {to=Int} headDim
      scale  = 1.0 / sqrt (cast {to=Double} headDim)
      qP     = q.tensorPtr
      kP     = k'.tensorPtr
      vP     = v.tensorPtr
      head0  = oneHeadCtx {ex} qP kP vP mask 0 headDimI scale
      ctxPtr = buildHeads {ex} qP kP vP mask headDimI scale (S k) headDimI head0
  pure (MkTensor ctxPtr Nothing)

-- One BERT layer: self-attention + residual + LayerNorm + FFN
-- (intermediate + GELU + output dense) + residual + LayerNorm.
applyLayer : {0 ex : Executor} -> UserExecutorCore ex => UserExecutorTraining ex
          => {seqLen, hidden, intermediate, numHeads, headDim : Nat}
          -> {auto prf : hidden = numHeads * headDim}
          -> BertLayerState hidden intermediate ex dt g
          -> (mask : Maybe AnyPtr)
          -> Tensor [seqLen, hidden] ex dt g
          -> IO (Tensor [seqLen, hidden] ex dt g)
applyLayer (MkBertLayer sa so im out) mask input = do
  attnCtx  <- applySelfAttn {numHeads} {headDim} sa mask input
  attnDen  <- applyBertLinear2d so.dense attnCtx
  postAttn <- tadd input attnDen
  postLN1  <- applyLN2d so.layerNorm postAttn
  ffnHid   <- applyBertLinear2d im.dense postLN1
  ffnAct   <- tgelu ffnHid
  ffnOut   <- applyBertLinear2d out.dense ffnAct
  postFfn  <- tadd postLN1 ffnOut
  applyLN2d out.layerNorm postFfn

-- Fold over the encoder layers.
applyEncoder : {0 ex : Executor} -> UserExecutorCore ex => UserExecutorTraining ex
            => {seqLen, hidden, intermediate, numHeads, headDim, numLayers : Nat}
            -> {auto prf : hidden = numHeads * headDim}
            -> Vect numLayers (BertLayerState hidden intermediate ex dt g)
            -> (mask : Maybe AnyPtr)
            -> Tensor [seqLen, hidden] ex dt g
            -> IO (Tensor [seqLen, hidden] ex dt g)
applyEncoder []        _    h = pure h
applyEncoder (l :: ls) mask h = do
  h' <- applyLayer {numHeads} {headDim} l mask h
  applyEncoder {numHeads} {headDim} ls mask h'

-- Pooler: take the [CLS] (row 0), apply dense + tanh.
export
applyPooler : {0 ex : Executor} -> UserExecutorCore ex => UserExecutorTraining ex
           => {seqLen, hidden : Nat}
           -> BertPoolerState hidden ex dt g
           -> Tensor [seqLen, hidden] ex dt g
           -> IO (Tensor [hidden] ex dt g)
applyPooler (MkBertPooler dn) input = do
  -- Extract row 0 — the [CLS] token's contextualised hidden state.
  cls    <- trowSelect input 0  -- [hidden]
  dense  <- tlinear dn.weightT cls dn.biasT
  ttanh dense

-- Embeddings forward: sum word + position + token-type, LayerNorm.
export
applyEmbeddings : {0 ex : Executor} -> UserExecutorCore ex => UserExecutorTraining ex
               => {seqLen, vocab, hidden, maxPos, typeVocab : Nat}
               -> BertEmbeddingsState vocab hidden maxPos typeVocab ex dt g
               -> (inputIds     : Tensor [seqLen] ex dt g)
               -> (positionIds  : Tensor [seqLen] ex dt g)
               -> (tokenTypeIds : Tensor [seqLen] ex dt g)
               -> IO (Tensor [seqLen, hidden] ex dt g)
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
|||
||| `attentionMask` is an optional `[seqLen, seqLen]` matrix; entries
||| `>= 0.5` are treated as "mask out" (`-1.0e20` filled into scores
||| pre-softmax, so attending to those positions returns ~0 weight).
||| Pass `Nothing` for the original no-mask behaviour — bit-identical
||| to the pre-RT1 forward on fixed-length / non-padded inputs.
export
hfBertForward : {0 ex : Executor} -> UserExecutorCore ex => UserExecutorTraining ex
             => {seqLen, vocab, hidden, numLayers, numHeads, headDim,
                 intermediate, maxPos, typeVocab : Nat}
             -> {auto prf : hidden = numHeads * headDim}
             -> BertModelState vocab hidden numLayers intermediate maxPos typeVocab ex dt g
             -> (inputIds     : Tensor [seqLen] ex dt g)
             -> (positionIds  : Tensor [seqLen] ex dt g)
             -> (tokenTypeIds : Tensor [seqLen] ex dt g)
             -> (attentionMask : Maybe (Tensor [seqLen, seqLen] ex dt g))
             -> IO (Tensor [hidden] ex dt g)
hfBertForward (MkBertModel emb layers pool) inputIds positionIds tokenTypeIds mask = do
  hEmb <- applyEmbeddings emb inputIds positionIds tokenTypeIds
  hEnc <- applyEncoder {numHeads} {headDim} layers (map (\m => m.tensorPtr) mask) hEmb
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
-- Linear [vocab, hidden] handle.

public export
record BertMlmHeadState
        (vocab, hidden : Nat) (0 ex : Executor) (0 dt : DType) (0 g : GradMode) where
  constructor MkBertMlmHead
  transformDense : Linear hidden hidden ex dt g
  transformLn    : LayerNorm hidden hidden ex dt g
  bias           : Tensor [vocab] ex dt g

||| Register the 5 MLM-head params under `<clsPrefix>.predictions.*`.
||| Real callers pass `"cls"` to match HF; tests pass a distinct prefix
||| to avoid C-side param-registry collisions.
makeMlmHead : UserExecutorTraining ex => RuntimeDType dt => Linked ex => Compatible ex dt => KnownGrad g
           => {vocab, hidden : Nat}
           -> (clsPrefix : String)
           -> IO (BertMlmHeadState vocab hidden ex dt g)
makeMlmHead clsPfx = do
  let p = clsPfx ++ ".predictions"
  td <- makeBertLinear {ex} {dt} {g} {i=hidden} {o=hidden} (p ++ ".transform.dense")
  tn <- makeBertLN     {ex} {dt} {g} {n=hidden}            (p ++ ".transform.LayerNorm")
  -- Standalone decoder bias. The decoder *weight* is tied to the word
  -- embedding and is not registered separately — only the bias is.
  bias <- tparam1dConst {ex} {dt} {n=vocab} (p ++ ".bias") 0.0
  -- td/tn are already at `g` (their make*s wove it); only the standalone
  -- bias needs the explicit grad-mode build.
  case sgrad {g} of
    SWithGrad => pure (MkBertMlmHead td tn bias)
    SNoGrad   => do
      bias' <- weakenGrad bias
      pure (MkBertMlmHead td tn bias')

public export
record BertForMaskedLmState
        (vocab, hidden, numLayers, intermediate, maxPos, typeVocab : Nat)
        (0 ex : Executor) (0 dt : DType) (0 g : GradMode) where
  constructor MkBertForMaskedLm
  base    : BertModelState vocab hidden numLayers intermediate maxPos typeVocab ex dt g
  mlmHead : BertMlmHeadState vocab hidden ex dt g

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
hfBertForMaskedLm : UserExecutorTraining ex => RuntimeDType dt => Linked ex => Compatible ex dt => KnownGrad g
                 => {vocab, hidden, numLayers, numHeads, intermediate, maxPos, typeVocab : Nat}
                 -> (paramPrefix : String)
                 -> IO (BertForMaskedLmState vocab hidden numLayers intermediate maxPos typeVocab ex dt g)
hfBertForMaskedLm pfx = do
  base <- hfBertModel {vocab} {hidden} {numLayers} {numHeads}
                      {intermediate} {maxPos} {typeVocab} pfx
  mlm  <- makeMlmHead {vocab} {hidden} "cls"
  pure (MkBertForMaskedLm base mlm)

-- Apply the MLM head to encoder output [seq, hidden] producing logits
-- [seq, vocab]. The tied decoder is reconstituted as a Linear
-- whose `weight` is the embedding tensor and whose `bias` is the head's
-- standalone bias.
applyMlmHead : {0 ex : Executor} -> UserExecutorCore ex => UserExecutorTraining ex
            => {seqLen, vocab, hidden : Nat}
            -> (head : BertMlmHeadState vocab hidden ex dt g)
            -> (wordEmb : Tensor [vocab, hidden] ex dt g)
            -> (encoderOut : Tensor [seqLen, hidden] ex dt g)
            -> IO (Tensor [seqLen, vocab] ex dt g)
applyMlmHead (MkBertMlmHead td tn b) wordEmb x = do
  h1 <- applyBertLinear2d td x       -- [seq, hidden]
  h2 <- tgelu h1
  h3 <- applyLN2d tn h2
  -- Tied decoder: word_emb is shape [vocab, hidden] — exactly the
  -- Linear hidden vocab weight shape.
  let decoder = MkLinear {i=hidden} {o=vocab} wordEmb b
  applyBertLinear2d decoder h3       -- [seq, vocab]

||| Full MLM forward: input IDs → per-token vocab logits. Caller
||| extracts the row at any `[MASK]` position and takes top-K to get
||| candidate fill-ins.
|||
||| `attentionMask` semantics match `hfBertForward`: `Just mat` injects
||| `-1.0e20` at any entry `>= 0.5`; `Nothing` runs un-masked.
export
hfBertMlmForward : {0 ex : Executor} -> UserExecutorCore ex => UserExecutorTraining ex
                => {seqLen, vocab, hidden, numLayers, numHeads, headDim,
                    intermediate, maxPos, typeVocab : Nat}
                -> {auto prf : hidden = numHeads * headDim}
                -> BertForMaskedLmState vocab hidden numLayers intermediate maxPos typeVocab ex dt g
                -> (inputIds     : Tensor [seqLen] ex dt g)
                -> (positionIds  : Tensor [seqLen] ex dt g)
                -> (tokenTypeIds : Tensor [seqLen] ex dt g)
                -> (attentionMask : Maybe (Tensor [seqLen, seqLen] ex dt g))
                -> IO (Tensor [seqLen, vocab] ex dt g)
hfBertMlmForward (MkBertForMaskedLm (MkBertModel emb layers _) head) i p t mask = do
  hEmb <- applyEmbeddings emb i p t
  hEnc <- applyEncoder {numHeads} {headDim} layers (map (\m => m.tensorPtr) mask) hEmb
  applyMlmHead head emb.wordEmb.weightT hEnc

||| Linear (`L IO`) twin of `hfBertMlmForward`: consume the model handle, run the
||| read-only MLM forward, return the `[seqLen, vocab]` logits (banged) beside
||| the rebuilt model (single-owner per forward; see `hfBertSeqClassifyForwardL`).
export
hfBertMlmForwardL : {0 ex : Executor} -> UserExecutorCore ex => UserExecutorTraining ex
                 => {seqLen, vocab, hidden, numLayers, numHeads, headDim,
                     intermediate, maxPos, typeVocab : Nat}
                 -> {auto prf : hidden = numHeads * headDim}
                 -> (1 _ : BertForMaskedLmState vocab hidden numLayers intermediate maxPos typeVocab ex dt g)
                 -> (inputIds     : Tensor [seqLen] ex dt g)
                 -> (positionIds  : Tensor [seqLen] ex dt g)
                 -> (tokenTypeIds : Tensor [seqLen] ex dt g)
                 -> (attentionMask : Maybe (Tensor [seqLen, seqLen] ex dt g))
                 -> L IO {use = 1} (LPair (!* (Tensor [seqLen, vocab] ex dt g))
                                         (BertForMaskedLmState vocab hidden numLayers intermediate maxPos typeVocab ex dt g))
hfBertMlmForwardL (MkBertForMaskedLm base head) i p t mask = do
  out <- liftIO1 (hfBertMlmForward {numHeads} {headDim} (MkBertForMaskedLm base head) i p t mask)
  pure1 (MkBang out # MkBertForMaskedLm base head)

----------------------------------------------------------------------
-- Grad-mode retype + `eval` (inference)
----------------------------------------------------------------------
--
-- The composite state records aren't the `Params` leaf-kind (they carry
-- many config Nats), so the generic `Nn.eval` doesn't apply directly.
-- These field-wise `castGrad` helpers thread the grad-mode retype through
-- the record tree, reusing the leaf Nn types' `Params.castGrad`; the
-- matching `params` traversals collect every leaf param for the C-side
-- `requires_grad` flip. `evalBertForMaskedLm` = flip + retype, yielding
-- a genuinely tape-free inference model (`NoGrad`) the optimizer rejects.

-- Field-wise grad-mode retype (pure — `g` is an erased phantom).
castEmbeddingsState : BertEmbeddingsState v h mp tv ex dt g -> BertEmbeddingsState v h mp tv ex dt g'
castEmbeddingsState (MkBertEmbeddings we pe te ln) =
  MkBertEmbeddings (castGrad we) (castGrad pe) (castGrad te) (castGrad ln)

castSelfAttn : BertSelfAttentionState h ex dt g -> BertSelfAttentionState h ex dt g'
castSelfAttn (MkBertSelfAttn q k v) = MkBertSelfAttn (castGrad q) (castGrad k) (castGrad v)

castSelfOut : BertSelfOutputState h ex dt g -> BertSelfOutputState h ex dt g'
castSelfOut (MkBertSelfOut d ln) = MkBertSelfOut (castGrad d) (castGrad ln)

castIntermed : BertIntermediateState h i ex dt g -> BertIntermediateState h i ex dt g'
castIntermed (MkBertIntermediate d) = MkBertIntermediate (castGrad d)

castOutput : BertOutputState h i ex dt g -> BertOutputState h i ex dt g'
castOutput (MkBertOut d ln) = MkBertOut (castGrad d) (castGrad ln)

castLayer : BertLayerState h i ex dt g -> BertLayerState h i ex dt g'
castLayer (MkBertLayer sa so im ou) =
  MkBertLayer (castSelfAttn sa) (castSelfOut so) (castIntermed im) (castOutput ou)

castPooler : BertPoolerState h ex dt g -> BertPoolerState h ex dt g'
castPooler (MkBertPooler d) = MkBertPooler (castGrad d)

||| Retype a whole BERT encoder/pooler `WithGrad <-> NoGrad` (pure).
export
castBertModel : BertModelState v h nl i mp tv ex dt g -> BertModelState v h nl i mp tv ex dt g'
castBertModel (MkBertModel emb layers pool) =
  MkBertModel (castEmbeddingsState emb) (map castLayer layers) (castPooler pool)

castMlmHead : BertMlmHeadState v h ex dt g -> BertMlmHeadState v h ex dt g'
castMlmHead (MkBertMlmHead td tn b) = MkBertMlmHead (castGrad td) (castGrad tn) (retypeGrad b)

||| Retype a whole `BertForMaskedLM` `WithGrad <-> NoGrad` (pure).
export
castBertForMaskedLm : BertForMaskedLmState v h nl i mp tv ex dt g ->
                      BertForMaskedLmState v h nl i mp tv ex dt g'
castBertForMaskedLm (MkBertForMaskedLm base mlm) =
  MkBertForMaskedLm (castBertModel base) (castMlmHead mlm)

-- Param traversals (leaf params via the Nn `Params` instances).
embeddingsStateParams : BertEmbeddingsState v h mp tv ex dt g -> List SomeParam
embeddingsStateParams (MkBertEmbeddings we pe te ln) =
  params we ++ params pe ++ params te ++ params ln

selfAttnParams : BertSelfAttentionState h ex dt g -> List SomeParam
selfAttnParams (MkBertSelfAttn q k v) = params q ++ params k ++ params v

selfOutParams : BertSelfOutputState h ex dt g -> List SomeParam
selfOutParams (MkBertSelfOut d ln) = params d ++ params ln

intermedParams : BertIntermediateState h i ex dt g -> List SomeParam
intermedParams (MkBertIntermediate d) = params d

outputStateParams : BertOutputState h i ex dt g -> List SomeParam
outputStateParams (MkBertOut d ln) = params d ++ params ln

layerStateParams : BertLayerState h i ex dt g -> List SomeParam
layerStateParams (MkBertLayer sa so im ou) =
  selfAttnParams sa ++ selfOutParams so ++ intermedParams im ++ outputStateParams ou

poolerStateParams : BertPoolerState h ex dt g -> List SomeParam
poolerStateParams (MkBertPooler d) = params d

bertModelParams : BertModelState v h nl i mp tv ex dt g -> List SomeParam
bertModelParams (MkBertModel emb layers pool) =
  embeddingsStateParams emb
    ++ concatMap layerStateParams (toList layers)
    ++ poolerStateParams pool

mlmHeadStateParams : BertMlmHeadState v h ex dt g -> List SomeParam
mlmHeadStateParams (MkBertMlmHead td tn b) = params td ++ params tn ++ [toParam b]

bertForMaskedLmParams : BertForMaskedLmState v h nl i mp tv ex dt g -> List SomeParam
bertForMaskedLmParams (MkBertForMaskedLm base mlm) =
  bertModelParams base ++ mlmHeadStateParams mlm

||| Inference-mode `BertForMaskedLM`: flip every param's C `requires_grad`
||| off and retype the model `WithGrad -> NoGrad`. The result runs
||| genuinely tape-free (no `withNoGrad` bracket needed) and the optimizer
||| can't accept it (it needs a `WithGrad` loss). The transformer-model
||| counterpart of `Nn.eval` (which only fits the leaf `Params` kind).
export
evalBertForMaskedLm : {0 ex : Executor} -> UserExecutorTraining ex =>
                      {0 vocab, hidden, numLayers, intermediate, maxPos, typeVocab : Nat} ->
                      {0 dt : DType} ->
                      BertForMaskedLmState vocab hidden numLayers intermediate maxPos typeVocab ex dt WithGrad ->
                      IO (BertForMaskedLmState vocab hidden numLayers intermediate maxPos typeVocab ex dt NoGrad)
evalBertForMaskedLm m = do
  traverse_ (\p => primIO (primSetRequiresGrad {ex} p.paramPtr 0)) (bertForMaskedLmParams m)
  pure (castBertForMaskedLm m)

----------------------------------------------------------------------
-- fromPretrained
----------------------------------------------------------------------

||| The `BertForMaskedLM` state at the dims carried by a `BertConfig` —
||| the honest "shapes determined by a file at runtime" type. Used as
||| the second component of `fromPretrained`'s dependent pair so the dims
||| stay tied to the config the caller can read back.
public export
BertForMaskedLm : (cfg : BertConfig) -> (0 ex : Executor) -> (0 dt : DType) -> (0 g : GradMode) -> Type
BertForMaskedLm cfg ex dt g =
  BertForMaskedLmState (vocabSize cfg) (hidden cfg) (numLayers cfg)
                       (intermediate cfg) (maxPosition cfg) (typeVocabSize cfg) ex dt g

||| Read a HuggingFace BERT `config.json` into a `BertConfig`. Pulls the
||| seven architecture dims by their HF keys; `type_vocab_size` defaults
||| to 2 (the HF default) when omitted. Field/parse failures surface as
||| `ConfigError` naming the offending key.
export
readBertConfig : String -> IO (Either LoadError BertConfig)
readBertConfig path = do
  Right j <- readConfigFile path
    | Left e => pure (Left e)
  pure $ do
    vocabSize     <- natField   j "vocab_size"
    hidden        <- natField   j "hidden_size"
    numLayers     <- natField   j "num_hidden_layers"
    numHeads      <- natField   j "num_attention_heads"
    intermediate  <- natField   j "intermediate_size"
    maxPosition   <- natField   j "max_position_embeddings"
    typeVocabSize <- natFieldOr j "type_vocab_size" 2
    pure (MkBertConfig vocabSize hidden numLayers numHeads intermediate maxPosition typeVocabSize)

||| Load a pretrained `BertForMaskedLM` from a local HF model directory.
|||
||| Reads `<dir>/config.json` to recover the architecture dims (so the
||| shapes come from the file, not a hardcoded literal), builds the model
||| at those dims, then fills every param from `<dir>/model.safetensors`.
||| The returned dependent pair carries the `BertConfig` so the caller can
||| read the runtime-determined dims back out and reuse them in the forward.
|||
||| The grad mode is the caller's choice: `{g = NoGrad}` builds a tape-free
||| inference model (params born `requires_grad=0`), `{g = WithGrad}` a
||| fine-tunable one. Weights load with `allowCast` so an F32 checkpoint
||| populates an F64 model (and vice versa) without a manual convert.
export
fromPretrained : Backend ex dt => KnownGrad g
              => (modelDir : String)
              -> IO (Either LoadError (cfg : BertConfig ** BertForMaskedLm cfg ex dt g))
fromPretrained dir = do
  Right cfg <- readBertConfig (dir ++ "/config.json")
    | Left e => pure (Left e)
  model <- hfBertForMaskedLm {ex} {dt} {g}
             {vocab        = vocabSize cfg}
             {hidden       = hidden cfg}
             {numLayers    = numLayers cfg}
             {numHeads     = numHeads cfg}
             {intermediate = intermediate cfg}
             {maxPos       = maxPosition cfg}
             {typeVocab    = typeVocabSize cfg}
             "bert"
  loaded <- load {ex} (dir ++ "/model.safetensors") ({ allowCast := True } defaultLoadOpts)
  case loaded of
    Left e   => pure (Left e)
    Right () => pure (Right (cfg ** model))
