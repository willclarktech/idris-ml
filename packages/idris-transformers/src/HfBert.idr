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
  let wCount = o * i
      wCountI = cast {to=Int} wCount
      oI = cast {to=Int} o
  weightVals <- traverse (\_ => map (* 0.02) normalSample) (Vect.replicate wCount ())
  let wBuf = prim__allocDoubles wCountI
      wBuf' = packDs wBuf 0 weightVals
      bBuf = prim__allocDoubles oI
      bBuf' = zeroBuf bBuf 0 oI
  w <- tparam2d {o} {i} (pfx ++ ".weight") wBuf'
  b <- tparam1d {n=o} (pfx ++ ".bias")     bBuf'
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
  let nTotal = vocab * dim
      nI = cast {to=Int} nTotal
  vals <- traverse (\_ => map (* 0.02) normalSample) (Vect.replicate nTotal ())
  let buf = prim__allocDoubles nI
      buf' = packDs buf 0 vals
  w <- tparam2d {o=vocab} {i=dim} (pfx ++ ".weight") buf'
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
  let nI = cast {to=Int} n
      gBuf = prim__allocDoubles nI
      gBuf' = fillConst gBuf 0 nI 1.0
      bBuf = prim__allocDoubles nI
      bBuf' = zeroBuf bBuf 0 nI
  g <- tparam1d {n} (pfx ++ ".weight") gBuf'
  b <- tparam1d {n} (pfx ++ ".bias")   bBuf'
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
