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
||| This first cut exposes the param-name catalogue + the `BertConfig`
||| record. The model state records, smart constructor, and forward
||| pass land in subsequent commits.
module HfBert

import Data.List
import Data.Vect


----------------------------------------------------------------------
-- Config
----------------------------------------------------------------------

||| HF BERT architecture knobs. The fields mirror HF's `BertConfig`
||| spelling (`hidden_size` → `hidden`, `num_hidden_layers` →
||| `numLayers`, etc.) — Idris naming convention shortens but the
||| mapping to HF is 1:1.
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
-- Param-name catalogue
----------------------------------------------------------------------
--
-- These helpers are the single source of truth for the strings the
-- model registers under. When the model constructor lands (next
-- commit) it must call `tparam2d` / `tparam1d` with exactly these
-- strings — drift between this catalogue and the actual registration
-- order is a unit-test failure, and drift between either and HF's
-- on-disk naming is a Phase 6 round-trip failure.

namePrefix : String -> String
namePrefix pfx = pfx  -- typically "bert" — keep as a function so a
                      -- non-default prefix is one edit away.

embeddingsPrefix : String -> String
embeddingsPrefix pfx = pfx ++ ".embeddings"

encoderLayerPrefix : String -> Nat -> String
encoderLayerPrefix pfx i = pfx ++ ".encoder.layer." ++ show i

poolerPrefix : String -> String
poolerPrefix pfx = pfx ++ ".pooler"

||| The 5 embedding-block param names — three lookup tables plus the
||| post-sum LayerNorm. Order matches `state_dict()` insertion order,
||| not lexicographic.
embeddingsParamNames : (pfx : String) -> List String
embeddingsParamNames pfx =
  let p = embeddingsPrefix pfx in
  [ p ++ ".word_embeddings.weight"
  , p ++ ".position_embeddings.weight"
  , p ++ ".token_type_embeddings.weight"
  , p ++ ".LayerNorm.weight"
  , p ++ ".LayerNorm.bias"
  ]

||| The 16 per-encoder-layer param names. Each Linear contributes
||| `.weight` + `.bias`; each LayerNorm contributes `.weight` +
||| `.bias`. Total per block: 3 Q/K/V × 2 + (attn-output dense + LN)
||| × 2 + intermediate × 1 + (output dense + LN) × 2 = 16.
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

||| The 2 pooler param names — one Linear (`dense.weight` +
||| `dense.bias`) projecting the `[CLS]` token output through a
||| `tanh`.
poolerParamNames : (pfx : String) -> List String
poolerParamNames pfx =
  let p = poolerPrefix pfx in
  [ p ++ ".dense.weight"
  , p ++ ".dense.bias"
  ]

-- Enumerate layer 0..n-1 in ascending order. Recursive because
-- `[0 .. pred n]` is wrong for n = 0 (gives `[0]`, should be empty).
encoderParamNames : (pfx : String) -> (numLayers : Nat) -> List String
encoderParamNames _   Z     = []
encoderParamNames pfx (S k) =
  encoderParamNames pfx k ++ encoderLayerParamNames pfx k

||| The complete BERT param catalogue for `cfg`, in the order the
||| `hfBertModel` constructor (next commit) registers them. Equality
||| against this list is the unit-test gate that catches naming
||| drift before it reaches the C loader.
|||
||| With `numLayers = N`, the total length is `5 + 16*N + 2`.
||| For `bertTinyConfig` (N=2) this is 39 names.
export
bertParamNames : (cfg : BertConfig) -> (paramPrefix : String) -> List String
bertParamNames cfg pfx =
  embeddingsParamNames pfx
    ++ encoderParamNames pfx cfg.numLayers
    ++ poolerParamNames pfx
