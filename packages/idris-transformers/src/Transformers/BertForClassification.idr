||| BERT sequence-classification head (HF `BertForSequenceClassification`).
|||
||| Mirrors the existing MLM-head pattern (Transformers.Bert.idr `BertMlmHeadState`):
||| one Linear-with-bias under `classifier.*` consuming the pooled
||| `[CLS]` embedding and producing per-class logits.
|||
||| Param naming on disk (HF `AutoModelForSequenceClassification` from
||| `google/bert_uncased_L-2_H-128_A-2`):
|||   bert.* (39 params at bertTinyConfig — backbone)
|||   classifier.weight                [numClasses, hidden]
|||   classifier.bias                  [numClasses]
||| Total: bertTinyConfig + 2-class head → 39 + 2 = 41 params.
|||
||| Forward composes `hfBertForward` (pooled [CLS]) with a 1-D
||| `tlinear` against the classifier head. Caller feeds the resulting
||| `[numClasses]` logits to `tnllLoss` against a one-hot target.
|||
||| Designed for the warm-start fine-tune workflow:
|||   loadModelPrefix "bert/model.safetensors" "bert."   -- backbone only
|||   -- (optional) freezeByPrefix opt "bert."
|||   -- train: backward through tnllLoss, nativeTrainStep
module Transformers.BertForClassification

import Data.Vect

import Executor
import Tensor
import Transformers.Bert

----------------------------------------------------------------------
-- Param-name catalogue
----------------------------------------------------------------------

||| The 2 classifier-head param names HF stores under `classifier.*`.
||| `classifierPrefix` lets unit tests register under a distinct
||| prefix to avoid colliding with prior test runs (mirrors the
||| `clsPrefix` knob in `mlmHeadParamNames`).
export
classifierHeadParamNames : (classifierPrefix : String) -> List String
classifierHeadParamNames pfx =
  [ pfx ++ ".weight"
  , pfx ++ ".bias"
  ]

||| Full catalogue for `BertForSequenceClassification` — the encoder +
||| pooler from `bertParamNames` plus the 2 classifier-head params
||| (`classifier.*`). For `bertTinyConfig` with `numClasses=3` the
||| total is 39 + 2 = 41 names.
export
bertForSequenceClassificationParamNames :
     (cfg : BertConfig)
  -> (bertPrefix : String)
  -> (classifierPrefix : String)
  -> List String
bertForSequenceClassificationParamNames cfg bertPfx clsPfx =
  bertParamNames cfg bertPfx ++ classifierHeadParamNames clsPfx

----------------------------------------------------------------------
-- State records
----------------------------------------------------------------------

||| Classifier head: a single Linear consuming the pooled [CLS]
||| embedding. The on-disk weight shape `[numClasses, hidden]` matches
||| `BertLinearWb hidden numClasses` exactly (the existing helper
||| already registers `<pfx>.weight` / `<pfx>.bias` under the right
||| HF-literal names).
|||
||| HF's `BertForSequenceClassification` interposes a `Dropout(p)`
||| between pooler and classifier. We omit it in the v1 head — dropout
||| at fine-tune time is a tuning knob the fine-tune row will revisit;
||| the worked example trains with `--freeze-backbone` first where
||| dropout is moot.
public export
record BertClassifierHeadState
        (hidden, numClasses : Nat)
        (0 ex : Executor) (0 dt : DType) (0 g : GradMode) where
  constructor MkBertClassifierHead
  dense : BertLinearWb hidden numClasses ex dt g

public export
record BertForSequenceClassificationState
        (vocab, hidden, numLayers, intermediate, maxPos, typeVocab, numClasses : Nat)
        (0 ex : Executor) (0 dt : DType) (0 g : GradMode) where
  constructor MkBertForSeqClassify
  base       : BertModelState vocab hidden numLayers intermediate maxPos typeVocab ex dt g
  classifier : BertClassifierHeadState hidden numClasses ex dt g

----------------------------------------------------------------------
-- Constructors
----------------------------------------------------------------------

||| Build a fresh classifier head under `<classifierPrefix>.weight` and
||| `<classifierPrefix>.bias`. Real callers pass `"classifier"` to
||| match HF; unit tests pass a distinct prefix (e.g. `"clftest"`) to
||| avoid C-side registry collisions.
export
makeClassifierHead :
     UserExecutorTraining ex => RuntimeDType dt => Linked ex => Compatible ex dt
  => {hidden, numClasses : Nat}
  -> (classifierPrefix : String)
  -> IO (BertClassifierHeadState hidden numClasses ex dt WithGrad)
makeClassifierHead pfx = do
  -- Classifier weight: shape [numClasses, hidden]; HF inits with
  -- Normal(0, 0.02). Classifier bias: shape [numClasses]; zero.
  -- (Mirrors the existing `makeBertLinear` shape but inlined here
  -- because that helper is private to `Transformers.Bert.idr`.)
  w <- tparam2dNormal {o=numClasses} {i=hidden} (pfx ++ ".weight") 0.0 0.02
  b <- tparam1dConst  {n=numClasses}            (pfx ++ ".bias")   0.0
  pure (MkBertClassifierHead (MkBertLinear w b))

||| Build a fresh BertForSequenceClassification. Backbone params live
||| under `<bertPrefix>.*` (pass `"bert"` to match HF on-disk names);
||| the classifier head lives under `<classifierPrefix>.*` (pass
||| `"classifier"` to match HF).
|||
||| Typical use:
|||   ` model <- hfBertForSequenceClassification {numClasses=3} "bert" "classifier"`
|||   ` _ <- loadModelPrefix "<path>/model.safetensors" "bert."`
|||
||| After construction, the registry contains `bertParamNames cfg
||| "bert"` (backbone) followed by `["classifier.weight",
||| "classifier.bias"]`. `loadModelPrefix _ "bert."` warm-starts the
||| former; the latter stays at its fresh-init (Normal(0,0.02) /
||| zero) — exactly what `BertForSequenceClassification.from_pretrained`
||| does on the Python side.
export
hfBertForSequenceClassification :
     UserExecutorTraining ex => RuntimeDType dt => Linked ex => Compatible ex dt
  => {vocab, hidden, numLayers, numHeads, intermediate, maxPos, typeVocab, numClasses : Nat}
  -> (bertPrefix : String)
  -> (classifierPrefix : String)
  -> IO (BertForSequenceClassificationState vocab hidden numLayers intermediate maxPos typeVocab numClasses ex dt WithGrad)
hfBertForSequenceClassification bertPfx clsPfx = do
  base <- hfBertModel {vocab} {hidden} {numLayers} {numHeads}
                      {intermediate} {maxPos} {typeVocab} bertPfx
  cls  <- makeClassifierHead {hidden} {numClasses} clsPfx
  pure (MkBertForSeqClassify base cls)

----------------------------------------------------------------------
-- Forward pass
----------------------------------------------------------------------

||| Full BertForSequenceClassification forward: input_ids (plus
||| position + tokenType IDs, and an optional attention mask) →
||| `[numClasses]` logits.
|||
|||   pooled  = hfBertForward(base, ids, position, tokenType, mask)  -- [hidden]
|||   logits  = classifier.dense.weight · pooled + classifier.dense.bias
|||
||| The caller composes this with `tnllLoss` against a one-hot target
||| at the example-level (no batching at this layer; the worked example
||| reduces over a `Vect n` of examples per epoch).
|||
||| Pass `Nothing` for `attentionMask` on fixed-length / non-padded
||| inputs — output is bit-identical to the pre-RT1 path.
export
hfBertSeqClassifyForward :
     {0 ex : Executor} -> UserExecutorCore ex => UserExecutorTraining ex
  => {seqLen, vocab, hidden, numLayers, numHeads, headDim,
      intermediate, maxPos, typeVocab, numClasses : Nat}
  -> {auto prf : hidden = numHeads * headDim}
  -> BertForSequenceClassificationState vocab hidden numLayers intermediate maxPos typeVocab numClasses ex dt g
  -> (inputIds     : Tensor [seqLen] ex dt g)
  -> (positionIds  : Tensor [seqLen] ex dt g)
  -> (tokenTypeIds : Tensor [seqLen] ex dt g)
  -> (attentionMask : Maybe (Tensor [seqLen, seqLen] ex dt g))
  -> IO (Tensor [numClasses] ex dt g)
hfBertSeqClassifyForward (MkBertForSeqClassify base (MkBertClassifierHead head)) i p t mask = do
  pooled <- hfBertForward {numHeads} {headDim} base i p t mask
  tlinear head.weight pooled head.bias
