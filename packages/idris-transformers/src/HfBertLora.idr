||| LoRA (Hu et al. 2021) adapter injection for `HfBert`. Provides
||| `BertLoraAdapters` — a value-level companion to `BertModelState`
||| holding per-layer `(A, B)` adapter pairs for the attention Q and
||| V projections (peft's canonical `target_modules=["query","value"]`)
||| — and `hfBertForwardWithLora` / `hfBertSeqClassifyForwardWithLora`,
||| which thread that companion through the encoder forward and
||| compose base Q/V output with the low-rank delta `(α/r) · B · A · x`.
|||
||| Key design choice: rather than duplicating the entire
||| `BertModelState` record hierarchy with LoRA-wrapped Linears
||| (which would require parallel `BertLayerLora` / `BertAttentionLora`
||| / etc. records and a parallel hfBertForward), the LoRA delta is
||| applied as a side-channel via the parallel `BertLoraAdapters`
||| record. Forward dispatches on `Maybe BertLoraAdapters`:
|||
|||  - `Nothing` → bit-identical to `hfBertForward`. Existing callers
|||    (loadHF + classify) keep working.
|||  - `Just ads` → at each layer's attention forward, the base Q
|||    output is augmented by `(α/r) · B_q · A_q · x` and similarly
|||    for V (matching peft's `bias="none"` default).
|||
||| Adapter param names follow HF / peft conventions exactly:
|||
|||     bert.encoder.layer.{i}.attention.self.query.lora_A
|||     bert.encoder.layer.{i}.attention.self.query.lora_B
|||     bert.encoder.layer.{i}.attention.self.value.lora_A
|||     bert.encoder.layer.{i}.attention.self.value.lora_B
|||
||| (the leading `base_model.model.` peft decoration + the
||| trailing `.default.weight` peft adapter-naming suffix are added
||| at the safetensors-IO boundary by the L4 `HfLoraIO` layer, NOT
||| baked into the in-memory paramId — keeps the in-Idris param
||| registry HF-aligned to match the backbone load path).
module HfBertLora

import Data.Vect

import Executor
import HfBert
import HfBertForClassification
import Tensor

----------------------------------------------------------------------
-- Adapter records
----------------------------------------------------------------------

||| A single LoRA adapter pair: `A : [r, i]` (Gaussian) + `B : [o, r]`
||| (zero). For BERT attention projections, `i = o = hidden`.
public export
record LoraAdapter (i : Nat) (o : Nat) (r : Nat)
                   (0 ex : Executor) (0 dt : DType) (0 g : GradMode) where
  constructor MkLoraAdapter
  loraA : Tensor [r, i] ex dt g
  loraB : Tensor [o, r] ex dt g

||| Per-encoder-layer LoRA adapter slots for the Q and V attention
||| projections (the peft default `target_modules=["query","value"]`).
||| `rankNat` and `alpha` are stored at value level so the forward
||| can compute the `(α/r)` scale at runtime — `r` itself is in the
||| type via the adapter `Vect`s' element types.
public export
record BertLoraAdapters (numLayers : Nat) (hidden : Nat) (r : Nat)
                       (0 ex : Executor) (0 dt : DType) (0 g : GradMode) where
  constructor MkBertLoraAdapters
  rankNat : Nat                                                       -- = r
  alpha   : Double                                                    -- = lora_alpha
  queries : Vect numLayers (LoraAdapter hidden hidden r ex dt g)
  values  : Vect numLayers (LoraAdapter hidden hidden r ex dt g)

----------------------------------------------------------------------
-- Constructor
----------------------------------------------------------------------

-- Build adapters for one attention module (e.g. "query" or "value")
-- across all encoder layers. Registers params under HF-aligned names.
buildPerLayerAdapters :
     {0 ex : Executor} -> UserExecutorTraining ex
  => RuntimeDType dt => Linked ex => Compatible ex dt
  => {hidden : Nat}
  -> (paramPrefix : String)
  -> (modName : String)
  -> (rank : Nat)
  -> (idx : Nat)
  -> (remaining : Nat)
  -> IO (Vect remaining (LoraAdapter hidden hidden rank ex dt WithGrad))
buildPerLayerAdapters _   _       _    _   Z     = pure []
buildPerLayerAdapters pfx modName rank idx (S k) = do
  let layerPfx = pfx ++ ".encoder.layer." ++ show idx ++ ".attention.self." ++ modName
  a <- tparam2dNormal {ex} {dt} {o=rank} {i=hidden}
                      (layerPfx ++ ".lora_A") 0.0 (1.0 / sqrt (cast rank))
  b <- tparam2dConst  {ex} {dt} {o=hidden} {i=rank}
                      (layerPfx ++ ".lora_B") 0.0
  rest <- buildPerLayerAdapters {hidden} pfx modName rank (S idx) k
  pure (MkLoraAdapter a b :: rest)

||| Construct a fresh `BertLoraAdapters` with per-layer Q and V
||| adapters registered under HF-aligned paramIds. `A` matrices are
||| Gaussian-init (std = 1/sqrt(rank), peft convention); `B`
||| matrices are zero-init so the t=0 LoRA contribution is identically
||| zero — `hfBertForwardWithLora model (Just lora) ...` produces
||| bit-identical output to `hfBertForward model ...` at construction.
|||
||| Caller MUST pass the same `hidden` and `numLayers` that the loaded
||| BertModelState was built with; the type system enforces this
||| because `BertLoraAdapters` shares its `(numLayers, hidden)`
||| params with `BertModelState`. Mismatched values fail to elaborate.
export
loraInjectBert :
     {0 ex : Executor} -> UserExecutorTraining ex
  => RuntimeDType dt => Linked ex => Compatible ex dt
  => {hidden : Nat}
  -> (paramPrefix : String)
  -> (numLayers : Nat)
  -> (rank : Nat)
  -> (alpha : Double)
  -> IO (BertLoraAdapters numLayers hidden rank ex dt WithGrad)
loraInjectBert pfx numLayers rank alpha = do
  qs <- buildPerLayerAdapters {hidden} pfx "query" rank 0 numLayers
  vs <- buildPerLayerAdapters {hidden} pfx "value" rank 0 numLayers
  pure (MkBertLoraAdapters rank alpha qs vs)

----------------------------------------------------------------------
-- Forward
----------------------------------------------------------------------

%default partial

-- Compute the LoRA delta for a single 2D linear: x[seq, hidden] +
-- (α/r) · B · A · x → returns delta only (not added to base yet).
-- Encoded as primMm + primTranspose2d to avoid materialising a
-- zero bias tensor. The full effective output `q + delta` is
-- computed by `addLoraDelta2d` below.
loraDelta2d :
     {0 ex : Executor} -> UserExecutorTraining ex
  => {seqLen, hidden, r : Nat}
  -> Tensor [seqLen, hidden] ex dt g
  -> LoraAdapter hidden hidden r ex dt g
  -> (rank : Nat)
  -> (alpha : Double)
  -> IO (Tensor [seqLen, hidden] ex dt g)
loraDelta2d input (MkLoraAdapter a b) rank alpha =
  ioRerun (\_ =>
    let aT     = primTranspose2d {ex} a.tensorPtr           -- [hidden, r]
        aOut   = primMm {ex} input.tensorPtr aT             -- [seq, r]
        bT     = primTranspose2d {ex} b.tensorPtr           -- [r, hidden]
        bOut   = primMm {ex} aOut bT                        -- [seq, hidden]
        scale  = alpha / cast rank
        scaled = primMulScalar {ex} bOut scale
    in MkTensor scaled Nothing)

-- Apply LoRA delta to a base linear output: `base + (α/r)·B·A·x`.
-- `Nothing` adapter degrades to a no-op (returns `base` as-is).
addLoraDelta2d :
     {0 ex : Executor} -> UserExecutorCore ex => UserExecutorTraining ex
  => {seqLen, hidden, r : Nat}
  -> (base : Tensor [seqLen, hidden] ex dt g)
  -> (input : Tensor [seqLen, hidden] ex dt g)
  -> (adapter : Maybe (LoraAdapter hidden hidden r ex dt g))
  -> (rank : Nat)
  -> (alpha : Double)
  -> IO (Tensor [seqLen, hidden] ex dt g)
addLoraDelta2d base _     Nothing     _    _     = pure base
addLoraDelta2d base input (Just adp)  rank alpha = do
  delta <- loraDelta2d input adp rank alpha
  tadd base delta

----------------------------------------------------------------------
-- LoRA-aware self-attention
----------------------------------------------------------------------

-- Mirror of `applySelfAttn` from HfBert.idr, with Maybe LoraAdapter
-- slots for Q and V (K + output dense are intentionally left at peft's
-- canonical default). When both adapters are Nothing AND
-- `Compatible Maybe = Just`, output bit-matches `applySelfAttn`.
applySelfAttnWithLora :
     {0 ex : Executor} -> UserExecutorCore ex => UserExecutorTraining ex
  => {seqLen, hidden, numHeads, headDim, r : Nat}
  -> {auto prf : hidden = numHeads * headDim}
  -> BertSelfAttentionState hidden ex dt g
  -> (qAdapter : Maybe (LoraAdapter hidden hidden r ex dt g))
  -> (vAdapter : Maybe (LoraAdapter hidden hidden r ex dt g))
  -> (rank : Nat)
  -> (alpha : Double)
  -> (mask : Maybe AnyPtr)
  -> Tensor [seqLen, hidden] ex dt g
  -> IO (Tensor [seqLen, hidden] ex dt g)
applySelfAttnWithLora {numHeads = Z} _ _ _ _ _ _ input                              = pure input
applySelfAttnWithLora {numHeads = S Z} {headDim} sa qAdp vAdp rank alpha mask input = do
  qBase <- tlinear2d sa.query.weight input sa.query.bias
  kBase <- tlinear2d sa.key.weight   input sa.key.bias
  vBase <- tlinear2d sa.value.weight input sa.value.bias
  q <- addLoraDelta2d qBase input qAdp rank alpha
  v <- addLoraDelta2d vBase input vAdp rank alpha
  ioRerun (\_ =>
    let scale  = 1.0 / sqrt (cast {to=Double} headDim)
        kT      = primTranspose2d {ex} kBase.tensorPtr
        scores  = primMulScalar {ex} (primMm {ex} q.tensorPtr kT) scale
        sMasked = case mask of
          Nothing => scores
          Just m  => primMaskedFill {ex} scores m (-1.0e20)
        attn = primSoftmax2d {ex} sMasked
        ctx  = primMm {ex} attn v.tensorPtr
    in MkTensor ctx Nothing)
applySelfAttnWithLora {numHeads = S (S k)} {headDim} sa qAdp vAdp rank alpha mask input = do
  qBase <- tlinear2d sa.query.weight input sa.query.bias
  kBase <- tlinear2d sa.key.weight   input sa.key.bias
  vBase <- tlinear2d sa.value.weight input sa.value.bias
  q <- addLoraDelta2d qBase input qAdp rank alpha
  v <- addLoraDelta2d vBase input vAdp rank alpha
  let headDimI = cast {to=Int} headDim
      scale  = 1.0 / sqrt (cast {to=Double} headDim)
      qP     = q.tensorPtr
      kP     = kBase.tensorPtr
      vP     = v.tensorPtr
      head0  = oneHeadCtx {ex} qP kP vP mask 0 headDimI scale
      ctxPtr = buildHeads {ex} qP kP vP mask headDimI scale (S k) headDimI head0
  pure (MkTensor ctxPtr Nothing)

-- LoRA-aware single layer.
applyLayerWithLora :
     {0 ex : Executor} -> UserExecutorCore ex => UserExecutorTraining ex
  => {seqLen, hidden, intermediate, numHeads, headDim, r : Nat}
  -> {auto prf : hidden = numHeads * headDim}
  -> BertLayerState hidden intermediate ex dt g
  -> (qAdapter : Maybe (LoraAdapter hidden hidden r ex dt g))
  -> (vAdapter : Maybe (LoraAdapter hidden hidden r ex dt g))
  -> (rank : Nat)
  -> (alpha : Double)
  -> (mask : Maybe AnyPtr)
  -> Tensor [seqLen, hidden] ex dt g
  -> IO (Tensor [seqLen, hidden] ex dt g)
applyLayerWithLora (MkBertLayer sa so im out) qAdp vAdp rank alpha mask input = do
  attnCtx  <- applySelfAttnWithLora {numHeads} {headDim} sa qAdp vAdp rank alpha mask input
  attnDen  <- tlinear2d so.dense.weight attnCtx so.dense.bias
  postAttn <- tadd input attnDen
  postLN1  <- applyLN2d so.layerNorm postAttn
  ffnHid   <- tlinear2d im.dense.weight postLN1 im.dense.bias
  ffnAct   <- tgelu ffnHid
  ffnOut   <- tlinear2d out.dense.weight ffnAct out.dense.bias
  postFfn  <- tadd postLN1 ffnOut
  applyLN2d out.layerNorm postFfn

-- LoRA-aware encoder fold: walks layers + adapter pairs in lockstep.
applyEncoderWithLora :
     {0 ex : Executor} -> UserExecutorCore ex => UserExecutorTraining ex
  => {seqLen, hidden, intermediate, numHeads, headDim, numLayers, r : Nat}
  -> {auto prf : hidden = numHeads * headDim}
  -> Vect numLayers (BertLayerState hidden intermediate ex dt g)
  -> Vect numLayers (LoraAdapter hidden hidden r ex dt g)   -- queries
  -> Vect numLayers (LoraAdapter hidden hidden r ex dt g)   -- values
  -> (rank : Nat)
  -> (alpha : Double)
  -> (mask : Maybe AnyPtr)
  -> Tensor [seqLen, hidden] ex dt g
  -> IO (Tensor [seqLen, hidden] ex dt g)
applyEncoderWithLora []        []         []         _    _     _    h   = pure h
applyEncoderWithLora (l :: ls) (qA :: qAs) (vA :: vAs) rank alpha mask h = do
  h' <- applyLayerWithLora {numHeads} {headDim} l (Just qA) (Just vA) rank alpha mask h
  applyEncoderWithLora {numHeads} {headDim} ls qAs vAs rank alpha mask h'

----------------------------------------------------------------------
-- Public forward functions
----------------------------------------------------------------------

||| Full BERT forward with optional LoRA adapters. `Nothing` for
||| `lora` is bit-identical to `hfBertForward` (re-routes through
||| `applyEncoder` directly); `Just adapters` composes per-layer
||| LoRA deltas into the Q and V outputs of every encoder layer.
||| `attentionMask` semantics mirror `hfBertForward` unchanged.
export
hfBertForwardWithLora :
     {0 ex : Executor} -> UserExecutorCore ex => UserExecutorTraining ex
  => {seqLen, vocab, hidden, numLayers, numHeads, headDim,
      intermediate, maxPos, typeVocab, r : Nat}
  -> {auto prf : hidden = numHeads * headDim}
  -> BertModelState vocab hidden numLayers intermediate maxPos typeVocab ex dt g
  -> Maybe (BertLoraAdapters numLayers hidden r ex dt g)
  -> (inputIds     : Tensor [seqLen] ex dt g)
  -> (positionIds  : Tensor [seqLen] ex dt g)
  -> (tokenTypeIds : Tensor [seqLen] ex dt g)
  -> (attentionMask : Maybe (Tensor [seqLen, seqLen] ex dt g))
  -> IO (Tensor [hidden] ex dt g)
hfBertForwardWithLora model Nothing inputIds positionIds tokenTypeIds mask =
  hfBertForward {numHeads} {headDim} model inputIds positionIds tokenTypeIds mask
hfBertForwardWithLora (MkBertModel emb layers pool) (Just lora)
                      inputIds positionIds tokenTypeIds mask = do
  hEmb <- applyEmbeddings emb inputIds positionIds tokenTypeIds
  hEnc <- applyEncoderWithLora {numHeads} {headDim} layers
                               lora.queries lora.values
                               lora.rankNat lora.alpha
                               (map (\m => m.tensorPtr) mask) hEmb
  applyPooler pool hEnc

||| Sequence-classification forward with optional LoRA adapters.
||| Composes `hfBertForwardWithLora` with the classifier head's
||| linear projection. The head's `head.weight` / `head.bias` stay
||| trainable in the canonical LoRA workflow (only the BERT backbone
||| is frozen + adapter-augmented).
export
hfBertSeqClassifyForwardWithLora :
     {0 ex : Executor} -> UserExecutorCore ex => UserExecutorTraining ex
  => {seqLen, vocab, hidden, numLayers, numHeads, headDim,
      intermediate, maxPos, typeVocab, numClasses, r : Nat}
  -> {auto prf : hidden = numHeads * headDim}
  -> BertForSequenceClassificationState vocab hidden numLayers intermediate
                                        maxPos typeVocab numClasses ex dt g
  -> Maybe (BertLoraAdapters numLayers hidden r ex dt g)
  -> (inputIds     : Tensor [seqLen] ex dt g)
  -> (positionIds  : Tensor [seqLen] ex dt g)
  -> (tokenTypeIds : Tensor [seqLen] ex dt g)
  -> (attentionMask : Maybe (Tensor [seqLen, seqLen] ex dt g))
  -> IO (Tensor [numClasses] ex dt g)
hfBertSeqClassifyForwardWithLora (MkBertForSeqClassify base (MkBertClassifierHead head))
                                 lora i p t mask = do
  pooled <- hfBertForwardWithLora {numHeads} {headDim} base lora i p t mask
  tlinear head.weight pooled head.bias
