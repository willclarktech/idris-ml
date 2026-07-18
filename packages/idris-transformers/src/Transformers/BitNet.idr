||| BitNet b1.58 (HF transformers `microsoft/bitnet-b1.58-2B-4T`),
||| HF-aligned.
|||
||| Target: `microsoft/bitnet-b1.58-2B-4T` (~2B params, ternary
||| linears throughout). The architecture is "BitNet-Llama" — Llama
||| 3.2-style decoder stack with `nn.Linear` everywhere swapped for
||| HF's `BitLinear` (ternary weight + scalar `weight_scale` + per-
||| token int8 activation quant) and two extra RmsNorms per block:
||| `attn_sub_norm` (between context aggregation and `o_proj`) and
||| `ffn_sub_norm` (between SwiGLU activation and `down_proj`).
|||
||| BitNet 2B-4T config (HF `BitNetConfig` defaults, microsoft/
||| bitnet-b1.58-2B-4T `config.json`):
|||
|||   vocab_size            = 128256
|||   hidden_size           = 2560
|||   num_hidden_layers     = 30
|||   num_attention_heads   = 20
|||   num_key_value_heads   = 5         (GQA, 4 Q heads per KV head)
|||   head_dim              = 128       (= hidden / num_attention_heads)
|||   intermediate_size     = 6912
|||   max_position_embeddings = 2048
|||   rope_theta            = 500000.0
|||   rms_norm_eps          = 1e-5
|||   tie_word_embeddings   = true      (lm_head shares embed_tokens.weight)
|||   attention_bias        = false     (all BitLinears are bias-free)
|||   hidden_act            = "relu2"   (squared ReLU; gate(x) * up(x))
|||
||| This module follows CONVENTIONS.md:
|||   - Param names match HF on-disk exactly: `model.embed_tokens.weight`,
|||     `model.layers.{i}.self_attn.q_proj.weight`,
|||     `model.layers.{i}.self_attn.q_proj.weight_scale`, etc.
|||   - Storage shapes match HF on disk. Notably:
|||       - BitLinear weights are packed uint8 axis-0:
|||         `[(out + 3) / 4, in]` raw on disk; we materialise them as
|||         `Tensor [out, in] ex Ternary NoGrad` via
|||         `tCreateTernaryFromHfPacked2d`.
|||       - `weight_scale` is a scalar `[1]` tensor in the model's
|||         compute dtype (F32 / F16 / BF16 — single value per linear).
|||       - All BitLinears are BIAS-FREE (`attention_bias=False`,
|||         MLP linears are explicit `bias=False`).
|||       - LM head is TIED to `embed_tokens.weight`
|||         (`tie_word_embeddings=True` in HF's config.json — there's no
|||         separate `lm_head.weight` on disk; `hfBitnetForwardLm`
|||         reuses the embedding tensor for the final projection).
|||   - One module-level state record per HF subtree (BitLinearHf,
|||     BitNetRmsNorm, Embedding, attention, MLP,
|||     block, model) so the type system pins shapes at construction.
|||
||| Forward composition (per layer):
|||   x' = x + o_proj(attn_sub_norm(SDPA(rope(q,k), v)))
|||              where q,k,v <- {q,k,v}_proj(RmsNorm_pre_attn(x))
|||   y  = x' + down_proj(ffn_sub_norm(silu(gate(x')) * up(x')))
|||              where gate,up <- {gate,up}_proj(RmsNorm_pre_mlp(x'))
|||
||| This commit lands the config + state records + param-name catalogue
||| + smart constructor only. Forward pass + checkpoint loading + the
||| roundtrip gate land in follow-up commits.
module Transformers.BitNet

import Data.Vect
import Language.Reflection
import Language.Reflection.Util

import Ml.Backend
import Ml.Checkpoint
import Ml.Compat.Random
import Ml.Executor
import Ml.GradMode
import Ml.Init
import Ml.Nn.Derive
import Ml.Nn.Embedding
import Ml.Nn.RmsNorm
import Ml.Nn.RoPE
import Ml.Sampler
import Ml.Tensor

import Transformers.Common
import Transformers.Config

%language ElabReflection

----------------------------------------------------------------------
-- Config
----------------------------------------------------------------------

||| HF BitNet architecture knobs. Mirrors HF's `BitNetConfig` 1:1.
||| Shape values are explicit Nats so they slot into Tensor dims at
||| construction; no inference from runtime values.
public export
record BitNetConfig where
  constructor MkBitNetConfig
  vocabSize    : Nat
  hidden       : Nat
  numLayers    : Nat
  numHeads     : Nat          -- query heads (= numAttentionHeads)
  numKvHeads   : Nat          -- key/value heads (GQA: <= numHeads)
  headDim      : Nat          -- = hidden / numHeads
  intermediate : Nat
  maxPosition  : Nat
  ropeBase     : Double       -- = rope_theta
  rmsNormEps   : Double

||| `microsoft/bitnet-b1.58-2B-4T` config.
public export
bitnet2B4T_Config : BitNetConfig
bitnet2B4T_Config = MkBitNetConfig
  { vocabSize    = 128256
  , hidden       = 2560
  , numLayers    = 30
  , numHeads     = 20
  , numKvHeads   = 5
  , headDim      = 128
  , intermediate = 6912
  , maxPosition  = 2048
  , ropeBase     = 500000.0
  , rmsNormEps   = 1.0e-5
  }

----------------------------------------------------------------------
-- Param-name catalogue (pure Idris — single source of truth)
----------------------------------------------------------------------
--
-- Each BitLinear contributes TWO params (weight + weight_scale); each
-- decoder block has 7 BitLinears (q/k/v/o + gate/up/down) + 4 RmsNorms
-- (input_layernorm, post_attention_layernorm, attn_sub_norm,
-- ffn_sub_norm). 7*2 + 4 = 18 params per layer. Plus 1 embedding +
-- 1 final norm at the top level (no `lm_head.weight` — tied).
--
-- For bitnet2B4T_Config (30 layers): 1 + 30*18 + 1 = 542 params.

layerPrefix : String -> Nat -> String
layerPrefix pfx i = pfx ++ ".layers." ++ show i

embeddingsParamName : (pfx : String) -> String
embeddingsParamName pfx = pfx ++ ".embed_tokens.weight"

finalNormParamName : (pfx : String) -> String
finalNormParamName pfx = pfx ++ ".norm.weight"

layerParamNames : (pfx : String) -> (i : Nat) -> List String
layerParamNames pfx i =
  let p = layerPrefix pfx i in
  [ p ++ ".input_layernorm.weight"
  , p ++ ".self_attn.q_proj.weight"
  , p ++ ".self_attn.q_proj.weight_scale"
  , p ++ ".self_attn.k_proj.weight"
  , p ++ ".self_attn.k_proj.weight_scale"
  , p ++ ".self_attn.v_proj.weight"
  , p ++ ".self_attn.v_proj.weight_scale"
  , p ++ ".self_attn.attn_sub_norm.weight"
  , p ++ ".self_attn.o_proj.weight"
  , p ++ ".self_attn.o_proj.weight_scale"
  , p ++ ".post_attention_layernorm.weight"
  , p ++ ".mlp.gate_proj.weight"
  , p ++ ".mlp.gate_proj.weight_scale"
  , p ++ ".mlp.up_proj.weight"
  , p ++ ".mlp.up_proj.weight_scale"
  , p ++ ".mlp.ffn_sub_norm.weight"
  , p ++ ".mlp.down_proj.weight"
  , p ++ ".mlp.down_proj.weight_scale"
  ]

||| All params Transformers.BitNet registers, in the order they're constructed.
||| For `bitnet2B4T_Config` (numLayers=30) this is 1 + 30*18 + 1
||| = 542 tensors. `pfx` is the HF on-disk prefix — typically `"model"`.
||| BitNet on disk uses `model.embed_tokens.weight`, `model.layers.{i}.…`,
||| and `model.norm.weight`. The LM head is NOT a separate weight —
||| `tie_word_embeddings=True` in `microsoft/bitnet-b1.58-2B-4T`'s
||| config.json, so the embedding's `[vocab, hidden]` weight is also
||| the LM-head projection at the top of `hfBitnetForwardLm`.
public export
hfBitnetParamNames : (cfg : BitNetConfig) -> (pfx : String) -> List String
hfBitnetParamNames cfg pfx =
  [embeddingsParamName pfx]
    ++ forBlocks cfg.numLayers (layerParamNames pfx)
    ++ [finalNormParamName pfx]

----------------------------------------------------------------------
-- HF-named building blocks (private — BitNet-specific layouts)
----------------------------------------------------------------------
--
-- Host-buffer helper (one private copy per Transformers.* module per CONVENTIONS
-- rule 4 — no cross-imports between Transformers.* modules).
fillBytesZero : AnyPtr -> Int -> Int -> AnyPtr
fillBytesZero buf _ 0   = buf
fillBytesZero buf off n =
  fillBytesZero (prim__setByte buf off 0) (off + 1) (n - 1)

||| BitLinear (HF-quant variant). Ternary weight `[out, in]` + scalar
||| `weight_scale` `[1]`. No bias — all BitNet linears are bias-free
||| under `attention_bias=False` + MLP `bias=False`.
|||
||| The two fields map to HF's two on-disk tensors:
|||   <pfx>.weight       -> uint8 packed [(out+3)/4, in]  ->  weightT
|||   <pfx>.weight_scale -> [1] in compute dtype          ->  weightScaleT
|||
||| `weightT` is `Ternary NoGrad` because BitNet b1.58 freezes both
||| the ternary weight and the dequant scale. The forward path reads
||| the scalar out of `weightScaleT` and passes it as a `Double` to
||| `tBitlinearFwdHfQuant`.
public export
record BitLinearHf (i, o : Nat) (0 ex : Executor) (0 dt : DType) (0 g : GradMode) where
  constructor MkBitLinearHf
  weightT      : Tensor [o, i] ex Ternary NoGrad
  weightScaleT : Tensor [1] ex dt NoGrad

-- Build a synthetic placeholder BitLinear. Ternary weight is all-zero
-- (zero-filled byte buffer); weight_scale is 1.0. These get overwritten
-- by the checkpoint load path; the constructor just needs valid handles
-- so subsequent forward calls don't fault on null pointers.
--
-- The weight_scale tensor IS registered under its HF paramId so
-- `loadModel` populates it. The weight tensor is NOT registered — the
-- ternary path goes through a custom load helper (filed under the
-- HfBitNetLoader follow-up) since the param registry's gradient
-- machinery assumes float-dtype storage. `weightName` is unused here
-- but kept in the signature so callers thread the HF param name and
-- the loader follow-up can pick it up without re-plumbing call sites.
-- `BitLinearHf`'s stored tensors are hardcoded `NoGrad` (BitNet freezes
-- both the ternary weight and the dequant scale), so its `g` index is a
-- pure phantom — construction is grad-mode-agnostic and needs no
-- `weakenGrad`. Only the float-typed makers below branch on `g`.
makeBitLinearHf : UserExecutorTraining ex => UserExecutorQuant ex
               => RuntimeDType dt => Linked ex => Compatible ex dt
               => {i, o : Nat}
               -> (weightName : String)
               -> (weightScaleName : String)
               -> IO (BitLinearHf i o ex dt g)
makeBitLinearHf weightName weightScaleName = do
  -- Ternary placeholder via zero-filled HF-packed buffer. The HF
  -- format is axis-0 packed `[(o+3)/4, i]`, so `((o+3) `div` 4) * i`
  -- bytes total. `prim__allocBytes` is calloc-backed → zero bytes →
  -- {0,0,0,0} ternary slots → all-zero weight.
  --
  -- `prim__allocBytes` returns AnyPtr (pure-typed), so a `let bytesPtr
  -- = prim__allocBytes …` reorders across sibling let-bindings via the
  -- gotcha in [[feedback_pure_typed_ffi_reorders]]. Wrap in `ioRerun`
  -- to force the allocation to fire at the right point in the IO
  -- sequence. Caught when adjacent `makeBitNetRmsNorm` calls in
  -- `makeBlock` (input_layernorm, post_attention_layernorm) stopped
  -- registering for the first ~17 of 30 layers.
  let oPackedI : Int
      oPackedI = cast {to=Int} ((o + 3) `div` 4)
      iI       : Int
      iI       = cast {to=Int} i
      totalBytes : Int
      totalBytes = oPackedI * iI
  bytesPtr <- ioRerun (\_ => prim__allocBytes totalBytes)
  w <- tCreateTernaryFromHfPacked2d {ex} {o} {i} bytesPtr
  -- Scalar weight_scale, registered under HF's `…weight_scale` name
  -- so `loadModel` populates it. Init to 1.0 (the post-load value
  -- will overwrite). Weaken to NoGrad — BitNet freezes weight_scale.
  scaleWg <- tparam1dConst {n=1} weightScaleName 1.0
  scale   <- weakenGrad scaleWg
  let _ = weightName  -- pinned in signature for the loader follow-up
  pure (MkBitLinearHf w scale)

||| HF-named RmsNorm. Same shape as Llama: one `[n]` weight, no bias,
||| eps from the model config (1e-5 for BitNet). Used for the four
||| RmsNorms per block (input_layernorm, post_attention_layernorm,
||| attn_sub_norm, ffn_sub_norm) plus the top-level `model.norm`.
makeBitNetRmsNorm : UserExecutorTraining ex => RuntimeDType dt => Linked ex => Compatible ex dt => KnownGrad g
                 => {n : Nat}
                 -> (paramFullName : String)
                 -> IO (RmsNorm n n ex dt g)
makeBitNetRmsNorm paramFullName = do
  w <- tparam1dConst {ex} {dt} {n} paramFullName 1.0
  case sgrad {g} of
    SWithGrad => pure (MkRmsNorm w)
    SNoGrad   => do
      w' <- weakenGrad w
      pure (MkRmsNorm w')

||| Token embedding: `[vocab, hidden]`. Stored under
||| `model.embed_tokens.weight`. This SAME tensor is reused as the
||| LM-head projection in `hfBitnetForwardLm`
||| (`tie_word_embeddings=True` — no separate `lm_head.weight`).
makeBitNetEmbedding : UserExecutorTraining ex => RuntimeDType dt => Linked ex => Compatible ex dt => KnownGrad g
                   => {vocab, hidden : Nat}
                   -> (paramFullName : String)
                   -> IO (Embedding vocab hidden ex dt g)
makeBitNetEmbedding paramFullName = do
  w <- tparam2dNormal {ex} {dt} {o=vocab} {i=hidden} paramFullName 0.0 0.02
  case sgrad {g} of
    SWithGrad => pure (MkEmbedding w)
    SNoGrad   => do
      w' <- weakenGrad w
      pure (MkEmbedding w')

-- LM head is tied to the token embedding (`tie_word_embeddings=True`).
-- `hfBitnetForwardLm` reuses `model.embedTokens.weightT` directly for
-- the final projection — no separate `BitNetLmHead` record.

----------------------------------------------------------------------
-- State records (one per HF BitNet subtree)
----------------------------------------------------------------------

||| Self-attention sublayer state: four BitLinears + the
||| BitNet-specific `attn_sub_norm` RmsNorm (applied between context
||| aggregation and `o_proj`). All BitLinears are bias-free; K/V
||| projections are narrower than Q under GQA.
public export
record BitNetAttentionState
        (hidden : Nat) (qOut : Nat) (kvOut : Nat)
        (0 ex : Executor) (0 dt : DType) (0 g : GradMode) where
  constructor MkBitNetAttention
  qProj : BitLinearHf hidden qOut ex dt g
  kProj : BitLinearHf hidden kvOut ex dt g
  vProj : BitLinearHf hidden kvOut ex dt g
  -- `attn_sub_norm` is applied to the post-SDPA `[seq, qOut]` tensor,
  -- so it's sized to `qOut` here, NOT `hidden`. For HF BitNet the
  -- config invariant is `hidden = numHeads * headDim = qOut`, so the
  -- safetensors-stored shape `[hidden_size]` loads cleanly under
  -- either typing — `qOut` is the one that makes the type-checked
  -- apply work without a proof. Same trick as Transformers.Llama's `qOut` in
  -- `oProj : LlamaLinearNoBias qOut hidden`.
  attnSubNorm : RmsNorm qOut qOut ex dt g
  oProj       : BitLinearHf qOut hidden ex dt g

||| MLP sublayer state. Three BitLinears (gate/up/down) + the BitNet-
||| specific `ffn_sub_norm` RmsNorm (over the intermediate dim, applied
||| between `act(gate)*up` and `down_proj`).
|||
||| Activation function is `hidden_act = "relu2"` (squared ReLU) per
||| HF's BitNet config defaults. The forward path applies this at
||| composition time — no state needed here.
public export
record BitNetMlpState
        (hidden : Nat) (intermediate : Nat)
        (0 ex : Executor) (0 dt : DType) (0 g : GradMode) where
  constructor MkBitNetMlp
  gateProj   : BitLinearHf hidden intermediate ex dt g
  upProj     : BitLinearHf hidden intermediate ex dt g
  ffnSubNorm : RmsNorm intermediate intermediate ex dt g
  downProj   : BitLinearHf intermediate hidden ex dt g

||| One decoder block: pre-norm + attention (with attn_sub_norm) +
||| residual; pre-norm + MLP (with ffn_sub_norm) + residual.
public export
record BitNetBlockState
        (hidden : Nat) (qOut : Nat) (kvOut : Nat) (intermediate : Nat)
        (0 ex : Executor) (0 dt : DType) (0 g : GradMode) where
  constructor MkBitNetBlock
  inputNorm    : RmsNorm hidden hidden ex dt g
  attn         : BitNetAttentionState hidden qOut kvOut ex dt g
  postAttnNorm : RmsNorm hidden hidden ex dt g
  mlp          : BitNetMlpState hidden intermediate ex dt g

||| Full BitNet model state: token embedding + N decoder blocks +
||| final RmsNorm + separate LM head (NOT tied).
public export
record BitNetModelState
        (vocab : Nat) (hidden : Nat) (numLayers : Nat)
        (qOut : Nat) (kvOut : Nat) (intermediate : Nat)
        (0 ex : Executor) (0 dt : DType) (0 g : GradMode) where
  constructor MkBitNetModel
  embedTokens : Embedding vocab hidden ex dt g
  blocks      : Vect numLayers (BitNetBlockState hidden qOut kvOut intermediate ex dt g)
  finalNorm   : RmsNorm hidden hidden ex dt g
  -- No `lmHead` field — `tie_word_embeddings=True` means the embed
  -- weight is also the LM-head projection (see `hfBitnetForwardLm`).

----------------------------------------------------------------------
-- Derived GCast (grad-mode retype + leaf-param traversal)
----------------------------------------------------------------------

-- `BitLinearHf`'s stored tensors are hardcoded `NoGrad` (the ternary
-- weight + dequant scale are frozen), so the record's `g` is a pure
-- phantom: the deriver's "field type doesn't mention `g`" rule passes
-- both fields through unchanged and contributes NO params. So `gparams`
-- on a BitNet model returns only the *float* params (embedding + the
-- RmsNorms) — exactly the set `eval`'s requires_grad flip cares about;
-- the frozen BitLinear tensors are correctly skipped. `gcast` is the
-- call-site-local rule wrapper around the imported pure `GCastImpl`.
gcast : List Name -> ParamTypeInfo -> Res (List TopLevel)
gcast nms p = GCastImpl nms p

%runElab derive `{BitLinearHf}          [gcast]
%runElab derive `{BitNetAttentionState} [gcast]
%runElab derive `{BitNetMlpState}       [gcast]
%runElab derive `{BitNetBlockState}     [gcast]
%runElab derive `{BitNetModelState}     [gcast]

||| Inference-mode BitNet: flip every (float) param's C `requires_grad`
||| off and retype `WithGrad -> NoGrad`. The BitLinear ternary weights +
||| scales are already frozen `NoGrad`, so `gparams` skips them.
export
evalBitnetModel : {0 ex : Executor} -> UserExecutorTraining ex =>
                  {0 vocab, hidden, numLayers, qOut, kvOut, intermediate : Nat} -> {0 dt : DType} ->
                  BitNetModelState vocab hidden numLayers qOut kvOut intermediate ex dt WithGrad ->
                  IO (BitNetModelState vocab hidden numLayers qOut kvOut intermediate ex dt NoGrad)
evalBitnetModel m = do
  traverse_ (\p => primIO (primSetRequiresGrad {ex} p.paramPtr 0)) (gparams m)
  pure (gcastGrad m)

----------------------------------------------------------------------
-- Smart constructors
----------------------------------------------------------------------

makeAttention : UserExecutorTraining ex => UserExecutorQuant ex
             => RuntimeDType dt => Linked ex => Compatible ex dt => KnownGrad g
             => {hidden, qOut, kvOut : Nat}
             -> (layerPfx : String)
             -> IO (BitNetAttentionState hidden qOut kvOut ex dt g)
makeAttention layerPfx = do
  q  <- makeBitLinearHf {i=hidden} {o=qOut}
          (layerPfx ++ ".self_attn.q_proj.weight")
          (layerPfx ++ ".self_attn.q_proj.weight_scale")
  k  <- makeBitLinearHf {i=hidden} {o=kvOut}
          (layerPfx ++ ".self_attn.k_proj.weight")
          (layerPfx ++ ".self_attn.k_proj.weight_scale")
  v  <- makeBitLinearHf {i=hidden} {o=kvOut}
          (layerPfx ++ ".self_attn.v_proj.weight")
          (layerPfx ++ ".self_attn.v_proj.weight_scale")
  sn <- makeBitNetRmsNorm {n=qOut}
          (layerPfx ++ ".self_attn.attn_sub_norm.weight")
  o  <- makeBitLinearHf {i=qOut} {o=hidden}
          (layerPfx ++ ".self_attn.o_proj.weight")
          (layerPfx ++ ".self_attn.o_proj.weight_scale")
  pure (MkBitNetAttention q k v sn o)

makeMlp : UserExecutorTraining ex => UserExecutorQuant ex
       => RuntimeDType dt => Linked ex => Compatible ex dt => KnownGrad g
       => {hidden, intermediate : Nat}
       -> (layerPfx : String)
       -> IO (BitNetMlpState hidden intermediate ex dt g)
makeMlp layerPfx = do
  g  <- makeBitLinearHf {i=hidden} {o=intermediate}
          (layerPfx ++ ".mlp.gate_proj.weight")
          (layerPfx ++ ".mlp.gate_proj.weight_scale")
  u  <- makeBitLinearHf {i=hidden} {o=intermediate}
          (layerPfx ++ ".mlp.up_proj.weight")
          (layerPfx ++ ".mlp.up_proj.weight_scale")
  fn <- makeBitNetRmsNorm {n=intermediate}
          (layerPfx ++ ".mlp.ffn_sub_norm.weight")
  dn <- makeBitLinearHf {i=intermediate} {o=hidden}
          (layerPfx ++ ".mlp.down_proj.weight")
          (layerPfx ++ ".mlp.down_proj.weight_scale")
  pure (MkBitNetMlp g u fn dn)

makeBlock : UserExecutorTraining ex => UserExecutorQuant ex
         => RuntimeDType dt => Linked ex => Compatible ex dt => KnownGrad g
         => {hidden, qOut, kvOut, intermediate : Nat}
         -> (layerPfx : String)
         -> IO (BitNetBlockState hidden qOut kvOut intermediate ex dt g)
makeBlock layerPfx = do
  ln1 <- makeBitNetRmsNorm {n=hidden} (layerPfx ++ ".input_layernorm.weight")
  at  <- makeAttention {hidden} {qOut} {kvOut} layerPfx
  ln2 <- makeBitNetRmsNorm {n=hidden} (layerPfx ++ ".post_attention_layernorm.weight")
  mp  <- makeMlp {hidden} {intermediate} layerPfx
  pure (MkBitNetBlock ln1 at ln2 mp)

makeBlocks : UserExecutorTraining ex => UserExecutorQuant ex
          => RuntimeDType dt => Linked ex => Compatible ex dt => KnownGrad g
          => {hidden, qOut, kvOut, intermediate : Nat}
          -> (modelPfx : String) -> (n : Nat) -> (offset : Nat)
          -> IO (Vect n (BitNetBlockState hidden qOut kvOut intermediate ex dt g))
makeBlocks _   Z     _      = pure []
makeBlocks pfx (S k) offset = do
  b  <- makeBlock {hidden} {qOut} {kvOut} {intermediate}
                  (pfx ++ ".layers." ++ show offset)
  bs <- makeBlocks pfx k (S offset)
  pure (b :: bs)

||| Construct a full BitNet model with named-param registration. The
||| param-prefix is typically `"model"` so registered names exactly
||| match HF on-disk (`model.embed_tokens.weight`, `model.layers.0.…`,
||| etc.). No LM-head param is created — `tie_word_embeddings=True`
||| means the embedding tensor serves as both.
|||
||| Like Transformers.Llama, `qOut = numHeads * headDim` and `kvOut = numKvHeads *
||| headDim` are explicit Nat args (not derived from the BitNetConfig)
||| so the type system catches dimension mismatches at construction
||| time. For `bitnet2B4T_Config`: qOut=2560 (= hidden, since 20*128),
||| kvOut=640 (= 5*128).
|||
||| BitLinear weights themselves are NOT populated by the standard
||| `loadModel` (which assumes float dtypes) — they're materialised
||| as zero-filled ternary placeholders here, and overwritten by
||| `loadHfBitnetCheckpoint`. The float-typed params (norms, embeddings,
||| weight_scales) ARE registered under their HF names and load via
||| the standard
||| safetensors path.
public export
hfBitnetModel : UserExecutorTraining ex => UserExecutorQuant ex
             => RuntimeDType dt => Linked ex => Compatible ex dt => KnownGrad g
             => {vocab, hidden, numLayers, qOut, kvOut, intermediate : Nat}
             -> (modelPrefix : String)
             -> IO (BitNetModelState vocab hidden numLayers qOut kvOut intermediate ex dt g)
hfBitnetModel pfx = do
  emb    <- makeBitNetEmbedding {vocab} {hidden} (pfx ++ ".embed_tokens.weight")
  blocks <- makeBlocks {hidden} {qOut} {kvOut} {intermediate} pfx numLayers 0
  ln     <- makeBitNetRmsNorm {n=hidden} (pfx ++ ".norm.weight")
  pure (MkBitNetModel emb blocks ln)

----------------------------------------------------------------------
-- Forward (composed from existing 2D primitives + Nn.RoPE +
-- tBitlinearFwdHfQuant for the BitLinears)
----------------------------------------------------------------------

%default partial

||| BitNet uses plain RoPE (no NTK / Llama-3 scaling). `factor=1.0`
||| in the LlamaRopeScaling record is the no-op short-circuit in
||| `applyLlamaFreqScaling`, so we get the standard RoPE table.
public export
bitnetRopeScaling : LlamaRopeScaling
bitnetRopeScaling = MkRopeScaling 1.0 1.0 1.0 0

||| Per-position RmsNorm on a `[seqLen, hidden]` tensor. Thin wrapper
||| around `Transformers.Common.applyRmsNorm2dRaw` that pattern-matches the
||| `BitNetRmsNorm` wrapper. The body (per-row fold over `primNarrow`
||| / `primMul` / `primSum` / scale) lives in `Transformers/Common.idr` and is
||| shared with Transformers.Llama.
export
applyRmsNorm2d : {0 ex : Executor} -> UserExecutorTraining ex => UserExecutorCore ex =>
                 {seqLen, hidden : Nat} ->
                 (eps : Double) ->
                 RmsNorm hidden hidden ex dt g ->
                 Tensor [seqLen, hidden] ex dt g ->
                 IO (Tensor [seqLen, hidden] ex dt g)
applyRmsNorm2d eps (MkRmsNorm weight) input =
  applyRmsNorm2dRaw eps weight input

||| 2D wrapper around the 1D `tBitlinearFwdHfQuant`. Walks `seqLen`
||| rows, calling the fused kernel per row, concatenating the [out]
||| outputs into a [seqLen, out] result. `useRmsNorm=False` here —
||| Transformers.BitNet applies the external `input_layernorm` /
||| `post_attention_layernorm` / `attn_sub_norm` / `ffn_sub_norm`
||| around the BitLinears, not the kernel's optional fused norm.
|||
||| Bias is bias-free in BitNet's BitLinears; the fused kernel takes
||| a bias arg unconditionally so we feed a zero-init [out] tensor.
||| Same zero-bias trick as Transformers.Llama's LM head + the kernel ignores
||| the rmsNormWeight when `useRmsNorm=False`, so we reuse the bias
||| placeholder as the placeholder rmsNormWeight (any [in]-shaped
||| tensor would do, but matching shapes keeps the call site simple).
||| One row of `applyBitLinearHf2d`. Lifted to top-level so its body
||| elaborates once at module compile, not per call site of the outer
||| `applyBitLinearHf2d` (7 BitLinears per block × 30 layers = 210
||| call sites on BitNet 2B-4T). Takes all dependencies (weight,
||| scale, bias, rms placeholder, input pointers + shapes) as plain
||| AnyPtr / Int / Double args so the body doesn't close over the
||| constraint-heavy `BitLinearHf i o ex dt g` record type.
private
bitlinearHfProcessRow : {0 ex : Executor} -> UserExecutorLinear ex
                     => UserExecutorQuant ex =>
                     (weightTPtr : AnyPtr) -> (scaleVal : Double) ->
                     (biasTPtr : AnyPtr) -> (rmsTPtr : AnyPtr) ->
                     (xPtr : AnyPtr) -> (iI : Int) -> (oI : Int) ->
                     (r : Int) -> AnyPtr
bitlinearHfProcessRow weightTPtr scaleVal biasTPtr rmsTPtr xPtr iI oI r =
  let row2d  = primNarrow {ex} xPtr 0 r 1                              -- [1, i]
      row1d  = primReshape1d {ex} row2d iI                             -- [i]
      rowOut = primBitlinearFwdHfQuant {ex} weightTPtr scaleVal
                 row1d biasTPtr 0 rmsTPtr 0.0
  in primReshape2d {ex} rowOut 1 oI

||| Row-folding helper for `applyBitLinearHf2d`. Lifted to top-level
||| (see `bitlinearHfProcessRow`).
private
bitlinearHfFoldRows : {0 ex : Executor} -> UserExecutorLinear ex
                   => UserExecutorQuant ex =>
                   (weightTPtr : AnyPtr) -> (scaleVal : Double) ->
                   (biasTPtr : AnyPtr) -> (rmsTPtr : AnyPtr) ->
                   (xPtr : AnyPtr) -> (iI : Int) -> (oI : Int) ->
                   (seqLenI : Int) -> (r : Int) -> (acc : AnyPtr) -> AnyPtr
bitlinearHfFoldRows weightTPtr scaleVal biasTPtr rmsTPtr xPtr iI oI seqLenI r acc =
  if r >= seqLenI
    then acc
    else bitlinearHfFoldRows {ex} weightTPtr scaleVal biasTPtr rmsTPtr xPtr iI oI seqLenI (r + 1)
           (primCat2 {ex} acc
             (bitlinearHfProcessRow {ex} weightTPtr scaleVal biasTPtr rmsTPtr xPtr iI oI r))

applyBitLinearHf2d : {0 ex : Executor} -> UserExecutorTraining ex => UserExecutorCore ex
                  => UserExecutorQuant ex => RuntimeDType dt
                  => Linked ex => Compatible ex dt
                  => {seqLen, i, o : Nat} ->
                  BitLinearHf i o ex dt g ->
                  Tensor [seqLen, i] ex dt g ->
                  IO (Tensor [seqLen, o] ex dt g)
applyBitLinearHf2d {seqLen} {i} {o} bl x = do
  let scaleVal : Double
      scaleVal = primItem {ex} bl.weightScaleT.tensorPtr
      oI       = cast {to=Int} o
      iI       = cast {to=Int} i
      -- Zero bias placeholder ([out], NoGrad). Calloc-backed buffer
      -- + dt-streamed creation, identical to Transformers.Llama's LM head trick.
      zBuf    = prim__allocDoubles oI
      biasPtr = dtCreateState1d {ex} {t=dt} oI zBuf (deviceStreamTag {ex})
      -- Placeholder rmsNormWeight ([in], NoGrad). C side won't read it
      -- since useRmsNorm=False, but the kernel signature still requires
      -- a non-null handle. Allocate a tiny zero buffer.
      rBuf    = prim__allocDoubles iI
      rmsPtr  = dtCreateState1d {ex} {t=dt} iI rBuf (deviceStreamTag {ex})
      wTPtr   = bl.weightT.tensorPtr
      xPtr    = x.tensorPtr
      seqLenI = cast {to=Int} seqLen
  -- Row loop: narrow → reshape to 1D → fused BitLinear → reshape to
  -- [1, out] → concat. Each layer's seven BitLinears × seqLen rows
  -- pays seqLen kernel launches per BitLinear. A fused 2D BitLinear
  -- kernel is the natural perf follow-up.
  ioRerun (\_ =>
    let out = if seqLen == 0
                then xPtr  -- impossible at well-typed call sites
                else bitlinearHfFoldRows {ex} wTPtr scaleVal biasPtr rmsPtr xPtr iI oI seqLenI 1
                       (bitlinearHfProcessRow {ex} wTPtr scaleVal biasPtr rmsPtr xPtr iI oI 0)
    in MkTensor out Nothing)

||| Embedding lookup: token IDs `[seqLen]` → `[seqLen, hidden]`.
||| Same pattern as Transformers.Llama's `applyEmbedLookup`.
export
applyEmbedLookup : {0 ex : Executor} -> UserExecutorTraining ex =>
                   {seqLen, vocab, hidden : Nat} ->
                   Embedding vocab hidden ex dt g ->
                   Tensor [seqLen] ex dt g ->
                   IO (Tensor [seqLen, hidden] ex dt g)
applyEmbedLookup {seqLen} {hidden} (MkEmbedding w) tokens = ioRerun (\_ =>
  let sI = cast {to=Int} seqLen
      hI  = cast {to=Int} hidden
      out = primEmbedding2d {ex} w.tensorPtr tokens.tensorPtr sI hI
  in MkTensor out Nothing)

-- `ropeAllHeadsFlat` is now imported from `Nn.RoPE` (consolidated from
-- the identical definitions Transformers.Llama and this module each carried).
-- BitNet uses RoPE only at prefill, so the call sites pass a fixed
-- positionOffset of 0 (the Nn.RoPE wrapper takes the offset that
-- Transformers.Llama's incremental-decode path threads through).

||| Full multi-head causal self-attention with GQA + RoPE +
||| BitNet-specific `attn_sub_norm` between context aggregation and
||| `o_proj`.
applyAttention : {0 ex : Executor} -> UserExecutorTraining ex => UserExecutorCore ex
              => UserExecutorQuant ex => RuntimeDType dt => Linked ex => Compatible ex dt
              => {seq, hidden, numHeads, numKvHeads, headDim, maxPos : Nat} ->
              (eps : Double) ->
              BitNetAttentionState hidden (numHeads * headDim) (numKvHeads * headDim) ex dt g ->
              RoPETables maxPos headDim ex dt ->
              Tensor [seq, hidden] ex dt g ->
              IO (Tensor [seq, hidden] ex dt g)
applyAttention {seq} {hidden} {numHeads} {numKvHeads} {headDim} {maxPos}
               eps attn tables input = do
  q <- applyBitLinearHf2d {seqLen=seq} attn.qProj input
  k <- applyBitLinearHf2d {seqLen=seq} attn.kProj input
  v <- applyBitLinearHf2d {seqLen=seq} attn.vProj input
  let sI    = cast {to=Int} seq
      hdI   = cast {to=Int} headDim
      nHI   = cast {to=Int} numHeads
      nKvHI = cast {to=Int} numKvHeads
  qRopedPtr <- ropeAllHeadsFlat {ex} {seq} {numH=numHeads}
                                {headDim} {maxPos} tables q.tensorPtr sI nHI   hdI 0
  kRopedPtr <- ropeAllHeadsFlat {ex} {seq} {numH=numKvHeads}
                                {headDim} {maxPos} tables k.tensorPtr sI nKvHI hdI 0
  ctxPtr <- ioRerun (\_ =>
              primSdpa2d {ex} qRopedPtr kRopedPtr v.tensorPtr
                         nHI nKvHI hdI 1)
  ctxT <- ioRerun (\_ => MkTensor ctxPtr Nothing)
  -- attn_sub_norm: BitNet-specific RmsNorm over the post-SDPA tensor,
  -- BEFORE the output projection. Diff with Llama. Sized to qOut to
  -- match ctxT's last dim (config invariant: hidden = numHeads *
  -- headDim = qOut).
  ctxNormed <- applyRmsNorm2d {seqLen=seq} {hidden=numHeads * headDim}
                              eps attn.attnSubNorm ctxT
  applyBitLinearHf2d {seqLen=seq} attn.oProj ctxNormed

-- BitNet's hidden_act is `relu2` — squared ReLU (`relu(x) ** 2`).
-- Composes from existing primitives. Element-wise.
applyRelu2 : {0 ex : Executor} -> UserExecutorCore ex =>
             {seqLen, n : Nat} ->
             Tensor [seqLen, n] ex dt g ->
             IO (Tensor [seqLen, n] ex dt g)
applyRelu2 x = do
  r <- trelu x
  tmul r r

||| MLP sublayer: gate/up BitLinears → relu² gate → ffn_sub_norm →
||| down BitLinear. Mirrors HF `BitNetMLP.forward`:
|||
|||   y = down_proj(ffn_sub_norm(act_fn(gate_proj(x)) * up_proj(x)))
applyMlp : {0 ex : Executor} -> UserExecutorTraining ex => UserExecutorCore ex
        => UserExecutorQuant ex => RuntimeDType dt => Linked ex => Compatible ex dt
        => {seqLen, hidden, intermediate : Nat} ->
        (eps : Double) ->
        BitNetMlpState hidden intermediate ex dt g ->
        Tensor [seqLen, hidden] ex dt g ->
        IO (Tensor [seqLen, hidden] ex dt g)
applyMlp {seqLen} {intermediate} eps mlp x = do
  g <- applyBitLinearHf2d {seqLen} mlp.gateProj x       -- [seq, intermediate]
  u <- applyBitLinearHf2d {seqLen} mlp.upProj   x       -- [seq, intermediate]
  ag <- applyRelu2 {seqLen} {n=intermediate} g          -- [seq, intermediate]
  gated <- tmul ag u                                     -- [seq, intermediate]
  normed <- applyRmsNorm2d {seqLen} {hidden=intermediate}
                           eps mlp.ffnSubNorm gated     -- [seq, intermediate]
  applyBitLinearHf2d {seqLen} mlp.downProj normed

||| One BitNet decoder block: pre-norm + attn (with attn_sub_norm) +
||| residual; pre-norm + MLP (with ffn_sub_norm) + residual.
export
applyBlock : {0 ex : Executor} -> UserExecutorTraining ex => UserExecutorCore ex
          => UserExecutorQuant ex => RuntimeDType dt => Linked ex => Compatible ex dt
          => {seq, hidden, numHeads, numKvHeads, headDim, intermediate, maxPos : Nat}
          -> (eps : Double)
          -> BitNetBlockState hidden (numHeads * headDim) (numKvHeads * headDim) intermediate ex dt g
          -> RoPETables maxPos headDim ex dt
          -> Tensor [seq, hidden] ex dt g
          -> IO (Tensor [seq, hidden] ex dt g)
applyBlock {seq} {hidden} {numHeads} {numKvHeads} {headDim} {intermediate}
           eps blk tables x =
  decoderBlockPreNorm
    (applyRmsNorm2d {seqLen=seq} {hidden} eps blk.inputNorm)
    (applyAttention {seq} {hidden} {numHeads} {numKvHeads} {headDim}
                    eps blk.attn tables)
    (applyRmsNorm2d {seqLen=seq} {hidden} eps blk.postAttnNorm)
    (applyMlp {seqLen=seq} {hidden} {intermediate} eps blk.mlp)
    x

applyBlocks : {0 ex : Executor} -> UserExecutorTraining ex => UserExecutorCore ex
           => UserExecutorQuant ex => RuntimeDType dt => Linked ex => Compatible ex dt
           => {seq, hidden, numHeads, numKvHeads, headDim, intermediate, maxPos, n : Nat}
           -> (eps : Double)
           -> Vect n (BitNetBlockState hidden (numHeads * headDim) (numKvHeads * headDim) intermediate ex dt g)
           -> RoPETables maxPos headDim ex dt
           -> Tensor [seq, hidden] ex dt g
           -> IO (Tensor [seq, hidden] ex dt g)
applyBlocks _   []        _      x = pure x
applyBlocks eps (b :: bs) tables x = do
  x' <- applyBlock {numHeads} {numKvHeads} {headDim} {intermediate} eps b tables x
  applyBlocks {numHeads} {numKvHeads} {headDim} {intermediate} eps bs tables x'

||| Forward pass: token IDs → final hidden state `[seq, hidden]`
||| post-`model.norm`. The LM head is a SEPARATE [vocab, hidden]
||| tensor (NOT tied to `embed_tokens.weight`), applied via
||| `hfBitnetForwardLm`.
public export
hfBitnetForward : {0 ex : Executor} -> UserExecutorTraining ex => UserExecutorCore ex
              => UserExecutorQuant ex => RuntimeDType dt => Linked ex => Compatible ex dt
              => {seq, vocab, hidden, numLayers, numHeads, numKvHeads, headDim, intermediate, maxPos : Nat}
              -> (eps : Double)
              -> BitNetModelState vocab hidden numLayers (numHeads * headDim) (numKvHeads * headDim) intermediate ex dt g
              -> RoPETables maxPos headDim ex dt
              -> Tensor [seq] ex dt g
              -> IO (Tensor [seq, hidden] ex dt g)
hfBitnetForward {numHeads} {numKvHeads} {headDim} {intermediate} eps model tables tokens = do
  emb   <- applyEmbedLookup model.embedTokens tokens
  hMid  <- applyBlocks {numHeads} {numKvHeads} {headDim} {intermediate}
                       eps model.blocks tables emb
  applyRmsNorm2d eps model.finalNorm hMid

||| LM head: tied to `embed_tokens.weight` ([vocab, hidden]).
||| Output `[seq, vocab]` logits per position. Bias-free, so we feed
||| a zero placeholder bias the same way Transformers.Llama does.
public export
hfBitnetForwardLm : {0 ex : Executor} -> UserExecutorTraining ex => UserExecutorCore ex
                => UserExecutorQuant ex => RuntimeDType dt => Linked ex => Compatible ex dt
                => {seq, vocab, hidden, numLayers, numHeads, numKvHeads, headDim, intermediate, maxPos : Nat}
                -> (eps : Double)
                -> BitNetModelState vocab hidden numLayers (numHeads * headDim) (numKvHeads * headDim) intermediate ex dt g
                -> RoPETables maxPos headDim ex dt
                -> Tensor [seq] ex dt g
                -> IO (Tensor [seq, vocab] ex dt g)
hfBitnetForwardLm {numHeads} {numKvHeads} {headDim} {intermediate} eps model tables tokens = do
  hFinal <- hfBitnetForward {numHeads} {numKvHeads} {headDim} {intermediate}
                            eps model tables tokens
  -- Tied LM head: HF's `tie_word_embeddings=True` means the embedding
  -- weight IS the LM-head projection. No separate `lm_head.weight`
  -- exists in the safetensors file for `microsoft/bitnet-b1.58-2B-4T`.
  projectTiedLmHead model.embedTokens.weightT hFinal

----------------------------------------------------------------------
-- Checkpoint load (B4.6 — HF-format ternary + float roundtrip)
----------------------------------------------------------------------
--
-- BitNet's safetensors checkpoint is a mix of two on-disk shapes:
--   1. Float-typed params (embed_tokens, all RmsNorm weights, every
--      BitLinear's weight_scale) — go through the standard
--      `load … {allowCast := True}` path. They're already in the C-side param
--      registry under their HF names from `makeBitNetEmbedding` /
--      `makeBitNetRmsNorm` / `makeBitLinearHf`.
--   2. Ternary BitLinear weights — stored as uint8 axis-0 packed
--      `[(out+3)/4, in]` with a custom 2-bit encoding the standard
--      `param_load*` path's dtype gate would refuse. We read the raw
--      bytes via `safetensorsReadRawBytes` and feed them to
--      `tCreateTernaryFromHfPacked2d` to materialise fresh Ternary
--      tensors, then splice them into a new `BitNetModelState`.
--
-- This pass returns a new model state with the ternary weights
-- overwritten; the float params are mutated in place by the C side.

%default partial

||| Load one HF-packed-uint8 ternary BitLinear weight by name. The
||| on-disk layout is `[(out+3)/4, in]` uint8 = `((out+3)/4) * in`
||| bytes; we allocate that, read the bytes, and route them through
||| `tCreateTernaryFromHfPacked2d` to get a `Tensor [o, i] ex Ternary
||| NoGrad`. Returns `Nothing` if the file/key is missing or the byte
||| count doesn't match — the caller keeps the placeholder weight.
loadHfTernaryWeight : {0 ex : Executor} -> UserExecutorQuant ex => Linked ex
                   => {o, i : Nat}
                   -> (path : String) -> (key : String)
                   -> IO (Maybe (Tensor [o, i] ex Ternary NoGrad))
loadHfTernaryWeight path key = do
  let oPackedI : Int
      oPackedI = cast {to=Int} ((o + 3) `div` 4)
      iI : Int
      iI = cast {to=Int} i
      expected : Int
      expected = oPackedI * iI
  buf <- ioRerun (\_ => prim__allocBytes expected)
  got <- safetensorsReadRawBytes path key buf expected
  if got /= expected
    then pure Nothing
    else do
      w <- tCreateTernaryFromHfPacked2d {ex} {o} {i} buf
      pure (Just w)

loadBitLinearTernary : {0 ex : Executor} -> UserExecutorQuant ex => Linked ex
                    => {i, o : Nat}
                    -> (path : String) -> (key : String)
                    -> BitLinearHf i o ex dt g
                    -> IO (BitLinearHf i o ex dt g, Bool)
loadBitLinearTernary path key bl = do
  mw <- loadHfTernaryWeight {ex} {o} {i} path key
  case mw of
    Nothing => pure (bl, False)
    Just w  => pure (MkBitLinearHf w bl.weightScaleT, True)

loadAttentionTernary : {0 ex : Executor} -> UserExecutorQuant ex => Linked ex
                    => {hidden, qOut, kvOut : Nat}
                    -> (path : String) -> (layerPfx : String)
                    -> BitNetAttentionState hidden qOut kvOut ex dt g
                    -> IO (BitNetAttentionState hidden qOut kvOut ex dt g, Nat)
loadAttentionTernary path lp (MkBitNetAttention q k v sn o) = do
  (q', okQ) <- loadBitLinearTernary {i=hidden} {o=qOut}  path (lp ++ ".self_attn.q_proj.weight") q
  (k', okK) <- loadBitLinearTernary {i=hidden} {o=kvOut} path (lp ++ ".self_attn.k_proj.weight") k
  (v', okV) <- loadBitLinearTernary {i=hidden} {o=kvOut} path (lp ++ ".self_attn.v_proj.weight") v
  (o', okO) <- loadBitLinearTernary {i=qOut}   {o=hidden} path (lp ++ ".self_attn.o_proj.weight") o
  let ok = (if okQ then 1 else 0) + (if okK then 1 else 0)
         + (if okV then 1 else 0) + (if okO then 1 else 0)
  pure (MkBitNetAttention q' k' v' sn o', ok)

loadMlpTernary : {0 ex : Executor} -> UserExecutorQuant ex => Linked ex
              => {hidden, intermediate : Nat}
              -> (path : String) -> (layerPfx : String)
              -> BitNetMlpState hidden intermediate ex dt g
              -> IO (BitNetMlpState hidden intermediate ex dt g, Nat)
loadMlpTernary path lp (MkBitNetMlp g u fn dn) = do
  (g',  okG) <- loadBitLinearTernary {i=hidden}       {o=intermediate} path (lp ++ ".mlp.gate_proj.weight") g
  (u',  okU) <- loadBitLinearTernary {i=hidden}       {o=intermediate} path (lp ++ ".mlp.up_proj.weight")   u
  (dn', okD) <- loadBitLinearTernary {i=intermediate} {o=hidden}       path (lp ++ ".mlp.down_proj.weight") dn
  let ok = (if okG then 1 else 0) + (if okU then 1 else 0) + (if okD then 1 else 0)
  pure (MkBitNetMlp g' u' fn dn', ok)

loadBlockTernary : {0 ex : Executor} -> UserExecutorQuant ex => Linked ex
                => {hidden, qOut, kvOut, intermediate : Nat}
                -> (path : String) -> (modelPfx : String) -> (idx : Nat)
                -> BitNetBlockState hidden qOut kvOut intermediate ex dt g
                -> IO (BitNetBlockState hidden qOut kvOut intermediate ex dt g, Nat)
loadBlockTernary path pfx idx (MkBitNetBlock ln1 at ln2 mp) = do
  let lp = layerPrefix pfx idx
  (at', nA) <- loadAttentionTernary {hidden} {qOut} {kvOut} path lp at
  (mp', nM) <- loadMlpTernary {hidden} {intermediate} path lp mp
  pure (MkBitNetBlock ln1 at' ln2 mp', nA + nM)

loadBlocksTernary : {0 ex : Executor} -> UserExecutorQuant ex => Linked ex
                 => {hidden, qOut, kvOut, intermediate : Nat}
                 -> (path : String) -> (modelPfx : String)
                 -> (offset : Nat)
                 -> Vect n (BitNetBlockState hidden qOut kvOut intermediate ex dt g)
                 -> IO (Vect n (BitNetBlockState hidden qOut kvOut intermediate ex dt g), Nat)
loadBlocksTernary _    _   _      []        = pure ([], 0)
loadBlocksTernary path pfx offset (b :: bs) = do
  (b',  nB)  <- loadBlockTernary {hidden} {qOut} {kvOut} {intermediate} path pfx offset b
  (bs', nBs) <- loadBlocksTernary {hidden} {qOut} {kvOut} {intermediate} path pfx (S offset) bs
  pure (b' :: bs', nB + nBs)

||| Load a microsoft/bitnet-b1.58-2B-4T checkpoint into a model state.
|||
|||   1. Walks every BitLinear in the model and replaces its ternary
|||      weight with one materialised from the safetensors file's
|||      packed-uint8 bytes (via `safetensorsReadRawBytes` +
|||      `tCreateTernaryFromHfPacked2d`).
|||   2. Loads all float-typed params (embed_tokens, all RmsNorms,
|||      every BitLinear `weight_scale`) in place via
|||      `load … {allowCast := True}`. The LM head is tied to `embed_tokens`,
|||      so no separate weight is loaded.
|||
||| Returns `(newModel, summary)` where `summary` is a triple of
||| `(ternaryLoaded, ternaryExpected, floatLoadOk)`. Callers typically
||| just check `ternaryLoaded == ternaryExpected && floatLoadOk`.
public export
loadHfBitnetCheckpoint :
  {0 ex : Executor} -> UserExecutorTraining ex => UserExecutorQuant ex
  => RuntimeDType dt => Linked ex => Compatible ex dt
  => {vocab, hidden, numLayers, qOut, kvOut, intermediate : Nat}
  -> (modelPrefix : String)
  -> (path : String)
  -> BitNetModelState vocab hidden numLayers qOut kvOut intermediate ex dt g
  -> IO ( BitNetModelState vocab hidden numLayers qOut kvOut intermediate ex dt g
        , (Nat, Nat, Bool))
loadHfBitnetCheckpoint pfx path model = do
  -- 1. Ternary BitLinear weights — read raw bytes per weight.
  (blocks', tnLoaded) <- loadBlocksTernary {hidden} {qOut} {kvOut} {intermediate}
                                            path pfx 0 model.blocks
  -- Each block carries 7 ternary weights: 4 in attention (q/k/v/o), 3 in
  -- MLP (gate/up/down). N layers × 7 = expected count.
  let tnExpected = numLayers * 7
  -- 2. Float-typed params via the standard allow_cast path. This
  -- mutates the existing param registry slots in place (the model
  -- record's float-typed Tensor fields keep their handles; their
  -- underlying C-side storage is overwritten).
  floatRes <- load {ex} path ({ allowCast := True } defaultLoadOpts)
  let floatOk  = case floatRes of { Right () => True; Left _ => False }
  let newModel = MkBitNetModel model.embedTokens blocks' model.finalNorm
  pure (newModel, (tnLoaded, tnExpected, floatOk))

----------------------------------------------------------------------
-- fromPretrained
----------------------------------------------------------------------

||| The BitNet model state at the dims a `BitNetConfig` carries. Like
||| Llama, the GQA out-dims (`numHeads * headDim` / `numKvHeads *
||| headDim`) are baked into the state type, so no extra proof is needed.
public export
BitNetModel : (cfg : BitNetConfig) -> (0 ex : Executor) -> (0 dt : DType) -> (0 g : GradMode) -> Type
BitNetModel cfg ex dt g =
  BitNetModelState (vocabSize cfg) (hidden cfg) (numLayers cfg)
                   (numHeads cfg * headDim cfg) (numKvHeads cfg * headDim cfg)
                   (intermediate cfg) ex dt g

||| Read a HuggingFace BitNet `config.json` into a `BitNetConfig`. Same
||| key set as Llama (BitNet's HF config mirrors `LlamaConfig`):
||| GQA head counts, `rope_theta` / `rms_norm_eps`, and `head_dim`
||| (defaulting to `hidden_size / num_attention_heads`).
export
readBitNetConfig : String -> IO (Either LoadError BitNetConfig)
readBitNetConfig path = do
  Right j <- readConfigFile path
    | Left e => pure (Left e)
  pure $ do
    vocabSize    <- natField    j "vocab_size"
    hidden       <- natField    j "hidden_size"
    numLayers    <- natField    j "num_hidden_layers"
    numHeads     <- natField    j "num_attention_heads"
    numKvHeads   <- natField    j "num_key_value_heads"
    intermediate <- natField    j "intermediate_size"
    maxPosition  <- natField    j "max_position_embeddings"
    headDim      <- natFieldOr  j "head_dim" (hidden `div` numHeads)
    ropeBase     <- doubleField j "rope_theta"
    rmsNormEps   <- doubleField j "rms_norm_eps"
    pure (MkBitNetConfig vocabSize hidden numLayers numHeads numKvHeads headDim
                         intermediate maxPosition ropeBase rmsNormEps)

||| Load a pretrained BitNet from a local HF model directory. Reads
||| `config.json` for the dims, builds the model (`{g}` = caller's grad
||| mode), then fills it via `loadHfBitnetCheckpoint` — which handles
||| BitNet's two-storage split (ternary BitLinear weights from raw packed
||| bytes; float params — norms / embeddings / weight_scales — via the
||| standard safetensors path). A partial load (missing ternary weights
||| or a float-load failure) surfaces as `ReadFailed`.
export
fromPretrained : Backend ex dt => UserExecutorQuant ex => KnownGrad g
              => (modelDir : String)
              -> IO (Either LoadError (cfg : BitNetConfig ** BitNetModel cfg ex dt g))
fromPretrained dir = do
  Right cfg <- readBitNetConfig (dir ++ "/config.json")
    | Left e => pure (Left e)
  model <- hfBitnetModel {ex} {dt} {g}
             {vocab        = vocabSize cfg}
             {hidden       = hidden cfg}
             {numLayers    = numLayers cfg}
             {qOut         = numHeads cfg * headDim cfg}
             {kvOut        = numKvHeads cfg * headDim cfg}
             {intermediate = intermediate cfg}
             "model"
  (model', (tnLoaded, tnExpected, floatOk)) <-
    loadHfBitnetCheckpoint {ex} {dt} "model" (dir ++ "/model.safetensors") model
  if floatOk && tnLoaded == tnExpected
    then pure (Right (cfg ** model'))
    else pure (Left ReadFailed)
