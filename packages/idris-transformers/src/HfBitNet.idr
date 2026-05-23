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
|||   tie_word_embeddings   = false     (lm_head is a separate weight)
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
|||         `Tensor [out, in] d Ternary NoGrad` via
|||         `tCreateTernaryFromHfPacked2d`.
|||       - `weight_scale` is a scalar `[1]` tensor in the model's
|||         compute dtype (F32 / F16 / BF16 — single value per linear).
|||       - All BitLinears are BIAS-FREE (`attention_bias=False`,
|||         MLP linears are explicit `bias=False`).
|||       - LM head is a SEPARATE `[vocab, hidden]` tensor under
|||         `lm_head.weight` (NOT tied to `embed_tokens.weight`).
|||   - One module-level state record per HF subtree (BitLinearHf,
|||     BitNetRmsNorm, BitNetEmbedding, BitNetLmHead, attention, MLP,
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
module HfBitNet

import Data.Vect

import Compat.Random
import Device
import Init
import Layer.RoPE
import Sampler
import Tensor


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
-- 1 final norm + 1 lm_head at the top level.
--
-- For bitnet2B4T_Config (30 layers): 1 + 30*18 + 1 + 1 = 543 params.

layerPrefix : String -> Nat -> String
layerPrefix pfx i = pfx ++ ".layers." ++ show i

embeddingsParamName : (pfx : String) -> String
embeddingsParamName pfx = pfx ++ ".embed_tokens.weight"

finalNormParamName : (pfx : String) -> String
finalNormParamName pfx = pfx ++ ".norm.weight"

||| LM head is NOT under `pfx` because HF stores it as a top-level
||| `lm_head.weight` (sibling to `model.…`), not under `model.…`.
lmHeadParamName : String
lmHeadParamName = "lm_head.weight"

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

||| All params HfBitNet registers, in the order they're constructed.
||| For `bitnet2B4T_Config` (numLayers=30) this is 1 + 30*18 + 1 + 1
||| = 543 tensors. `pfx` is the HF on-disk prefix — typically `"model"`.
||| BitNet on disk uses `model.embed_tokens.weight`, `model.layers.{i}.…`,
||| `model.norm.weight`, AND `lm_head.weight` (NOT under `model.`).
public export
hfBitnetParamNames : (cfg : BitNetConfig) -> (pfx : String) -> List String
hfBitnetParamNames cfg pfx =
  let mkLayer = layerParamNames pfx
  in [embeddingsParamName pfx]
  ++ concatMap mkLayer (rangeNat cfg.numLayers)
  ++ [finalNormParamName pfx]
  ++ [lmHeadParamName]

  where
    rangeNat : Nat -> List Nat
    rangeNat n = go n 0 []
      where
        go : Nat -> Nat -> List Nat -> List Nat
        go Z _ acc = reverse acc
        go (S k) i acc = go k (S i) (i :: acc)


----------------------------------------------------------------------
-- HF-named building blocks (private — BitNet-specific layouts)
----------------------------------------------------------------------
--
-- Host-buffer helper (one private copy per Hf* module per CONVENTIONS
-- rule 4 — no cross-imports between Hf* modules).
fillBytesZero : AnyPtr -> Int -> Int -> AnyPtr
fillBytesZero buf _ 0 = buf
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
record BitLinearHf (i, o : Nat) (0 d : Device) (0 dt : DType) (0 g : GradMode) where
  constructor MkBitLinearHf
  weightT      : Tensor [o, i] d Ternary NoGrad
  weightScaleT : Tensor [1] d dt NoGrad


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
makeBitLinearHf : UserDeviceTraining d => UserDeviceQuant d
               => RuntimeDType dt => Linked d => Compatible d dt
               => {i, o : Nat}
               -> (weightName : String)
               -> (weightScaleName : String)
               -> IO (BitLinearHf i o d dt WithGrad)
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
  w <- tCreateTernaryFromHfPacked2d {d} {o} {i} bytesPtr
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
public export
record BitNetRmsNorm (n : Nat) (0 d : Device) (0 dt : DType) (0 g : GradMode) where
  constructor MkBitNetRmsNorm
  weight : Tensor [n] d dt g

makeBitNetRmsNorm : UserDeviceTraining d => RuntimeDType dt => Linked d => Compatible d dt
                 => {n : Nat}
                 -> (paramFullName : String)
                 -> IO (BitNetRmsNorm n d dt WithGrad)
makeBitNetRmsNorm paramFullName = do
  w <- tparam1dConst {n} paramFullName 1.0
  pure (MkBitNetRmsNorm w)


||| Token embedding: `[vocab, hidden]`. Stored under
||| `model.embed_tokens.weight`. BitNet does NOT tie this to the LM
||| head (`tie_word_embeddings=False`); the LM head is a separate
||| `lm_head.weight` tensor.
public export
record BitNetEmbedding (vocab, hidden : Nat) (0 d : Device) (0 dt : DType) (0 g : GradMode) where
  constructor MkBitNetEmbedding
  weight : Tensor [vocab, hidden] d dt g

makeBitNetEmbedding : UserDeviceTraining d => RuntimeDType dt => Linked d => Compatible d dt
                   => {vocab, hidden : Nat}
                   -> (paramFullName : String)
                   -> IO (BitNetEmbedding vocab hidden d dt WithGrad)
makeBitNetEmbedding paramFullName = do
  w <- tparam2dNormal {o=vocab} {i=hidden} paramFullName 0.0 0.02
  pure (MkBitNetEmbedding w)


||| LM head: separate `[vocab, hidden]` tensor stored at `lm_head.weight`
||| (top-level, NOT under `model.…`). Bias-free.
public export
record BitNetLmHead (vocab, hidden : Nat) (0 d : Device) (0 dt : DType) (0 g : GradMode) where
  constructor MkBitNetLmHead
  weight : Tensor [vocab, hidden] d dt g

makeBitNetLmHead : UserDeviceTraining d => RuntimeDType dt => Linked d => Compatible d dt
                => {vocab, hidden : Nat}
                -> (paramFullName : String)
                -> IO (BitNetLmHead vocab hidden d dt WithGrad)
makeBitNetLmHead paramFullName = do
  w <- tparam2dNormal {o=vocab} {i=hidden} paramFullName 0.0 0.02
  pure (MkBitNetLmHead w)


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
        (0 d : Device) (0 dt : DType) (0 g : GradMode) where
  constructor MkBitNetAttention
  qProj       : BitLinearHf hidden qOut  d dt g
  kProj       : BitLinearHf hidden kvOut d dt g
  vProj       : BitLinearHf hidden kvOut d dt g
  attnSubNorm : BitNetRmsNorm hidden d dt g
  oProj       : BitLinearHf qOut hidden d dt g


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
        (0 d : Device) (0 dt : DType) (0 g : GradMode) where
  constructor MkBitNetMlp
  gateProj   : BitLinearHf hidden intermediate d dt g
  upProj     : BitLinearHf hidden intermediate d dt g
  ffnSubNorm : BitNetRmsNorm intermediate d dt g
  downProj   : BitLinearHf intermediate hidden d dt g


||| One decoder block: pre-norm + attention (with attn_sub_norm) +
||| residual; pre-norm + MLP (with ffn_sub_norm) + residual.
public export
record BitNetBlockState
        (hidden : Nat) (qOut : Nat) (kvOut : Nat) (intermediate : Nat)
        (0 d : Device) (0 dt : DType) (0 g : GradMode) where
  constructor MkBitNetBlock
  inputNorm    : BitNetRmsNorm hidden d dt g
  attn         : BitNetAttentionState hidden qOut kvOut d dt g
  postAttnNorm : BitNetRmsNorm hidden d dt g
  mlp          : BitNetMlpState hidden intermediate d dt g


||| Full BitNet model state: token embedding + N decoder blocks +
||| final RmsNorm + separate LM head (NOT tied).
public export
record BitNetModelState
        (vocab : Nat) (hidden : Nat) (numLayers : Nat)
        (qOut : Nat) (kvOut : Nat) (intermediate : Nat)
        (0 d : Device) (0 dt : DType) (0 g : GradMode) where
  constructor MkBitNetModel
  embedTokens : BitNetEmbedding vocab hidden d dt g
  blocks      : Vect numLayers (BitNetBlockState hidden qOut kvOut intermediate d dt g)
  finalNorm   : BitNetRmsNorm hidden d dt g
  lmHead      : BitNetLmHead vocab hidden d dt g


----------------------------------------------------------------------
-- Smart constructors
----------------------------------------------------------------------

makeAttention : UserDeviceTraining d => UserDeviceQuant d
             => RuntimeDType dt => Linked d => Compatible d dt
             => {hidden, qOut, kvOut : Nat}
             -> (layerPfx : String)
             -> IO (BitNetAttentionState hidden qOut kvOut d dt WithGrad)
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
  sn <- makeBitNetRmsNorm {n=hidden}
          (layerPfx ++ ".self_attn.attn_sub_norm.weight")
  o  <- makeBitLinearHf {i=qOut} {o=hidden}
          (layerPfx ++ ".self_attn.o_proj.weight")
          (layerPfx ++ ".self_attn.o_proj.weight_scale")
  pure (MkBitNetAttention q k v sn o)

makeMlp : UserDeviceTraining d => UserDeviceQuant d
       => RuntimeDType dt => Linked d => Compatible d dt
       => {hidden, intermediate : Nat}
       -> (layerPfx : String)
       -> IO (BitNetMlpState hidden intermediate d dt WithGrad)
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

makeBlock : UserDeviceTraining d => UserDeviceQuant d
         => RuntimeDType dt => Linked d => Compatible d dt
         => {hidden, qOut, kvOut, intermediate : Nat}
         -> (layerPfx : String)
         -> IO (BitNetBlockState hidden qOut kvOut intermediate d dt WithGrad)
makeBlock layerPfx = do
  ln1 <- makeBitNetRmsNorm {n=hidden} (layerPfx ++ ".input_layernorm.weight")
  at  <- makeAttention {hidden} {qOut} {kvOut} layerPfx
  ln2 <- makeBitNetRmsNorm {n=hidden} (layerPfx ++ ".post_attention_layernorm.weight")
  mp  <- makeMlp {hidden} {intermediate} layerPfx
  pure (MkBitNetBlock ln1 at ln2 mp)

makeBlocks : UserDeviceTraining d => UserDeviceQuant d
          => RuntimeDType dt => Linked d => Compatible d dt
          => {hidden, qOut, kvOut, intermediate : Nat}
          -> (modelPfx : String) -> (n : Nat) -> (offset : Nat)
          -> IO (Vect n (BitNetBlockState hidden qOut kvOut intermediate d dt WithGrad))
makeBlocks _   Z     _      = pure []
makeBlocks pfx (S k) offset = do
  b  <- makeBlock {hidden} {qOut} {kvOut} {intermediate}
                  (pfx ++ ".layers." ++ show offset)
  bs <- makeBlocks pfx k (S offset)
  pure (b :: bs)


||| Construct a full BitNet model with named-param registration. The
||| param-prefix is typically `"model"` so registered names exactly
||| match HF on-disk (`model.embed_tokens.weight`, `model.layers.0.…`,
||| etc.). The LM head is registered under top-level `lm_head.weight`,
||| NOT under `pfx`, matching HF on-disk.
|||
||| Like HfLlama, `qOut = numHeads * headDim` and `kvOut = numKvHeads *
||| headDim` are explicit Nat args (not derived from the BitNetConfig)
||| so the type system catches dimension mismatches at construction
||| time. For `bitnet2B4T_Config`: qOut=2560 (= hidden, since 20*128),
||| kvOut=640 (= 5*128).
|||
||| BitLinear weights themselves are NOT populated by the standard
||| `loadModel` (which assumes float dtypes) — they're materialised
||| as zero-filled ternary placeholders here, and overwritten by a
||| custom load helper (filed under the HfBitNetLoader follow-up).
||| The float-typed params (norms, embeddings, weight_scales, lm_head)
||| ARE registered under their HF names and load via the standard
||| safetensors path.
public export
hfBitnetModel : UserDeviceTraining d => UserDeviceQuant d
             => RuntimeDType dt => Linked d => Compatible d dt
             => {vocab, hidden, numLayers, qOut, kvOut, intermediate : Nat}
             -> (modelPrefix : String)
             -> IO (BitNetModelState vocab hidden numLayers qOut kvOut intermediate d dt WithGrad)
hfBitnetModel pfx = do
  emb    <- makeBitNetEmbedding {vocab} {hidden} (pfx ++ ".embed_tokens.weight")
  blocks <- makeBlocks {hidden} {qOut} {kvOut} {intermediate} pfx numLayers 0
  ln     <- makeBitNetRmsNorm {n=hidden} (pfx ++ ".norm.weight")
  lm     <- makeBitNetLmHead {vocab} {hidden} lmHeadParamName
  pure (MkBitNetModel emb blocks ln lm)
