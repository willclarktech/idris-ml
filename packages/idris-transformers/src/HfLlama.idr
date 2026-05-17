||| Llama (3 / 3.1 / 3.2 family), HF-aligned.
|||
||| Target: `meta-llama/Llama-3.2-1B` (base, not Instruct — skips the
||| chat-template requirement for v1). Architecture covers the Llama 3
||| family; only the dims at construction time change between 1B / 3B
||| / 8B.
|||
||| Llama 3.2 1B specifics (from HF's `config.json` for the public-
||| facing checkpoint, accepted-license required to fetch the actual
||| weights via `HF_TOKEN`):
|||
|||   vocab_size            = 128256
|||   hidden_size           = 2048
|||   num_hidden_layers     = 16
|||   num_attention_heads   = 32
|||   num_key_value_heads   = 8        (GQA, 4 Q heads per KV head)
|||   head_dim              = 64       (= hidden / num_attention_heads)
|||   intermediate_size     = 8192
|||   max_position_embeddings = 131072 (with RoPE NTK scaling)
|||   rope_theta            = 500000.0
|||   rope_scaling          = { type: "llama3", factor: 32,
|||                              low_freq_factor: 1.0,
|||                              high_freq_factor: 4.0,
|||                              original_max_position: 8192 }
|||   rms_norm_eps          = 1e-5
|||   tie_word_embeddings   = true    (lm_head.weight = embed_tokens.weight)
|||
||| This module follows CONVENTIONS.md:
|||   - Param names match HF on-disk exactly: `model.embed_tokens.weight`,
|||     `model.layers.{i}.self_attn.q_proj.weight`, etc.
|||   - Storage shapes match HF on disk. Notably:
|||       - Q/K/V/O are SEPARATE linears (not fused — Llama doesn't
|||         fuse QKV the way GPT-2 does).
|||       - All Linears are BIAS-FREE (matches Llama's
|||         `nn.Linear(..., bias=False)`).
|||       - K and V projections are NARROWER than Q (GQA):
|||         `k_proj.weight : [numKvHeads * headDim, hidden]`
|||         `q_proj.weight : [numHeads * headDim, hidden]`
|||       - LM head is tied to `model.embed_tokens.weight`; NOT stored
|||         separately on disk (mirrors HfBert's `applyMlmHead`).
|||
||| Forward composition (per layer):
|||   x' = x + self_attn(RmsNorm(x))      -- pre-norm + residual
|||   y  = x' + mlp(RmsNorm(x'))           -- pre-norm + residual
|||
||| Self-attention is multi-head with GQA + RoPE on Q/K. The RoPE
||| tables + multi-head attention machinery land alongside this
||| module as the typed-wrapper layer + Layer/RoPE.idr work
||| absorbed from Phase 3. For now this file lands the config + state
||| records + param-name catalogue + smart constructor (no forward
||| pass yet); forward + KV cache + example come in follow-up commits.
module HfLlama

import Data.Vect

import Compat.Random
import Device
import Init
import Sampler
import Tensor


----------------------------------------------------------------------
-- Config
----------------------------------------------------------------------

||| HF Llama architecture knobs. Mirrors HF's `LlamaConfig` 1:1.
public export
record LlamaConfig where
  constructor MkLlamaConfig
  vocabSize    : Nat
  hidden       : Nat
  numLayers    : Nat
  numHeads     : Nat          -- query heads
  numKvHeads   : Nat          -- key/value heads (GQA: <= numHeads)
  headDim      : Nat          -- = hidden / numHeads
  intermediate : Nat
  maxPosition  : Nat
  ropeBase     : Double       -- = rope_theta
  rmsNormEps   : Double


||| `meta-llama/Llama-3.2-1B` config.
public export
llama32_1B_Config : LlamaConfig
llama32_1B_Config = MkLlamaConfig
  { vocabSize    = 128256
  , hidden       = 2048
  , numLayers    = 16
  , numHeads     = 32
  , numKvHeads   = 8
  , headDim      = 64
  , intermediate = 8192
  , maxPosition  = 131072
  , ropeBase     = 500000.0
  , rmsNormEps   = 1.0e-5
  }


----------------------------------------------------------------------
-- Param-name catalogue (pure Idris — single source of truth)
----------------------------------------------------------------------

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
  , p ++ ".self_attn.k_proj.weight"
  , p ++ ".self_attn.v_proj.weight"
  , p ++ ".self_attn.o_proj.weight"
  , p ++ ".post_attention_layernorm.weight"
  , p ++ ".mlp.gate_proj.weight"
  , p ++ ".mlp.up_proj.weight"
  , p ++ ".mlp.down_proj.weight"
  ]

||| All params HfLlama registers, in the order they're constructed.
||| For `llama32_1B_Config` (numLayers=16) this is 1 + 16*9 + 1 = 146
||| tensors. The LM head is tied to `embed_tokens.weight` (Llama-3.2's
||| `tie_word_embeddings=true`) and is NOT registered separately
||| (mirrors HfBert's MlmHead).
|||
||| Pfx is the HF on-disk prefix — typically `"model"`. Llama on disk
||| uses `model.embed_tokens.weight`, `model.layers.{i}.…`, etc.
||| Pass `""` to drop the prefix (testing).
public export
hfLlamaParamNames : (cfg : LlamaConfig) -> (pfx : String) -> List String
hfLlamaParamNames cfg pfx =
  let mkLayer = layerParamNames pfx
  in [embeddingsParamName pfx]
  ++ concatMap mkLayer (rangeNat cfg.numLayers)
  ++ [finalNormParamName pfx]

  where
    rangeNat : Nat -> List Nat
    rangeNat n = go n 0 []
      where
        go : Nat -> Nat -> List Nat -> List Nat
        go Z _ acc = reverse acc
        go (S k) i acc = go k (S i) (i :: acc)


----------------------------------------------------------------------
-- HF-named building blocks (private — Llama-specific layouts)
----------------------------------------------------------------------

-- Host-buffer helpers (one private copy per Hf* module per CONVENTIONS
-- rule 4 — no cross-imports between Hf* modules).
packDs : AnyPtr -> Int -> Vect n Double -> AnyPtr
packDs buf _   []        = buf
packDs buf off (x :: xs) = packDs (prim__setDouble buf off x) (off + 1) xs

fillConst : AnyPtr -> Int -> Int -> Double -> AnyPtr
fillConst buf _ 0 _ = buf
fillConst buf off n v =
  fillConst (prim__setDouble buf off v) (off + 1) (n - 1) v


||| Bias-free Linear with weight shape `[out, in]`. Matches Llama's
||| `nn.Linear(..., bias=False)` storage layout. Used for q_proj /
||| k_proj / v_proj / o_proj and the SwiGLU sublayer's gate / up /
||| down projections.
public export
record LlamaLinearNoBias (i, o : Nat) (0 d : Device) (0 dt : DType) (0 g : GradMode) where
  constructor MkLlamaLinear
  weight : Tensor [o, i] d dt g

makeLlamaLinear : UserDeviceTraining d => RuntimeDType dt => Linked d => Compatible d dt
               => {i, o : Nat}
               -> (paramFullName : String)
               -> IO (LlamaLinearNoBias i o d dt WithGrad)
makeLlamaLinear paramFullName = do
  let wCount = o * i
      wCountI = cast {to=Int} wCount
  weightVals <- traverse (\_ => map (* 0.02) normalSample) (Vect.replicate wCount ())
  let wBuf = prim__allocDoubles wCountI
      wBuf' = packDs wBuf 0 weightVals
  w <- tparam2d {o} {i} paramFullName wBuf'
  pure (MkLlamaLinear w)


||| HF-named RmsNorm: one weight tensor, no bias. The param name comes
||| from HF on-disk (`model.norm.weight`, `…input_layernorm.weight`,
||| `…post_attention_layernorm.weight`). The eps comes from the model
||| config (1e-5 for Llama 3).
public export
record LlamaRmsNorm (n : Nat) (0 d : Device) (0 dt : DType) (0 g : GradMode) where
  constructor MkLlamaRmsNorm
  weight : Tensor [n] d dt g

makeLlamaRmsNorm : UserDeviceTraining d => RuntimeDType dt => Linked d => Compatible d dt
                => {n : Nat}
                -> (paramFullName : String)
                -> IO (LlamaRmsNorm n d dt WithGrad)
makeLlamaRmsNorm paramFullName = do
  let nI = cast {to=Int} n
      wBuf = prim__allocDoubles nI
      wBuf' = fillConst wBuf 0 nI 1.0
  w <- tparam1d {n} paramFullName wBuf'
  pure (MkLlamaRmsNorm w)


||| Token embedding: `[vocab, hidden]`. Used for both the input
||| embedding lookup AND the (tied) LM head at forward time.
public export
record LlamaEmbedding (vocab, hidden : Nat) (0 d : Device) (0 dt : DType) (0 g : GradMode) where
  constructor MkLlamaEmbedding
  weight : Tensor [vocab, hidden] d dt g

makeLlamaEmbedding : UserDeviceTraining d => RuntimeDType dt => Linked d => Compatible d dt
                  => {vocab, hidden : Nat}
                  -> (paramFullName : String)
                  -> IO (LlamaEmbedding vocab hidden d dt WithGrad)
makeLlamaEmbedding paramFullName = do
  let nTotal = vocab * hidden
      nI = cast {to=Int} nTotal
  vals <- traverse (\_ => map (* 0.02) normalSample) (Vect.replicate nTotal ())
  let buf = prim__allocDoubles nI
      buf' = packDs buf 0 vals
  w <- tparam2d {o=vocab} {i=hidden} paramFullName buf'
  pure (MkLlamaEmbedding w)


----------------------------------------------------------------------
-- State records (one per HF Llama subtree)
----------------------------------------------------------------------

||| Attention sublayer state: four bias-free linears. K/V are narrower
||| than Q under GQA: their out-dim is `numKvHeads * headDim`, NOT
||| `numHeads * headDim`. The shapes here are explicit so the
||| compile-time check matches HF on-disk exactly.
public export
record LlamaAttentionState
        (hidden : Nat) (qOut : Nat) (kvOut : Nat)
        (0 d : Device) (0 dt : DType) (0 g : GradMode) where
  constructor MkLlamaAttention
  qProj : LlamaLinearNoBias hidden qOut  d dt g     -- [numHeads * headDim, hidden]
  kProj : LlamaLinearNoBias hidden kvOut d dt g     -- [numKvHeads * headDim, hidden]
  vProj : LlamaLinearNoBias hidden kvOut d dt g     -- [numKvHeads * headDim, hidden]
  oProj : LlamaLinearNoBias qOut hidden d dt g      -- [hidden, numHeads * headDim]


||| SwiGLU MLP sublayer. All three projections are bias-free. Mirrors
||| Layer.SwiGLU's record shape but with HF-aligned names re-bound at
||| construction.
public export
record LlamaMlpState
        (hidden : Nat) (intermediate : Nat)
        (0 d : Device) (0 dt : DType) (0 g : GradMode) where
  constructor MkLlamaMlp
  gateProj : LlamaLinearNoBias hidden intermediate d dt g
  upProj   : LlamaLinearNoBias hidden intermediate d dt g
  downProj : LlamaLinearNoBias intermediate hidden d dt g


||| One decoder block: pre-norm + attention + residual; pre-norm +
||| MLP + residual.
public export
record LlamaBlockState
        (hidden : Nat) (qOut : Nat) (kvOut : Nat) (intermediate : Nat)
        (0 d : Device) (0 dt : DType) (0 g : GradMode) where
  constructor MkLlamaBlock
  inputNorm    : LlamaRmsNorm hidden d dt g
  attn         : LlamaAttentionState hidden qOut kvOut d dt g
  postAttnNorm : LlamaRmsNorm hidden d dt g
  mlp          : LlamaMlpState hidden intermediate d dt g


||| Full Llama model state: token embedding + N decoder blocks +
||| final RmsNorm. LM head is tied to embed_tokens.weight; not stored
||| separately.
public export
record LlamaModelState
        (vocab : Nat) (hidden : Nat) (numLayers : Nat)
        (qOut : Nat) (kvOut : Nat) (intermediate : Nat)
        (0 d : Device) (0 dt : DType) (0 g : GradMode) where
  constructor MkLlamaModel
  embedTokens : LlamaEmbedding vocab hidden d dt g
  blocks      : Vect numLayers (LlamaBlockState hidden qOut kvOut intermediate d dt g)
  finalNorm   : LlamaRmsNorm hidden d dt g


----------------------------------------------------------------------
-- Smart constructors
----------------------------------------------------------------------

makeAttention : UserDeviceTraining d => RuntimeDType dt => Linked d => Compatible d dt
             => {hidden, qOut, kvOut : Nat}
             -> (layerPfx : String)
             -> IO (LlamaAttentionState hidden qOut kvOut d dt WithGrad)
makeAttention layerPfx = do
  q <- makeLlamaLinear {i=hidden} {o=qOut}  (layerPfx ++ ".self_attn.q_proj.weight")
  k <- makeLlamaLinear {i=hidden} {o=kvOut} (layerPfx ++ ".self_attn.k_proj.weight")
  v <- makeLlamaLinear {i=hidden} {o=kvOut} (layerPfx ++ ".self_attn.v_proj.weight")
  o <- makeLlamaLinear {i=qOut}   {o=hidden} (layerPfx ++ ".self_attn.o_proj.weight")
  pure (MkLlamaAttention q k v o)

makeMlp : UserDeviceTraining d => RuntimeDType dt => Linked d => Compatible d dt
       => {hidden, intermediate : Nat}
       -> (layerPfx : String)
       -> IO (LlamaMlpState hidden intermediate d dt WithGrad)
makeMlp layerPfx = do
  g  <- makeLlamaLinear {i=hidden}       {o=intermediate} (layerPfx ++ ".mlp.gate_proj.weight")
  u  <- makeLlamaLinear {i=hidden}       {o=intermediate} (layerPfx ++ ".mlp.up_proj.weight")
  dn <- makeLlamaLinear {i=intermediate} {o=hidden}       (layerPfx ++ ".mlp.down_proj.weight")
  pure (MkLlamaMlp g u dn)

makeBlock : UserDeviceTraining d => RuntimeDType dt => Linked d => Compatible d dt
         => {hidden, qOut, kvOut, intermediate : Nat}
         -> (layerPfx : String)
         -> IO (LlamaBlockState hidden qOut kvOut intermediate d dt WithGrad)
makeBlock layerPfx = do
  ln1 <- makeLlamaRmsNorm {n=hidden} (layerPfx ++ ".input_layernorm.weight")
  at  <- makeAttention {hidden} {qOut} {kvOut} layerPfx
  ln2 <- makeLlamaRmsNorm {n=hidden} (layerPfx ++ ".post_attention_layernorm.weight")
  mp  <- makeMlp {hidden} {intermediate} layerPfx
  pure (MkLlamaBlock ln1 at ln2 mp)

makeBlocks : UserDeviceTraining d => RuntimeDType dt => Linked d => Compatible d dt
          => {hidden, qOut, kvOut, intermediate : Nat}
          -> (modelPfx : String) -> (n : Nat) -> (offset : Nat)
          -> IO (Vect n (LlamaBlockState hidden qOut kvOut intermediate d dt WithGrad))
makeBlocks _   Z     _      = pure []
makeBlocks pfx (S k) offset = do
  b  <- makeBlock {hidden} {qOut} {kvOut} {intermediate}
                  (pfx ++ ".layers." ++ show offset)
  bs <- makeBlocks pfx k (S offset)
  pure (b :: bs)


||| Construct a full Llama model. Param-prefix is typically `"model"`
||| so registered names exactly match HF on-disk
||| (`model.embed_tokens.weight`, `model.layers.0.…`, etc.).
|||
||| `qOut = numHeads * headDim` and `kvOut = numKvHeads * headDim` are
||| explicit Nat args (not derived from the LlamaConfig) so the type
||| system catches dimension mismatches at construction time. For
||| `llama32_1B_Config`: qOut=2048, kvOut=512.
public export
hfLlamaModel : UserDeviceTraining d => RuntimeDType dt => Linked d => Compatible d dt
            => {vocab, hidden, numLayers, qOut, kvOut, intermediate : Nat}
            -> (modelPrefix : String)
            -> IO (LlamaModelState vocab hidden numLayers qOut kvOut intermediate d dt WithGrad)
hfLlamaModel pfx = do
  emb    <- makeLlamaEmbedding {vocab} {hidden} (pfx ++ ".embed_tokens.weight")
  blocks <- makeBlocks {hidden} {qOut} {kvOut} {intermediate} pfx numLayers 0
  ln     <- makeLlamaRmsNorm {n=hidden} (pfx ++ ".norm.weight")
  pure (MkLlamaModel emb blocks ln)
