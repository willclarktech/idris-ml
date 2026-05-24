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
import HfCommon
import Init
import Layer.RoPE
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
  -- Fused C-side normal(0, 0.02) init (commit 085348d). Replaces the
  -- `traverse normalSample` + `packDs` chain that, at Llama-3.2-1B's
  -- 1.24B-element scale, took 58 min for the full hfLlamaModel state
  -- (per the head-to-head in scripts/time_inference_llama.py). With
  -- the fused primitive, libtorch's torch::nn::init::normal_ runs the
  -- entire fill at memory-bandwidth speed in C — no per-element FFI.
  w <- tparam2dNormal {o} {i} paramFullName 0.0 0.02
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
  -- Fused C-side const fill (weight = 1.0). Replaces fillConst loop
  -- + per-element FFI.
  w <- tparam1dConst {n} paramFullName 1.0
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
  -- Fused C-side normal(0, 0.02) init. Llama 3.2's embed_tokens is
  -- [128256, 2048] = 263M elements — the single largest tensor in
  -- the model. At per-element FFI rates the host-side fill alone was
  -- ~10 min; under the fused-init primitive it's a libtorch in-place
  -- kernel that completes in ~ms.
  w <- tparam2dNormal {o=vocab} {i=hidden} paramFullName 0.0 0.02
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


----------------------------------------------------------------------
-- Forward (composed from existing 2D primitives + Layer.RoPE)
----------------------------------------------------------------------

%default partial

||| Per-position RmsNorm on a `[seqLen, hidden]` tensor. Thin wrapper
||| around `HfCommon.applyRmsNorm2dRaw` that pattern-matches the
||| `LlamaRmsNorm` wrapper. The body lives in `HfCommon.idr` so
||| HfBitNet (and any future adapter using the same per-row fold)
||| shares the implementation.
applyRmsNorm2d : {0 d : Device} -> UserDeviceTraining d => UserDeviceCore d =>
                 {seqLen, hidden : Nat} ->
                 (eps : Double) ->
                 LlamaRmsNorm hidden d dt g ->
                 Tensor [seqLen, hidden] d dt g ->
                 IO (Tensor [seqLen, hidden] d dt g)
applyRmsNorm2d eps (MkLlamaRmsNorm weight) input =
  applyRmsNorm2dRaw eps weight input


||| Bias-free Linear forward on `[seqLen, in] -> [seqLen, out]`.
||| Plain matmul `x @ W^T`. Used for q/k/v/o_proj and gate/up/down_proj.
applyLinear2d : {0 d : Device} -> UserDeviceTraining d =>
                LlamaLinearNoBias i o d dt g ->
                Tensor [seqLen, i] d dt g ->
                IO (Tensor [seqLen, o] d dt g)
applyLinear2d (MkLlamaLinear w) x = ioRerun (\_ =>
  let wT  = primTranspose2d {d} w.tensorPtr        -- [i, o]
      out = primMm {d} x.tensorPtr wT              -- [seqLen, o]
  in MkTensor out Nothing)


||| Embedding lookup: token IDs `[seqLen]` → `[seqLen, hidden]`. Same
||| pattern as HfBert.idr's applyEmbedLookup2d.
applyEmbedLookup : {0 d : Device} -> UserDeviceTraining d =>
                   {seqLen, vocab, hidden : Nat} ->
                   LlamaEmbedding vocab hidden d dt g ->
                   Tensor [seqLen] d dt g ->
                   IO (Tensor [seqLen, hidden] d dt g)
applyEmbedLookup {seqLen} {hidden} (MkLlamaEmbedding w) tokens = ioRerun (\_ =>
  let sI = cast {to=Int} seqLen
      hI = cast {to=Int} hidden
      flat = primEmbedding {d} w.tensorPtr tokens.tensorPtr sI hI
      twoD = primReshape2d {d} flat sI hI
  in MkTensor twoD Nothing)


-- Build the strict-upper-triangle causal mask (1.0 above diagonal,
-- 0.0 elsewhere). Same routine as Layer/Transformer.idr / HfGpt2.
writeCausalMask : AnyPtr -> Int -> Int -> Int -> AnyPtr
writeCausalMask buf i j n =
  if i >= n then buf
  else if j >= n then writeCausalMask buf (i + 1) (i + 2) n
  else let buf' = prim__setDouble buf (i * n + j) 1.0
       in writeCausalMask buf' i (j + 1) n


-- All-heads RoPE helper: reshape flat [seq, numH*headDim] projection
-- to rank-3 [seq, numH, headDim], call `applyRopeAllHeads` (which
-- handles all heads in one call via broadcast cos/sin over the head
-- axis), reshape back to flat. The flat ↔ rank-3 reshapes are
-- metadata-only on torch + mlx (view-with-strides) and copy-free on
-- tape (shape metadata only).
--
-- Replaces the per-head `buildRopedHeads` loop (closed 2026-05-30,
-- predecessor commit `6850366`) which emitted ~31 per-head concats
-- per layer × 2 (Q + K) × 16 layers ≈ ~1,000 concats per forward.
-- All-heads RoPE collapses that to ~7 ops per Q or K per layer
-- (broadcast muls + narrow + reshape + 1 concat for the rotate-half).
ropeAllHeadsFlat :
     {0 d : Device} -> UserDeviceTraining d =>
     {seq, numH, headDim, maxPos : Nat} ->
     RoPETables maxPos headDim d dt g ->
     (full : AnyPtr) ->                     -- [seq, numH * headDim]
     (sI, nHI, hdI : Int) ->
     IO AnyPtr
ropeAllHeadsFlat {d} {seq} {numH} {headDim} {maxPos} tables full sI nHI hdI = do
  full3 <- ioRerun (\_ =>
            the (Tensor [seq, numH, headDim] d dt g)
                (MkTensor (primReshape3d {d} full sI nHI hdI) Nothing))
  rot3 <- applyRopeAllHeads {seq} {numHeads=numH} {headDim} {maxPos} tables 0 full3
  ioRerun (\_ => primReshape2d {d} rot3.tensorPtr sI (nHI * hdI))


||| Full multi-head causal self-attention with GQA + RoPE.
applyAttention : {0 d : Device} -> UserDeviceTraining d => RuntimeDType dt => Linked d => Compatible d dt =>
                 {seq, hidden, numHeads, numKvHeads, headDim, maxPos : Nat} ->
                 -- NB: previously had `{auto qPrf : hidden = numHeads * headDim}`
                 -- and `{auto ratio : numHeads = numKvHeads * (div numHeads numKvHeads)}`.
                 -- Idris's Peano elaboration chokes at Llama 3.2 1B's
                 -- 32 * 64 = 2048 (above ~1000-Nat threshold per
                 -- gotchas.md). Dropped — caller is responsible for
                 -- passing coherent dims; misconfigured ratios become
                 -- runtime issues (garbage logits, no crash). -->
                 LlamaAttentionState hidden (numHeads * headDim) (numKvHeads * headDim) d dt g ->
                 RoPETables maxPos headDim d dt g ->
                 Tensor [seq, hidden] d dt g ->
                 IO (Tensor [seq, hidden] d dt g)
applyAttention {seq} {hidden} {numHeads} {numKvHeads} {headDim} {maxPos} attn tables input = do
  q <- applyLinear2d attn.qProj input  -- [seq, numHeads   * headDim]
  k <- applyLinear2d attn.kProj input  -- [seq, numKvHeads * headDim]
  v <- applyLinear2d attn.vProj input  -- [seq, numKvHeads * headDim]
  let sI     = cast {to=Int} seq
      hdI    = cast {to=Int} headDim
      nHI    = cast {to=Int} numHeads
      nKvHI  = cast {to=Int} numKvHeads
  -- All-heads RoPE on Q and K (V skips RoPE). One call per Q/K replaces
  -- the per-head loop (#399 Commit B-followup 2026-05-30): kills the
  -- ~62 per-layer `primConcat2dAxis1` calls that were ~80% of the
  -- forward op count post-SDPA. Net per layer drops from ~518 RoPE ops
  -- to ~15 (rank-3 broadcast cos/sin over [seq, numH, headDim] halves).
  qRopedPtr <- ropeAllHeadsFlat {d} {seq} {numH=numHeads}   {headDim} {maxPos} tables q.tensorPtr sI nHI   hdI
  kRopedPtr <- ropeAllHeadsFlat {d} {seq} {numH=numKvHeads} {headDim} {maxPos} tables k.tensorPtr sI nKvHI hdI
  -- ONE fused SDPA call replaces the per-head matmul/scale/mask/
  -- softmax/matmul loop. On torch-mps this routes to MPSGraph's fused
  -- attention kernel (~1 op/layer vs ~5/head/layer = 160/layer prior);
  -- mlx routes to its fast::sdpa; tape composes the existing kernels
  -- in one C call (saves Idris↔C FFI hops, same math).
  ctxPtr <- ioRerun (\_ =>
              primSdpa2d {d} qRopedPtr kRopedPtr v.tensorPtr
                         nHI nKvHI hdI 1)  -- isCausal=1
  ctxT <- ioRerun (\_ => MkTensor ctxPtr Nothing)
  applyLinear2d attn.oProj ctxT


||| SwiGLU MLP on `[seq, hidden]`. Three bias-free linears plus a
||| fused `primSwiGlu2d` (silu(gate) * up) middle stage — collapses the
||| previous `tsilu` + `tmul` pair into one FFI call per block.
applyMlp : {0 d : Device} -> UserDeviceTraining d => UserDeviceCore d =>
           LlamaMlpState hidden intermediate d dt g ->
           Tensor [seqLen, hidden] d dt g ->
           IO (Tensor [seqLen, hidden] d dt g)
applyMlp mlp x = do
  g <- applyLinear2d mlp.gateProj x       -- [seq, intermediate]
  u <- applyLinear2d mlp.upProj   x       -- [seq, intermediate]
  mid <- ioRerun (\_ =>
           let out = primSwiGlu2d {d} g.tensorPtr u.tensorPtr
           in MkTensor out Nothing)        -- [seq, intermediate]
  applyLinear2d mlp.downProj mid


||| One Llama decoder block: pre-norm + attn + residual; pre-norm +
||| MLP + residual.
applyBlock : {0 d : Device} -> UserDeviceTraining d => UserDeviceCore d
          => RuntimeDType dt => Linked d => Compatible d dt
          => {seq, hidden, numHeads, numKvHeads, headDim, intermediate, maxPos : Nat}
          -- qPrf / ratio proofs dropped — see applyAttention. -->
          -> (eps : Double)
          -> LlamaBlockState hidden (numHeads * headDim) (numKvHeads * headDim) intermediate d dt g
          -> RoPETables maxPos headDim d dt g
          -> Tensor [seq, hidden] d dt g
          -> IO (Tensor [seq, hidden] d dt g)
applyBlock {seq} {hidden} {numHeads} {numKvHeads} {headDim} eps blk tables x = do
  xLn1   <- applyRmsNorm2d eps blk.inputNorm x
  aOut   <- applyAttention {seq} {hidden} {numHeads} {numKvHeads} {headDim} blk.attn tables xLn1
  xMid   <- tadd x aOut
  xLn2   <- applyRmsNorm2d eps blk.postAttnNorm xMid
  mOut   <- applyMlp blk.mlp xLn2
  tadd xMid mOut


applyBlocks : {0 d : Device} -> UserDeviceTraining d => UserDeviceCore d
           => RuntimeDType dt => Linked d => Compatible d dt
           => {seq, hidden, numHeads, numKvHeads, headDim, intermediate, maxPos, n : Nat}
           -- qPrf / ratio proofs dropped — see applyAttention. -->
           -> (eps : Double)
           -> Vect n (LlamaBlockState hidden (numHeads * headDim) (numKvHeads * headDim) intermediate d dt g)
           -> RoPETables maxPos headDim d dt g
           -> Tensor [seq, hidden] d dt g
           -> IO (Tensor [seq, hidden] d dt g)
applyBlocks _   []        _      x = pure x
applyBlocks eps (b :: bs) tables x = do
  x' <- applyBlock {numHeads} {numKvHeads} {headDim} eps b tables x
  applyBlocks {numHeads} {numKvHeads} {headDim} eps bs tables x'


||| Forward pass: token IDs → final hidden state `[seq, hidden]`
||| post-`model.norm`. The LM head (tied to embed_tokens) is applied
||| separately via `hfLlamaForwardLm`.
public export
hfLlamaForward : {0 d : Device} -> UserDeviceTraining d => UserDeviceCore d
              => RuntimeDType dt => Linked d => Compatible d dt
              => {seq, vocab, hidden, numLayers, numHeads, numKvHeads, headDim, intermediate, maxPos : Nat}
              -- qPrf / ratio proofs dropped — see applyAttention. -->
              -> (eps : Double)
              -> LlamaModelState vocab hidden numLayers (numHeads * headDim) (numKvHeads * headDim) intermediate d dt g
              -> RoPETables maxPos headDim d dt g
              -> Tensor [seq] d dt g
              -> IO (Tensor [seq, hidden] d dt g)
hfLlamaForward {numHeads} {numKvHeads} {headDim} eps model tables tokens = do
  emb   <- applyEmbedLookup model.embedTokens tokens
  hMid  <- applyBlocks {numHeads} {numKvHeads} {headDim} eps model.blocks tables emb
  applyRmsNorm2d eps model.finalNorm hMid


||| LM head: tied to `embed_tokens.weight`. Output `[seq, vocab]`
||| logits per position. Reuses the embedding tensor as the LM
||| projection weight (same pattern as HfBert's applyMlmHead).
public export
hfLlamaForwardLm : {0 d : Device} -> UserDeviceTraining d => UserDeviceCore d
                => RuntimeDType dt => Linked d => Compatible d dt
                => {seq, vocab, hidden, numLayers, numHeads, numKvHeads, headDim, intermediate, maxPos : Nat}
                -- qPrf / ratio proofs dropped — see applyAttention. -->
                -> (eps : Double)
                -> LlamaModelState vocab hidden numLayers (numHeads * headDim) (numKvHeads * headDim) intermediate d dt g
                -> RoPETables maxPos headDim d dt g
                -> Tensor [seq] d dt g
                -> IO (Tensor [seq, vocab] d dt g)
hfLlamaForwardLm {numHeads} {numKvHeads} {headDim} eps model tables tokens = do
  hFinal <- hfLlamaForward {numHeads} {numKvHeads} {headDim} eps model tables tokens
  -- LM head via the tied embed_tokens.weight (shape [vocab, hidden]).
  -- tlinear2d expects weight [out, in] = [vocab, hidden] which matches.
  let vI = cast {to=Int} vocab
      zBuf = prim__allocDoubles vI  -- calloc-backed → already zeros
      zeroBias : Tensor [vocab] d dt g
      zeroBias = MkTensor (dtCreateState1d {d} {t=dt} vI zBuf (deviceStreamTag {d})) Nothing
  tlinear2d model.embedTokens.weight hFinal zeroBias
