||| GPT-2 (decoder-only), HF-aligned.
|||
||| Target: `sshleifer/tiny-gpt2` (the HuggingFace CI fixture — degenerate
||| dims, but exercises every GPT-2 architectural piece) and any GPT-2-family
||| checkpoint sharing the same `state_dict()` naming convention.
|||
||| This module follows the rules in `CONVENTIONS.md`:
|||   - Param names are literal HF-on-disk strings — `transformer.h.0.attn.c_attn.weight`,
|||     `transformer.wte.weight`, `transformer.ln_f.weight`, …
|||   - Storage shapes match HF on disk. GPT-2 has two warts the BERT module
|||     deliberately doesn't exercise:
|||       (1) **Fused QKV**: `attn.c_attn.weight` is one tensor of shape
|||           `[hidden, 3*hidden]` storing Q‖K‖V concatenated along axis=1.
|||           We split via `primNarrow ... 1 ...` at forward time. Pinned by
|||           `linear_shape_narrow::axis1_correctness_rank2` in the
|||           common-backend test suite.
|||       (2) **Conv1D transpose wart**: HF stores `c_attn`, `c_proj`,
|||           `mlp.c_fc`, `mlp.c_proj` weights as `[in, out]` (the transpose
|||           of `nn.Linear`'s `[out, in]`). We store them HF-natively and
|||           apply `y = x @ W + b` (`applyConv1D`); core `tlinear2d` is
|||           the wrong shape for this and stays bypassed.
|||
||| GPT-2 also differs from BERT on three structural points:
|||   - **Pre-norm**: `ln_1` runs BEFORE attention, `ln_2` BEFORE MLP;
|||     residual adds happen around each. BERT was post-norm.
|||   - **Causal mask**: applied via `primMaskedFill` on scores BEFORE softmax.
|||     BERT is bidirectional (no mask).
|||   - **Tied LM head**: `lm_head.weight` is tied to `transformer.wte.weight`
|||     — it isn't on disk separately. Mirrors the HfBert `MlmHead` pattern
|||     (`applyMlmHead` at Transformers.Bert.idr:769).
module Transformers.Gpt2

import Data.Vect

import Compat.Random
import Executor
import Transformers.Common
import Init
import Nn.Embedding
import Nn.LayerNorm
import Sampler
import Tensor

----------------------------------------------------------------------
-- Config
----------------------------------------------------------------------

||| HF GPT-2 architecture knobs. Field names mirror HF's `GPT2Config`
||| (`n_embd` → `hidden`, `n_layer` → `numLayers`, etc.). `headDim`
||| is the per-head Q/K/V dim — derived from `hidden / numHeads` by
||| caller convention; we keep it explicit so the type-level shape
||| math stays straightforward.
public export
record Gpt2Config where
  constructor MkGpt2Config
  vocabSize    : Nat
  hidden       : Nat
  numLayers    : Nat
  numHeads     : Nat
  headDim      : Nat   -- = hidden / numHeads
  intermediate : Nat   -- = 4 * hidden (GPT-2 default)
  maxPosition  : Nat

||| `distilgpt2` — the proof-of-concept target anchored by
||| `scripts/save_oracle_gpt2.py`. Pretrained GPT-2 distilled by HF;
||| 6 layers (half of gpt2-small's 12), hidden=768, n_head=12,
||| head_dim=64, intermediate=3072, max_pos=1024, vocab=50257. Same
||| HF-on-disk naming as gpt2 / gpt2-medium / gpt2-large / gpt2-xl, so
||| this module covers the whole GPT-2 family by swapping dims.
public export
distilGpt2Config : Gpt2Config
distilGpt2Config = MkGpt2Config
  { vocabSize    = 50257
  , hidden       = 768
  , numLayers    = 6
  , numHeads     = 12
  , headDim      = 64
  , intermediate = 3072
  , maxPosition  = 1024
  }

----------------------------------------------------------------------
-- Param-name catalogue (pure Idris — single source of truth)
----------------------------------------------------------------------

blockPrefix : String -> Nat -> String
blockPrefix pfx i = pfx ++ ".h." ++ show i

embeddingsParamNames : (pfx : String) -> List String
embeddingsParamNames pfx =
  [ pfx ++ ".wte.weight"
  , pfx ++ ".wpe.weight"
  ]

blockParamNames : (pfx : String) -> (i : Nat) -> List String
blockParamNames pfx i =
  let p = blockPrefix pfx i in
  [ p ++ ".ln_1.weight"
  , p ++ ".ln_1.bias"
  , p ++ ".attn.c_attn.weight"
  , p ++ ".attn.c_attn.bias"
  , p ++ ".attn.c_proj.weight"
  , p ++ ".attn.c_proj.bias"
  , p ++ ".ln_2.weight"
  , p ++ ".ln_2.bias"
  , p ++ ".mlp.c_fc.weight"
  , p ++ ".mlp.c_fc.bias"
  , p ++ ".mlp.c_proj.weight"
  , p ++ ".mlp.c_proj.bias"
  ]

finalNormParamNames : (pfx : String) -> List String
finalNormParamNames pfx =
  [ pfx ++ ".ln_f.weight"
  , pfx ++ ".ln_f.bias"
  ]

||| All params HfGpt2 registers, in the order they're constructed.
||| For `tinyGpt2Config` (numLayers=2) this is 2 + 2*12 + 2 = 28 params.
||| Note: the LM-head weight is tied to `wte.weight` and is NOT
||| registered separately (mirrors HfBert's MlmHead).
|||
||| Empty `pfx` produces unprefixed HF-native names (`transformer.wte.weight`).
||| Non-empty `pfx` produces `<pfx>.transformer.…` for scoped multi-network
||| examples (rare for HF-aligned modules).
public export
hfGpt2ParamNames : (cfg : Gpt2Config) -> (pfx : String) -> List String
hfGpt2ParamNames cfg pfx =
  let pfx_t = if pfx == "" then "transformer" else pfx ++ ".transformer"
  in embeddingsParamNames pfx_t
  ++ forBlocks cfg.numLayers (blockParamNames pfx_t)
  ++ finalNormParamNames pfx_t

----------------------------------------------------------------------
-- HF-named building blocks (mirror HfBert pattern but GPT-2-named)
----------------------------------------------------------------------

-- Small host-buffer helpers (copied from Transformers.Bert.idr's private region).
-- Each `Hf*` module owns these privately per CONVENTIONS rule 4
-- ("no cross-imports between Hf* modules"). The duplication is the
-- cost; the benefit is each module is independently readable
-- side-by-side with HF's reference Python.

packDs : AnyPtr -> Int -> Vect n Double -> AnyPtr
packDs buf _   []        = buf
packDs buf off (x :: xs) = packDs (prim__setDouble buf off x) (off + 1) xs

zeroBuf : AnyPtr -> Int -> Int -> AnyPtr
zeroBuf buf _   0 = buf
zeroBuf buf off n = zeroBuf (prim__setDouble buf off 0.0) (off + 1) (n - 1)

fillConst : AnyPtr -> Int -> Int -> Double -> AnyPtr
fillConst buf _ 0 _   = buf
fillConst buf off n v =
  fillConst (prim__setDouble buf off v) (off + 1) (n - 1) v

||| HF Conv1D: stored as `[in, out]` (transpose of `nn.Linear`).
||| At forward time: `y = x @ W + b` (`x` is `[batch, in]`, `W` is
||| `[in, out]`, `b` is `[out]`, result is `[batch, out]`).
public export
record Gpt2Conv1D (i, o : Nat) (0 ex : Executor) (0 dt : DType) (0 g : GradMode) where
  constructor MkGpt2Conv1D
  weight : Tensor [i, o] ex dt g  -- HF-native storage shape
  bias   : Tensor [o] ex dt g

makeConv1D : UserExecutorTraining ex => RuntimeDType dt => Linked ex => Compatible ex dt
          => {i, o : Nat}
          -> (paramPrefix : String)
          -> IO (Gpt2Conv1D i o ex dt WithGrad)
makeConv1D pfx = do
  -- HF GPT-2 uses normal(0, 0.02) init for Conv1D weights, zero bias.
  -- Fused C-side init via tparam2dNormal / tparam1dConst (commit
  -- 085348d); see Transformers.Bert.idr's makeBertLinear for the bottleneck
  -- replaced.
  w <- tparam2dNormal {o=i} {i=o} (pfx ++ ".weight") 0.0 0.02
  b <- tparam1dConst  {n=o}       (pfx ++ ".bias")   0.0
  pure (MkGpt2Conv1D w b)

||| HF-named LayerNorm: registers `<pfx>.weight` (γ, init 1.0) and
||| `<pfx>.bias` (β, init 0.0). GPT-2's `ln_*` / `ln_f` are standard
||| LayerNorms with affine params (unlike Llama's RMSNorm which only
||| has a weight).
makeGpt2LN : UserExecutorTraining ex => RuntimeDType dt => Linked ex => Compatible ex dt
          => {n : Nat}
          -> (paramPrefix : String)
          -> IO (LayerNorm n n ex dt WithGrad)
makeGpt2LN pfx = do
  -- Fused C-side const fill (γ = 1.0, β = 0.0); replaces fillConst /
  -- zeroBuf host-side loops.
  g <- tparam1dConst {n} (pfx ++ ".weight") 1.0
  b <- tparam1dConst {n} (pfx ++ ".bias")   0.0
  pure (MkLayerNorm g b)

||| Token / positional embedding: `[count, hidden]`.
makeGpt2Embedding : UserExecutorTraining ex => RuntimeDType dt => Linked ex => Compatible ex dt
                 => {count, hidden : Nat}
                 -> (paramPrefix : String)
                 -> IO (Embedding count hidden ex dt WithGrad)
makeGpt2Embedding pfx = do
  -- Fused C-side normal(0, 0.02) init.
  w <- tparam2dNormal {o=count} {i=hidden} (pfx ++ ".weight") 0.0 0.02
  pure (MkEmbedding w)

----------------------------------------------------------------------
-- State records
----------------------------------------------------------------------

public export
record Gpt2AttentionState
        (hidden : Nat)
        (0 ex : Executor) (0 dt : DType) (0 g : GradMode) where
  constructor MkGpt2Attention
  cAttn : Gpt2Conv1D hidden (3 * hidden) ex dt g  -- fused QKV
  cProj : Gpt2Conv1D hidden hidden ex dt g

public export
record Gpt2MlpState
        (hidden, intermediate : Nat)
        (0 ex : Executor) (0 dt : DType) (0 g : GradMode) where
  constructor MkGpt2Mlp
  cFc   : Gpt2Conv1D hidden intermediate ex dt g
  cProj : Gpt2Conv1D intermediate hidden ex dt g

public export
record Gpt2BlockState
        (hidden, intermediate : Nat)
        (0 ex : Executor) (0 dt : DType) (0 g : GradMode) where
  constructor MkGpt2Block
  ln1  : LayerNorm hidden hidden ex dt g
  attn : Gpt2AttentionState hidden ex dt g
  ln2  : LayerNorm hidden hidden ex dt g
  mlp  : Gpt2MlpState hidden intermediate ex dt g

public export
record Gpt2ModelState
        (vocab, hidden, numLayers, intermediate, maxPos : Nat)
        (0 ex : Executor) (0 dt : DType) (0 g : GradMode) where
  constructor MkGpt2Model
  wte    : Embedding vocab hidden ex dt g
  wpe    : Embedding maxPos hidden ex dt g
  blocks : Vect numLayers (Gpt2BlockState hidden intermediate ex dt g)
  lnF    : LayerNorm hidden hidden ex dt g

----------------------------------------------------------------------
-- Smart constructors
----------------------------------------------------------------------

makeAttention : UserExecutorTraining ex => RuntimeDType dt => Linked ex => Compatible ex dt
             => {hidden : Nat}
             -> (paramPrefix : String)
             -> IO (Gpt2AttentionState hidden ex dt WithGrad)
makeAttention pfx = do
  ca <- makeConv1D {i=hidden} {o=3 * hidden} (pfx ++ ".c_attn")
  cp <- makeConv1D {i=hidden} {o=hidden}     (pfx ++ ".c_proj")
  pure (MkGpt2Attention ca cp)

makeMlp : UserExecutorTraining ex => RuntimeDType dt => Linked ex => Compatible ex dt
       => {hidden, intermediate : Nat}
       -> (paramPrefix : String)
       -> IO (Gpt2MlpState hidden intermediate ex dt WithGrad)
makeMlp pfx = do
  cf <- makeConv1D {i=hidden}       {o=intermediate} (pfx ++ ".c_fc")
  cp <- makeConv1D {i=intermediate} {o=hidden}       (pfx ++ ".c_proj")
  pure (MkGpt2Mlp cf cp)

makeBlock : UserExecutorTraining ex => RuntimeDType dt => Linked ex => Compatible ex dt
         => {hidden, intermediate : Nat}
         -> (paramPrefix : String)
         -> IO (Gpt2BlockState hidden intermediate ex dt WithGrad)
makeBlock pfx = do
  l1 <- makeGpt2LN {n=hidden} (pfx ++ ".ln_1")
  at <- makeAttention {hidden} (pfx ++ ".attn")
  l2 <- makeGpt2LN {n=hidden} (pfx ++ ".ln_2")
  mp <- makeMlp {hidden} {intermediate} (pfx ++ ".mlp")
  pure (MkGpt2Block l1 at l2 mp)

makeBlocks : UserExecutorTraining ex => RuntimeDType dt => Linked ex => Compatible ex dt
          => {hidden, intermediate : Nat}
          -> (paramPrefix : String)
          -> (n : Nat)
          -> (offset : Nat)
          -> IO (Vect n (Gpt2BlockState hidden intermediate ex dt WithGrad))
makeBlocks _   Z     _      = pure []
makeBlocks pfx (S k) offset = do
  b  <- makeBlock {hidden} {intermediate} (blockPrefix pfx offset)
  bs <- makeBlocks pfx k (S offset)
  pure (b :: bs)

||| Construct a full GPT-2 model. All params register under HF-native
||| names with the supplied `paramPrefix` (typically `""` so the
||| registered names are exactly HF's on-disk names — `transformer.wte.weight`,
||| `transformer.h.0.attn.c_attn.weight`, etc.).
public export
hfGpt2Model : UserExecutorTraining ex => RuntimeDType dt => Linked ex => Compatible ex dt
           => {vocab, hidden, numLayers, numHeads, headDim, intermediate, maxPos : Nat}
           -> {auto prfH : hidden = numHeads * headDim}
           -> (paramPrefix : String)
           -> IO (Gpt2ModelState vocab hidden numLayers intermediate maxPos ex dt WithGrad)
hfGpt2Model pfx = do
  let pfx_t = if pfx == "" then "transformer" else pfx ++ ".transformer"
  wte    <- makeGpt2Embedding {count=vocab}  {hidden} (pfx_t ++ ".wte")
  wpe    <- makeGpt2Embedding {count=maxPos} {hidden} (pfx_t ++ ".wpe")
  blocks <- makeBlocks {hidden} {intermediate} pfx_t numLayers 0
  lnF    <- makeGpt2LN {n=hidden} (pfx_t ++ ".ln_f")
  pure (MkGpt2Model wte wpe blocks lnF)

----------------------------------------------------------------------
-- Forward primitives (per-block, untyped at the AnyPtr boundary
-- inside the per-head loops — same pattern as HfBert's applySelfAttn)
----------------------------------------------------------------------

-- ε for LayerNorm. HF GPT-2 defaults to 1e-5 (per modeling_gpt2.py).
gpt2LnEps : Double
gpt2LnEps = 1.0e-5

applyLN2d : {0 ex : Executor} -> UserExecutorTraining ex =>
            {0 seqLen, hidden : Nat} ->
            LayerNorm hidden hidden ex dt g
         -> Tensor [seqLen, hidden] ex dt g
         -> IO (Tensor [seqLen, hidden] ex dt g)
applyLN2d (MkLayerNorm g b) input = ioRerun (\_ =>
  MkTensor (primLayerNorm2d {ex} input.tensorPtr g.tensorPtr b.tensorPtr gpt2LnEps)
           Nothing)

||| Conv1D forward on `[seqLen, i] -> [seqLen, o]`. Bias broadcasts
||| `[o]` across the seqLen axis via `primAdd`'s standard
||| numpy-style broadcasting (verified on all three backends).
applyConv1D2d : {0 ex : Executor} -> UserExecutorTraining ex =>
                Gpt2Conv1D i o ex dt g
             -> Tensor [seqLen, i] ex dt g
             -> IO (Tensor [seqLen, o] ex dt g)
applyConv1D2d (MkGpt2Conv1D w b) x = ioRerun (\_ =>
  let mm = primMm {ex} x.tensorPtr w.tensorPtr           -- [seqLen, o]
      withBias = primAdd {ex} mm b.tensorPtr             -- + [o] broadcast
  in MkTensor withBias Nothing)

-- Fill the strict upper triangle of an n×n buffer with 1.0. Used for
-- the causal mask. Mirrors `Layer/Transformer.idr:114` `writeCausalMask`.
writeCausalMask : AnyPtr -> Int -> Int -> Int -> AnyPtr
writeCausalMask buf i j n =
  if i >= n then buf
  else if j >= n then writeCausalMask buf (i + 1) (i + 2) n
  else let buf' = prim__setDouble buf (i * n + j) 1.0
       in writeCausalMask buf' i (j + 1) n

-- Per-head attention math (causal, with multi-head split via
-- axis=1 narrow). Caller supplies the prebuilt causal mask pointer.
-- Returns AnyPtr to a `[seqLen, headDim]` block.
oneHeadCausalCtx : {0 ex : Executor} -> UserExecutorTraining ex =>
                   (qFull, kFull, vFull : AnyPtr)
                -> (causalMask : AnyPtr)
                -> (startI, headDimI : Int)
                -> (scale : Double)
                -> AnyPtr
oneHeadCausalCtx qFull kFull vFull causalMask startI headDimI scale =
  let qh     = primNarrow {ex} qFull 1 startI headDimI
      kh     = primNarrow {ex} kFull 1 startI headDimI
      vh     = primNarrow {ex} vFull 1 startI headDimI
      kT     = primTranspose2d {ex} kh
      scores = primMulScalar {ex} (primMm {ex} qh kT) scale
      masked = primMaskedFill {ex} scores causalMask (-1.0e20)
      attn   = primSoftmax2d {ex} masked
  in primMm {ex} attn vh

-- Concatenate per-head outputs along axis=1 to recover `[seqLen, hidden]`.
buildCausalHeads : {0 ex : Executor} -> UserExecutorTraining ex =>
                   (qFull, kFull, vFull, causalMask : AnyPtr)
                -> (headDimI : Int) -> (scale : Double)
                -> (remaining : Nat) -> (startI : Int) -> (acc : AnyPtr)
                -> AnyPtr
buildCausalHeads _ _ _ _ _ _ Z _ acc                                          = acc
buildCausalHeads qFull kFull vFull causalMask headDimI scale (S k) startI acc =
  let nextCtx = oneHeadCausalCtx {ex} qFull kFull vFull causalMask startI headDimI scale
      newAcc  = primConcat2dAxis1 {ex} acc nextCtx
  in buildCausalHeads {ex} qFull kFull vFull causalMask headDimI scale k (startI + headDimI) newAcc

||| GPT-2 self-attention. Pre-norm caller already applied `ln_1`.
||| Forward: split fused QKV via axis=1 narrow → per-head split via
||| axis=1 narrow (twice nested) → causal-masked scaled-dot-product →
||| concat heads back along axis=1 → c_proj.
|||
||| `numHeads = S Z` (single head) special-cases out of the narrow
||| loop (one less narrow + concat round trip); `numHeads = S (S _)`
||| goes through the full multi-head path which exercises the
||| commit-1 narrow-axis-1 fix on torch/mlx.
applySelfAttn : {0 ex : Executor} -> UserExecutorTraining ex =>
                {seqLen, hidden, numHeads, headDim : Nat}
             -> {auto prf : hidden = numHeads * headDim}
             -> Gpt2AttentionState hidden ex dt g
             -> (causalMask : AnyPtr)
             -> Tensor [seqLen, hidden] ex dt g
             -> IO (Tensor [seqLen, hidden] ex dt g)
applySelfAttn {numHeads = Z} _ _ input                                = pure input
applySelfAttn {numHeads = S Z} {hidden} {headDim} sa causalMask input = do
  -- Single head: still need fused-QKV split (the storage doesn't
  -- depend on numHeads), but skip the per-head narrow loop.
  qkv <- applyConv1D2d sa.cAttn input  -- [seq, 3*hidden]
  ctxT <- ioRerun (\_ =>
    let hI = cast {to=Int} hidden
        q      = primNarrow {ex} qkv.tensorPtr 1 0       hI
        k'     = primNarrow {ex} qkv.tensorPtr 1 hI      hI
        v      = primNarrow {ex} qkv.tensorPtr 1 (2*hI)  hI
        scale  = 1.0 / sqrt (cast {to=Double} headDim)
        kT     = primTranspose2d {ex} k'
        scores = primMulScalar {ex} (primMm {ex} q kT) scale
        masked = primMaskedFill {ex} scores causalMask (-1.0e20)
        attn   = primSoftmax2d {ex} masked
        ctx    = primMm {ex} attn v
    in MkTensor ctx Nothing)
  applyConv1D2d sa.cProj ctxT
applySelfAttn {numHeads = S (S k)} {hidden} {headDim} sa causalMask input = do
  qkv <- applyConv1D2d sa.cAttn input  -- [seq, 3*hidden]
  let hI    = cast {to=Int} hidden
      hdI   = cast {to=Int} headDim
      scale = 1.0 / sqrt (cast {to=Double} headDim)
  ctxT <- ioRerun (\_ =>
    let qFull = primNarrow {ex} qkv.tensorPtr 1 0       hI
        kFull = primNarrow {ex} qkv.tensorPtr 1 hI      hI
        vFull = primNarrow {ex} qkv.tensorPtr 1 (2*hI)  hI
        h0    = oneHeadCausalCtx {ex} qFull kFull vFull causalMask 0 hdI scale
        full  = buildCausalHeads {ex} qFull kFull vFull causalMask hdI scale (S k) hdI h0
    in MkTensor full Nothing)
  applyConv1D2d sa.cProj ctxT

||| MLP: c_proj(gelu(c_fc(x))).
applyMlp : {0 ex : Executor} -> UserExecutorTraining ex =>
           {0 seqLen, hidden, intermediate : Nat} ->
           Gpt2MlpState hidden intermediate ex dt g
        -> Tensor [seqLen, hidden] ex dt g
        -> IO (Tensor [seqLen, hidden] ex dt g)
applyMlp mlp x = do
  hFc  <- applyConv1D2d mlp.cFc x
  -- HF GPT-2 uses gelu_new (tanh approximation); tgelu matches.
  hAct <- tgelu hFc
  applyConv1D2d mlp.cProj hAct

||| One decoder block. Pre-norm + residual on both attention and MLP
||| sublayers. (Contrast HfBert which is post-norm.)
applyBlock : {0 ex : Executor} -> UserExecutorTraining ex =>
             {seqLen, hidden, numHeads, headDim, intermediate : Nat}
          -> {auto prf : hidden = numHeads * headDim}
          -> Gpt2BlockState hidden intermediate ex dt g
          -> (causalMask : AnyPtr)
          -> Tensor [seqLen, hidden] ex dt g
          -> IO (Tensor [seqLen, hidden] ex dt g)
applyBlock {hidden} {numHeads} {headDim} blk causalMask x = do
  -- Attention sublayer
  xLn1   <- applyLN2d blk.ln1 x
  aOut   <- applySelfAttn {numHeads} {headDim} blk.attn causalMask xLn1
  xMid   <- tadd x aOut
  -- MLP sublayer
  xLn2   <- applyLN2d blk.ln2 xMid
  mOut   <- applyMlp blk.mlp xLn2
  tadd xMid mOut

applyBlocks : {0 ex : Executor} -> UserExecutorTraining ex =>
              {seqLen, hidden, numHeads, headDim, intermediate, n : Nat}
           -> {auto prf : hidden = numHeads * headDim}
           -> Vect n (Gpt2BlockState hidden intermediate ex dt g)
           -> (causalMask : AnyPtr)
           -> Tensor [seqLen, hidden] ex dt g
           -> IO (Tensor [seqLen, hidden] ex dt g)
applyBlocks []        _ x  = pure x
applyBlocks (b :: bs) cm x = do
  x' <- applyBlock {numHeads} {headDim} b cm x
  applyBlocks {numHeads} {headDim} bs cm x'

-- Embedding lookup returning `[seqLen, hidden]`. Same wrapping pattern
-- as HfBert's `applyEmbedLookup2d`.
applyEmbedLookup2d : {0 ex : Executor} -> UserExecutorTraining ex =>
                     {seqLen, vocab, hidden : Nat}
                  -> Embedding vocab hidden ex dt g
                  -> Tensor [seqLen] ex dt g
                  -> IO (Tensor [seqLen, hidden] ex dt g)
applyEmbedLookup2d {seqLen} {hidden} (MkEmbedding w) tokens = ioRerun (\_ =>
  let sI = cast {to=Int} seqLen
      hI  = cast {to=Int} hidden
      out = primEmbedding2d {ex} w.tensorPtr tokens.tensorPtr sI hI
  in MkTensor out Nothing)

----------------------------------------------------------------------
-- Top-level forward (encoder)
----------------------------------------------------------------------

||| Forward pass: token IDs → final hidden state `[seqLen, hidden]`
||| (post-`ln_f`). Mirrors HF GPT-2's `last_hidden_state`. The LM head
||| is applied separately (see `hfGpt2ForwardLm`).
|||
||| `posIds` is a `[seqLen]` tensor of Double-typed position indices
||| (matching the convention used by HfBert's `posIds`). The caller
||| materialises `[0, 1, ..., seqLen-1]`.
public export
hfGpt2Forward : {0 ex : Executor} -> UserExecutorTraining ex => RuntimeDType dt => Linked ex => Compatible ex dt =>
                {seqLen, vocab, hidden, numLayers, numHeads, headDim, intermediate, maxPos : Nat}
             -> {auto prf : hidden = numHeads * headDim}
             -> Gpt2ModelState vocab hidden numLayers intermediate maxPos ex dt g
             -> Tensor [seqLen] ex dt g  -- token IDs
             -> Tensor [seqLen] ex dt g  -- position IDs (0, 1, ..., seqLen-1)
             -> IO (Tensor [seqLen, hidden] ex dt g)
hfGpt2Forward {seqLen} {hidden} {numHeads} {headDim} model tokenIds posIds = do
  -- Embedding lookups
  hTok <- applyEmbedLookup2d model.wte tokenIds
  hPos <- applyEmbedLookup2d model.wpe posIds
  -- Sum of token + positional embeddings (GPT-2 has no token-type embedding;
  -- BERT did).
  hEmb <- tadd hTok hPos
  -- Build causal mask once for the sequence (seqLen × seqLen). Routed
  -- through dtCreateState2d to land in the persistent-state path
  -- (same pattern as Layer/Transformer.idr:411-414) so the buffer isn't
  -- clobbered by tape_reset on grad-requiring backends.
  let sI = cast {to=Int} seqLen
      maskBuf  = prim__allocDoubles (sI * sI)
      maskBuf' = writeCausalMask maskBuf 0 1 sI
      mask     = dtCreateState2d {ex} {t=dt} sI sI maskBuf' (deviceStreamTag {ex})
  -- Decoder stack
  hMid <- applyBlocks {numHeads} {headDim} model.blocks mask hEmb
  -- Final LayerNorm
  applyLN2d model.lnF hMid

||| GPT-2 LM head: tied to `wte.weight`. Output `[seqLen, vocab]`
||| logits for each position. Same reconstitution pattern as HfBert's
||| `applyMlmHead`.
public export
hfGpt2ForwardLm : {0 ex : Executor} -> UserExecutorTraining ex => RuntimeDType dt => Linked ex => Compatible ex dt =>
                  {seqLen, vocab, hidden, numLayers, numHeads, headDim, intermediate, maxPos : Nat}
               -> {auto prf : hidden = numHeads * headDim}
               -> Gpt2ModelState vocab hidden numLayers intermediate maxPos ex dt g
               -> Tensor [seqLen] ex dt g  -- token IDs
               -> Tensor [seqLen] ex dt g  -- position IDs
               -> IO (Tensor [seqLen, vocab] ex dt g)
hfGpt2ForwardLm {hidden} {vocab} {numHeads} {headDim} model tokenIds posIds = do
  hFinal <- hfGpt2Forward {numHeads} {headDim} model tokenIds posIds  -- [seqLen, hidden]
  -- LM head: x @ wte.weight^T → [seqLen, vocab]. wte.weight is
  -- [vocab, hidden]; tlinear2d wants weight as [out, in] = [vocab, hidden].
  -- Bias is zero (no separate LM bias in GPT-2).
  projectTiedLmHead model.wte.weightT hFinal
