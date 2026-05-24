||| HfBitNetInference — load `microsoft/bitnet-b1.58-2B-4T` and run
||| inference through the typed-tensor stack with ternary BitLinears.
|||
||| BitNet 2B-4T: vocab=128256, hidden=2560, n_layer=30, n_head=20,
||| n_kv_heads=5 (GQA 4:1), head_dim=128, intermediate=6912,
||| max_position=4096, rope_theta=500000, rms_norm_eps=1e-5,
||| hidden_act="relu2", tie_word_embeddings=true. On-disk ~1.2 GB
||| (packed-uint8 ternary linears + BF16 embed/norms/scales).
|||
||| Two modes:
|||
|||   --dump-logits        CI gate. Fixed two-token prompt [9906, 1917]
|||                        (= "Hello world" under the Llama-3 BPE).
|||                        Forward once. Print the last-position
|||                        logits (vocab=128256 floats) to stdout, one
|||                        per line. Comparator in
|||                        scripts/compare_inference.py.
|||
|||   (default)            Greedy decode demo. Tokenize the prompt
|||                        (default "The capital of France is"),
|||                        generate `--num-tokens` next tokens by
|||                        repeated argmax (no KV cache — re-runs the
|||                        full forward on the growing sequence each
|||                        step), detokenize, print the resulting
|||                        text. Each forward at seq=N+k is ~3-15s
|||                        depending on backend/device; keep N small
|||                        for a quick demo.
|||
||| Pre-requisites:
|||   - models/microsoft/bitnet-b1.58-2B-4T/model.safetensors
|||     (1.18 GB, fetch via `bash packages/idris-transformers/scripts/
|||      hf-download.sh microsoft/bitnet-b1.58-2B-4T` — not gated).
|||   - models/microsoft/bitnet-b1.58-2B-4T/tokenizer.json (default
|||     mode only — dump-logits mode doesn't need a tokenizer).
module Example.HfBitNetInference

import Data.Fin
import Data.List
import Data.String
import Data.Vect
import System
import System.Clock
import System.File

import Array
import BuildConfig
import Checkpoint
import Device
import HfBitNet
import Layer.RoPE
import Tensor
import Tokenizer
import Util


----------------------------------------------------------------------
-- BitNet 2B-4T config, pinned at the type level
----------------------------------------------------------------------

VocabSize : Nat
VocabSize = 128256

Hidden : Nat
Hidden = 2560

NumLayers : Nat
NumLayers = 30

NumHeads : Nat
NumHeads = 20

NumKvHeads : Nat
NumKvHeads = 5

HeadDim : Nat
HeadDim = 128

QOut : Nat
QOut = NumHeads * HeadDim       -- = 2560 (= Hidden)

KvOut : Nat
KvOut = NumKvHeads * HeadDim    -- = 640

Intermediate : Nat
Intermediate = 6912

-- The model's max_position is 4096. Default demo runs prompt~7 +
-- 5 generated = 12 tokens, well under 32. Cap small so the cos/sin
-- tables stay tiny (32 × 64 = 2K floats per table). Users wanting
-- longer continuations would bump this.
MaxPos : Nat
MaxPos = 32

RopeBase : Double
RopeBase = 500000.0

RmsNormEps : Double
RmsNormEps = 1.0e-5

ModelRepo : String
ModelRepo = "microsoft/bitnet-b1.58-2B-4T"

modelDir : String
modelDir = "models/" ++ ModelRepo

hfWeightsPath : String
hfWeightsPath = modelDir ++ "/model.safetensors"


----------------------------------------------------------------------
-- Input-ID tensor helper (mirrors HfLlamaInference)
----------------------------------------------------------------------

mkIds : {n : Nat} -> Vect n Double
     -> Tensor [n] ExampleDevice ExampleDType WithGrad
mkIds xs =
  let raw = bulkToTensor {d=ExampleDevice} {dt=ExampleDType}
                         (VArray (map SArray xs))
  in tinput1d {n} raw


toExistVect : (xs : List a) -> (n : Nat ** Vect n a)
toExistVect xs = (length xs ** fromList xs)


----------------------------------------------------------------------
-- Stdout dump of [vocab]-shape row, one float per line (dump-logits gate)
----------------------------------------------------------------------

printRow : Int -> Int -> AnyPtr -> IO ()
printRow end i p =
  if i >= end
    then pure ()
    else do
      let v = primItem1d {d=ExampleDevice} p i
      putStrLn (show v)
      printRow end (i + 1) p


-- Helpers for --bisect-blocks mode: collect a 1D tensor's values as
-- a List of `show`-formatted strings, then writeFile the whole thing
-- once. Avoids per-element file syscalls for large dumps (the largest
-- single dump is the [128256] logits = 128k lines).
collectShown : Int -> Int -> AnyPtr -> IO (List String)
collectShown end startIdx ptr = go startIdx []
  where
    go : Int -> List String -> IO (List String)
    go i acc =
      if i >= end
        then pure (reverse acc)
        else do
          let v = primItem1d {d=ExampleDevice} ptr i
          go (i + 1) (show v :: acc)

dumpRowToFile : String -> Int -> AnyPtr -> IO ()
dumpRowToFile path nElems ptr = do
  xs <- collectShown nElems 0 ptr
  res <- writeFile path (unlines xs)
  case res of
    Right () => pure ()
    Left  err =>
      putStrLn ("ERR: writeFile " ++ path ++ ": " ++ show err)


-- Manual per-block iteration with dumps after each block. Calls
-- `applyBlock` (exported from HfBitNet) once per Vect element and
-- invokes `dumpFn` with a "block_NN" label. Idris-2's elaborator hung
-- (>90 min, killed) when this iteration was attempted as a new
-- polymorphic helper inside HfBitNet.idr; moving it here with the
-- ExampleDevice/ExampleDType types pinned concretely lets the
-- elaborator skip the per-call constraint specialisation that the
-- BitNet quant-typeclass surface forces.
iterateBlocksDumping :
     {n : Nat}
  -> Vect n (BitNetBlockState Hidden QOut KvOut Intermediate
                              ExampleDevice ExampleDType WithGrad)
  -> RoPETables MaxPos HeadDim ExampleDevice ExampleDType WithGrad
  -> Tensor [2, Hidden] ExampleDevice ExampleDType WithGrad
  -> (idx : Nat)
  -> (dumpFn : String -> AnyPtr -> Int -> IO ())
  -> IO (Tensor [2, Hidden] ExampleDevice ExampleDType WithGrad)
iterateBlocksDumping []        _      x _   _      = pure x
iterateBlocksDumping (b :: bs) tables x idx dumpFn = do
  x' <- applyBlock {numHeads=NumHeads} {numKvHeads=NumKvHeads}
                   {headDim=HeadDim}   {intermediate=Intermediate}
                   RmsNormEps b tables x
  let label = "block_" ++ (if idx < 10 then "0" ++ show idx else show idx)
      n     = cast {to=Int} 2 * cast {to=Int} Hidden
  dumpFn label x'.tensorPtr n
  iterateBlocksDumping bs tables x' (S idx) dumpFn


----------------------------------------------------------------------
-- Greedy generation
----------------------------------------------------------------------

argmaxRow : (vocab : Nat) -> AnyPtr -> IO Nat
argmaxRow vocab p = go (cast {to=Int} vocab) 0 0 (-1.0e300)
  where
    go : Int -> Int -> Int -> Double -> IO Nat
    go end i bestI bestV =
      if i >= end
        then pure (cast {to=Nat} bestI)
        else let v = primItem1d {d=ExampleDevice} p i
             in if v > bestV
                  then go end (i + 1) i v
                  else go end (i + 1) bestI bestV


genOneStep : BitNetModelState VocabSize Hidden NumLayers QOut KvOut Intermediate
                              ExampleDevice ExampleDType WithGrad
          -> RoPETables MaxPos HeadDim ExampleDevice ExampleDType WithGrad
          -> List (Fin VocabSize)
          -> IO (Maybe (Fin VocabSize))
genOneStep model tables toksList = do
  let idsList = map (cast {to=Double} . finToNat) toksList
  case toExistVect idsList of
    (curLen ** idDoubles) =>
      -- Each forward materialises a [out, in]-float dequant of the
      -- ternary weights per BitLinear (210 per layer-set). With
      -- autograd on, libtorch holds those for backward — 5 generation
      -- steps accumulate enough to OOM-kill the process on 16 GB.
      -- Plain withNoGrad here (not Keep): the step's result is a
      -- Maybe (Fin VocabSize), no Tensor needs to survive the
      -- bracket exit.
      withNoGrad {d=ExampleDevice} $ do
        let inputIds = mkIds idDoubles
        logits <- hfBitnetForwardLm {d=ExampleDevice} {dt=ExampleDType}
                                    {seq          = curLen}
                                    {vocab        = VocabSize}
                                    {hidden       = Hidden}
                                    {numLayers    = NumLayers}
                                    {numHeads     = NumHeads}
                                    {numKvHeads   = NumKvHeads}
                                    {headDim      = HeadDim}
                                    {intermediate = Intermediate}
                                    {maxPos       = MaxPos}
                                    RmsNormEps model tables inputIds
        lastRow <- trowSelect logits (cast {to=Int} curLen - 1)
        nextN   <- argmaxRow VocabSize lastRow.tensorPtr
        pure (natToFin nextN VocabSize)


genLoop : BitNetModelState VocabSize Hidden NumLayers QOut KvOut Intermediate
                           ExampleDevice ExampleDType WithGrad
       -> RoPETables MaxPos HeadDim ExampleDevice ExampleDType WithGrad
       -> List (Fin VocabSize)
       -> (remaining : Nat)
       -> IO (List (Fin VocabSize))
genLoop _     _      tokens Z     = pure tokens
genLoop model tables tokens (S k) = do
  mNext <- genOneStep model tables tokens
  case mNext of
    Nothing => do
      putStrLn "  (argmax produced out-of-range token; stopping)"
      pure tokens
    Just next => do
      resetForEval {d=ExampleDevice}
      genLoop model tables (tokens ++ [next]) k


----------------------------------------------------------------------
-- --prompt + --num-tokens argv parsing
----------------------------------------------------------------------

extractPrompt : List String -> String
extractPrompt args = go args
  where
    go : List String -> String
    go ("--prompt" :: p :: _) = p
    go (_ :: rest)            = go rest
    go []                     = "The capital of France is"

extractNumTokens : List String -> Nat
extractNumTokens args = go args
  where
    go : List String -> Nat
    go ("--num-tokens" :: n :: _) =
      fromMaybe 5 (parsePositive {a=Nat} n)
    go (_ :: rest)                = go rest
    go []                         = 5


----------------------------------------------------------------------
-- Default mode: greedy generation demo
----------------------------------------------------------------------

runGenerate : Tokenizer VocabSize
           -> BitNetModelState VocabSize Hidden NumLayers QOut KvOut Intermediate
                               ExampleDevice ExampleDType WithGrad
           -> RoPETables MaxPos HeadDim ExampleDevice ExampleDType WithGrad
           -> (prompt : String) -> (numTokens : Nat) -> IO ()
runGenerate tok model tables prompt numTokens = do
  Right (promptLen ** promptIds) <- tokenize tok prompt
    | Left err => putStrLn ("ERR: tokenize: " ++ show err)
  putStrLn ""
  putStrLn "BitNet b1.58 2B-4T greedy generation"
  putStrLn "===================================="
  putStrLn ""
  putStrLn ("Prompt:    " ++ prompt)
  putStrLn ("Tokens in: " ++ show promptLen ++ ", generating: " ++ show numTokens)
  putStrLn ""
  let promptList = toList promptIds
  finalList <- genLoop model tables promptList numTokens
  Right text <- detokenize tok (fromList finalList)
    | Left err => putStrLn ("ERR: detokenize: " ++ show err)
  putStrLn ("Output:    " ++ text)


----------------------------------------------------------------------
-- main
----------------------------------------------------------------------

stageStamp : (label : String) -> Clock Monotonic -> IO ()
stageStamp label t0 = do
  now <- clockTime Monotonic
  putStrLn ("[stage] " ++ formatElapsed t0 now ++ " " ++ label)


main : IO ()
main = do
  args <- getArgs
  let dumpLogits   = elem "--dump-logits"   args
  let bisectBlocks = elem "--bisect-blocks" args
  let prompt       = extractPrompt args
  let numTokens    = extractNumTokens args
  t0 <- clockTime Monotonic

  -- Probe the tokenizer up-front so a missing tokenizer fails fast,
  -- before the ~8s param load. dump-logits mode skips this since
  -- the gate uses a hardcoded prompt.
  tokR <- mkTokenizer ModelRepo VocabSize
  case tokR of
    Left err =>
      if not dumpLogits
        then do
          putStrLn ("ERR: mkTokenizer failed: " ++ show err)
          putStrLn ("     Likely missing tokenizer files at models/" ++ ModelRepo ++ "/")
          putStrLn ("     Run: bash packages/idris-transformers/scripts/hf-download.sh "
                    ++ ModelRepo)
          exitFailure
        else
          putStrLn ("WARN: mkTokenizer failed (continuing — dump-logits doesn't need it): "
                    ++ show err)
    Right _ => pure ()
  stageStamp "tokenizer probe ok" t0

  putStrLn ("[stage] hfBitnetModel — constructing 542-param state ("
            ++ "embed/norms/scales + ternary placeholders)...")
  model <- hfBitnetModel {d=ExampleDevice} {dt=ExampleDType}
                         {vocab        = VocabSize}
                         {hidden       = Hidden}
                         {numLayers    = NumLayers}
                         {qOut         = QOut}
                         {kvOut        = KvOut}
                         {intermediate = Intermediate}
                         "model"
  stageStamp "hfBitnetModel ok" t0

  putStrLn ("[stage] loadHfBitnetCheckpoint — reading "
            ++ hfWeightsPath ++ " (~1.18 GB)...")
  (loaded, (tnLoaded, tnExpected, floatOk)) <-
    loadHfBitnetCheckpoint {d=ExampleDevice} {dt=ExampleDType}
                           {vocab        = VocabSize}
                           {hidden       = Hidden}
                           {numLayers    = NumLayers}
                           {qOut         = QOut}
                           {kvOut        = KvOut}
                           {intermediate = Intermediate}
                           "model" hfWeightsPath model
  putStrLn ("  ternary weights loaded: " ++ show tnLoaded ++ "/" ++ show tnExpected)
  putStrLn ("  float-typed params via loadModelAllowCast: "
            ++ (if floatOk then "ok" else "FAILED"))
  if tnLoaded /= tnExpected || not floatOk
    then do
      putStrLn "ERR: checkpoint load incomplete"
      exitFailure
    else pure ()
  stageStamp "loadHfBitnetCheckpoint ok" t0

  putStrLn "[stage] buildLlamaRoPETables — precomputing cos/sin tables..."
  tables <- buildLlamaRoPETables {d=ExampleDevice} {dt=ExampleDType}
                                  {maxPos  = MaxPos}
                                  {headDim = HeadDim}
                                  RopeBase bitnetRopeScaling
  stageStamp "buildLlamaRoPETables ok" t0

  if bisectBlocks
    then do
      -- Per-block divergence-bisection mode: run the forward dumping
      -- post-embedding, post-each-block, post-final-norm hidden states
      -- to models/idris-bisect/<label>.txt (one float per line). Also
      -- dump the final last-position logits to logits.txt. The companion
      -- script `save_oracle_bitnet_blocks.py` produces matching oracle
      -- safetensors under models/bitnet-2b-4t-bisect/; `compare_bitnet_blocks.py`
      -- walks both directories and reports per-label max-rel-diff.
      _ <- system "mkdir -p models/idris-bisect"
      let inputIds = mkIds (the (Vect 2 Double) [9906.0, 1917.0])
      let dumpFn : String -> AnyPtr -> Int -> IO ()
          dumpFn label ptr nElems = do
            let path = "models/idris-bisect/" ++ label ++ ".txt"
            dumpRowToFile path nElems ptr
            putStrLn ("[bisect] wrote " ++ label ++
                      " (" ++ show nElems ++ " floats)")
      putStrLn "[stage] bisect-blocks forward..."
      let nHidden = cast {to=Int} 2 * cast {to=Int} Hidden
      -- Step 1: embedding
      emb <- applyEmbedLookup loaded.embedTokens inputIds
      dumpFn "embedding" emb.tensorPtr nHidden
      -- Step 2: iterate the 30 decoder blocks, dumping after each
      hMid <- iterateBlocksDumping loaded.blocks tables emb Z dumpFn
      -- Step 3: final RmsNorm
      hFinal <- applyRmsNorm2d {seqLen=2} {hidden=Hidden}
                               RmsNormEps loaded.finalNorm hMid
      dumpFn "final_norm" hFinal.tensorPtr nHidden
      -- Step 4: tied LM head — project hFinal through embed_tokens.weight
      let vI = cast {to=Int} VocabSize
          zBuf = prim__allocDoubles vI
          zeroBias : Tensor [VocabSize] ExampleDevice ExampleDType WithGrad
          zeroBias = MkTensor (dtCreateState1d {d=ExampleDevice} {t=ExampleDType}
                                vI zBuf (deviceStreamTag {d=ExampleDevice})) Nothing
      logits <- tlinear2d loaded.embedTokens.weight hFinal zeroBias
      lastRow <- trowSelect logits 1
      dumpFn "logits" lastRow.tensorPtr vI
      stageStamp "bisect-blocks done" t0
      pure ()
    else if dumpLogits
      then do
        -- CI gate path: fixed two-token prompt [9906, 1917] = "Hello world"
        -- under Llama-3 BPE. Forward once, dump all 128256 last-position
        -- logits one per line so compare_inference.py can read them back.
        let inputIds = mkIds (the (Vect 2 Double) [9906.0, 1917.0])
        putStrLn "[stage] hfBitnetForwardLm — single forward pass (seq=2)..."
        logits <- withNoGradKeep {d=ExampleDevice} $
          hfBitnetForwardLm {d=ExampleDevice} {dt=ExampleDType}
                            {seq          = 2}
                            {vocab        = VocabSize}
                            {hidden       = Hidden}
                            {numLayers    = NumLayers}
                            {numHeads     = NumHeads}
                            {numKvHeads   = NumKvHeads}
                            {headDim      = HeadDim}
                            {intermediate = Intermediate}
                            {maxPos       = MaxPos}
                            RmsNormEps loaded tables inputIds
        stageStamp "hfBitnetForwardLm ok" t0
        lastRow <- trowSelect logits 1
        printRow (cast {to=Int} VocabSize) 0 lastRow.tensorPtr
        stageStamp "dump-logits done" t0
        pure ()
      else do
        case tokR of
          Left _ => exitFailure
          Right tokVal => do
            putStrLn ("[stage] runGenerate — greedy decode loop (" ++
                      show numTokens ++ " tokens)...")
            runGenerate tokVal loaded tables prompt numTokens
            stageStamp "runGenerate done" t0
            pure ()
