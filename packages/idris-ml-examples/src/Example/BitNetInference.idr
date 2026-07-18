||| BitNetInference — load `microsoft/bitnet-b1.58-2B-4T` and run
||| inference through the typed-tensor stack with ternary BitLinears.
|||
||| BitNet 2B-4T: vocab=128256, hidden=2560, n_layer=30, n_head=20,
||| n_kv_heads=5 (GQA 4:1), head_dim=128, intermediate=6912,
||| max_position=4096, rope_theta=500000, rms_norm_eps=1e-5,
||| hidden_act="relu2", tie_word_embeddings=true. On-disk ~1.2 GB
||| (packed-uint8 ternary linears + BF16 embed/norms/scales).
|||
||| The model is loaded with `Transformers.BitNet.fromPretrained`: it
||| reads `<dir>/config.json` for the dims, builds the model, and fills
||| it via `loadHfBitnetCheckpoint` (ternary BitLinear weights from raw
||| packed bytes + float params via the safetensors path). The returned
||| `(cfg ** model)` ties the model's type to the file. Built `NoGrad`
||| (tape-free by construction): each forward dequants the ternary weights
||| to float, and a `WithGrad` model would have autograd retain those
||| dequants and OOM the process over a few decode steps. The `{g=NoGrad}`
||| type makes that impossible structurally — no tape, no retained
||| dequants — so no runtime `withNoGrad` brackets are needed.
|||
||| Two modes:
|||
|||   --dump-logits        CI gate. Fixed two-token prompt [9906, 1917]
|||                        (= "Hello world" under the Llama-3 BPE).
|||                        Forward once. Print the last-position
|||                        logits (vocab floats) to stdout, one per
|||                        line. Comparator in scripts/compare_inference.py.
|||
|||   (default)            Greedy decode demo. Tokenize the prompt
|||                        (default "The capital of France is"),
|||                        generate `--num-tokens` next tokens by
|||                        repeated argmax (no KV cache), detokenize,
|||                        print the resulting text.
|||
||| Pre-requisites:
|||   - models/microsoft/bitnet-b1.58-2B-4T/{config.json,model.safetensors}
|||     (fetch via `bash packages/idris-transformers/scripts/hf-download.sh
|||      microsoft/bitnet-b1.58-2B-4T` — not gated).
|||   - models/microsoft/bitnet-b1.58-2B-4T/tokenizer.json (default
|||     mode only — dump-logits mode doesn't need a tokenizer).
module Example.BitNetInference

import Data.Fin
import Data.List
import Data.String
import Data.Vect
import System
import System.Clock
import System.File

import Ml.Array
import Ml.Checkpoint
import Ml.Executor
import Ml.Nn.Embedding
import Ml.Nn.RoPE
import Ml.Tensor
import Ml.Util
import Transformers.BitNet
import Transformers.Tokenizer

import BuildConfig
import Example.Common.InferenceHelper

----------------------------------------------------------------------
-- Model location (dims come from the file, not from here)
----------------------------------------------------------------------

-- Clamped RoPE-table context. The model's max_position is larger; the
-- default demo runs ~12 tokens, well under 32. Cap small so the cos/sin
-- tables stay tiny. A table-size knob, separate from `cfg.maxPosition`,
-- so it stays a local literal (it also appears at type position in
-- `RoPETables MaxPos (headDim cfg) ...`).
MaxPos : Nat
MaxPos = 32

ModelRepo : String
ModelRepo = "microsoft/bitnet-b1.58-2B-4T"

modelDir : String
modelDir = "models/" ++ ModelRepo

-- Manual per-block iteration with dumps after each block (bisect mode).
-- Calls `applyBlock` (exported from Transformers.BitNet) once per Vect
-- element and invokes `dumpFn` with a "block_NN" label. Pinned to the
-- concrete ExampleExecutor/ExampleDType (not a polymorphic helper inside
-- Transformers.BitNet) so the elaborator skips the per-call constraint
-- specialisation the BitNet quant-typeclass surface forces.
iterateBlocksDumping :
     (cfg : BitNetConfig)
  -> {n : Nat}
  -> Vect n (BitNetBlockState (hidden cfg) (numHeads cfg * headDim cfg)
                              (numKvHeads cfg * headDim cfg) (intermediate cfg)
                              ExampleExecutor ExampleDType NoGrad)
  -> RoPETables MaxPos (headDim cfg) ExampleExecutor ExampleDType
  -> Tensor [2, hidden cfg] ExampleExecutor ExampleDType NoGrad
  -> (idx : Nat)
  -> (dumpFn : String -> AnyPtr -> Int -> IO ())
  -> IO (Tensor [2, hidden cfg] ExampleExecutor ExampleDType NoGrad)
iterateBlocksDumping _   []        _      x _   _      = pure x
iterateBlocksDumping cfg (b :: bs) tables x idx dumpFn = do
  x' <- applyBlock {numHeads     = numHeads cfg} {numKvHeads = numKvHeads cfg}
                   {headDim      = headDim cfg}  {intermediate = intermediate cfg}
                   (rmsNormEps cfg) b tables x
  let label = "block_" ++ (if idx < 10 then "0" ++ show idx else show idx)
      n     = cast {to=Int} 2 * cast {to=Int} (hidden cfg)
  dumpFn label x'.tensorPtr n
  iterateBlocksDumping cfg bs tables x' (S idx) dumpFn

----------------------------------------------------------------------
-- Greedy generation (helpers live in Example.Common.InferenceHelper)
----------------------------------------------------------------------

genOneStep : (cfg : BitNetConfig)
          -> BitNetModel cfg ExampleExecutor ExampleDType NoGrad
          -> RoPETables MaxPos (headDim cfg) ExampleExecutor ExampleDType
          -> List (Fin (vocabSize cfg))
          -> IO (Maybe (Fin (vocabSize cfg)))
genOneStep cfg model tables toksList = do
  let idsList = map (cast {to=Double} . finToNat) toksList
  case toExistVect idsList of
    (curLen ** idDoubles) => do
      -- Each forward materialises a [out, in]-float dequant of the
      -- ternary weights per BitLinear. The model is `NoGrad`, so those
      -- dequants are never autograd-retained — no tape, no bracket, no
      -- OOM over decode steps (the `withNoGrad` this used to need is now
      -- the model's `{g=NoGrad}` type).
      let inputIds = retypeGrad (mkIds idDoubles)
      logits <- hfBitnetForwardLm {ex=ExampleExecutor} {dt=ExampleDType}
                                  {seq          = curLen}
                                  {vocab        = vocabSize cfg}
                                  {hidden       = hidden cfg}
                                  {numLayers    = numLayers cfg}
                                  {numHeads     = numHeads cfg}
                                  {numKvHeads   = numKvHeads cfg}
                                  {headDim      = headDim cfg}
                                  {intermediate = intermediate cfg}
                                  {maxPos       = MaxPos}
                                  (rmsNormEps cfg) model tables inputIds
      lastRow <- trowSelect logits (cast {to=Int} curLen - 1)
      nextN   <- argmaxRow (vocabSize cfg) lastRow.tensorPtr
      pure (natToFin nextN (vocabSize cfg))

genLoop : (cfg : BitNetConfig)
       -> BitNetModel cfg ExampleExecutor ExampleDType NoGrad
       -> RoPETables MaxPos (headDim cfg) ExampleExecutor ExampleDType
       -> List (Fin (vocabSize cfg))
       -> (remaining : Nat)
       -> IO (List (Fin (vocabSize cfg)))
genLoop _   _     _      tokens Z     = pure tokens
genLoop cfg model tables tokens (S k) = do
  mNext <- genOneStep cfg model tables tokens
  case mNext of
    Nothing => do
      putStrLn "  (argmax produced out-of-range token; stopping)"
      pure tokens
    Just next => do
      resetForEval {ex=ExampleExecutor}
      genLoop cfg model tables (tokens ++ [next]) k

----------------------------------------------------------------------
-- Default mode: greedy generation demo
----------------------------------------------------------------------

runGenerate : (cfg : BitNetConfig)
           -> Tokenizer (vocabSize cfg)
           -> BitNetModel cfg ExampleExecutor ExampleDType NoGrad
           -> RoPETables MaxPos (headDim cfg) ExampleExecutor ExampleDType
           -> (prompt : String) -> (numTokens : Nat) -> IO ()
runGenerate cfg tok model tables prompt numTokens = do
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
  finalList <- genLoop cfg model tables promptList numTokens
  Right text <- detokenize tok (fromList finalList)
    | Left err => putStrLn ("ERR: detokenize: " ++ show err)
  putStrLn ("Output:    " ++ text)

----------------------------------------------------------------------
-- main
----------------------------------------------------------------------

main : IO ()
main = do
  requireMachine {m = ChosenMachine}
  args <- getArgs
  let dumpLogits   = elem "--dump-logits"   args
  let bisectBlocks = elem "--bisect-blocks" args
  let prompt       = extractPrompt "The capital of France is" args
  let numTokens    = extractNumTokens 5 args
  t0 <- clockTime Monotonic

  -- Load BitNet straight from the HF checkpoint dir. fromPretrained
  -- reads config.json for the dims, builds the 542-param state, and
  -- fills it via loadHfBitnetCheckpoint (ternary BitLinears from raw
  -- packed bytes + float params via safetensors). A partial load
  -- surfaces as ReadFailed.
  putStrLn ("[stage] fromPretrained — config.json + 542-param state + checkpoint (~1.18 GB)...")
  Right (cfg ** model) <- fromPretrained {ex=ExampleExecutor} {dt=ExampleDType} {g=NoGrad} modelDir
    | Left err => do
        putStrLn ("ERR: fromPretrained " ++ modelDir ++ ": " ++ show err)
        exitFailure
  stageStamp "fromPretrained ok" t0

  -- Probe the tokenizer (dump-logits mode skips needing it — its prompt
  -- is hardcoded).
  tokR <- mkTokenizer ModelRepo (vocabSize cfg)
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

  putStrLn "[stage] buildLlamaRoPETables — precomputing cos/sin tables..."
  tables <- buildLlamaRoPETables {ex=ExampleExecutor} {dt=ExampleDType}
                                  {maxPos  = MaxPos}
                                  {headDim = headDim cfg}
                                  (ropeBase cfg) bitnetRopeScaling
  stageStamp "buildLlamaRoPETables ok" t0

  if bisectBlocks
    then do
      -- Per-block divergence-bisection mode: run the forward dumping
      -- post-embedding, post-each-block, post-final-norm hidden states
      -- to models/idris-bisect/<label>.txt (one float per line). The
      -- companion `save_oracle_bitnet_blocks.py` produces matching oracle
      -- safetensors; `compare_bitnet_blocks.py` reports per-label diff.
      _ <- system "mkdir -p models/idris-bisect"
      let inputIds = retypeGrad (mkIds (the (Vect 2 Double) [9906.0, 1917.0]))
      let dumpFn : String -> AnyPtr -> Int -> IO ()
          dumpFn label ptr nElems = do
            let path = "models/idris-bisect/" ++ label ++ ".txt"
            dumpRowToFile path nElems ptr
            putStrLn ("[bisect] wrote " ++ label ++
                      " (" ++ show nElems ++ " floats)")
      putStrLn "[stage] bisect-blocks forward..."
      let nHidden = cast {to=Int} 2 * cast {to=Int} (hidden cfg)
      -- Step 1: embedding
      emb <- applyEmbedLookup model.embedTokens inputIds
      dumpFn "embedding" emb.tensorPtr nHidden
      -- Step 2: iterate the decoder blocks, dumping after each
      hMid <- iterateBlocksDumping cfg model.blocks tables emb Z dumpFn
      -- Step 3: final RmsNorm
      hFinal <- applyRmsNorm2d {seqLen=2} {hidden=hidden cfg}
                               (rmsNormEps cfg) model.finalNorm hMid
      dumpFn "final_norm" hFinal.tensorPtr nHidden
      -- Step 4: tied LM head — project hFinal through embed_tokens.weight.
      -- Pattern-match the Embedding to extract its weight: a chained
      -- `model.embedTokens.weightT` projection is ambiguous here because
      -- `weightT` is a field on both Embedding and BitLinearHf and the
      -- `BitNetModel cfg` alias doesn't reduce for projection resolution.
      let MkEmbedding embWeight = model.embedTokens
      let vI                    = cast {to=Int} (vocabSize cfg)
          zBuf = prim__allocDoubles vI
          zeroBias : Tensor [vocabSize cfg] ExampleExecutor ExampleDType NoGrad
          zeroBias = MkTensor (dtCreateState1d {ex=ExampleExecutor} {t=ExampleDType}
                                vI zBuf (deviceStreamTag {ex=ExampleExecutor})) Nothing
      logits <- tlinear2d embWeight hFinal zeroBias
      lastRow <- trowSelect logits 1
      dumpFn "logits" lastRow.tensorPtr vI
      stageStamp "bisect-blocks done" t0
      pure ()
    else if dumpLogits
      then do
        -- CI gate path: fixed two-token prompt [9906, 1917] = "Hello world"
        -- under Llama-3 BPE. Forward once, dump all last-position logits
        -- one per line so compare_inference.py can read them back.
        let inputIds = retypeGrad (mkIds (the (Vect 2 Double) [9906.0, 1917.0]))
        putStrLn "[stage] hfBitnetForwardLm — single forward pass (seq=2)..."
        logits <- hfBitnetForwardLm {ex=ExampleExecutor} {dt=ExampleDType}
                                    {seq          = 2}
                                    {vocab        = vocabSize cfg}
                                    {hidden       = hidden cfg}
                                    {numLayers    = numLayers cfg}
                                    {numHeads     = numHeads cfg}
                                    {numKvHeads   = numKvHeads cfg}
                                    {headDim      = headDim cfg}
                                    {intermediate = intermediate cfg}
                                    {maxPos       = MaxPos}
                                    (rmsNormEps cfg) model tables inputIds
        stageStamp "hfBitnetForwardLm ok" t0
        lastRow <- trowSelect logits 1
        printRow (cast {to=Int} (vocabSize cfg)) 0 lastRow.tensorPtr
        stageStamp "dump-logits done" t0
        pure ()
      else do
        case tokR of
          Left _       => exitFailure
          Right tokVal => do
            putStrLn ("[stage] runGenerate — greedy decode loop (" ++
                      show numTokens ++ " tokens)...")
            runGenerate cfg tokVal model tables prompt numTokens
            stageStamp "runGenerate done" t0
            pure ()
