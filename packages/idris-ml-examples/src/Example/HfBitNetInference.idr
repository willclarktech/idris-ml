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
|||   (default)            Demo: just runs the dump-logits forward and
|||                        prints the top-5 token IDs. No greedy decode
|||                        loop in v1 — that lands once the dump-mode
|||                        roundtrip is green.
|||
||| Pre-requisites:
|||   - models/microsoft/bitnet-b1.58-2B-4T/model.safetensors
|||     (1.18 GB, fetch via `bash packages/idris-transformers/scripts/
|||      hf-download.sh microsoft/bitnet-b1.58-2B-4T` — not gated).
|||   - The example targets torch-mps (F32) or mlx-gpu (F32) — tape's
|||     F64 lingua franca makes embed_tokens 1.3 GB; not yet validated
|||     end-to-end on tape due to ~10 GB working-set.
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

-- The model's max_position is 4096, but the dump-mode gate only needs
-- seq=2. Cap at 32 to keep the cos/sin tables tiny.
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


----------------------------------------------------------------------
-- Stdout dump of [vocab]-shape row, one float per line
----------------------------------------------------------------------

printRow : Int -> Int -> AnyPtr -> IO ()
printRow end i p =
  if i >= end
    then pure ()
    else do
      let v = primItem1d {d=ExampleDevice} p i
      putStrLn (show v)
      printRow end (i + 1) p


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
  let dumpLogits = elem "--dump-logits" args
  t0 <- clockTime Monotonic

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

  -- Fixed two-token prompt: [9906, 1917] = "Hello world" under Llama-3
  -- BPE (verified by save_oracle_bitnet.py's tokenizer drift assertion).
  -- The oracle dumps `model(input_ids).logits[0, -1, :]` for this exact
  -- input; we run the same forward and dump the same vector.
  let inputIds = mkIds (the (Vect 2 Double) [9906.0, 1917.0])
  putStrLn "[stage] hfBitnetForwardLm — single forward pass (seq=2)..."
  logits <- hfBitnetForwardLm {d=ExampleDevice} {dt=ExampleDType}
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

  -- Last-position row = logits[1, :] — the position-1 prediction the
  -- oracle saved.
  lastRow <- trowSelect logits 1
  if dumpLogits
    then do
      printRow (cast {to=Int} VocabSize) 0 lastRow.tensorPtr
      stageStamp "dump-logits done" t0
      pure ()
    else do
      putStrLn ""
      putStrLn "BitNet b1.58 2B-4T forward demo"
      putStrLn "================================"
      putStrLn ""
      putStrLn ("Prompt token IDs: [9906, 1917]")
      putStrLn ("Logits at position 1 — first 5 values:")
      let n5 = the Int 5
      printRow n5 0 lastRow.tensorPtr
      stageStamp "demo done" t0
      pure ()
