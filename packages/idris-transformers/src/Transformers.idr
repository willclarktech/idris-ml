||| idris-transformers — HuggingFace-aligned model library on top of
||| idris-ml.
|||
||| Each HF architecture is one module under this package; param
||| names and storage shapes match HF's on-disk safetensors format,
||| so loading is plain `loadModel "model.safetensors"` from
||| `Checkpoint.idr` — no rename map, no shape-split adapter.
|||
||| See `CONVENTIONS.md` for the design rules every `Transformers.*`
||| module follows.
|||
||| The per-architecture modules — import directly:
|||
|||   import Transformers.Bert    -- BERT encoder + pooler + MLM head
|||   import Transformers.Gpt2    -- GPT-2 family (distilgpt2)
|||   import Transformers.Llama   -- Llama 3.x (GQA + RoPE + KV cache)
|||   import Transformers.BitNet  -- BitNet b1.58 (ternary BitLinears)
|||
||| Each exposes `fromPretrained : Backend ex dt => KnownGrad g => String
||| -> IO (Either LoadError (cfg ** Model cfg ex dt g))` — point it at a
||| local HF model dir and the dims come from its `config.json`.
module Transformers
