||| idris-transformers — HuggingFace-aligned model library on top of
||| idris-ml.
|||
||| Each HF architecture is one module under this package; param
||| names and storage shapes match HF's on-disk safetensors format,
||| so loading is plain `loadModel "model.safetensors"` from
||| `Checkpoint.idr` — no rename map, no shape-split adapter.
|||
||| See `CONVENTIONS.md` for the design rules every Hf-prefixed
||| module follows.
|||
||| Currently this top-level module is a placeholder. As HF model
||| modules land they will be re-exported from here:
|||
|||   import HfBert   -- BERT encoder + pooler (prajjwal1/bert-tiny)
|||   import HfGpt2   -- (follow-up row)
|||   import HfLlama  -- (Row 7 — LLM-class example)
module Transformers
