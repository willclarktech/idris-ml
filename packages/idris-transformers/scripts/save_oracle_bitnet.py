"""Produce the integration-test oracle for Example/HfBitNetInference.

Loads `microsoft/bitnet-b1.58-2B-4T` via HuggingFace transformers, runs
forward on a fixed token sequence, and writes the last-position logits
to `models/bitnet-2b-4t-oracle.safetensors`. The Idris side reads the
same file and asserts max-abs-difference against its own forward output.

`microsoft/bitnet-b1.58-2B-4T` is the canonical BitNet b1.58 release
from Microsoft (~2B params, ternary BitLinears throughout, ~1.2 GB
safetensors on disk). It uses Llama 3-style tokenization (vocab=128256),
30 decoder layers, hidden=2560, head_dim=128, GQA 20->5, intermediate=
6912, RoPE theta=500000, RmsNorm eps=1e-5, hidden_act="relu2", and
crucially `tie_word_embeddings=True` -- `lm_head` shares storage with
`embed_tokens.weight` (no separate `lm_head.weight` on disk).

Element-wise logit diff over 30 layers + 2560 hidden in BF16 lands
around ~1e-3 in practice; the gate tolerates that.

Usage:
    cd packages/pytorch && uv run python \\
        ../idris-transformers/scripts/save_oracle_bitnet.py

    # or via the Makefile:
    make test-hf-bitnet-roundtrip
"""

from __future__ import annotations

from pathlib import Path

import torch
from safetensors.torch import save_file
from transformers import AutoModelForCausalLM, AutoTokenizer

SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT  = SCRIPT_DIR.parent.parent.parent   # <repo-root>
MODELS_DIR = REPO_ROOT / "models"
ORACLE_PATH = MODELS_DIR / "bitnet-2b-4t-oracle.safetensors"
MODEL_LOCAL = MODELS_DIR / "microsoft" / "bitnet-b1.58-2B-4T"

MODEL_ID = "microsoft/bitnet-b1.58-2B-4T"

# Two tokens to keep the oracle small while exercising both position 0
# (no prior context) and position 1 (attends to position 0 via the
# causal mask). The IDs below are the Llama-3-style BPE encodings for
# "Hello world" -- the tokenizer drift assertion will catch upstream
# rebuilds.
FIXED_INPUT_IDS: list[int] = [9906, 1917]
HIDDEN: int = 2560
VOCAB:  int = 128256


def main() -> None:
    MODELS_DIR.mkdir(parents=True, exist_ok=True)

    torch.manual_seed(42)

    print(f"loading {MODEL_ID} from {MODEL_LOCAL} ...")
    assert MODEL_LOCAL.is_dir(), (
        f"{MODEL_LOCAL} not found -- run `bash packages/idris-transformers/"
        f"scripts/hf-download.sh {MODEL_ID}` first."
    )
    tokenizer = AutoTokenizer.from_pretrained(str(MODEL_LOCAL))
    model = AutoModelForCausalLM.from_pretrained(
        str(MODEL_LOCAL), torch_dtype=torch.bfloat16,
    )
    model.train(False)  # inference mode (== model.eval())

    # Workaround for transformers 5.9.0's CPU BitNet load bug:
    # `BitNetDeserialize.convert` (transformers/integrations/bitnet.py)
    # threads `target_dtype = weight.dtype` (uint8 on disk) through
    # `unpack_weights(... , dtype=target_dtype)` instead of using the
    # model's BF16 dtype. The result is U8 [out, in] with the ternary
    # codes wrap-encoded: {-1 -> 255, 0 -> 0, +1 -> 1}. Reading that
    # as int8 reinterprets 255 as -1, recovering the intended ternary,
    # which we then cast to BF16 for the forward pass. The condition
    # below picks every AutoBitLinear-typed module (skipping
    # `lm_head` which is tied) and rewrites its weight buffer.
    bitlinear_class_name = "AutoBitLinear"
    fixed = 0
    for name, mod in model.named_modules():
        if type(mod).__name__ != bitlinear_class_name:
            continue
        if mod.weight.dtype == torch.uint8:
            w_i8 = mod.weight.data.view(torch.int8)
            mod.weight.data = w_i8.to(torch.bfloat16)
            fixed += 1
    print(f"  patched {fixed} AutoBitLinear weights uint8 -> bfloat16 ternary")

    cfg = model.config
    print(
        f"  config: vocab_size={cfg.vocab_size} hidden_size={cfg.hidden_size} "
        f"num_hidden_layers={cfg.num_hidden_layers} "
        f"num_attention_heads={cfg.num_attention_heads} "
        f"num_key_value_heads={cfg.num_key_value_heads} "
        f"intermediate_size={cfg.intermediate_size} "
        f"hidden_act={cfg.hidden_act} "
        f"tie_word_embeddings={cfg.tie_word_embeddings}"
    )
    assert cfg.hidden_size == HIDDEN
    assert cfg.vocab_size == VOCAB
    assert cfg.hidden_act == "relu2"
    assert cfg.tie_word_embeddings is True, (
        "BitNet 2B-4T should tie embeddings; oracle relies on that."
    )

    actual_ids = tokenizer.encode("Hello world", add_special_tokens=False)
    assert actual_ids == FIXED_INPUT_IDS, (
        f"Tokenizer drift: expected {FIXED_INPUT_IDS}, got {actual_ids}."
    )

    input_ids = torch.tensor([FIXED_INPUT_IDS], dtype=torch.long)

    with torch.no_grad():
        outputs = model(input_ids=input_ids)

    logits = outputs.logits  # [1, seq, vocab]
    assert logits.shape == (1, len(FIXED_INPUT_IDS), VOCAB), (
        f"logits shape {logits.shape} != [1, {len(FIXED_INPUT_IDS)}, {VOCAB}]"
    )
    assert torch.isfinite(logits).all()

    # Save last-position logits in F32 (the on-disk dtype is BF16 but
    # the comparison runs against Idris-side F32/F64 -- promote here to
    # keep the comparator dtype-uniform).
    output_vec = logits[0, -1, :].to(torch.float32).contiguous()
    assert output_vec.shape == (VOCAB,)

    save_file({"output": output_vec}, str(ORACLE_PATH))
    print(f"wrote {ORACLE_PATH}")
    print(f"  shape: {list(output_vec.shape)}")
    print(f"  dtype: {output_vec.dtype}")
    print(f"  first 5 values: {output_vec[:5].tolist()}")
    print(f"  argmax token id: {output_vec.argmax().item()}")
    top5 = output_vec.topk(5)
    print(f"  top-5 token ids: {top5.indices.tolist()}")
    print(f"  top-5 token strs: {[tokenizer.decode([i]) for i in top5.indices.tolist()]}")


if __name__ == "__main__":
    main()
