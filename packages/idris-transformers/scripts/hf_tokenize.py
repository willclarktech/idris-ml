"""Tokenizer subprocess used by idris-transformers' Tokenizer.idr.

Wraps HuggingFace's `transformers.AutoTokenizer` so the Idris side gets
the same vocab + tokenization Python ships, without needing to bind
the Rust `huggingface/tokenizers` crate via FFI. Three subcommands:

  python hf_tokenize.py <repo> vocab
      Prints the tokenizer's vocab size to stdout (one integer, no
      trailing whitespace beyond the print's newline).

  python hf_tokenize.py <repo> encode --input-file <path>
      Reads UTF-8 text from `<path>`, prints space-separated token IDs
      to stdout. (`--input-file` over an argv string sidesteps shell
      quoting on arbitrary inputs.)

  python hf_tokenize.py <repo> decode --input-file <path>
      Reads space-separated IDs from `<path>`, writes the decoded
      string to stdout WITHOUT a trailing newline (so captured output
      is exactly the decoded text).

Failure modes return non-zero exit + an error line on stderr.

The Idris-side wrapper (`Tokenizer.idr`) calls this script via
`System.system` with stdout redirected to a temp file, then reads +
parses. ~1s of Python startup per call — acceptable because the
caller pattern is "tokenize the prompt once, generate N tokens
in-process, detokenize the result once". For perf-critical workloads
a Rust-FFI backing is filed as a future row in TODO.md.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from transformers import AutoTokenizer


SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT  = SCRIPT_DIR.parent.parent.parent   # <repo-root>
MODELS_DIR = REPO_ROOT / "models"


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("repo", help="HF repo name (e.g. distilgpt2)")
    ap.add_argument("mode", choices=["vocab", "encode", "decode"])
    ap.add_argument(
        "--input-file",
        default=None,
        help="Read input from a file rather than argv. Required for encode/decode.",
    )
    args = ap.parse_args()

    # Load from the local <repo-root>/models/<repo>/ dir populated by
    # hf-download.sh. Single physical copy shared with Idris's loadModel
    # (no separate ~/.cache/huggingface).
    local_path = MODELS_DIR / args.repo
    if not local_path.is_dir():
        print(
            f"error: {local_path} not found — run\n"
            f"  bash packages/idris-transformers/scripts/hf-download.sh {args.repo}\n"
            f"first.",
            file=sys.stderr,
        )
        return 1
    tok = AutoTokenizer.from_pretrained(str(local_path))

    if args.mode == "vocab":
        # `vocab_size` reports the model's expected embedding-table size,
        # not the number of entries in the underlying BPE/WordPiece dict.
        # For HF-aligned modules we want the embedding-table size — it's
        # what the model.wte.weight first dim is.
        print(tok.vocab_size)
        return 0

    if args.input_file is None:
        print("error: encode/decode require --input-file", file=sys.stderr)
        return 1

    with open(args.input_file, encoding="utf-8") as f:
        raw = f.read()

    if args.mode == "encode":
        # add_special_tokens=True so [CLS]/[SEP] (BERT), <|endoftext|>
        # (GPT-2), <|begin_of_text|> (Llama-3) etc. get added by default.
        # Callers that want to NOT add them can subtract them after the
        # fact; we err on the side of "behaves like AutoTokenizer".
        ids = tok.encode(raw, add_special_tokens=True)
        # Space-separated IDs on a single line.
        sys.stdout.write(" ".join(str(i) for i in ids))
        return 0

    if args.mode == "decode":
        ids = [int(x) for x in raw.split() if x.strip()]
        out = tok.decode(ids, skip_special_tokens=False)
        # No trailing newline — what we write IS the decoded string.
        sys.stdout.write(out)
        return 0

    print(f"error: unknown mode {args.mode!r}", file=sys.stderr)
    return 1


if __name__ == "__main__":
    sys.exit(main())
