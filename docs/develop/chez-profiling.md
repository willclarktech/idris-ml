# Chez Source-Level Profiling for Idris-Generated Code

Use this when you need to know **which Idris-side code is hot** —
the Idris VM time between FFI calls, not the C-side compute. Took
~20 minutes to set up from scratch on 2026-05-14 and immediately
localised a 22× perf regression to `Layer/Transformer.idr`'s
positional encoding (recursive `Data.Nat` arithmetic recomputed
every forward pass — see `perf-changes.md` 2026-05-14 entries).

## How it works

Idris 2's Chez backend emits two artefacts under
`build/exec/<example>_app/`:

- `<example>.ss` — the generated Scheme source (~thousands of lines
  for a typical example)
- `<example>.so` — Chez's compiled bytecode
- `compileChez` — a script that does
  `(parameterize ([optimize-level 3] [compile-file-message #f])
     (compile-program "<example>.ss"))`

Chez's `compile-profile` parameter, set to `'source`, embeds per-line
counters into the compiled binary. After the binary runs and main
returns, `(profile-dump-html "<prefix>")` writes an HTML heatmap of
per-line execution counts.

The Idris-generated `.ss` already starts with `(import (chezscheme))`,
so all Chez profile procedures (`compile-profile`, `profile-dump-html`,
etc.) are available.

## Recipe

```bash
# 1. Pick an example. Make sure the binary is up to date.
EX=gpt-large
make BACKEND=mlx MLX_SITE=/path/to/pip/mlx example-$EX  # or your build

# 2. Clone the generated Scheme, add a trailing profile-dump call.
cp build/exec/${EX}_app/${EX}.ss /tmp/${EX}-prof.ss

# Find the line that calls main (Idris emits exactly one such call):
#   (PrimIO-unsafePerformIO (lambda (eta-0) (ExampleC-45<Name>-main eta-0)))
# Append after it (no harm if compile-profile is off):
cat >>/tmp/${EX}-prof.ss <<'PROFILE'
(profile-dump-html "/tmp/${EX}-prof")
PROFILE
# In practice use sed -i '/<main call regex>/a\ ...' instead.

# 3. Compile with profile counters enabled. optimize-level 3 stays.
CHEZ=$(which scheme)
$CHEZ -q <<EOF
(parameterize ([compile-profile 'source]
               [optimize-level 3]
               [compile-file-message #f])
  (compile-program "/tmp/${EX}-prof.ss"))
EOF

# 4. Run. Use the same env / dylib paths as the regular binary.
DYLD_LIBRARY_PATH=$PWD/build/exec/${EX}_app:/path/to/mlx/lib \
IDRIS2_INC_SRC=$PWD/build/exec/${EX}_app \
MLX_DEVICE=cpu \
  $CHEZ --program /tmp/${EX}-prof.so --seed 99 --epochs 1

# 5. Open the heatmap.
open /tmp/${EX}-profprofile.html  # summary by line, sorted hottest first
# Per-line counts also at /tmp/${EX}-prof${EX}-prof.ss.html
```

## Reading the output

`profile.html` lists hottest lines first, color-coded:

| Class | Count range |
|------|---:|
| pc12  | ~1B |
| pc11  | ~965M-980M |
| pc10  | ~482M-490M |
| pc9   | ~30M |
| pc8   | ~18M |
| pc7   | ~15M |
| pc6   | ~9M-10M |
| pc5   | ~6M-7M |
| pc4   | ~3M |
| pc3   | ~1M |
| pc2   | 1-~80K |
| pc1   | 0 |

Click any line for the per-line context in the `.ss.html` source view.
The Idris codegen mangles names: `DataC-45Nat-lte` is `Data.Nat.lte`,
`ExampleC-45GptLarge-main` is `Example.GptLarge.main`. The `C-45` is a
hyphen, `C-39` is an apostrophe, `C-60`/`C-62` are `<`/`>`, and so on.

## What it caught (2026-05-14, GptLarge)

Top entries pointed at `Data.Nat.lte` at 1.9B, `Data.Nat.divC-39` and
`modC-39` at 490M each — all firing from `Layer/Transformer.idr`'s
`posEncVal` being called 1M+ times per epoch with recursive Peano
arithmetic. Caching the positional encoding + replacing the recursive
Nat arithmetic with `Int` gave a 22× wall reduction on every backend.

## Caveats

- `compile-profile 'source` requires the source `.ss` file path to
  remain accessible at runtime. The Chez profile data tags each
  counter with `(file, line, column)` triples.
- The instrumented binary runs slower than the optimised one
  (probably 2-5× depending on hot-loop density) — only use for
  diagnosis, not perf measurement of fixes.
- The counter values are *execution counts*, not wallclock. A line
  executed 1B times in 100ms is fine; a line executed 100 times in
  1s is the real problem. Use the counts as a *call-frequency*
  signal and combine with wallclock numbers from
  `perf-log.jsonl`.
- For per-procedure (not per-line) flame-graph–style output, use
  `(profile-dump-list)` or `(profile-query-weight)` from a Chez REPL.
