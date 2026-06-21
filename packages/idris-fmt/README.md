# idris-fmt

A compiler-native source formatter for Idris 2 — and the repo's own `make fmt-idris`. It parses
with the compiler's own parser and lexer (`Idris.Parser`, `Parser.Lexer.Source`), so it sees the
code exactly as `idris2` does, and **gates every reformat behind a round-trip oracle**: it can
never emit code that differs in meaning from the input.

## What it does

Whitespace hygiene, import sorting, `:`/`=`/`=>` alignment, and FC-driven reindentation — each
pass independently oracle-gated with an identity fallback (if a pass can't prove it preserved
meaning, the input is returned unchanged).

## CLI

```bash
idris-fmt --write FILE...        # format files in place
idris-fmt --check FILE...        # exit 1 if any file is not already formatted
idris-fmt --parse-check FILE...  # parse every file, no formatting (coverage probe)
idris-fmt --help                 # usage (via idris-args)
```

Positional arguments are the file paths. Depends on `base`, `idris2` (the compiler as a library),
and [idris-args](../idris-args/).

## In the repo

```bash
make fmt-idris    # format every .idr in place
make check-fmt    # fail if anything is unformatted (the test-integration-lint-fmt CI gate)
```

`make fmt` / `make check-fmt` are the cross-language umbrellas; the Idris arm is this tool. See
the formatter conventions in [CLAUDE.md](../../CLAUDE.md) ("Formatters").

## The round-trip oracle

`Format.Roundtrip` is the safety net. Two independent signatures of a source string must match
before and after a reformat:

- `codeSig` — the lexer token signature (whitespace-insensitive token stream).
- `astSig` / `deepSig` — a structural signature from the compiler's parser.

Whitespace-only passes are checked with the token oracle; reindentation (which leaves the token
stream unchanged) uses the deeper AST oracle. If either signature would change, the pass falls
back to the original text. This is why the tool is safe to run unattended in CI.
