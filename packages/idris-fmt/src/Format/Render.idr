||| The reformatter. The current pass is conservative whitespace hygiene
||| (no reindentation, no alignment, no import sorting yet — those layer on
||| top in later passes); every result is gated by `safeReformat` so the
||| tool can only ever emit code that lexes+parses to the same tokens as its
||| input, falling back to the original text otherwise.
module Format.Render

import Data.List
import Data.String

import Format.Align
import Format.Imports
import Format.Roundtrip

||| Drop trailing spaces/tabs from a single line (no newline expected).
rstrip : String -> String
rstrip = pack . reverse . dropWhile isSpace . reverse . unpack

||| Collapse runs of 2+ blank lines down to a single blank line.
collapseBlanks : List String -> List String
collapseBlanks = go False
  where
    go : (prevBlank : Bool) -> List String -> List String
    go _ []                = []
    go prevBlank (l :: ls) =
      let blank = l == "" in
      if blank && prevBlank
        then go True ls
        else l :: go blank ls

||| Drop leading and trailing blank lines.
trimBlanks : List String -> List String
trimBlanks =
  reverse . dropWhile (== "") . reverse . dropWhile (== "")

||| Whitespace hygiene: strip trailing whitespace, collapse blank runs,
||| trim leading/trailing blank lines, and end with exactly one newline.
||| (A file of only blank lines normalises to empty.)
hygiene : String -> String
hygiene src =
  let ls = trimBlanks (collapseBlanks (map rstrip (lines src)))
  in concat (map (++ "\n") ls)

||| Apply the import-sort pass, gated by `safeImportSort`; on any doubt
||| (bail, or oracle rejection) return the input unchanged.
sortImportsSafe : String -> String
sortImportsSafe src =
  case sortImports src of
    Nothing   => src
    Just cand => if safeImportSort src cand then cand else src

||| Apply one alignment pass, gated by `safeReformat` (pure spacing, so the
||| token stream + parse must be identical); fall back on rejection.
alignPass : (String -> String) -> String -> String
alignPass pass src =
  let out = pass src
  in if safeReformat src out then out else src

||| Run every alignment pass in turn, each independently oracle-gated.
||| Colons (annotations), then equals (bindings / multi-clause defs), then
||| arrows (case / with arms). The passes target disjoint lines in practice.
alignSafe : String -> String
alignSafe = alignPass alignArrows . alignPass alignEquals . alignPass alignColons

||| Reformat a source string: whitespace hygiene, import sort/dedup, then
||| colon alignment. Guaranteed safe — each pass is gated by its round-trip
||| oracle and falls back to its input, so the result can never change the
||| code's meaning.
export
format : String -> String
format src =
  let hy = hygiene src
      hy2 = if safeReformat src hy then hy else src
  in alignSafe (sortImportsSafe hy2)

||| Is the source already in formatted (fixed-point) form?
export
isFormatted : String -> Bool
isFormatted src = format src == src
