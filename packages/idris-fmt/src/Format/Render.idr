||| The reformatter. Passes run in order — whitespace hygiene, import
||| sort/dedup, column alignment, then reindentation — and each is gated by its
||| own round-trip oracle, falling back to its input on any doubt, so the tool
||| can only ever emit code that means the same as its input.
module Format.Render

import Data.List
import Data.String

import Format.Align
import Format.Imports
import Format.Reindent
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
sortImportsSafe : (localMods : List String) -> String -> String
sortImportsSafe localMods src =
  case sortImports localMods src of
    Nothing   => src
    Just cand => if safeImportSort src cand then cand else src

||| Apply one transform, gated by a round-trip oracle; fall back to the input
||| on rejection. The oracle differs per transform (spacing vs reindent), so it
||| is a parameter.
gatedPass : (String -> String -> Bool) -> (String -> String) -> String -> String
gatedPass safe pass src =
  let out = pass src
  in if safe src out then out else src

||| Alignment passes are pure spacing, so the token stream + parse must be
||| identical: gate with `safeReformat`.
alignPass : (String -> String) -> String -> String
alignPass = gatedPass safeReformat

||| Run every alignment pass in turn, each independently oracle-gated.
||| Colons (annotations), then equals (bindings / multi-clause defs), then
||| arrows (case / with arms). The passes target disjoint lines in practice.
alignSafe : String -> String
alignSafe = alignPass alignArrows . alignPass alignEquals . alignPass alignColons

||| Reindentation changes leading whitespace only; gate with `safeReindent`
||| (parse + deep AST equality + imports unchanged — `codeSig` is trivially
||| preserved here so it cannot oracle a layout change).
reindentSafe : String -> String
reindentSafe = gatedPass safeReindent reindent

||| Reformat a source string: whitespace hygiene, import sort/dedup, column
||| alignment, then reindentation. Guaranteed safe — each pass is gated by its
||| round-trip oracle and falls back to its input, so the result can never
||| change the code's meaning. `localMods` is the owning package's module
||| list for the import grouping's local tier (empty = two tiers).
export
formatWith : (localMods : List String) -> String -> String
formatWith localMods src =
  let hy = hygiene src
      hy2 = if safeReformat src hy then hy else src
  in reindentSafe (alignSafe (sortImportsSafe localMods hy2))

||| `formatWith` with no known local modules.
export
format : String -> String
format = formatWith []

||| Is the source already in formatted (fixed-point) form?
export
isFormattedWith : (localMods : List String) -> String -> Bool
isFormattedWith localMods src = formatWith localMods src == src

||| `isFormattedWith` with no known local modules.
export
isFormatted : String -> Bool
isFormatted = isFormattedWith []
