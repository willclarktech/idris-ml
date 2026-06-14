||| The reformatter. The current pass is conservative whitespace hygiene
||| (no reindentation, no alignment, no import sorting yet — those layer on
||| top in later passes); every result is gated by `safeReformat` so the
||| tool can only ever emit code that lexes+parses to the same tokens as its
||| input, falling back to the original text otherwise.
module Format.Render

import Data.List
import Data.String

import Format.Roundtrip

||| Drop trailing spaces/tabs from a single line (no newline expected).
rstrip : String -> String
rstrip = pack . reverse . dropWhile isSpace . reverse . unpack

||| Collapse runs of 2+ blank lines down to a single blank line.
collapseBlanks : List String -> List String
collapseBlanks = go False
  where
    go : (prevBlank : Bool) -> List String -> List String
    go _ [] = []
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

||| Reformat a source string. Guaranteed safe: if the hygiene pass ever
||| produced output that did not round-trip, the original is returned
||| unchanged.
export
format : String -> String
format src =
  let out = hygiene src
  in if safeReformat src out then out else src

||| Is the source already in formatted (fixed-point) form?
export
isFormatted : String -> Bool
isFormatted src = format src == src
