||| Column-alignment pass (currently: type-annotation colons).
|||
||| Aligns the `:` across a run of consecutive same-indentation lines that
||| are each a simple `name(s) : type` annotation — i.e. record-field blocks,
||| data-constructor blocks, and adjacent type signatures (the columnar
||| style used by e.g. the HfBert records). Only the spaces *before* the
||| colon change, so the token stream and AST are untouched; the caller's
||| `safeReformat` gate makes mis-grouping impossible to turn into a code
||| change (worst case it's a no-op).
|||
||| The colon is located with the compiler lexer (not regex): the first
||| depth-0 `Symbol ":"` (the lexer tokenises `::` / `:=` distinctly) whose
||| preceding tokens are all identifiers/commas. That precisely picks out
||| annotation colons and skips `::`, `:=`, and colons inside brackets or
||| expressions.
module Format.Align

import Data.List
import Data.Maybe
import Data.String
import Libraries.Text.Bounded

import Parser.Lexer.Source
import Parser.Source

leadingSpaces : String -> Nat
leadingSpaces s = length (takeWhile (== ' ') (unpack s))

isNameTok : Token -> Bool
isNameTok (Ident _) = True
isNameTok (Symbol ",") = True
isNameTok _ = False

opens : String -> Bool
opens s = s == "(" || s == "[" || s == "{"

closes : String -> Bool
closes s = s == ")" || s == "]" || s == "}"

||| Char offset of the annotation colon, or Nothing if the line is not a
||| simple `name(s) : type` line.
colonCol : String -> Maybe Nat
colonCol line = case lex line of
  Left _ => Nothing
  Right (_, toks) => go 0 [] toks
  where
    go : Integer -> List Token -> List (WithBounds Token) -> Maybe Nat
    go _ _ [] = Nothing
    go d pre (t :: ts) = case t.val of
      Symbol ":" =>
        if d == 0 && not (isNil pre) && all isNameTok pre
          then Just (cast (snd (start t)))
          else Nothing
      Symbol s =>
        if opens s then go (d + 1) (t.val :: pre) ts
        else if closes s then go (d - 1) (t.val :: pre) ts
        else go d (t.val :: pre) ts
      other => go d (other :: pre) ts

-- Length of the line up to (and excluding) the colon, trailing spaces dropped.
preLen : (String, Nat) -> Nat
preLen (line, col) = length (pack (reverse (dropWhile (== ' ') (reverse (unpack (substr 0 col line))))))

-- Re-emit a line with its colon at `target` (>= preLen+1), spaces after
-- the colon untouched.
alignLine : (target : Nat) -> (String, Nat) -> String
alignLine target (line, col) =
  let pre  = pack (reverse (dropWhile (== ' ') (reverse (unpack (substr 0 col line)))))
      post = substr (S col) (length line) line
      pad  = target `minus` length pre
  in pre ++ pack (replicate pad ' ') ++ ":" ++ post

-- Align one group (>= 2 lines) of (line, colonCol) to a common column.
alignGroup : List (String, Nat) -> List String
alignGroup grp =
  let target = S (foldl max 0 (map preLen grp))
  in map (alignLine target) grp

||| Align type-annotation colons across consecutive same-indent annotation
||| lines. Pure spacing; gate the result through `safeReformat`.
export
alignColons : String -> String
alignColons src = concat (map (++ "\n") (go (lines src)))
  where
    -- collect a maximal run of same-indent annotation lines, then align it
    go : List String -> List String
    go [] = []
    go (l :: ls) = case colonCol l of
      Nothing => l :: go ls
      Just _ =>
        let ind = leadingSpaces l
            isMember : String -> Bool
            isMember x = leadingSpaces x == ind && isJust (colonCol x)
            run = l :: takeWhile isMember ls
            rest = drop (length run `minus` 1) ls
            withCols = mapMaybe (\x => map (MkPair x) (colonCol x)) run
        in if length run >= 2
             then alignGroup withCols ++ go rest
             else l :: go ls
