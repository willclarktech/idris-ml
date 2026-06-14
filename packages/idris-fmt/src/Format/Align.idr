||| Column-alignment passes (type-annotation `:`, binding/clause `=`,
||| case/with-arm `=>`).
|||
||| Each pass aligns a "key" token across a run of consecutive same-indent
||| lines that each carry that token at depth 0. Only the spaces *before* the
||| key change, so the token stream and AST are untouched; the caller's
||| `safeReformat` gate makes mis-grouping impossible to turn into a code
||| change (worst case it's a no-op).
|||
||| The key is located with the compiler lexer (not regex): the first
||| depth-0 token matching the pass predicate whose preceding tokens pass the
||| pass guard. The lexer tokenises `::` / `:=` / `==` / `=>` distinctly, so
||| matching `Symbol ":"` / `Symbol "="` / `Symbol "=>"` precisely picks out
||| the intended token and skips look-alikes, bracketed colons, etc.
|||
||| Scope of the `=` pass: aligns indented
||| binding groups (let / where / local defs) unconditionally, and top-level
||| multi-clause definitions only when the run shares one LHS head name —
||| unrelated top-level defs are left alone to avoid churn.
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
isNameTok (Ident _)    = True
isNameTok (Symbol ",") = True
isNameTok _            = False

opens : String -> Bool
opens s = s == "(" || s == "[" || s == "{"

closes : String -> Bool
closes s = s == ")" || s == "]" || s == "}"

-- Width (char count) of a token's surface text. Only symbols vary in length
-- among the keys we align (`:` / `=` = 1, `=>` = 2).
tokWidth : Token -> Nat
tokWidth (Symbol s) = length s
tokWidth _          = 1

isColon : Token -> Bool
isColon (Symbol ":") = True
isColon _            = False

isEquals : Token -> Bool
isEquals (Symbol "=") = True
isEquals _            = False

isArrow : Token -> Bool
isArrow (Symbol "=>") = True
isArrow _             = False

-- The leading identifier of a line (its LHS head), if any — used to keep the
-- top-level `=` pass to genuine multi-clause definitions.
headIdent : String -> Maybe String
headIdent line = case lex line of
  Right (_, (t :: _)) => case t.val of
    Ident n => Just n
    _       => Nothing
  _ => Nothing

||| `(charOffset, width)` of the first depth-0 token satisfying `match` whose
||| preceding tokens satisfy `okPre`, or Nothing if no such token exists (or
||| the first depth-0 match fails `okPre` — the line is then not a member).
locTok : (match : Token -> Bool) -> (okPre : List Token -> Bool) ->
         String -> Maybe (Nat, Nat)
locTok match okPre line = case lex line of
  Left _          => Nothing
  Right (_, toks) => go 0 [] toks
  where
    go : Integer -> List Token -> List (WithBounds Token) -> Maybe (Nat, Nat)
    go _ _ []          = Nothing
    go d pre (t :: ts) =
      if d == 0 && match t.val
        then (if okPre pre then Just (cast (snd (start t)), tokWidth t.val)
                           else Nothing)
        else case t.val of
          Symbol s =>
            if opens s then go (d + 1) (t.val :: pre) ts
            else if closes s then go (d - 1) (t.val :: pre) ts
            else go d (t.val :: pre) ts
          other => go d (other :: pre) ts

-- Length of the line up to (and excluding) the key, trailing spaces dropped.
preLen : (String, Nat, Nat) -> Nat
preLen (line, col, _) =
  length (pack (reverse (dropWhile (== ' ') (reverse (unpack (substr 0 col line))))))

-- Re-emit a line with its key at `target` (>= preLen+1); the key text and
-- everything after it are sliced from the original, so width is preserved.
alignLine : (target : Nat) -> (String, Nat, Nat) -> String
alignLine target (line, col, w) =
  let pre  = pack (reverse (dropWhile (== ' ') (reverse (unpack (substr 0 col line)))))
      key  = substr col w line
      post = substr (col + w) (length line) line
      pad  = target `minus` length pre
  in pre ++ pack (replicate pad ' ') ++ key ++ post

-- Align one group (>= 2 lines) of (line, keyCol, keyWidth) to a common column.
alignGroup : List (String, Nat, Nat) -> List String
alignGroup grp =
  let target = S (foldl max 0 (map preLen grp))
  in map (alignLine target) grp

||| Generic alignment driver. `match`/`okPre` locate the key; `extends first
||| cand` is the extra membership rule binding a candidate line into the run
||| started by `first` (beyond same-indent + has-key). Pure spacing; gate the
||| result through `safeReformat`.
alignWith : (match : Token -> Bool) -> (okPre : List Token -> Bool) ->
            (extends : String -> String -> Bool) -> String -> String
alignWith match okPre extends src = concat (map (++ "\n") (go (lines src)))
  where
    key : String -> Maybe (Nat, Nat)
    key = locTok match okPre

    go : List String -> List String
    go []        = []
    go (l :: ls) = case key l of
      Nothing => l :: go ls
      Just _  =>
        let ind = leadingSpaces l
            isMember : String -> Bool
            isMember x = leadingSpaces x == ind && isJust (key x) && extends l x
            run        = l :: takeWhile isMember ls
            rest       = drop (length run `minus` 1) ls
            withCols   = mapMaybe (\x => map (\(c, w) => (x, c, w)) (key x)) run
        in if length run >= 2
             then alignGroup withCols ++ go rest
             else l :: go ls

||| Align type-annotation colons across consecutive same-indent annotation
||| lines (record fields, data constructors, adjacent signatures).
export
alignColons : String -> String
alignColons = alignWith isColon (\pre => not (isNil pre) && all isNameTok pre)
                        (\_, _ => True)

||| Align binding/clause `=` across consecutive same-indent lines. Indented
||| groups always; top-level runs only when all lines share one LHS head
||| (genuine multi-clause defs, not unrelated definitions).
export
alignEquals : String -> String
alignEquals = alignWith isEquals (\pre => not (isNil pre)) extends
  where
    extends : String -> String -> Bool
    extends first cand =
      if leadingSpaces first == 0
        then case (headIdent first, headIdent cand) of
               (Just a, Just b) => a == b
               _                => False
        else True

||| Align `=>` across consecutive same-indent case / with arms.
export
alignArrows : String -> String
alignArrows = alignWith isArrow (\pre => not (isNil pre)) (\_, _ => True)
