||| Reindentation: recompute each line's leading whitespace to the repo's
||| canonical offside layout, driven by the parsed AST's FCs — no pretty-printer.
|||
||| The walk threads the canonical *column* a block's items should sit at (the
||| Reader value) and emits, per declaration / clause / constructor / field, an
||| `Anchor` = (source line, canonical column) (the Writer output) — an RWS-lite
||| shape borrowed from `gvnkd/idris2-fmt`, applied to a span walk rather than a
||| `Doc` printer. Repo conventions (calibrated against the corpus): nested
||| declaration blocks (data / record / interface / implementation / mutual /
||| namespace / parameters / using / failing) indent +2; a function `where`
||| block indents its items +4 (own-line `where` at +2, items +2 beyond);
||| `with`-clause sub-clauses +2.
|||
||| Each source line is then shifted by the delta of the nearest preceding
||| anchor, so anchor lines land on their canonical column while every
||| continuation / term-internal line (do / case / let / multi-line type) moves
||| by the *same* delta — relative layout is preserved, never reflowed. Term
||| internals are not descended; they ride on their declaration's shift.
|||
||| `reindent` is the pure transform (parse-failure ⇒ identity). The caller
||| gates it with `Format.Roundtrip.safeReindent`, so a mis-shift can only ever
||| fall back to the input, never change meaning.
module Format.Reindent

import Data.List
import Data.Maybe
import Data.String

import Core.FC
import Idris.Syntax

import Format.Roundtrip

%default covering

-- A declaration/clause/constructor/field head: (source line 0-indexed,
-- canonical column it should sit at).
0 Anchor : Type
Anchor = (Int, Int)

-- Concrete start line of a node's span, or Nothing for virtual / empty FCs
-- (compiler-generated — never a real source anchor).
fcLine : FC -> Maybe Int
fcLine fc = (fst . startPos) <$> isConcreteFC fc

anchorAt : FC -> Int -> List Anchor
anchorAt fc col = toList ((\l => (l, col)) <$> fcLine fc)

mutual
  collectDecls : (col : Int) -> List PDecl -> List Anchor
  collectDecls col = concatMap (collectDecl col)

  collectDecl : (col : Int) -> PDecl -> List Anchor
  collectDecl col d = anchorAt d.fc col ++ collectBody col d.val

  collectBody : (col : Int) -> PDeclNoFC -> List Anchor
  collectBody col (PDef cls)         = concatMap (collectClause col) cls
  collectBody col (PData _ _ _ dd)   = collectData col dd
  collectBody col (PRecord _ _ _ rd) = collectRecord col rd
  collectBody col (PParameters _ ds) = collectDecls (col + 2) ds
  collectBody col (PUsing _ ds)      = collectDecls (col + 2) ds
  collectBody col (PInterface _ _ _ _ _ _ _ ds) = collectDecls (col + 2) ds
  collectBody col (PImplementation _ _ _ _ _ _ _ _ _ mds) =
    maybe [] (collectDecls (col + 2)) mds
  collectBody col (PFail _ ds)       = collectDecls (col + 2) ds
  collectBody col (PMutual ds)       = collectDecls (col + 2) ds
  collectBody col (PNamespace _ ds)  = collectDecls (col + 2) ds
  collectBody _   _                  = []

  collectClause : (col : Int) -> PClause -> List Anchor
  collectClause col (MkPatClause fc _ _ wb) =
    anchorAt fc col ++ collectDecls (col + 4) wb
  collectClause col (MkWithClause fc _ _ _ cs) =
    anchorAt fc col ++ concatMap (collectClause (col + 2)) cs
  collectClause col (MkImpossible fc _) = anchorAt fc col

  collectData : (col : Int) -> PDataDecl -> List Anchor
  collectData col (MkPData _ _ _ _ cons) =
    concatMap (\c => anchorAt c.fc (col + 2)) cons
  collectData _ (MkPLater _ _ _) = []

  collectRecord : (col : Int) -> PRecordDecl' Name -> List Anchor
  collectRecord col (MkPRecord _ _ _ _ flds) =
    concatMap (\f => anchorAt f.fc (col + 2)) flds
  collectRecord _ (MkPRecordLater _ _) = []

indexList : Nat -> List a -> Maybe a
indexList _     []         = Nothing
indexList Z     (x :: _)   = Just x
indexList (S k) (_ :: xs)  = indexList k xs

leadingInt : String -> Int
leadingInt s = cast (length (takeWhile (== ' ') (unpack s)))

-- Turn anchors into (line, delta) where delta = canonicalCol - currentLeading.
toDeltas : List String -> List Anchor -> List (Int, Int)
toDeltas srcLines = mapMaybe mk
  where
    mk : Anchor -> Maybe (Int, Int)
    mk (ln, col) =
      if ln < 0 then Nothing
      else (\l => (ln, col - leadingInt l)) <$> indexList (integerToNat (cast ln)) srcLines

-- Re-emit a line with `d` added to its leading-space count (blank lines stay
-- blank; negative results clamp to 0).
shiftLine : Int -> String -> String
shiftLine d line =
  let (sp, rest) = span (== ' ') (unpack line) in
  if rest == []
    then ""
    else pack (replicate (integerToNat (cast (max 0 (cast (length sp) + d)))) ' ' ++ rest)

-- Apply deltas in source order: each line takes the delta of the nearest
-- preceding anchor (0 before the first anchor — module header / imports).
reflow : List (Int, Int) -> List String -> List String
reflow deltas = go 0 0 (sortBy (\x, y => compare (fst x) (fst y)) deltas)
  where
    go : (idx : Int) -> (cur : Int) -> List (Int, Int) -> List String -> List String
    go _   _   _  []        = []
    go idx cur ds (l :: ls) =
      let (cur', ds') = advance cur ds in
      shiftLine cur' l :: go (idx + 1) cur' ds' ls
      where
        advance : Int -> List (Int, Int) -> (Int, List (Int, Int))
        advance c ((aln, ad) :: rest) = if aln <= idx then advance ad rest else (c, (aln, ad) :: rest)
        advance c []                  = (c, [])

export
reindent : String -> String
reindent src = case parseModule src of
  Nothing => src
  Just m  =>
    let srcLines = lines src
        anchors  = collectDecls 0 m.decls
        deltas   = toDeltas srcLines anchors
    in concat (map (++ "\n") (reflow deltas srcLines))
