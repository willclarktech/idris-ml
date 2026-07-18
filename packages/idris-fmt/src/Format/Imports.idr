||| Import sort + dedup pass.
|||
||| Imports are grouped in three tiers — external/stdlib first, then
||| cross-package libraries, then package-local modules — sorted
||| alphabetically by module path within each group with a blank line
||| between groups, and exact duplicates removed. "Local" is exact
||| membership in the owning ipkg's module list (see `Format.Ipkg`);
||| the library tier is everything neither external nor local. An empty
||| local set degrades to two tiers.
||| `Show Import` already emits the canonical `import [public] Path [as NS]`
||| line, so rendering is just `show`; any mis-render is caught by
||| `safeImportSort` (the caller's gate).
|||
||| Conservative: the pass only fires when the import block is a clean run
||| of import lines + blanks (no interleaved comments). Anything else →
||| `Nothing` (the caller keeps the original), so a comment between imports
||| is never lost.
module Format.Imports

import Data.List
import Data.List1
import Data.String
import Idris.Syntax

import Format.Roundtrip

-- Top-level namespaces treated as external (stdlib + curated deps); these
-- group before project-internal imports. Misclassification only affects
-- grouping aesthetics — safeImportSort still guarantees no import is
-- added/lost, so it can never corrupt code.
externalRoots : List String
externalRoots =
  [ "Data", "System", "Control", "Decidable", "Language", "Text"
  , "Debug", "Builtin", "Prelude", "Syntax", "Deriving", "Network"
  , "Libraries", "Hedgehog", "Core", "Idris", "Parser" ]

pathString : Import -> String
pathString imp = show imp.path

firstSeg : Import -> String
firstSeg imp = head (split (== '.') (pathString imp))

isExternal : Import -> Bool
isExternal imp = firstSeg imp `elem` externalRoots

-- Sorted/grouped/deduped import lines (external, library, local tiers).
-- `localMods` is the owning package's module list (Format.Ipkg); imports
-- in it are the local tier, non-external non-local imports the library
-- tier. Empty local set degrades to two tiers.
renderBlock : (localMods : List String) -> List Import -> List String
renderBlock localMods imps =
  let deduped = nubBy (\a, b => show a == show b) imps
      byPath = sortBy (\a, b => compare (pathString a) (pathString b))
      isLocal : Import -> Bool
      isLocal imp = pathString imp `elem` localMods
      ext         = byPath (filter isExternal deduped)
      lib         = byPath (filter (\i => not (isExternal i) && not (isLocal i)) deduped)
      loc         = byPath (filter (\i => isLocal i && not (isExternal i)) deduped)
      joinGroups : List (List String) -> List String
      joinGroups gs = case filter (not . null) gs of
                        []          => []
                        (g :: rest) => g ++ concatMap ([""] ++) rest
  in joinGroups [map show ext, map show lib, map show loc]

isImportLine : String -> Bool
isImportLine l = "import " `isPrefixOf` l

isCommentLine : String -> Bool
isCommentLine l =
  let t = trim l
  in ("--" `isPrefixOf` t) || ("{-" `isPrefixOf` t) || ("|||" `isPrefixOf` t)

||| Produce a candidate with imports sorted/deduped, or `Nothing` to bail
||| (no imports, or a comment is interleaved in the import block). Assumes
||| hygiene-normalised input (LF lines, single trailing newline). The caller
||| MUST gate the result through `Format.Roundtrip.safeImportSort`.
export
sortImports : (localMods : List String) -> String -> Maybe String
sortImports localMods src =
  case parseModule src of
    Nothing  => Nothing
    Just mod => case mod.imports of
      []   => Nothing
      imps =>
        let ls   = lines src
            idxs = findIndices isImportLine ls
        in case idxs of
             []        => Nothing
             (i0 :: _) =>
               let lastIdx = foldl (\_, x => x) i0 idxs
                   block   = take (S (lastIdx `minus` i0)) (drop i0 ls)
               in if any isCommentLine block
                    then Nothing
                    else let out = take i0 ls ++ renderBlock localMods imps ++ drop (S lastIdx) ls
                         in Just (concat (map (++ "\n") out))
