||| Owning-ipkg discovery for the import grouping's local tier.
|||
||| The three-tier import grouping needs to know which imports are
||| package-local. Rather than asking the caller (or hardcoding project
||| names), derive it: the ipkg that owns the file being formatted lists
||| every module of its package, so "local" is exact membership in that
||| list. External stays the curated stdlib/compiler set, and the middle
||| (cross-package library) tier is everything else — no configuration.
|||
||| Discovery walks up from the file to the nearest directory containing
||| `.ipkg` files (the colocated dual-ipkg pattern means several: library +
||| tests sharing one sourcedir). The owner is the ipkg whose `modules`
||| list contains the file's own module; its module list (plus `main`) is
||| the local set. No ipkg found → empty local set → the grouping degrades
||| to two tiers. Misclassification is aesthetic only — `safeImportSort`
||| still guarantees no import is added or lost.
module Format.Ipkg

import Data.List
import Data.String
import System.Directory
import System.File

||| The fields of an ipkg the grouping needs.
public export
record IpkgInfo where
  constructor MkIpkgInfo
  sourcedir : String
  modules   : List String

-- The value after the first `=` on a line, trimmed, quotes stripped.
valueAfterEq : String -> String
valueAfterEq l =
  let stripQ = pack . filter (/= '"') . unpack
  in stripQ (trim (pack (drop 1 (dropWhile (/= '=') (unpack l)))))

||| Parse ipkg text (line-based). `modules =` starts a block whose
||| continuation lines begin with `,`; `main = X` counts as a module;
||| `sourcedir = "src"` defaults to ".".
export
parseIpkg : String -> IpkgInfo
parseIpkg src = go False (MkIpkgInfo "." []) (lines src)
  where
    go : (inModules : Bool) -> IpkgInfo -> List String -> IpkgInfo
    go _ acc []                = acc
    go inModules acc (l :: ls) =
      let t = trim l in
      if "sourcedir" `isPrefixOf` t
        then go False ({ sourcedir := valueAfterEq t } acc) ls
      else if "modules" `isPrefixOf` t
        then go True ({ modules $= (++ [valueAfterEq t]) } acc) ls
      else if "main" `isPrefixOf` t
        then go False ({ modules $= (++ [valueAfterEq t]) } acc) ls
      else if inModules && ("," `isPrefixOf` t)
        then go True ({ modules $= (++ [trim (pack (drop 1 (unpack t)))]) } acc) ls
      else go False acc ls

||| The module name a file would have under `<ipkgDir>/<sourcedir>/`, or
||| Nothing if the file does not live there.
export
moduleNameFor : (ipkgDir : String) -> (sourcedir : String) -> (file : String) -> Maybe String
moduleNameFor ipkgDir sourcedir file =
  let prefixDir = (if ipkgDir == "" then "" else ipkgDir ++ "/")
               ++ (if sourcedir == "." || sourcedir == "" then "" else sourcedir ++ "/")
  in if (prefixDir `isPrefixOf` file) && (".idr" `isSuffixOf` file)
       then let rel = pack (drop (length (unpack prefixDir)) (unpack file))
                stem = pack (reverse (drop 4 (reverse (unpack rel))))
            in Just (pack (map (\c => if c == '/' then '.' else c) (unpack stem)))
       else Nothing

||| Parent directory of a relative path ("" when none is left).
export
parentDir : String -> String
parentDir p = case break (== '/') (reverse (unpack p)) of
  (_, [])        => ""
  (_, _ :: rest) => pack (reverse rest)

||| The local-module set for a file: walk up to the nearest directory with
||| ipkg files, prefer the owner (the ipkg listing this file's module),
||| union the matching module lists. `[]` when no ipkg is found.
export
localModulesFor : (file : String) -> IO (List String)
localModulesFor file = go 10 (parentDir file)
  where
    ipkgsIn : String -> IO (List String)
    ipkgsIn dir = do
      Right entries <- listDir (if dir == "" then "." else dir)
        | Left _ => pure []
      pure (map (\e => (if dir == "" then "" else dir ++ "/") ++ e)
                (filter (".ipkg" `isSuffixOf`) entries))

    slurp : String -> IO (Maybe IpkgInfo)
    slurp fn = do
      Right s <- readFile fn | Left _ => pure Nothing
      pure (Just (parseIpkg s))

    ownerOf : String -> IpkgInfo -> Bool
    ownerOf dir inf = case moduleNameFor dir inf.sourcedir file of
      Just mn => mn `elem` inf.modules
      Nothing => False

    go : Nat -> String -> IO (List String)
    go Z _       = pure []
    go (S k) dir = do
      paths <- ipkgsIn dir
      case paths of
        [] => if dir == "" then pure [] else go k (parentDir dir)
        _  => do
          mInfos <- traverse slurp paths
          let infos  = mapMaybe id mInfos
          let owners = filter (ownerOf dir) infos
          let chosen = if null owners then infos else owners
          pure (concatMap (\i => i.modules) chosen)
