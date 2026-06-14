||| The round-trip safety oracle: proof that a reformat changed only
||| layout + comments, never the code.
|||
||| A formatter for a layout-sensitive language is all-or-nothing — a pass
||| that silently drops a token or alters parse structure is worse than no
||| formatter. Every transform in idris-fmt is gated by `safeReformat`
||| below, so the tool can never emit code that means something different
||| from its input.
|||
||| Two independent checks, because neither alone is sufficient:
|||
|||   * `codeSig` (token-level): the compiler lexer's token stream, rendered
|||     to strings. `lex` already drops `Space` and line/block comments, so
|||     equal signatures means identical tokens modulo whitespace+comments.
|||     Catches dropped/added/mangled tokens (e.g. a botched FFI string).
|||
|||   * `parses` (structure-level): the formatted text still parses with the
|||     compiler's own parser. Catches layout damage that the token check
|||     misses — `lex` tokenizes *without* applying the offside rule, so a
|||     reindent that breaks a `where`/`do`/`case` block leaves the token
|||     stream identical yet changes (or breaks) the parse.
module Format.Roundtrip

import Core.Core
import Core.FC
import Parser.Source
import Idris.Parser
import Idris.Syntax

import Parser.Lexer.Source
import Libraries.Text.Bounded

import Data.List

tokShow : WithBounds Token -> String
tokShow t = show t.val

||| Token-level signature of a source string: the lexer token stream
||| rendered to strings. `Nothing` if the string does not lex.
export
codeSig : String -> Maybe (List String)
codeSig src = case lex src of
  Left _    => Nothing
  Right res => Just (map tokShow (snd res))

||| Does the string parse as a module with the compiler's own parser?
export
parses : String -> Bool
parses src =
  let origin = Virtual Interactive in
  case runParser origin Nothing src (prog origin) of
    Left _  => False
    Right _ => True

||| Parse to the compiler's surface `Module`, or `Nothing` on a parse error.
export
parseModule : String -> Maybe Module
parseModule src =
  let origin = Virtual Interactive in
  case runParser origin Nothing src (prog origin) of
    Left _            => Nothing
    Right (_, _, mod) => Just mod

||| FC-insensitive structural signature of the top-level declarations: each
||| decl rendered via `Show PDeclNoFC`, which omits FC. Equal `astSig` means
||| the same declaration tree modulo source positions + comments — used to
||| prove a layout/import transform left the *code* untouched.
|||
||| Caveat: `Show PDeclNoFC` / `Show PClause` are *shallow* — they collapse
||| `PData`/`PRecord`/`PInterface`/`PImplementation`/`PParameters`/`PMutual`/
||| `PNamespace`/`PFail` to bare constructor names, and any `where`-bearing or
||| `with` clause to `"MkPatClause"`/`"MkWithClause"`. So `astSig` suffices for
||| transforms that leave declarations textually intact (import-sort, alignment)
||| but is blind to reindentation, which can silently move a clause out of a
||| `where` block or a method out of an interface. Use `deepSig` for that.
export
astSig : String -> Maybe (List String)
astSig src = (\m => map (\d => show d.val) m.decls) <$> parseModule src

||| The module's imports, each rendered via `Show Import` (order-sensitive).
||| Compared as a multiset for import-sort (which reorders + dedups).
export
importSig : String -> Maybe (List String)
importSig src = (\m => map show m.imports) <$> parseModule src

||| `safeReformat original formatted` holds when `formatted` is a faithful
||| reformat of `original`: it still parses, and its token stream is
||| byte-identical (modulo whitespace + line/block comments). This is the
||| gate every transform must pass before its output is trusted.
export
safeReformat : (original : String) -> (formatted : String) -> Bool
safeReformat original formatted =
  parses formatted &&
  (case (codeSig original, codeSig formatted) of
     (Just a, Just b) => a == b
     _                => False)

||| Safety gate for the import-sort pass, which deliberately *reorders* (so
||| `codeSig` differs). Holds when the declarations are untouched and the
||| imports are the same set (a reordering with exact duplicates removed) —
||| i.e. no import was added or lost.
export
safeImportSort : (original : String) -> (formatted : String) -> Bool
safeImportSort original formatted =
  parses formatted &&
  astSig original == astSig formatted &&
  (case (importSig original, importSig formatted) of
     (Just a, Just b) => sort (nub a) == sort (nub b)
     _                => False)

-- Deep, FC-insensitive structural signature ----------------------------------
-- `astSig`/`Show` are shallow: `Show PDeclNoFC` collapses data/record/interface/
-- implementation/parameters/mutual/namespace/failing blocks to bare names, and
-- `Show PClause` collapses any `where`-bearing or `with` clause to
-- "MkPatClause"/"MkWithClause". The functions below descend that declaration +
-- clause layer by hand, reusing the compiler's already-deep `Show PTerm`,
-- `Show PTypeDecl`, and `Show (PiBindData ·)` at every term leaf — so a false
-- "equal" cannot hide a block-membership change (a clause leaving a `where`
-- block, a method leaving an interface, a field leaving a record, …). Term
-- internals (do/case/let/with layout) ride for free on `Show PTerm`.

mutual
  sigDecls : List PDecl -> String
  sigDecls ds = concatMap (\d => sigDeclNoFC d.val ++ ";") ds

  sigClauses : List PClause -> String
  sigClauses cs = concatMap (\c => sigClause c ++ "|") cs

  sigClause : PClause -> String
  sigClause (MkPatClause _ lhs rhs wb) =
    "PC(" ++ show lhs ++ "=" ++ show rhs ++ "){" ++ sigDecls wb ++ "}"
  sigClause (MkWithClause _ lhs _ _ cs) =
    "WC(" ++ show lhs ++ "){" ++ sigClauses cs ++ "}"
  sigClause (MkImpossible _ lhs) = "IMP(" ++ show lhs ++ ")"

  sigField : PField -> String
  sigField pf = concatMap (show . val) pf.names ++ ":" ++ show pf.val

  sigData : PDataDecl -> String
  sigData (MkPData _ n _ _ cons) = "data " ++ show n ++ "=" ++ show cons
  sigData (MkPLater _ n ty)      = "dataLater " ++ show n ++ ":" ++ show ty

  sigRecord : PRecordDecl' Name -> String
  sigRecord (MkPRecord n _ _ _ flds) =
    "rec " ++ show n ++ "{" ++ concatMap sigField flds ++ "}"
  sigRecord (MkPRecordLater n _) = "recLater " ++ show n

  ||| Deep signature of one declaration. Constructors with hidden nested decls /
  ||| clauses are descended explicitly; the rest fall back to `Show PDeclNoFC`
  ||| (`PClaim` is already fully deep there; `PFixity`/`PDirective`/`PBuiltin`
  ||| are single-line and reindent-inert).
  sigDeclNoFC : PDeclNoFC -> String
  sigDeclNoFC (PDef cls)                    = "PDef{" ++ sigClauses cls ++ "}"
  sigDeclNoFC (PData _ _ _ dd)              = "PData{" ++ sigData dd ++ "}"
  sigDeclNoFC (PParameters _ ds)            = "PParameters{" ++ sigDecls ds ++ "}"
  sigDeclNoFC (PUsing _ ds)                 = "PUsing{" ++ sigDecls ds ++ "}"
  sigDeclNoFC (PInterface _ _ n _ _ _ _ ds) =
    "PInterface " ++ show n ++ "{" ++ sigDecls ds ++ "}"
  sigDeclNoFC (PImplementation _ _ _ _ _ n _ _ _ mds) =
    "PImpl " ++ show n ++ "{" ++ maybe "" sigDecls mds ++ "}"
  sigDeclNoFC (PRecord _ _ _ rd) = "PRecord{" ++ sigRecord rd ++ "}"
  sigDeclNoFC (PFail _ ds)       = "PFail{" ++ sigDecls ds ++ "}"
  sigDeclNoFC (PMutual ds)       = "PMutual{" ++ sigDecls ds ++ "}"
  sigDeclNoFC (PNamespace _ ds)  = "PNamespace{" ++ sigDecls ds ++ "}"
  sigDeclNoFC (PTransform s a b) = "PTransform " ++ s ++ " " ++ show a ++ " " ++ show b
  sigDeclNoFC (PRunElabDecl t)   = "PRunElabDecl " ++ show t
  sigDeclNoFC d                  = show d

||| Deep, FC-insensitive structural signature of the top-level declarations —
||| the reindentation oracle. Unlike `astSig` it descends `where` blocks and
||| every nested-declaration block, so it detects a layout change that silently
||| re-parents a clause/method/field.
export
deepSig : String -> Maybe (List String)
deepSig src = (\m => map (\d => sigDeclNoFC d.val) m.decls) <$> parseModule src

||| Safety gate for the reindentation pass, which changes only leading
||| whitespace. `codeSig` is trivially preserved by a whitespace-only edit so it
||| is *not* a sufficient oracle here; `deepSig` is. Holds when the output still
||| parses, its deep declaration signature is unchanged, and its imports are
||| untouched.
export
safeReindent : (original : String) -> (formatted : String) -> Bool
safeReindent original formatted =
  parses formatted &&
  deepSig original == deepSig formatted &&
  importSig original == importSig formatted
