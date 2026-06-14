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
  Left _ => Nothing
  Right res => Just (map tokShow (snd res))

||| Does the string parse as a module with the compiler's own parser?
export
parses : String -> Bool
parses src =
  let origin = Virtual Interactive in
  case runParser origin Nothing src (prog origin) of
    Left _ => False
    Right _ => True

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
     _ => False)
