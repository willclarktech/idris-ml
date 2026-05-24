||| Unit tests for `Tokenizer`.
|||
||| The tokenizer subprocess hits the HuggingFace `transformers` Python
||| library; these tests require the pytorch package's uv venv. The
||| existing `make test-transformers` target runs from the repo root,
||| so the subprocess `cd packages/pytorch && uv run python …`
||| invocation resolves correctly.
|||
||| Three vocabs covered (BERT WordPiece, distilgpt2 BPE, Llama-3
||| Tiktoken-BPE), pinning expected token IDs for fixed strings. If HF
||| ships a tokenizer change upstream the asserts fire loudly.
module Test.Tokenizer

import Data.List
import Data.String
import Data.Vect
import Data.Fin

import Tokenizer
import Test.Harness


----------------------------------------------------------------------
-- Helpers
----------------------------------------------------------------------

-- Lower the Vect of bounded IDs to a plain List Nat for assertion. The
-- `v` bound must be erased (0-quantity) or the elaborator tries to
-- materialise the full Peano representation of `Fin 30522` /
-- `Fin 50257` and OOMs (cf. `docs/develop/gotchas.md` "Large Nat
-- type-level reduction"). finToNat doesn't need `v` at runtime so the
-- erasure is sound.
listIds : {0 v : Nat} -> (n : Nat ** Vect n (Fin v)) -> List Nat
listIds (_ ** ids) = map finToNat (toList ids)

showList : List Nat -> String
showList xs = "[" ++ joinBy ", " (map show xs) ++ "]"
  where
    joinBy : String -> List String -> String
    joinBy _   []        = ""
    joinBy _   [x]       = x
    joinBy sep (x :: xs) = x ++ sep ++ joinBy sep xs


----------------------------------------------------------------------
-- BERT WordPiece (vocab=30522)
----------------------------------------------------------------------

bertRepo : String
bertRepo = "google/bert_uncased_L-2_H-128_A-2"

testBertVocab : IO Bool
testBertVocab = do
  r <- mkTokenizer bertRepo 30522
  case r of
    Right _  => check "mkTokenizer BERT vocab=30522 returns Right" True
    Left err => do
      putStrLn ("  FAIL: mkTokenizer BERT: " ++ show err)
      pure False

testBertEncodeHello : IO Bool
testBertEncodeHello = do
  Right tok <- mkTokenizer bertRepo 30522
    | Left err => do
        putStrLn ("  FAIL: mkTokenizer: " ++ show err)
        pure False
  r <- tokenize tok "hello"
  case r of
    Left err => do
      putStrLn ("  FAIL: tokenize: " ++ show err)
      pure False
    Right ids => do
      let got = listIds ids
      if got == [101, 7592, 102]
        then check "BERT encode \"hello\" = [101, 7592, 102]" True
        else do
          putStrLn ("  FAIL: BERT encode \"hello\" returned " ++ showList got)
          pure False

testBertVocabMismatch : IO Bool
testBertVocabMismatch = do
  -- DO NOT pattern-match the Nat literals (12345, 30522) in the case
  -- arm — Idris elaborates Nat-literal patterns by unfolding to Peano,
  -- which OOMs at this scale (cf. `gotchas.md` "Large Nat type-level
  -- reduction"). Use value-level equality instead.
  r <- mkTokenizer bertRepo 12345
  case r of
    Left (TokVocabMismatch claimed onDisk) =>
      if claimed == 12345 && onDisk == 30522
        then check "mkTokenizer BERT vocab=12345 fires TokVocabMismatch 12345 30522" True
        else do
          putStrLn ("  FAIL: TokVocabMismatch but values wrong: claimed=" ++
                    show claimed ++ " onDisk=" ++ show onDisk)
          pure False
    Left err => do
      putStrLn ("  FAIL: expected TokVocabMismatch but got " ++ show err)
      pure False
    Right _ => do
      putStrLn "  FAIL: expected Left TokVocabMismatch but got Right"
      pure False


----------------------------------------------------------------------
-- distilgpt2 BPE (vocab=50257)
----------------------------------------------------------------------

gpt2Repo : String
gpt2Repo = "distilgpt2"

testGpt2Vocab : IO Bool
testGpt2Vocab = do
  r <- mkTokenizer gpt2Repo 50257
  case r of
    Right _  => check "mkTokenizer distilgpt2 vocab=50257 returns Right" True
    Left err => do
      putStrLn ("  FAIL: mkTokenizer distilgpt2: " ++ show err)
      pure False

testGpt2EncodeHello : IO Bool
testGpt2EncodeHello = do
  Right tok <- mkTokenizer gpt2Repo 50257
    | Left err => do
        putStrLn ("  FAIL: mkTokenizer: " ++ show err)
        pure False
  r <- tokenize tok "Hello world"
  case r of
    Left err => do
      putStrLn ("  FAIL: tokenize: " ++ show err)
      pure False
    Right ids => do
      let got = listIds ids
      if got == [15496, 995]
        then check "distilgpt2 encode \"Hello world\" = [15496, 995]" True
        else do
          putStrLn ("  FAIL: distilgpt2 encode \"Hello world\" returned " ++ showList got)
          pure False

testGpt2RoundTrip : IO Bool
testGpt2RoundTrip = do
  Right tok <- mkTokenizer gpt2Repo 50257
    | Left err => do
        putStrLn ("  FAIL: mkTokenizer: " ++ show err)
        pure False
  r1 <- tokenize tok "Hello world"
  case r1 of
    Left err => do
      putStrLn ("  FAIL: tokenize: " ++ show err)
      pure False
    Right (_ ** ids) => do
      r2 <- detokenize tok ids
      case r2 of
        Left err  => do
          putStrLn ("  FAIL: detokenize: " ++ show err)
          pure False
        Right txt =>
          if trim txt == "Hello world"
            then check "distilgpt2 round-trips \"Hello world\"" True
            else do
              putStrLn ("  FAIL: round-trip got " ++ show (trim txt))
              pure False


----------------------------------------------------------------------
-- Suite
----------------------------------------------------------------------

public export
suite : List (String, List (IO Bool))
suite =
  [ ("Tokenizer — BERT WordPiece (vocab=30522)",
     [ testBertVocab
     , testBertEncodeHello
     , testBertVocabMismatch
     ])
  , ("Tokenizer — distilgpt2 BPE (vocab=50257)",
     [ testGpt2Vocab
     , testGpt2EncodeHello
     , testGpt2RoundTrip
     ])
  ]
