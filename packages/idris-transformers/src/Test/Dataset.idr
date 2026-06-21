||| Unit tests for `Transformers.Dataset` — the TSV loader + padding + 2D
||| attention-mask builder used by the SST-2 BERT-classify fine-tune
||| (RT3).
|||
||| Fixture: `test-fixtures/sst2-mini/mini.tsv` carries 3 hand-curated
||| BERT WordPiece-tokenized lines. The downloader emits the same
||| format on a real `glue sst2 train` pull; the fixture keeps the
||| test self-contained (no network, no Python subprocess).
module Test.Dataset

import Data.List
import Data.Vect

import Test.Harness
import Transformers.Dataset

-- The fixture path resolves relative to where the test binary runs
-- (`make test-unit-idris-transformers` invokes `./packages/idris-transformers/build/exec/idris-transformers-test`,
-- so the cwd at run time is the repo root — same as how
-- save_oracle.py-driven gates resolve their paths).
fixturePath : String
fixturePath = "packages/idris-transformers/test-fixtures/sst2-mini/mini.tsv"

-- Expected loaded examples (matches mini.tsv line-for-line).
expected : List (Nat, List Nat)
expected =
  [ (1, [101, 1037, 2204, 3185, 102])
  , (0, [101, 1037, 2919, 3185, 102, 4083])
  , (1, [101, 1996, 3185, 2003, 5875, 102])
  ]

testLoadCount : IO Bool
testLoadCount = do
  xs <- loadHfDataset fixturePath
  let n = length xs
  if n == 3
    then check ("loadHfDataset returned 3 examples (got " ++ show n ++ ")") True
    else do
      putStrLn ("  FAIL: expected 3 examples, got " ++ show n)
      pure False

testLoadValues : IO Bool
testLoadValues = do
  xs <- loadHfDataset fixturePath
  let got : List (Nat, List Nat)
      got = map (\ex => (ex.label, ex.tokenIds)) xs
  if got == expected
    then check "loadHfDataset preserves labels + tokenIds exactly" True
    else do
      putStrLn "  FAIL: parsed examples don't match the fixture"
      putStrLn ("    got:      " ++ show got)
      putStrLn ("    expected: " ++ show expected)
      pure False

-- Pads to seqLen=8 with padId=0. The first example has 5 tokens, so
-- 3 padding slots get the padId; the mask is [1,1,1,1,1,0,0,0].
testPadPadsShortExample : IO Bool
testPadPadsShortExample = do
  let ex                           = MkTokenizedExample [101, 1037, 2204, 3185, 102] 1
  let (ids, mask, lbl)             = padToSeqLen 8 0 ex
  let idsExpected  : Vect 8 Nat    = [101, 1037, 2204, 3185, 102, 0, 0, 0]
  let maskExpected : Vect 8 Double = [1.0, 1.0, 1.0, 1.0, 1.0, 0.0, 0.0, 0.0]
  if ids == idsExpected && mask == maskExpected && lbl == 1
    then check "padToSeqLen pads with padId + correct mask + preserved label" True
    else do
      putStrLn "  FAIL: padding mismatch"
      putStrLn ("    ids got:       " ++ show ids)
      putStrLn ("    mask got:      " ++ show mask)
      pure False

-- Truncates a longer example to exactly seqLen. The 6-token example
-- truncated to 4 drops the tail; mask is all-1.
testPadTruncatesLongExample : IO Bool
testPadTruncatesLongExample = do
  let ex                           = MkTokenizedExample [101, 1037, 2919, 3185, 102, 4083] 0
  let (ids, mask, lbl)             = padToSeqLen 4 0 ex
  let idsExpected  : Vect 4 Nat    = [101, 1037, 2919, 3185]
  let maskExpected : Vect 4 Double = [1.0, 1.0, 1.0, 1.0]
  if ids == idsExpected && mask == maskExpected && lbl == 0
    then check "padToSeqLen truncates long sequence + mask stays all-1s" True
    else do
      putStrLn "  FAIL: truncation mismatch"
      putStrLn ("    ids got:       " ++ show ids)
      putStrLn ("    mask got:      " ++ show mask)
      pure False

-- A 1D padding mask `[1, 1, 1, 0, 0]` (positions 0..2 real, 3..4 pad)
-- becomes a 5×5 attention-mask matrix. Each ROW is identical to the
-- inverted 1D mask: real positions get `0.0` (no mask), padding gets
-- `1.0` (masked out). Total flat length = 25.
testToAttentionMask2d : IO Bool
testToAttentionMask2d = do
  let pos : Vect 5 Double = [1.0, 1.0, 1.0, 0.0, 0.0]
  let flat                = toAttentionMask2d {seqLen=5} pos
  -- Every row is [0, 0, 0, 1, 1] (real, real, real, pad, pad inverted).
  let row : Vect 5 Double           = [0.0, 0.0, 0.0, 1.0, 1.0]
  let expectedFlat : Vect 25 Double = row ++ row ++ row ++ row ++ row
  if flat == expectedFlat
    then check "toAttentionMask2d builds 5x5 mask with padding columns = 1.0" True
    else do
      putStrLn "  FAIL: 2D mask construction mismatch"
      putStrLn ("    got:      " ++ show flat)
      putStrLn ("    expected: " ++ show expectedFlat)
      pure False

export
suite : List (String, List (IO Bool))
suite =
  [ ("Transformers.Dataset TSV loader",
     [ testLoadCount
     , testLoadValues
     ])
  , ("Transformers.Dataset padding + mask",
     [ testPadPadsShortExample
     , testPadTruncatesLongExample
     , testToAttentionMask2d
     ])
  ]
