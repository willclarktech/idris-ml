||| Simple tokenized text-classification dataset loader for HF
||| datasets pre-processed by `scripts/hf-download-dataset.sh`.
|||
||| File format (TAB-separated, one example per line):
|||
|||   <label>\t<comma-separated token ids>
|||
||| Choosing TSV over JSONL / parquet keeps the Idris parser to a
||| single `Data.String.split` chain — no JSON dependency. The
||| downloader script bakes the tokenization step in (so the Idris
||| side reads integer IDs directly), matching what
||| `transformers.AutoTokenizer.encode(...)` would produce at runtime
||| without paying the ~1s/call subprocess startup of `Tokenizer.idr`.
||| For datasets that ship multiple text fields (e.g. NLI premise +
||| hypothesis), extend the downloader to concatenate before tokenizing
||| — this module stays single-text on the Idris side.
module HfDataset

import Data.List
import Data.List1
import Data.String
import Data.Vect
import System.File

----------------------------------------------------------------------
-- Record + parsing
----------------------------------------------------------------------

||| One tokenized text-classification example: a variable-length list
||| of token IDs and an integer label. The downloader has already
||| inserted any special tokens ([CLS] / [SEP] for BERT-family) so
||| consumers don't need a tokenizer at training time.
public export
record TokenizedExample where
  constructor MkTokenizedExample
  tokenIds : List Nat
  label    : Nat

-- Parse a comma-separated token-id string. Empty input → []. Any
-- non-Nat chunk fails the whole parse (returns Nothing).
parseIdList : String -> Maybe (List Nat)
parseIdList s =
  case trim s of
    "" => Just []
    body =>
      let parts = forget (split (== ',') body)
          parsed : List (Maybe Nat) = map parseNat parts
      in sequence parsed
  where
    parseNat : String -> Maybe Nat
    parseNat str =
      case parseInteger {a=Integer} (trim str) of
        Nothing => Nothing
        Just n  =>
          if n < 0
            then Nothing
            else Just (cast n)

-- Parse a single TSV line. Returns Nothing on malformed input
-- (missing tab, non-Nat label, bad token list).
parseLine : String -> Maybe TokenizedExample
parseLine line =
  case forget (split (== '\t') line) of
    [labelStr, idsStr] =>
      case (parseInteger {a=Integer} (trim labelStr), parseIdList idsStr) of
        (Just lbl, Just ids) =>
          if lbl < 0 then Nothing else Just (MkTokenizedExample ids (cast lbl))
        _ => Nothing
    _ => Nothing

||| Load a tokenized TSV file emitted by `hf-download-dataset.sh`.
||| Returns the parsed examples (best-effort: malformed lines are
||| dropped silently — callers asserting against a fixed count gate
||| catch downloader regressions). Empty lines are skipped.
export
loadHfDataset : (tsvPath : String) -> IO (List TokenizedExample)
loadHfDataset path = do
  res <- readFile path
  let contents : String
      contents = either (const "") id res
  let nonEmpty = filter (not . null . trim) (lines contents)
      parsed   = mapMaybe parseLine nonEmpty
  pure parsed

----------------------------------------------------------------------
-- Padding / truncation
----------------------------------------------------------------------

-- Pad/truncate a List Nat to exactly `seqLen` entries. Pads at the
-- end with `padId`; truncates from the start? No — preserve [CLS]
-- at position 0 by truncating from the end. The mask follows the
-- same alignment (1.0 for real positions, 0.0 for padding).
takePadEnd : (seqLen : Nat) -> (padId : Nat) -> List Nat
          -> (Vect seqLen Nat, Vect seqLen Double)
takePadEnd Z      _    _        = ([], [])
takePadEnd (S k)  pid  []       =
  let (rest, restMask) = takePadEnd k pid []
  in (pid :: rest, 0.0 :: restMask)
takePadEnd (S k)  pid  (x :: xs) =
  let (rest, restMask) = takePadEnd k pid xs
  in (x :: rest, 1.0 :: restMask)

||| Pad/truncate a single example to exactly `seqLen` tokens. Returns
||| `(ids, mask, label)`:
|||   - `ids`: Vect seqLen Nat (padded with `padId`)
|||   - `mask`: Vect seqLen Double — `1.0` at real positions, `0.0`
|||     at padding (HF's standard convention)
|||   - `label`: Nat (unchanged)
|||
||| For sequences longer than `seqLen`, truncation drops the tail —
||| BERT's [CLS] at position 0 is preserved. Special-token alignment
||| is the downloader's responsibility.
export
padToSeqLen : (seqLen : Nat) -> (padId : Nat)
           -> TokenizedExample
           -> (Vect seqLen Nat, Vect seqLen Double, Nat)
padToSeqLen seqLen padId ex =
  let (ids, mask) = takePadEnd seqLen padId ex.tokenIds
  in (ids, mask, ex.label)

----------------------------------------------------------------------
-- Batching utilities
----------------------------------------------------------------------

||| Build a 2D attention-mask `[seqLen, seqLen]` from a 1D position
||| mask `[seqLen]` (HF convention: 1.0 = real, 0.0 = padding).
|||
||| Output entries `>= 0.5` are interpreted by `hfBertForward` as
||| "mask out" — so the 2D matrix should be 1.0 wherever the column
||| `j` is padding, and 0.0 elsewhere. The row index `i` doesn't
||| matter (every query sees the same padding-mask).
|||
||| Returns the row-major flat `Vect (seqLen * seqLen) Double` —
||| compatible with `tparam2d` / `bulkToTensor2d`-style consumers.
export
toAttentionMask2d : {seqLen : Nat} -> Vect seqLen Double
                 -> Vect (seqLen * seqLen) Double
toAttentionMask2d {seqLen} posMask =
  -- For row i, column j: 1.0 if posMask[j] < 0.5 (= padding), else 0.0.
  flatten (replicate seqLen invertedPosMask)
  where
    -- The inverted 1D mask, used as every row of the 2D mask matrix.
    invertedPosMask : Vect seqLen Double
    invertedPosMask = map (\v => if v < 0.5 then 1.0 else 0.0) posMask

    flatten : Vect m (Vect n a) -> Vect (m * n) a
    flatten []        = []
    flatten (r :: rs) = r ++ flatten rs
