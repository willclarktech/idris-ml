||| Shared test helpers across `Test.Hf{Bert,Gpt2,Llama,BitNet}`.
||| Runtime adapter helpers live in `Transformers.Common.idr`; this is the
||| test-side counterpart so test bodies can stay one-liners.
module Test.Common

import Data.List

import Test.Harness

||| Standard param-count assertion. Mirrors the per-adapter
||| `testParamCount` shape exactly — every adapter test had this same
||| 5-line block with only `label` and `expected` differing.
|||
|||   pass : "<label> length = <expected> (got <got>)"
|||   fail : "<label> length = <expected> (got <got>)"
export
assertHfModelParamCount : (label    : String)
                       -> (names    : List String)
                       -> (expected : Nat)
                       -> IO Bool
assertHfModelParamCount label names expected =
  let got = length names
  in check (label ++ " length = " ++ show expected ++ " (got " ++ show got ++ ")")
           (got == expected)
