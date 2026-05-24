||| Shared helpers for the HF inference example wrappers.
|||
||| HfLlamaInference and HfBitNetInference each carry the same small
||| pile of utility functions for argv parsing, fixed-prompt tensor
||| construction, per-row dumps, argmax, stage timing. None of these
||| depend on the per-adapter model type or forward function, so they
||| live here and the per-arch example files import this module.
|||
||| The bigger `genOneStep` / `genLoop` / `runGenerate` helpers stay
||| per-adapter because they vary on the model state type, the forward
||| function, withNoGrad bracketing policy, and per-arch banner text.
||| If a third adapter wants greedy decode in the future, *that's* the
||| signal to lift them with a parametric closure API.
|||
||| All values use the `ExampleDevice` / `ExampleDType` pair from
||| `BuildConfig` so the helpers pick up the per-build target lane
||| automatically (e.g. F32 on torch-mps / mlx-gpu, F64 on tape).
module Example.HfInferenceHelper

import Data.String
import Data.Vect
import System.Clock
import System.File

import Array
import BuildConfig
import Device
import Tensor
import Util


----------------------------------------------------------------------
-- Token-id tensor construction
----------------------------------------------------------------------

||| Build a `[n] Double` tensor from a Vect of token IDs (already cast
||| to Double — the autograd surface is float-typed).
public export
mkIds : {n : Nat} -> Vect n Double
     -> Tensor [n] ExampleDevice ExampleDType WithGrad
mkIds xs =
  let raw = bulkToTensor {d=ExampleDevice} {dt=ExampleDType}
                         (VArray (map SArray xs))
  in tinput1d {n} raw

||| Lift a List to a length-indexed Vect with an existential length.
||| Used to bridge the runtime-known token-list length to the
||| compile-time `seq` parameter of the forward function.
public export
toExistVect : (xs : List a) -> (n : Nat ** Vect n a)
toExistVect xs = (length xs ** fromList xs)


----------------------------------------------------------------------
-- Stdout dump of a [n]-shape row, one float per line
----------------------------------------------------------------------

||| Walk a row pointer end-exclusive printing each element. Used by
||| --dump-final-hidden / --dump-logits modes for the cross-language
||| CI gates (oracle comparator reads one float per line from stdout).
public export
printRow : Int -> Int -> AnyPtr -> IO ()
printRow end i p =
  if i >= end
    then pure ()
    else do
      let v = primItem1d {d=ExampleDevice} p i
      putStrLn (show v)
      printRow end (i + 1) p


----------------------------------------------------------------------
-- Argmax over a 1D row pointer
----------------------------------------------------------------------

||| Linear argmax over a [vocab]-shape row pointer. Returns the index
||| as a Nat. Used by greedy decode (`genOneStep`) to pick the next
||| token from the LM-head logits.
public export
argmaxRow : (vocab : Nat) -> AnyPtr -> IO Nat
argmaxRow vocab p = go (cast {to=Int} vocab) 0 0 (-1.0e300)
  where
    go : Int -> Int -> Int -> Double -> IO Nat
    go end i bestI bestV =
      if i >= end
        then pure (cast {to=Nat} bestI)
        else let v = primItem1d {d=ExampleDevice} p i
             in if v > bestV
                  then go end (i + 1) i v
                  else go end (i + 1) bestI bestV


----------------------------------------------------------------------
-- File dump (used by BitNet's --bisect-blocks)
----------------------------------------------------------------------

||| Collect a 1D row's contents as `show`-formatted strings between
||| `startIdx` and `end` (exclusive). Used in batch by `dumpRowToFile`
||| to avoid one syscall per element when the row is large
||| (vocab=128256 in BitNet's logits dump).
public export
collectShown : Int -> Int -> AnyPtr -> IO (List String)
collectShown end startIdx ptr = go startIdx []
  where
    go : Int -> List String -> IO (List String)
    go i acc =
      if i >= end
        then pure (reverse acc)
        else do
          let v = primItem1d {d=ExampleDevice} ptr i
          go (i + 1) (show v :: acc)

||| Write a 1D row of `nElems` floats from `ptr` to `path`, one float
||| per line. Batches the read via `collectShown` then issues one
||| `writeFile`.
public export
dumpRowToFile : String -> Int -> AnyPtr -> IO ()
dumpRowToFile path nElems ptr = do
  xs <- collectShown nElems 0 ptr
  res <- writeFile path (unlines xs)
  case res of
    Right () => pure ()
    Left  err =>
      putStrLn ("ERR: writeFile " ++ path ++ ": " ++ show err)


----------------------------------------------------------------------
-- argv parsing for --prompt / --num-tokens
----------------------------------------------------------------------

||| Extract the `--prompt <string>` argv pair; fall back to `dflt` if
||| absent.
public export
extractPrompt : (dflt : String) -> List String -> String
extractPrompt dflt args = go args
  where
    go : List String -> String
    go ("--prompt" :: p :: _) = p
    go (_ :: rest)            = go rest
    go []                     = dflt

||| Extract the `--num-tokens <N>` argv pair; fall back to `dflt` if
||| absent or unparseable.
public export
extractNumTokens : (dflt : Nat) -> List String -> Nat
extractNumTokens dflt args = go args
  where
    go : List String -> Nat
    go ("--num-tokens" :: n :: _) =
      fromMaybe dflt (parsePositive {a=Nat} n)
    go (_ :: rest)                = go rest
    go []                         = dflt


----------------------------------------------------------------------
-- Stage timer
----------------------------------------------------------------------

||| Print a "[stage] [hh:mm:ss] <label>" diagnostic line. Used to
||| break up multi-minute HF-inference runs into observable stages
||| (tokenizer probe / model construction / checkpoint load / RoPE
||| table build / forward / generate). Pairs with
||| `time_inference_{llama,bitnet}.py` which emits identical labels.
public export
stageStamp : (label : String) -> Clock Monotonic -> IO ()
stageStamp label t0 = do
  now <- clockTime Monotonic
  putStrLn ("[stage] " ++ formatElapsed t0 now ++ " " ++ label)
