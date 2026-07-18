||| BringYourOwn — worked example of a user-supplied backend.
|||
||| Walks through the recipe from `docs/users/grad-mode-and-device-typing.md`'s
||| "Custom devices: user-supplied backends" section: declare your
||| own device tag type, bind your dylib's C symbols via `%foreign`,
||| implement `UserExecutorCore` for the type, and `Tensor [..] MyDev`
||| is a valid type that dispatches all ops to your backend.
|||
||| This example uses the libbyo.dylib (see
||| `packages/backends/backend_byo.c`) — a 100-line stub backend
||| that logs each op call and returns a placeholder scalar. The
||| goal isn't to compute anything useful, it's to *see* the
||| dispatch fire when ops are called on `Tensor [..] BYO`.
|||
||| Build + run via `make example-bring-your-own`.
module Example.BringYourOwn

import BuildConfig
import Ml.Executor
import Ml.Executor.Core

----------------------------------------------------------------------
-- Step 1: declare your device tag type.
--
-- This is a 0-quantity phantom on Tensor. The constructor is just a
-- handle for instance resolution — `Tensor [4] BYO` is the value-less
-- type that says "this tensor lives on the BYO backend."
----------------------------------------------------------------------

public export
data BYO : Type where MkBYO : BYO

----------------------------------------------------------------------
-- Step 2: bind the C symbols your backend exports.
--
-- libbyo.dylib must be on the dynamic-link path at runtime. The
-- Makefile's `example-bring-your-own` target arranges this.
----------------------------------------------------------------------

%foreign "C:byo_tensor_create_scalar,libbyo"
prim__createScalarBYO : Double -> Int -> AnyPtr

%foreign "C:byo_tensor_free,libbyo"
prim__freeBYO : AnyPtr -> ()

%foreign "C:byo_tensor_item,libbyo"
prim__itemBYO : AnyPtr -> Double

%foreign "C:byo_tensor_item1d,libbyo"
prim__item1dBYO : AnyPtr -> Int -> Double

%foreign "C:byo_tensor_clone,libbyo"
prim__cloneBYO : AnyPtr -> AnyPtr

%foreign "C:byo_tensor_add,libbyo"
prim__addBYO : AnyPtr -> AnyPtr -> AnyPtr

%foreign "C:byo_tensor_sub,libbyo"
prim__subBYO : AnyPtr -> AnyPtr -> AnyPtr

%foreign "C:byo_tensor_mul,libbyo"
prim__mulBYO : AnyPtr -> AnyPtr -> AnyPtr

%foreign "C:byo_tensor_div,libbyo"
prim__divBYO : AnyPtr -> AnyPtr -> AnyPtr

%foreign "C:byo_tensor_neg,libbyo"
prim__negBYO : AnyPtr -> AnyPtr

%foreign "C:byo_tensor_abs,libbyo"
prim__absBYO : AnyPtr -> AnyPtr

%foreign "C:byo_tensor_exp,libbyo"
prim__expBYO : AnyPtr -> AnyPtr

%foreign "C:byo_tensor_log,libbyo"
prim__logBYO : AnyPtr -> AnyPtr

%foreign "C:byo_tensor_sqrt,libbyo"
prim__sqrtBYO : AnyPtr -> AnyPtr

%foreign "C:byo_tensor_pow,libbyo"
prim__powBYO : AnyPtr -> AnyPtr -> AnyPtr

%foreign "C:byo_tensor_sigmoid,libbyo"
prim__sigmoidBYO : AnyPtr -> AnyPtr

%foreign "C:byo_tensor_tanh,libbyo"
prim__tanhBYO : AnyPtr -> AnyPtr

%foreign "C:byo_tensor_add_scalar,libbyo"
prim__addScalarBYO : AnyPtr -> Double -> AnyPtr

%foreign "C:byo_tensor_mul_scalar,libbyo"
prim__mulScalarBYO : AnyPtr -> Double -> AnyPtr

%foreign "C:byo_tensor_clamp_min,libbyo"
prim__clampMinBYO : AnyPtr -> Double -> AnyPtr

%foreign "C:byo_tensor_clamp,libbyo"
prim__clampBYO : AnyPtr -> Double -> Double -> AnyPtr

%foreign "C:byo_tensor_round,libbyo"
prim__roundBYO : AnyPtr -> AnyPtr

----------------------------------------------------------------------
-- Step 3: implement `UserExecutorCore` for your type.
--
-- One method per op. The method body forwards to your backend's
-- corresponding `%foreign` binding. `deviceName` is the human tag
-- shown in logs.
----------------------------------------------------------------------

public export
UserExecutorCore BYO where
  deviceName       = "byo"
  primCreateScalar = prim__createScalarBYO
  primFree         = prim__freeBYO
  primItem         = prim__itemBYO
  primItem1d       = prim__item1dBYO
  primClone        = prim__cloneBYO
  primAdd          = prim__addBYO
  primSub          = prim__subBYO
  primMul          = prim__mulBYO
  primDiv          = prim__divBYO
  primNeg          = prim__negBYO
  primAbs          = prim__absBYO
  primExp          = prim__expBYO
  primLog          = prim__logBYO
  primSqrt         = prim__sqrtBYO
  primPow          = prim__powBYO
  primSigmoid      = prim__sigmoidBYO
  primTanh         = prim__tanhBYO
  primAddScalar    = prim__addScalarBYO
  primMulScalar    = prim__mulScalarBYO
  primClampMin     = prim__clampMinBYO
  primClamp        = prim__clampBYO
  primRound        = prim__roundBYO

-- A BYO author self-declares `Linked` for their device: by compiling it
-- in, it is available by definition (the generated HwConfig only
-- withholds instances for the built-in backends not in BACKEND). With
-- this instance, `BYO` tensors are spellable at the Linked-gated tensor
-- constructors too — not just the raw UserExecutorCore methods `fma` uses.
public export
Linked BYO where

----------------------------------------------------------------------
-- Step 4: use it. Any function that's generic in `UserExecutorCore ex`
-- resolves to your instance when called with `{ex=BYO}` (or with
-- `BYO`-typed Tensor arguments that drive the inference).
----------------------------------------------------------------------

||| Compute `(a + b) * c` using only `UserExecutorCore` methods.
||| Polymorphic in `ex`; works with any backend that implements the
||| interface.
fma : (0 ex : Executor) -> UserExecutorCore ex => Double -> Double -> Double -> Double
fma ex a b c =
  primItem {ex}
    (primMul {ex}
      (primAdd {ex} (primCreateScalar {ex} a 0) (primCreateScalar {ex} b 0))
      (primCreateScalar {ex} c 0))

main : IO ()
main = do
  requireMachine {m = ChosenMachine}
  putStrLn "=== BringYourOwn: dispatch demo ==="
  putStrLn ("device tag: " ++ deviceName {ex=BYO})
  putStrLn ""
  putStrLn "Computing (2 + 3) * 5 on the BYO backend — watch stderr"
  putStrLn "for the per-op log lines libbyo emits as ops fire."
  putStrLn ""
  let result = fma BYO 2.0 3.0 5.0
  putStrLn ""
  putStrLn ("(2 + 3) * 5 = " ++ show result)
  putStrLn ""
  putStrLn "Same expression on the build's primary backend (libidrisml)"
  putStrLn "for contrast — no [byo] lines, because the dispatch"
  putStrLn "goes through whatever ExampleExecutor resolves to in this build."
  putStrLn ""
  let viaPrimary = fma ExampleExecutor 2.0 3.0 5.0
  putStrLn ""
  putStrLn ("(2 + 3) * 5 on " ++ deviceName {ex=ExampleExecutor} ++ " = " ++ show viaPrimary)
