||| Internal construction ABI — host buffer prims + the device × dtype
||| create dispatch. For backend authors (bring-your-own-backend) and
||| Tensor.idr's own smart constructors; user code constructs tensors
||| through the typed surface (`tensor` / `param` × `InitSpec`,
||| `bulkToTensor*`, `tparam*`) instead. Re-exported via
||| `import public` from Tensor.idr, so `import Tensor` keeps every
||| existing call site compiling.
module Ml.Tensor.Internal

import Ml.Executor

----------------------------------------------------------------------
-- C-side allocation + bulk-load helpers
----------------------------------------------------------------------

%foreign "C:tensor_alloc_doubles,libidrisml"
export prim__allocDoubles : Int -> AnyPtr

-- Wrapper that returns the buffer pointer for threading through let chains
%foreign "C:tensor_write_double_return,libidrisml"
export
prim__setDouble : AnyPtr -> Int -> Double -> AnyPtr

%foreign "C:tensor_alloc_ints,libidrisml"
export
prim__allocInts : Int -> AnyPtr

%foreign "C:tensor_write_int_return,libidrisml"
export
prim__setInt : AnyPtr -> Int -> Int -> AnyPtr

-- Byte buffer (#411 B2). Used for the packed-ternary byte buffer that
-- feeds `tensor_create_ternary_packed_2d`. `prim__setByte` takes
-- Idris-level Int (no Bits8 FFI type); shared_utils.c narrows to uint8.
%foreign "C:tensor_alloc_bytes,libidrisml"
export
prim__allocBytes : Int -> AnyPtr

%foreign "C:tensor_write_byte_return,libidrisml"
export
prim__setByte : AnyPtr -> Int -> Int -> AnyPtr

----------------------------------------------------------------------
-- dtCreate* — device × dtype create dispatch
----------------------------------------------------------------------

-- `ex` selects the backend (via the `primCreate*Streamed` method),
-- `t` selects the dtype (via `dtypeTag`). Both implicits are pinned
-- at the call site: `{ex}` from the enclosing device context, `{t=dt}`
-- by the caller. Signatures match the former RuntimeDType methods
-- (trailing `streamTag : Int`) so existing call sites only gain `{ex}`.

public export
dtCreateScalar : {0 ex : Executor} -> {0 t : Type} -> Backend ex t =>
                 Double -> Int -> Int -> AnyPtr
dtCreateScalar v rg stream = primCreateScalarStreamed {ex} v rg stream (dtypeTag {t})

public export
dtCreate : {0 ex : Executor} -> {0 t : Type} -> Backend ex t =>
           AnyPtr -> AnyPtr -> Int -> Int -> Int -> AnyPtr
dtCreate dat sh r rg stream = primCreateStreamed {ex} dat sh r rg stream (dtypeTag {t})

public export
dtCreate1d : {0 ex : Executor} -> {0 t : Type} -> Backend ex t =>
             Int -> AnyPtr -> Int -> Int -> AnyPtr
dtCreate1d n dat rg stream = primCreate1dStreamed {ex} n dat rg stream (dtypeTag {t})

public export
dtCreate2d : {0 ex : Executor} -> {0 t : Type} -> Backend ex t =>
             Int -> Int -> AnyPtr -> Int -> Int -> AnyPtr
dtCreate2d r c dat rg stream = primCreate2dStreamed {ex} r c dat rg stream (dtypeTag {t})

public export
dtCreateParam1d : {0 ex : Executor} -> {0 t : Type} -> Backend ex t =>
                  Int -> AnyPtr -> Int -> AnyPtr
dtCreateParam1d n dat stream = primCreateParam1dStreamed {ex} n dat stream (dtypeTag {t})

public export
dtCreateParam2d : {0 ex : Executor} -> {0 t : Type} -> Backend ex t =>
                  Int -> Int -> AnyPtr -> Int -> AnyPtr
dtCreateParam2d r c dat stream = primCreateParam2dStreamed {ex} r c dat stream (dtypeTag {t})

public export
dtCreateParam3d : {0 ex : Executor} -> {0 t : Type} -> Backend ex t =>
                  Int -> Int -> Int -> AnyPtr -> Int -> AnyPtr
dtCreateParam3d a b c dat stream = primCreateParam3dStreamed {ex} a b c dat stream (dtypeTag {t})

public export
dtCreateParam4d : {0 ex : Executor} -> {0 t : Type} -> Backend ex t =>
                  Int -> Int -> Int -> Int -> AnyPtr -> Int -> AnyPtr
dtCreateParam4d a b c e dat stream = primCreateParam4dStreamed {ex} a b c e dat stream (dtypeTag {t})

-- Fused param create + in-place init. Each
-- `dtCreateParam<rank>{Normal,Const}` dispatches to the backend's
-- `primCreateParam<rank><Init>Streamed` instance method, threading
-- the dtypeTag from `RuntimeDType t`. Replaces the per-element
-- Idris-side sampler + per-element `prim__setDouble` FFI in callers
-- (the Transformers.{Bert,Gpt2,Llama} smart constructors + the core
-- Nn.{Linear,RmsNorm,SwiGLU,Embedding}). The actual init runs in
-- the C backend (libtorch's `torch::nn::init::normal_` or
-- `t.fill_`), at memory-bandwidth speed.
public export
dtCreateParam1dNormal : {0 ex : Executor} -> {0 t : Type} -> Backend ex t =>
                        Int -> Double -> Double -> Int -> AnyPtr
dtCreateParam1dNormal n mean std stream = primCreateParam1dNormalStreamed {ex} n mean std stream (dtypeTag {t})

public export
dtCreateParam2dNormal : {0 ex : Executor} -> {0 t : Type} -> Backend ex t =>
                        Int -> Int -> Double -> Double -> Int -> AnyPtr
dtCreateParam2dNormal r c mean std stream = primCreateParam2dNormalStreamed {ex} r c mean std stream (dtypeTag {t})

public export
dtCreateParam3dNormal : {0 ex : Executor} -> {0 t : Type} -> Backend ex t =>
                        Int -> Int -> Int -> Double -> Double -> Int -> AnyPtr
dtCreateParam3dNormal a b c mean std stream = primCreateParam3dNormalStreamed {ex} a b c mean std stream (dtypeTag {t})

public export
dtCreateParam4dNormal : {0 ex : Executor} -> {0 t : Type} -> Backend ex t =>
                        Int -> Int -> Int -> Int -> Double -> Double -> Int -> AnyPtr
dtCreateParam4dNormal a b c e mean std stream = primCreateParam4dNormalStreamed {ex} a b c e mean std stream (dtypeTag {t})

public export
dtCreateParam1dConst : {0 ex : Executor} -> {0 t : Type} -> Backend ex t =>
                       Int -> Double -> Int -> AnyPtr
dtCreateParam1dConst n value stream = primCreateParam1dConstStreamed {ex} n value stream (dtypeTag {t})

public export
dtCreateParam2dConst : {0 ex : Executor} -> {0 t : Type} -> Backend ex t =>
                       Int -> Int -> Double -> Int -> AnyPtr
dtCreateParam2dConst r c value stream = primCreateParam2dConstStreamed {ex} r c value stream (dtypeTag {t})

public export
dtCreateParam3dConst : {0 ex : Executor} -> {0 t : Type} -> Backend ex t =>
                       Int -> Int -> Int -> Double -> Int -> AnyPtr
dtCreateParam3dConst a b c value stream = primCreateParam3dConstStreamed {ex} a b c value stream (dtypeTag {t})

public export
dtCreateParam4dConst : {0 ex : Executor} -> {0 t : Type} -> Backend ex t =>
                       Int -> Int -> Int -> Int -> Double -> Int -> AnyPtr
dtCreateParam4dConst a b c e value stream = primCreateParam4dConstStreamed {ex} a b c e value stream (dtypeTag {t})

public export
dtCreateState1d : {0 ex : Executor} -> {0 t : Type} -> Backend ex t =>
                  Int -> AnyPtr -> Int -> AnyPtr
dtCreateState1d n dat stream = primCreateState1dStreamed {ex} n dat stream (dtypeTag {t})

public export
dtCreateState2d : {0 ex : Executor} -> {0 t : Type} -> Backend ex t =>
                  Int -> Int -> AnyPtr -> Int -> AnyPtr
dtCreateState2d r c dat stream = primCreateState2dStreamed {ex} r c dat stream (dtypeTag {t})

public export
dtCastFrom : {0 ex : Executor} -> {0 t : Type} -> Backend ex t =>
             AnyPtr -> Int -> AnyPtr
dtCastFrom tns stream = primCastStreamed {ex} tns stream (dtypeTag {t})
