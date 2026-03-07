module Variable

import Data.List
import Data.Maybe
import Data.SortedMap
import Data.Vect
import System.Random

import Floating
import Tensor
import Util


----------------------------------------------------------------------
-- Operation Tags
----------------------------------------------------------------------

public export
data TapeOp = ConstOp
            | NegOp | AbsOp | ExpOp | LogOp | SqrtOp
            | AddOp | SubOp | MulOp | DivOp | PowOp
            | MatVecOp | DotOp
            | SoftmaxOp | LogSoftmaxOp
            | BatchCosSimOp | ReadOpOp | WriteOpOp
            | InterpWriteOp
            | SigmoidOp | TanhOp

toTag : TapeOp -> Int
toTag ConstOp      = 0
toTag NegOp        = 1
toTag AbsOp        = 2
toTag ExpOp        = 3
toTag LogOp        = 4
toTag SqrtOp       = 5
toTag AddOp        = 6
toTag SubOp        = 7
toTag MulOp        = 8
toTag DivOp        = 9
toTag PowOp        = 10
toTag MatVecOp     = 11
toTag DotOp        = 12
toTag SoftmaxOp    = 13
toTag LogSoftmaxOp = 14
toTag BatchCosSimOp = 15
toTag ReadOpOp      = 16
toTag WriteOpOp     = 17
toTag InterpWriteOp = 18
toTag SigmoidOp    = 19
toTag TanhOp       = 20

fromTag : Int -> TapeOp
fromTag 1  = NegOp
fromTag 2  = AbsOp
fromTag 3  = ExpOp
fromTag 4  = LogOp
fromTag 5  = SqrtOp
fromTag 6  = AddOp
fromTag 7  = SubOp
fromTag 8  = MulOp
fromTag 9  = DivOp
fromTag 10 = PowOp
fromTag 11 = MatVecOp
fromTag 12 = DotOp
fromTag 13 = SoftmaxOp
fromTag 14 = LogSoftmaxOp
fromTag 15 = BatchCosSimOp
fromTag 16 = ReadOpOp
fromTag 17 = WriteOpOp
fromTag 18 = InterpWriteOp
fromTag 19 = SigmoidOp
fromTag 20 = TanhOp
fromTag _  = ConstOp


----------------------------------------------------------------------
-- Tape FFI (C-backed storage, Scheme FFI wrappers)
----------------------------------------------------------------------

-- All tape storage and backward pass live in C (csrc/tensor.c).
-- The init guard loads the shared library on first call.
-- Subsequent foreign-procedure calls find symbols in the loaded library.

-- Init guard: loads build/libidrisml.dylib once.
-- Embedded in prim__tapeGen and prim__tapeAppendConst (the two entry points).
-- Other functions rely on the library being loaded by one of these.

%foreign "scheme:(lambda (dummy) (when (not (top-level-bound? 'idrisml-loaded)) (begin (load-shared-object \"build/libidrisml.dylib\") (set-top-level-value! 'idrisml-loaded #t))) ((foreign-procedure \"tape_get_gen\" (int) int) dummy))"
prim__tapeGen : Int -> Int

%foreign "scheme:(lambda (val pid) (when (not (top-level-bound? 'idrisml-loaded)) (begin (load-shared-object \"build/libidrisml.dylib\") (set-top-level-value! 'idrisml-loaded #t))) ((foreign-procedure \"tape_append_const\" (double string) int) val pid))"
prim__tapeAppendConst : Double -> String -> Int

%foreign "scheme:(lambda (tag a1 val) ((foreign-procedure \"tape_append_unary\" (int int double) int) tag a1 val))"
prim__tapeAppendUnary : Int -> Int -> Double -> Int

%foreign "scheme:(lambda (tag a1 a2 val) ((foreign-procedure \"tape_append_binary\" (int int int double) int) tag a1 a2 val))"
prim__tapeAppendBinary : Int -> Int -> Int -> Double -> Int

%foreign "scheme:(lambda (tag count meta out) ((foreign-procedure \"tape_append_tensor_op\" (int int void* void*) void*) tag count meta out))"
prim__tapeAppendTensorOp : Int -> Int -> AnyPtr -> AnyPtr -> AnyPtr

%foreign "scheme:(lambda (meta val) ((foreign-procedure \"tape_append_dot_op\" (void* double) int) meta val))"
prim__tapeAppendDotOp : AnyPtr -> Double -> Int

%foreign "scheme:(lambda (idx pid) ((foreign-procedure \"tape_set_pid\" (int string) int) idx pid))"
prim__tapeSetParamId : Int -> String -> Int

-- Gradient array allocation (C-allocated)
%foreign "scheme:(lambda (n) ((foreign-procedure \"grad_alloc\" (int) void*) n))"
prim__gradAlloc : Int -> AnyPtr

-- gradAdd: accumulate gradient at index. Returns handle for threading.
%foreign "scheme:(lambda (g idx val) ((foreign-procedure \"grad_add\" (void* int double) void*) g idx val))"
prim__gradAdd : AnyPtr -> Int -> Double -> AnyPtr

-- C-backed backward pass: walks tape in C, returns number of collected params
%foreign "scheme:(lambda (g sz) ((foreign-procedure \"walk_backward_and_reset\" (void* int) int) g sz))"
prim__walkBackwardAndReset : AnyPtr -> Int -> Int

-- Access collected (pid, grad) results from walk_backward
%foreign "scheme:(lambda (i) ((foreign-procedure \"result_get_pid\" (int) string) i))"
prim__resultGetPid : Int -> String

%foreign "scheme:(lambda (i) ((foreign-procedure \"result_get_val\" (int) double) i))"
prim__resultGetVal : Int -> Double


----------------------------------------------------------------------
-- Weight Buffer FFI (C-backed)
----------------------------------------------------------------------

export
%foreign "scheme:(lambda (n) ((foreign-procedure \"weight_buf_alloc\" (int) void*) n))"
prim__weightBufAlloc : Int -> AnyPtr

%foreign "scheme:(lambda (wb) ((foreign-procedure \"weight_buf_vals\" (void*) void*) wb))"
prim__weightBufVals : AnyPtr -> AnyPtr

%foreign "scheme:(lambda (wb idx val) ((foreign-procedure \"weight_buf_set_val\" (void* int double) void*) wb idx val))"
prim__weightBufSetVal : AnyPtr -> Int -> Double -> AnyPtr

%foreign "scheme:(lambda (wb idx pid) ((foreign-procedure \"weight_buf_set_pid\" (void* int string) void*) wb idx pid))"
prim__weightBufSetPid : AnyPtr -> Int -> String -> AnyPtr

%foreign "scheme:(lambda (wb) ((foreign-procedure \"weight_buf_ensure\" (void*) int) wb))"
prim__tapeEnsureBulkConst : AnyPtr -> Int


----------------------------------------------------------------------
-- Tensor FFI
----------------------------------------------------------------------

-- Buffer allocation (calloc'd, must be freed)
%foreign "scheme:(lambda (n) ((foreign-procedure \"tensor_alloc\" (int) void*) n))"
prim__tensorAlloc : Int -> AnyPtr

%foreign "scheme:(lambda (ptr) (begin ((foreign-procedure \"tensor_free\" (void*) void) ptr) 0))"
prim__tensorFree : AnyPtr -> Int

%foreign "scheme:(lambda (ptr idx) ((foreign-procedure \"tensor_read\" (void* int) double) ptr idx))"
prim__tensorRead : AnyPtr -> Int -> Double

-- Force evaluation of first arg, return second. Chez Scheme evaluates all
-- function arguments strictly, so this creates an ordering dependency.
%foreign "scheme:(lambda (a b) b)"
prim__seq : AnyPtr -> AnyPtr -> AnyPtr

-- Scheme-native memory writes (no C FFI crossing per element)
-- foreign-set! 'double writes 8 bytes; 'integer-32 writes 4 bytes.
-- Returns the pointer for threading.
%foreign "scheme:(lambda (ptr idx val) (begin (foreign-set! 'double ptr (* idx 8) val) ptr))"
prim__setDouble : AnyPtr -> Int -> Double -> AnyPtr

%foreign "scheme:(lambda (ptr idx val) (begin (foreign-set! 'integer-32 ptr (* idx 4) val) ptr))"
prim__setInt32 : AnyPtr -> Int -> Int -> AnyPtr

-- MatVec meta: alloc, get internal pointers, compute, backward
%foreign "scheme:(lambda (m n) ((foreign-procedure \"matvec_meta_alloc\" (int int) void*) m n))"
prim__matvecMetaAlloc : Int -> Int -> AnyPtr

-- Allocate meta for persistent buffer path (no arena w_vals/w_tape copy)
%foreign "scheme:(lambda (m n wptr wstart) ((foreign-procedure \"matvec_meta_alloc_buf\" (int int void* int) void*) m n wptr wstart))"
prim__matvecMetaAllocBuf : Int -> Int -> AnyPtr -> Int -> AnyPtr

-- Raw array accessors (one C call each, cached for bulk writes)
%foreign "scheme:(lambda (meta) ((foreign-procedure \"matvec_meta_w_vals\" (void*) void*) meta))"
prim__matvecWVals : AnyPtr -> AnyPtr
%foreign "scheme:(lambda (meta) ((foreign-procedure \"matvec_meta_w_tape\" (void*) void*) meta))"
prim__matvecWTape : AnyPtr -> AnyPtr
%foreign "scheme:(lambda (meta) ((foreign-procedure \"matvec_meta_x_vals\" (void*) void*) meta))"
prim__matvecXVals : AnyPtr -> AnyPtr
%foreign "scheme:(lambda (meta) ((foreign-procedure \"matvec_meta_x_tape\" (void*) void*) meta))"
prim__matvecXTape : AnyPtr -> AnyPtr

%foreign "scheme:(lambda (meta out) ((foreign-procedure \"matvec_meta_compute\" (void* void*) void*) meta out))"
prim__matvecCompute : AnyPtr -> AnyPtr -> AnyPtr

-- Dot meta: alloc, get internal pointers, compute
%foreign "scheme:(lambda (n) ((foreign-procedure \"dot_meta_alloc\" (int) void*) n))"
prim__dotMetaAlloc : Int -> AnyPtr

%foreign "scheme:(lambda (meta) ((foreign-procedure \"dot_meta_a_vals\" (void*) void*) meta))"
prim__dotAVals : AnyPtr -> AnyPtr
%foreign "scheme:(lambda (meta) ((foreign-procedure \"dot_meta_a_tape\" (void*) void*) meta))"
prim__dotATape : AnyPtr -> AnyPtr
%foreign "scheme:(lambda (meta) ((foreign-procedure \"dot_meta_b_vals\" (void*) void*) meta))"
prim__dotBVals : AnyPtr -> AnyPtr
%foreign "scheme:(lambda (meta) ((foreign-procedure \"dot_meta_b_tape\" (void*) void*) meta))"
prim__dotBTape : AnyPtr -> AnyPtr

%foreign "scheme:(lambda (meta) ((foreign-procedure \"dot_meta_compute\" (void*) double) meta))"
prim__dotCompute : AnyPtr -> Double

-- Softmax/LogSoftmax meta: alloc, get internal pointers, compute
%foreign "scheme:(lambda (n) ((foreign-procedure \"softmax_meta_alloc\" (int) void*) n))"
prim__softmaxMetaAlloc : Int -> AnyPtr

%foreign "scheme:(lambda (meta) ((foreign-procedure \"softmax_meta_x_vals\" (void*) void*) meta))"
prim__softmaxXVals : AnyPtr -> AnyPtr
%foreign "scheme:(lambda (meta) ((foreign-procedure \"softmax_meta_x_tape\" (void*) void*) meta))"
prim__softmaxXTape : AnyPtr -> AnyPtr

%foreign "scheme:(lambda (meta out) ((foreign-procedure \"softmax_meta_compute\" (void* void*) void*) meta out))"
prim__softmaxCompute : AnyPtr -> AnyPtr -> AnyPtr

%foreign "scheme:(lambda (meta out) ((foreign-procedure \"logsoftmax_meta_compute\" (void* void*) void*) meta out))"
prim__logsoftmaxCompute : AnyPtr -> AnyPtr -> AnyPtr

-- BatchCosSim meta: alloc, get internal pointers, set beta, compute
%foreign "scheme:(lambda (n w) ((foreign-procedure \"batch_cossim_meta_alloc\" (int int) void*) n w))"
prim__batchCosSimMetaAlloc : Int -> Int -> AnyPtr

%foreign "scheme:(lambda (meta) ((foreign-procedure \"batch_cossim_meta_mem_vals\" (void*) void*) meta))"
prim__batchCosSimMemVals : AnyPtr -> AnyPtr
%foreign "scheme:(lambda (meta) ((foreign-procedure \"batch_cossim_meta_mem_tape\" (void*) void*) meta))"
prim__batchCosSimMemTape : AnyPtr -> AnyPtr
%foreign "scheme:(lambda (meta) ((foreign-procedure \"batch_cossim_meta_key_vals\" (void*) void*) meta))"
prim__batchCosSimKeyVals : AnyPtr -> AnyPtr
%foreign "scheme:(lambda (meta) ((foreign-procedure \"batch_cossim_meta_key_tape\" (void*) void*) meta))"
prim__batchCosSimKeyTape : AnyPtr -> AnyPtr

%foreign "scheme:(lambda (meta val tidx) ((foreign-procedure \"batch_cossim_meta_set_beta\" (void* double int) void*) meta val tidx))"
prim__batchCosSimSetBeta : AnyPtr -> Double -> Int -> AnyPtr

%foreign "scheme:(lambda (meta out) ((foreign-procedure \"batch_cossim_compute\" (void* void*) void*) meta out))"
prim__batchCosSimCompute : AnyPtr -> AnyPtr -> AnyPtr

-- ReadOp meta: alloc, get internal pointers, compute
%foreign "scheme:(lambda (n w) ((foreign-procedure \"readop_meta_alloc\" (int int) void*) n w))"
prim__readOpMetaAlloc : Int -> Int -> AnyPtr

%foreign "scheme:(lambda (meta) ((foreign-procedure \"readop_meta_mem_vals\" (void*) void*) meta))"
prim__readOpMemVals : AnyPtr -> AnyPtr
%foreign "scheme:(lambda (meta) ((foreign-procedure \"readop_meta_mem_tape\" (void*) void*) meta))"
prim__readOpMemTape : AnyPtr -> AnyPtr
%foreign "scheme:(lambda (meta) ((foreign-procedure \"readop_meta_weight_vals\" (void*) void*) meta))"
prim__readOpWeightVals : AnyPtr -> AnyPtr
%foreign "scheme:(lambda (meta) ((foreign-procedure \"readop_meta_weight_tape\" (void*) void*) meta))"
prim__readOpWeightTape : AnyPtr -> AnyPtr

%foreign "scheme:(lambda (meta out) ((foreign-procedure \"readop_compute\" (void* void*) void*) meta out))"
prim__readOpCompute : AnyPtr -> AnyPtr -> AnyPtr

-- WriteOp meta: alloc, get internal pointers, compute
%foreign "scheme:(lambda (n w) ((foreign-procedure \"writeop_meta_alloc\" (int int) void*) n w))"
prim__writeOpMetaAlloc : Int -> Int -> AnyPtr

%foreign "scheme:(lambda (meta) ((foreign-procedure \"writeop_meta_mem_vals\" (void*) void*) meta))"
prim__writeOpMemVals : AnyPtr -> AnyPtr
%foreign "scheme:(lambda (meta) ((foreign-procedure \"writeop_meta_mem_tape\" (void*) void*) meta))"
prim__writeOpMemTape : AnyPtr -> AnyPtr
%foreign "scheme:(lambda (meta) ((foreign-procedure \"writeop_meta_weight_vals\" (void*) void*) meta))"
prim__writeOpWeightVals : AnyPtr -> AnyPtr
%foreign "scheme:(lambda (meta) ((foreign-procedure \"writeop_meta_weight_tape\" (void*) void*) meta))"
prim__writeOpWeightTape : AnyPtr -> AnyPtr
%foreign "scheme:(lambda (meta) ((foreign-procedure \"writeop_meta_erase_vals\" (void*) void*) meta))"
prim__writeOpEraseVals : AnyPtr -> AnyPtr
%foreign "scheme:(lambda (meta) ((foreign-procedure \"writeop_meta_erase_tape\" (void*) void*) meta))"
prim__writeOpEraseTape : AnyPtr -> AnyPtr
%foreign "scheme:(lambda (meta) ((foreign-procedure \"writeop_meta_add_vals\" (void*) void*) meta))"
prim__writeOpAddVals : AnyPtr -> AnyPtr
%foreign "scheme:(lambda (meta) ((foreign-procedure \"writeop_meta_add_tape\" (void*) void*) meta))"
prim__writeOpAddTape : AnyPtr -> AnyPtr

%foreign "scheme:(lambda (meta out) ((foreign-procedure \"writeop_compute\" (void* void*) void*) meta out))"
prim__writeOpCompute : AnyPtr -> AnyPtr -> AnyPtr


-- InterpWrite meta: alloc, get internal pointers, compute
%foreign "scheme:(lambda (n w) ((foreign-procedure \"interp_write_meta_alloc\" (int int) void*) n w))"
prim__interpWriteMetaAlloc : Int -> Int -> AnyPtr

%foreign "scheme:(lambda (meta) ((foreign-procedure \"interp_write_meta_mem_vals\" (void*) void*) meta))"
prim__interpWriteMemVals : AnyPtr -> AnyPtr
%foreign "scheme:(lambda (meta) ((foreign-procedure \"interp_write_meta_mem_tape\" (void*) void*) meta))"
prim__interpWriteMemTape : AnyPtr -> AnyPtr
%foreign "scheme:(lambda (meta) ((foreign-procedure \"interp_write_meta_weight_vals\" (void*) void*) meta))"
prim__interpWriteWeightVals : AnyPtr -> AnyPtr
%foreign "scheme:(lambda (meta) ((foreign-procedure \"interp_write_meta_weight_tape\" (void*) void*) meta))"
prim__interpWriteWeightTape : AnyPtr -> AnyPtr
%foreign "scheme:(lambda (meta) ((foreign-procedure \"interp_write_meta_add_vals\" (void*) void*) meta))"
prim__interpWriteAddVals : AnyPtr -> AnyPtr
%foreign "scheme:(lambda (meta) ((foreign-procedure \"interp_write_meta_add_tape\" (void*) void*) meta))"
prim__interpWriteAddTape : AnyPtr -> AnyPtr

%foreign "scheme:(lambda (meta out) ((foreign-procedure \"interp_write_compute\" (void* void*) void*) meta out))"
prim__interpWriteCompute : AnyPtr -> AnyPtr -> AnyPtr


----------------------------------------------------------------------
-- Idris Tape Wrappers
----------------------------------------------------------------------

-- Must pass a varying argument to prim__tapeGen to prevent CSE.
-- Zero-arg definitions and constant-body functions are evaluated once.
%noinline
tapeGeneration : Nat -> Nat
tapeGeneration dummy = cast (prim__tapeGen (cast dummy))

-- Read current tape size (number of entries). Must pass a varying dummy
-- argument to prevent CSE, same as tapeGeneration.
%foreign "scheme:(lambda (dummy) ((foreign-procedure \"tape_get_size\" (int) int) dummy))"
prim__tapeSize : Int -> Int

export
%noinline
tapeSize : Nat -> Nat
tapeSize dummy = cast (prim__tapeSize (cast dummy))

%noinline
tapeAppendConst : Double -> String -> Nat
tapeAppendConst val pid = cast (prim__tapeAppendConst val pid)

%noinline
tapeAppendUnary : TapeOp -> Nat -> Double -> Nat
tapeAppendUnary op a1 val = cast (prim__tapeAppendUnary (toTag op) (cast a1) val)

%noinline
tapeAppendBinary : TapeOp -> Nat -> Nat -> Double -> Nat
tapeAppendBinary op a1 a2 val = cast (prim__tapeAppendBinary (toTag op) (cast a1) (cast a2) val)

-- Append tensor op entry (MatVec, Softmax, etc.) + set meta->out.
-- tape_append_tensor_op handles set_out internally based on tag.
%noinline
tapeAppendMatVecOp : Int -> AnyPtr -> AnyPtr -> AnyPtr
tapeAppendMatVecOp count meta outBuf = prim__tapeAppendTensorOp 11 count meta outBuf

-- Ensure weight entries are on tape (cached within same epoch). Returns start index.
%noinline
tapeEnsureBulkConst : AnyPtr -> Int -> Int
tapeEnsureBulkConst wBuf _ = prim__tapeEnsureBulkConst wBuf

-- Append DotOp + output ConstOp + set meta->out_tape_idx. Returns ConstOp tape index.
%noinline
tapeAppendDotOp : AnyPtr -> Double -> Nat
tapeAppendDotOp meta val = cast (prim__tapeAppendDotOp meta val)

%noinline
tapeAppendSoftmaxOp : Int -> Int -> AnyPtr -> AnyPtr -> AnyPtr
tapeAppendSoftmaxOp tag count meta outBuf = prim__tapeAppendTensorOp tag count meta outBuf

%noinline
tapeAppendBatchCosSimOp : Int -> AnyPtr -> AnyPtr -> AnyPtr
tapeAppendBatchCosSimOp count meta outBuf = prim__tapeAppendTensorOp 15 count meta outBuf

%noinline
tapeAppendReadOpOp : Int -> AnyPtr -> AnyPtr -> AnyPtr
tapeAppendReadOpOp count meta outBuf = prim__tapeAppendTensorOp 16 count meta outBuf

%noinline
tapeAppendWriteOpOp : Int -> AnyPtr -> AnyPtr -> AnyPtr
tapeAppendWriteOpOp count meta outBuf = prim__tapeAppendTensorOp 17 count meta outBuf

%noinline
tapeAppendInterpWriteOp : Int -> AnyPtr -> AnyPtr -> AnyPtr
tapeAppendInterpWriteOp count meta outBuf = prim__tapeAppendTensorOp 18 count meta outBuf


----------------------------------------------------------------------
-- Record Definition
----------------------------------------------------------------------

public export
record Variable where
  constructor Var
  tapeIdx : Nat
  tapeGen : Nat
  paramId : Maybe String
  value : Double


----------------------------------------------------------------------
-- Stale Variable Re-registration
----------------------------------------------------------------------

-- Fresh variables (current gen) return their existing tapeIdx.
-- Stale variables (previous gen) get re-appended as Const entries.
%noinline
ensureOnTape : Variable -> Nat
ensureOnTape v =
  if v.tapeGen == tapeGeneration v.tapeIdx
    then v.tapeIdx
    else tapeAppendConst v.value (fromMaybe "" v.paramId)


----------------------------------------------------------------------
-- Instances
----------------------------------------------------------------------

export
Show Variable where
  show v =
    "Var" ++
    (case v.paramId of (Just pid) => "<" ++ pid ++ ">"; Nothing => "") ++
    "(" ++ show v.value ++ ")"

public export
implementation Eq Variable where
  v1 == v2 = v1.value == v2.value

public export
implementation Ord Variable where
  v1 < v2 = v1.value < v2.value

public export
implementation FromDouble Variable where
  fromDouble n =
    let idx = tapeAppendConst n ""
    in Var idx (tapeGeneration idx) Nothing n

public export
implementation Cast Variable Double where
  cast v = v.value

public export
implementation Cast Double Variable where
  cast = fromDouble

public export
implementation Random Variable where
  randomIO = map fromDouble randomIO
  randomRIO (lo, hi) = map fromDouble (randomRIO (lo.value, hi.value))


----------------------------------------------------------------------
-- Arithmetic Instances
----------------------------------------------------------------------

public export
implementation Num Variable where
  v1 + v2 =
    let idx1 = ensureOnTape v1
        idx2 = ensureOnTape v2
        val = v1.value + v2.value
        idx = tapeAppendBinary AddOp idx1 idx2 val
    in Var idx (tapeGeneration idx) Nothing val

  v1 * v2 =
    let idx1 = ensureOnTape v1
        idx2 = ensureOnTape v2
        val = v1.value * v2.value
        idx = tapeAppendBinary MulOp idx1 idx2 val
    in Var idx (tapeGeneration idx) Nothing val

  fromInteger v =
    let val = fromInteger v
        idx = tapeAppendConst val ""
    in Var idx (tapeGeneration idx) Nothing val

public export
implementation Neg Variable where
  v1 - v2 =
    let idx1 = ensureOnTape v1
        idx2 = ensureOnTape v2
        val = v1.value - v2.value
        idx = tapeAppendBinary SubOp idx1 idx2 val
    in Var idx (tapeGeneration idx) Nothing val

  negate v =
    let idx0 = ensureOnTape v
        val = negate v.value
        idx = tapeAppendUnary NegOp idx0 val
    in Var idx (tapeGeneration idx) Nothing val

public export
implementation Abs Variable where
  abs v =
    let idx0 = ensureOnTape v
        val = abs v.value
        idx = tapeAppendUnary AbsOp idx0 val
    in Var idx (tapeGeneration idx) Nothing val

public export
implementation Fractional Variable where
  v1 / v2 =
    let idx1 = ensureOnTape v1
        idx2 = ensureOnTape v2
        val = v1.value / v2.value
        idx = tapeAppendBinary DivOp idx1 idx2 val
    in Var idx (tapeGeneration idx) Nothing val

public export
implementation Floating Variable where
  exp v =
    let idx0 = ensureOnTape v
        val = exp v.value
        idx = tapeAppendUnary ExpOp idx0 val
    in Var idx (tapeGeneration idx) Nothing val

  log v =
    let idx0 = ensureOnTape v
        val = log v.value
        idx = tapeAppendUnary LogOp idx0 val
    in Var idx (tapeGeneration idx) Nothing val

  pow v1 v2 =
    let idx1 = ensureOnTape v1
        idx2 = ensureOnTape v2
        val = pow v1.value v2.value
        idx = tapeAppendBinary PowOp idx1 idx2 val
    in Var idx (tapeGeneration idx) Nothing val

  sqrt v =
    let idx0 = ensureOnTape v
        val = sqrt v.value
        idx = tapeAppendUnary SqrtOp idx0 val
    in Var idx (tapeGeneration idx) Nothing val


----------------------------------------------------------------------
-- Clamping
----------------------------------------------------------------------

||| Clamp a variable to [lo, hi] with straight-through gradient.
||| When clamped, the output is a detached constant (no gradient).
||| When within bounds, the original variable passes through unchanged.
export
clampVar : Double -> Double -> Variable -> Variable
clampVar lo hi v =
  if v.value < lo then fromDouble lo
  else if v.value > hi then fromDouble hi
  else v


----------------------------------------------------------------------
-- Native Activation Ops
----------------------------------------------------------------------

||| Sigmoid with a single tape entry (SigmoidOp).
||| Forward: σ(x) = 1/(1+exp(-x))
||| Backward: grad * σ(x) * (1 - σ(x))
export
sigmoidVar : Variable -> Variable
sigmoidVar v =
  let idx0 = ensureOnTape v
      val = 1.0 / (1.0 + exp (negate v.value))
      idx = tapeAppendUnary SigmoidOp idx0 val
  in Var idx (tapeGeneration idx) Nothing val

||| Tanh with a single tape entry (TanhOp).
||| Forward: tanh(x)
||| Backward: grad * (1 - tanh(x)^2)
export
tanhVar : Variable -> Variable
tanhVar v =
  let idx0 = ensureOnTape v
      val = 2.0 / (1.0 + exp (negate (2.0 * v.value))) - 1.0
      idx = tapeAppendUnary TanhOp idx0 val
  in Var idx (tapeGeneration idx) Nothing val


----------------------------------------------------------------------
-- Parameter Naming
----------------------------------------------------------------------

%noinline
setParamId : String -> Variable -> Variable
setParamId pid v =
  let idx = cast {to=Nat} (prim__tapeSetParamId (cast v.tapeIdx) pid)
  in Var idx v.tapeGen (Just pid) v.value

export
param : String -> Double -> Variable
param pid = setParamId pid . fromDouble

export
nameParam : String -> Nat -> Variable -> Variable
nameParam prefx i p = setParamId (prefx ++ show i) p


----------------------------------------------------------------------
-- Tensor-level Operations (C-backed forward + tape recording)
----------------------------------------------------------------------

-- Pack a row of scalar Variables into value/tape-index arrays.
-- Uses Scheme-native foreign-set! (no C FFI crossing per element).
-- Returns the value pointer for threading (prevents dead-code elimination).
packRow : AnyPtr -> AnyPtr -> Int -> Vect k (Scalar Variable) -> AnyPtr
packRow vp tp _ [] = vp
packRow vp tp off (STensor v :: rest) =
  let tIdx = ensureOnTape v
      vp' = prim__setDouble vp off v.value
      tp' = prim__setInt32 tp off (cast tIdx)
  in packRow vp' tp' (off + 1) rest

-- Pack all rows of a matrix into value/tape-index arrays (row-major).
-- Threads the value pointer through each row to force evaluation.
packMatrix : AnyPtr -> AnyPtr -> Int -> {n : Nat} -> Vect m (Vector n Variable) -> AnyPtr
packMatrix vp tp _ {m=Z} [] = vp
packMatrix vp tp off {m=S k} {n} (VTensor row :: rows) =
  let vp' = packRow vp tp off row
  in packMatrix vp' tp (off + cast {to=Int} n) rows

-- Pack a vector of scalar Variables into value/tape-index arrays.
packVec : AnyPtr -> AnyPtr -> Int -> Vect k (Scalar Variable) -> AnyPtr
packVec vp tp _ [] = vp
packVec vp tp off (STensor v :: rest) =
  let tIdx = ensureOnTape v
      vp' = prim__setDouble vp off v.value
      tp' = prim__setInt32 tp off (cast tIdx)
  in packVec vp' tp' (off + 1) rest

-- Build k output Scalars by reading values from a C buffer and appending
-- ConstOp entries. off is the current index into the buffer.
buildOutputScalars : AnyPtr -> Int -> (k : Nat) -> Vect k (Scalar Variable)
buildOutputScalars outBuf off Z = []
buildOutputScalars outBuf off (S k) =
  let val = prim__tensorRead outBuf off
      idx = tapeAppendConst val ""
      gen = tapeGeneration idx
  in STensor (Var idx gen Nothing val) :: buildOutputScalars outBuf (off + 1) k

||| Matrix-vector multiply using C BLAS, recording a single MatVecOp
||| tape entry instead of m*n scalar entries.
export
matrixVectorMultiplyVar : {m, n : Nat} -> Matrix m n Variable -> Vector n Variable -> Vector m Variable
matrixVectorMultiplyVar {m} {n} (VTensor rows) (VTensor xs) =
  let mI = cast {to=Int} m
      nI = cast {to=Int} n
      -- Allocate meta (arena) and output buffer (heap)
      meta = prim__matvecMetaAlloc mI nI
      outBuf = prim__tensorAlloc mI
      -- Get raw array pointers (4 C calls, then Scheme-native writes)
      wvPtr = prim__matvecWVals meta
      wtPtr = prim__matvecWTape meta
      xvPtr = prim__matvecXVals meta
      xtPtr = prim__matvecXTape meta
      -- Pack weight values and tape indices (row-major, Scheme foreign-set!)
      wvPtr' = packMatrix wvPtr wtPtr 0 rows
      -- Pack input values and tape indices (xvPtr depends on wvPtr' via seq)
      xvPtr' = packVec (prim__seq wvPtr' xvPtr) xtPtr 0 xs
      -- Compute forward: writes results into outBuf.
      -- prim__seq forces packing to complete before compute reads the arrays.
      outBuf' = prim__matvecCompute meta (prim__seq xvPtr' outBuf)
      -- Append MatVecOp entry + set meta->out_tape_start.
      outBuf'' = tapeAppendMatVecOp mI meta outBuf'
  in VTensor $ buildOutputScalars outBuf'' 0 m


||| Matrix-vector multiply using persistent weight buffer.
||| Bulk-registers all weights in one FFI call instead of per-element packing.
export
matrixVectorMultiplyVarBuf : {m, n : Nat} -> AnyPtr -> Vector n Variable -> Vector m Variable
matrixVectorMultiplyVarBuf {m} {n} wBuf (VTensor xs) =
  let mI = cast {to=Int} m
      nI = cast {to=Int} n
      -- Ensure weights on tape (cached within epoch): 0-1 FFI calls
      wTapeStart = tapeEnsureBulkConst wBuf (mI * nI)
      -- Allocate meta using persistent buffer path
      wValsPtr = prim__weightBufVals wBuf
      meta = prim__matvecMetaAllocBuf mI nI wValsPtr wTapeStart
      outBuf = prim__tensorAlloc mI
      -- Pack input values and tape indices (unchanged)
      xvPtr = prim__matvecXVals meta
      xtPtr = prim__matvecXTape meta
      xvPtr' = packVec xvPtr xtPtr 0 xs
      -- Compute forward
      outBuf' = prim__matvecCompute meta (prim__seq xvPtr' outBuf)
      -- Append MatVecOp entry
      outBuf'' = tapeAppendMatVecOp mI meta outBuf'
  in VTensor $ buildOutputScalars outBuf'' 0 m


----------------------------------------------------------------------
-- Weight Buffer Helpers
----------------------------------------------------------------------

-- Write initial values and pids from a matrix of Variables into a weight buffer.
-- Returns the buffer pointer for threading.
initWeightBufRow : AnyPtr -> Int -> Vect k (Scalar Variable) -> AnyPtr
initWeightBufRow wBuf _ [] = wBuf
initWeightBufRow wBuf off (STensor v :: rest) =
  let wBuf' = prim__weightBufSetVal wBuf off v.value
      wBuf'' = prim__weightBufSetPid wBuf' off (fromMaybe "" v.paramId)
  in initWeightBufRow wBuf'' (off + 1) rest

export
initWeightBuf : AnyPtr -> Int -> {n : Nat} -> Vect m (Vector n Variable) -> AnyPtr
initWeightBuf wBuf _ {m=Z} [] = wBuf
initWeightBuf wBuf off {m=S k} {n} (VTensor row :: rows) =
  let wBuf' = initWeightBufRow wBuf off row
  in initWeightBuf wBuf' (off + cast {to=Int} n) rows

-- Sync updated values from Variables into the C buffer after applyDeltas.
syncWeightBufRow : AnyPtr -> Int -> Vect k (Scalar Variable) -> AnyPtr
syncWeightBufRow wBuf _ [] = wBuf
syncWeightBufRow wBuf off (STensor v :: rest) =
  let wBuf' = prim__weightBufSetVal wBuf off v.value
  in syncWeightBufRow wBuf' (off + 1) rest

export
syncWeightBuf : AnyPtr -> Int -> {n : Nat} -> Vect m (Vector n Variable) -> AnyPtr
syncWeightBuf wBuf _ {m=Z} [] = wBuf
syncWeightBuf wBuf off {m=S k} {n} (VTensor row :: rows) =
  let wBuf' = syncWeightBufRow wBuf off row
  in syncWeightBuf wBuf' (off + cast {to=Int} n) rows


||| Dot product using C BLAS, recording a single DotOp tape entry.
export
dotProductVar : {n : Nat} -> Vector n Variable -> Vector n Variable -> Variable
dotProductVar {n} (VTensor as) (VTensor bs) =
  let nI = cast {to=Int} n
      -- Allocate meta (arena)
      meta = prim__dotMetaAlloc nI
      -- Get raw array pointers (4 C calls)
      avPtr = prim__dotAVals meta
      atPtr = prim__dotATape meta
      bvPtr = prim__dotBVals meta
      btPtr = prim__dotBTape meta
      -- Pack both vectors (Scheme-native foreign-set!)
      avPtr' = packVec avPtr atPtr 0 as
      bvPtr' = packVec (prim__seq avPtr' bvPtr) btPtr 0 bs
      -- Compute forward (prim__seq forces packing before compute)
      val = prim__dotCompute (prim__seq bvPtr' meta)
      -- Append DotOp + ConstOp entries + set meta->out_tape_idx.
      outIdx = tapeAppendDotOp meta val
      gen = tapeGeneration outIdx
  in Var outIdx gen Nothing val


----------------------------------------------------------------------
-- Softmax / LogSoftmax (C-backed)
----------------------------------------------------------------------

||| Softmax using C kernel, recording a single SoftmaxOp tape entry.
export
softmaxVar : {n : Nat} -> Vector n Variable -> Vector n Variable
softmaxVar {n} (VTensor xs) =
  let nI = cast {to=Int} n
      meta = prim__softmaxMetaAlloc nI
      outBuf = prim__tensorAlloc nI
      xvPtr = prim__softmaxXVals meta
      xtPtr = prim__softmaxXTape meta
      xvPtr' = packVec xvPtr xtPtr 0 xs
      outBuf' = prim__softmaxCompute meta (prim__seq xvPtr' outBuf)
      outBuf'' = tapeAppendSoftmaxOp (toTag SoftmaxOp) nI meta outBuf'
  in VTensor $ buildOutputScalars outBuf'' 0 n

||| LogSoftmax using C kernel, recording a single LogSoftmaxOp tape entry.
export
logSoftmaxVar : {n : Nat} -> Vector n Variable -> Vector n Variable
logSoftmaxVar {n} (VTensor xs) =
  let nI = cast {to=Int} n
      meta = prim__softmaxMetaAlloc nI
      outBuf = prim__tensorAlloc nI
      xvPtr = prim__softmaxXVals meta
      xtPtr = prim__softmaxXTape meta
      xvPtr' = packVec xvPtr xtPtr 0 xs
      outBuf' = prim__logsoftmaxCompute meta (prim__seq xvPtr' outBuf)
      outBuf'' = tapeAppendSoftmaxOp (toTag LogSoftmaxOp) nI meta outBuf'
  in VTensor $ buildOutputScalars outBuf'' 0 n


----------------------------------------------------------------------
-- NTM Memory Operations (C-backed)
----------------------------------------------------------------------

-- Build matrix output: N rows of W output Variables from a flat C buffer.
buildOutputMatrix : AnyPtr -> Int -> (rows : Nat) -> (cols : Nat) -> Vect rows (Vector cols Variable)
buildOutputMatrix outBuf off Z cols = []
buildOutputMatrix outBuf off (S k) cols =
  let row = VTensor (buildOutputScalars outBuf off cols)
  in row :: buildOutputMatrix outBuf (off + cast {to=Int} cols) k cols

||| Batch cosine similarity: out[i] = beta * cos_sim(key, mem[i])
||| Records a single BatchCosSimOp tape entry instead of ~4NW scalar entries.
export
batchCosineSimilarityVar : {n, w : Nat} -> Variable -> Matrix n w Variable -> Vector w Variable -> Vector n Variable
batchCosineSimilarityVar {n} {w} beta (VTensor memRows) (VTensor keyElems) =
  let nI = cast {to=Int} n
      wI = cast {to=Int} w
      meta = prim__batchCosSimMetaAlloc nI wI
      outBuf = prim__tensorAlloc nI
      mvPtr = prim__batchCosSimMemVals meta
      mtPtr = prim__batchCosSimMemTape meta
      kvPtr = prim__batchCosSimKeyVals meta
      ktPtr = prim__batchCosSimKeyTape meta
      mvPtr' = packMatrix mvPtr mtPtr 0 memRows
      kvPtr' = packVec (prim__seq mvPtr' kvPtr) ktPtr 0 keyElems
      betaIdx = ensureOnTape beta
      meta' = prim__batchCosSimSetBeta (prim__seq kvPtr' meta) beta.value (cast betaIdx)
      outBuf' = prim__batchCosSimCompute meta' outBuf
      outBuf'' = tapeAppendBatchCosSimOp nI meta' outBuf'
  in VTensor $ buildOutputScalars outBuf'' 0 n

||| C-backed read operation: out[j] = sum_i weight[i] * mem[i*w+j]
||| Records a single ReadOpOp tape entry instead of ~2NW scalar entries.
export
readOpVar : {n, w : Nat} -> Vector n Variable -> Matrix n w Variable -> Vector w Variable
readOpVar {n} {w} (VTensor weightElems) (VTensor memRows) =
  let nI = cast {to=Int} n
      wI = cast {to=Int} w
      meta = prim__readOpMetaAlloc nI wI
      outBuf = prim__tensorAlloc wI
      mvPtr = prim__readOpMemVals meta
      mtPtr = prim__readOpMemTape meta
      wvPtr = prim__readOpWeightVals meta
      wtPtr = prim__readOpWeightTape meta
      mvPtr' = packMatrix mvPtr mtPtr 0 memRows
      wvPtr' = packVec (prim__seq mvPtr' wvPtr) wtPtr 0 weightElems
      outBuf' = prim__readOpCompute meta (prim__seq wvPtr' outBuf)
      outBuf'' = tapeAppendReadOpOp wI meta outBuf'
  in VTensor $ buildOutputScalars outBuf'' 0 w

||| C-backed write operation: out[i][j] = mem[i][j]*(1-w[i]*e[j]) + w[i]*a[j]
||| Records a single WriteOpOp tape entry instead of ~4NW scalar entries.
export
writeOpVar : {n, w : Nat} -> Vector n Variable -> Matrix n w Variable -> Vector w Variable -> Vector w Variable -> Matrix n w Variable
writeOpVar {n} {w} (VTensor weightElems) (VTensor memRows) (VTensor eraseElems) (VTensor addElems) =
  let nI = cast {to=Int} n
      wI = cast {to=Int} w
      meta = prim__writeOpMetaAlloc nI wI
      outBuf = prim__tensorAlloc (nI * wI)
      mvPtr = prim__writeOpMemVals meta
      mtPtr = prim__writeOpMemTape meta
      wvPtr = prim__writeOpWeightVals meta
      wtPtr = prim__writeOpWeightTape meta
      evPtr = prim__writeOpEraseVals meta
      etPtr = prim__writeOpEraseTape meta
      avPtr = prim__writeOpAddVals meta
      atPtr = prim__writeOpAddTape meta
      mvPtr' = packMatrix mvPtr mtPtr 0 memRows
      wvPtr' = packVec (prim__seq mvPtr' wvPtr) wtPtr 0 weightElems
      evPtr' = packVec (prim__seq wvPtr' evPtr) etPtr 0 eraseElems
      avPtr' = packVec (prim__seq evPtr' avPtr) atPtr 0 addElems
      outBuf' = prim__writeOpCompute meta (prim__seq avPtr' outBuf)
      outBuf'' = tapeAppendWriteOpOp (nI * wI) meta outBuf'
  in VTensor $ buildOutputMatrix outBuf'' 0 n w


||| C-backed interpolation write: out[i][j] = (1-w[i])*mem[i][j] + w[i]*add[j]
||| Records a single InterpWriteOp tape entry.
export
interpolationWriteVar : {n, w : Nat} -> Vector n Variable -> Matrix n w Variable -> Vector w Variable -> Matrix n w Variable
interpolationWriteVar {n} {w} (VTensor weightElems) (VTensor memRows) (VTensor addElems) =
  let nI = cast {to=Int} n
      wI = cast {to=Int} w
      meta = prim__interpWriteMetaAlloc nI wI
      outBuf = prim__tensorAlloc (nI * wI)
      mvPtr = prim__interpWriteMemVals meta
      mtPtr = prim__interpWriteMemTape meta
      wvPtr = prim__interpWriteWeightVals meta
      wtPtr = prim__interpWriteWeightTape meta
      avPtr = prim__interpWriteAddVals meta
      atPtr = prim__interpWriteAddTape meta
      mvPtr' = packMatrix mvPtr mtPtr 0 memRows
      wvPtr' = packVec (prim__seq mvPtr' wvPtr) wtPtr 0 weightElems
      avPtr' = packVec (prim__seq wvPtr' avPtr) atPtr 0 addElems
      outBuf' = prim__interpWriteCompute meta (prim__seq avPtr' outBuf)
      outBuf'' = tapeAppendInterpWriteOp (nI * wI) meta outBuf'
  in VTensor $ buildOutputMatrix outBuf'' 0 n w


----------------------------------------------------------------------
-- Backpropagation (C-backed)
----------------------------------------------------------------------

-- Build SortedMap from C result buffer (populated by walk_backward).
-- walk_backward collects (pid, grad) pairs into result_pids/result_vals.
-- Duplicate pids (same param re-registered after stale detection) are
-- accumulated with (+) via mergeWith.
buildGradMap : Int -> Int -> SortedMap String Double -> SortedMap String Double
buildGradMap n i acc = if i >= n then acc
  else let pid = prim__resultGetPid i
           val = prim__resultGetVal i
       in buildGradMap n (i + 1) (mergeWith (+) acc (singleton pid val))

export
collectGrads : Double -> Variable -> SortedMap String Double
collectGrads initGrad root =
  let size = cast {to=Int} root.tapeIdx + 1
      g = prim__gradAlloc size
      g' = prim__gradAdd g (cast root.tapeIdx) initGrad
      nParams = prim__walkBackwardAndReset g' size
  in buildGradMap nParams 0 empty
