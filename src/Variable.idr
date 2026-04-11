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
-- Backend FFI (libtorch via libidrisml)
----------------------------------------------------------------------

-- Lifecycle
%foreign "C:tensor_create_scalar,libidrisml"
prim__createScalar : Double -> Int -> AnyPtr

%foreign "C:tensor_free,libidrisml"
prim__free : AnyPtr -> ()

%foreign "C:tensor_item,libidrisml"
prim__item : AnyPtr -> Double

-- Arithmetic (all return new tensors — libtorch builds autograd graph)
%foreign "C:tensor_add,libidrisml"
prim__add : AnyPtr -> AnyPtr -> AnyPtr

%foreign "C:tensor_sub,libidrisml"
prim__sub : AnyPtr -> AnyPtr -> AnyPtr

%foreign "C:tensor_mul,libidrisml"
prim__mul : AnyPtr -> AnyPtr -> AnyPtr

%foreign "C:tensor_div,libidrisml"
prim__div : AnyPtr -> AnyPtr -> AnyPtr

%foreign "C:tensor_neg,libidrisml"
prim__neg : AnyPtr -> AnyPtr

%foreign "C:tensor_abs,libidrisml"
prim__abs : AnyPtr -> AnyPtr

%foreign "C:tensor_exp,libidrisml"
export prim__exp : AnyPtr -> AnyPtr

%foreign "C:tensor_log,libidrisml"
export prim__log : AnyPtr -> AnyPtr

%foreign "C:tensor_sqrt,libidrisml"
prim__sqrt : AnyPtr -> AnyPtr

%foreign "C:tensor_pow,libidrisml"
prim__pow : AnyPtr -> AnyPtr -> AnyPtr

%foreign "C:tensor_sigmoid,libidrisml"
export prim__sigmoid : AnyPtr -> AnyPtr

%foreign "C:tensor_tanh,libidrisml"
prim__tanh : AnyPtr -> AnyPtr

-- Linear algebra
%foreign "C:tensor_mv,libidrisml"
prim__mv : AnyPtr -> AnyPtr -> AnyPtr

%foreign "C:tensor_dot,libidrisml"
prim__dot : AnyPtr -> AnyPtr -> AnyPtr

%foreign "C:tensor_outer,libidrisml"
prim__outer : AnyPtr -> AnyPtr -> AnyPtr

%foreign "C:tensor_matmul,libidrisml"
prim__matmul : AnyPtr -> AnyPtr -> AnyPtr

-- Activation
%foreign "C:tensor_softmax,libidrisml"
export prim__softmax : AnyPtr -> Int -> AnyPtr

%foreign "C:tensor_log_softmax,libidrisml"
prim__logSoftmax : AnyPtr -> Int -> AnyPtr

-- Loss
%foreign "C:tensor_bce_with_logits,libidrisml"
prim__bceWithLogits : AnyPtr -> AnyPtr -> AnyPtr

-- Reduction
%foreign "C:tensor_sum,libidrisml"
prim__sum : AnyPtr -> AnyPtr

%foreign "C:tensor_mean,libidrisml"
prim__mean : AnyPtr -> AnyPtr

-- Tensor creation/accessors
%foreign "C:tensor_create,libidrisml"
prim__create : AnyPtr -> AnyPtr -> Int -> Int -> AnyPtr

%foreign "C:tensor_numel,libidrisml"
prim__numel : AnyPtr -> Int

%foreign "C:tensor_size,libidrisml"
prim__size : AnyPtr -> Int -> Int

%foreign "C:tensor_select,libidrisml"
export prim__select : AnyPtr -> Int -> Int -> AnyPtr

%foreign "C:tensor_unsqueeze,libidrisml"
prim__unsqueeze : AnyPtr -> Int -> AnyPtr

%foreign "C:tensor_stack,libidrisml"
prim__stack : AnyPtr -> Int -> Int -> AnyPtr

%foreign "C:tensor_detach,libidrisml"
prim__detach : AnyPtr -> AnyPtr

%foreign "C:tensor_with_grad,libidrisml"
prim__withGrad : AnyPtr -> AnyPtr

%foreign "C:tensor_mul_scalar,libidrisml"
prim__mulScalar : AnyPtr -> Double -> AnyPtr

%foreign "C:tensor_add_scalar,libidrisml"
export prim__addScalar : AnyPtr -> Double -> AnyPtr

%foreign "C:tensor_clamp_min,libidrisml"
prim__clampMin : AnyPtr -> Double -> AnyPtr

-- NTM
%foreign "C:tensor_cosine_similarity,libidrisml"
prim__cosineSimilarity : AnyPtr -> AnyPtr -> Int -> AnyPtr

%foreign "C:tensor_conv1d_circular,libidrisml"
prim__conv1dCircular : AnyPtr -> AnyPtr -> AnyPtr

-- Autograd
-- Returns the input pointer for threading (prevents dead code elimination).
%foreign "scheme:(lambda (t) ((foreign-procedure \"tensor_backward\" (void*) void) t) t)"
prim__backward : AnyPtr -> AnyPtr

%foreign "C:tensor_grad,libidrisml"
prim__grad : AnyPtr -> AnyPtr

%foreign "C:tensor_zero_grad,libidrisml"
prim__zeroGrad : AnyPtr -> ()

-- Parameter registry
-- Registers a parameter: enables requires_grad and adds to the registry.
-- Returns the tensorPtr for threading (prevents dead code elimination).
%foreign "scheme:(lambda (name t) ((foreign-procedure \"tensor_set_requires_grad\" (void* int) void) t 1) ((foreign-procedure \"param_register\" (string void*) void) name t) t)"
export
prim__paramRegister : String -> AnyPtr -> AnyPtr

%foreign "C:param_clear,libidrisml"
prim__paramClear : ()

%foreign "C:param_count,libidrisml"
prim__paramCount : Int

%foreign "C:param_name,libidrisml"
prim__paramName : Int -> String

%foreign "C:param_grad_item,libidrisml"
prim__paramGradItem : Int -> Double

%foreign "C:param_grad_item_and_zero,libidrisml"
prim__paramGradItemAndZero : Int -> Double

%foreign "scheme:(lambda () ((foreign-procedure \"param_zero_all_grads\" () void)) 0)"
prim__paramZeroAllGrads : Int

%foreign "C:param_subtract_delta,libidrisml"
prim__paramSubtractDelta : Int -> Double -> ()

-- In-place scalar subtract on a tensor (under no_grad). Returns tensor for threading.
%foreign "C:tensor_subtract_scalar_inplace,libidrisml"
export
prim__tensorSubScalarInplace : AnyPtr -> Double -> AnyPtr

-- Tensor-level parameter creation
%foreign "C:tensor_create_param_2d,libidrisml"
export
prim__createParam2d : Int -> Int -> AnyPtr -> AnyPtr

%foreign "C:tensor_create_param_1d,libidrisml"
export
prim__createParam1d : Int -> AnyPtr -> AnyPtr

-- Persistent non-param tensors (for non-learnable state like NTM memory)
%foreign "C:tensor_create_state_2d,libidrisml"
export
prim__createState2d : Int -> Int -> AnyPtr -> AnyPtr

%foreign "C:tensor_create_state_1d,libidrisml"
export
prim__createState1d : Int -> AnyPtr -> AnyPtr

%foreign "C:tensor_view_2d,libidrisml"
export
prim__view2d : AnyPtr -> Int -> Int -> AnyPtr

%foreign "C:tensor_view_1d,libidrisml"
export
prim__view1d : AnyPtr -> Int -> AnyPtr

%foreign "C:tensor_item_2d,libidrisml"
export
prim__item2d : AnyPtr -> Int -> Int -> Double

%foreign "C:tensor_item_1d,libidrisml"
export
prim__item1d : AnyPtr -> Int -> Double

-- Fused LSTM gates: takes combined [4*o] tensor + prev_cell [o], returns pair handle
%foreign "C:tensor_lstm_gates_pair,libidrisml"
export
prim__lstmGatesPair : AnyPtr -> AnyPtr -> Int -> AnyPtr

%foreign "C:tensor_pair_first,libidrisml"
export
prim__pairFirst : AnyPtr -> AnyPtr

%foreign "C:tensor_pair_second,libidrisml"
export
prim__pairSecond : AnyPtr -> AnyPtr

-- Fused NTM read head: entire addressing pipeline in one C call
%foreign "C:tensor_ntm_read_head,libidrisml"
export
prim__ntmReadHead : AnyPtr -> AnyPtr -> AnyPtr -> AnyPtr -> AnyPtr -> AnyPtr -> AnyPtr -> AnyPtr

-- NTM interpolation write: memory + outer(weights, add)
%foreign "C:tensor_ntm_interp_write,libidrisml"
export
prim__ntmInterpWrite : AnyPtr -> AnyPtr -> AnyPtr -> AnyPtr

-- Tensor-level forward ops (used by layers with consolidated weight tensors)
||| Matrix-vector multiply on raw tensor pointers.
export
tensorMv : AnyPtr -> AnyPtr -> AnyPtr
tensorMv = prim__mv

||| Add two raw tensor pointers.
export
tensorAdd : AnyPtr -> AnyPtr -> AnyPtr
tensorAdd = prim__add

-- No-grad scope
%foreign "C:tensor_no_grad_begin,libidrisml"
prim__noGradBegin : ()

%foreign "C:tensor_no_grad_end,libidrisml"
prim__noGradEnd : ()

-- LSTM
%foreign "C:tensor_lstm_cell,libidrisml"
prim__lstmCell : AnyPtr -> AnyPtr -> AnyPtr -> AnyPtr -> AnyPtr -> AnyPtr -> AnyPtr -> AnyPtr -> AnyPtr -> ()

-- Debug
%foreign "C:tensor_print,libidrisml"
prim__print : AnyPtr -> ()


----------------------------------------------------------------------
-- Sequencing helper
----------------------------------------------------------------------

-- Force evaluation of first arg, return second.
-- Must use concrete AnyPtr types — polymorphic types cause Chez
-- to miscount arguments when b is itself a function type.
%foreign "scheme:(lambda (a b) b)"
export
prim__seq : AnyPtr -> AnyPtr -> AnyPtr


----------------------------------------------------------------------
-- Helpers: pack/unpack between Variable vectors and libtorch tensors
----------------------------------------------------------------------

-- Use C-side allocation to avoid Scheme/C FFI evaluation order issues.
-- Scheme-side foreign-alloc + foreign-set! can be reordered by the
-- Chez Scheme optimizer, causing C functions to read stale pointers.

%foreign "C:tensor_alloc_doubles,libidrisml"
export
prim__allocDoubles : Int -> AnyPtr

%foreign "C:tensor_write_double,libidrisml"
prim__writeDouble : AnyPtr -> Int -> Double -> ()

%foreign "C:tensor_read_double,libidrisml"
prim__readDouble : AnyPtr -> Int -> Double

-- Wrapper that returns the buffer pointer for threading through let chains
%foreign "scheme:(lambda (buf off val) ((foreign-procedure \"tensor_write_double\" (void* int double) void) buf off val) buf)"
prim__setDouble : AnyPtr -> Int -> Double -> AnyPtr

%foreign "C:tensor_create_1d,libidrisml"
prim__create1d : Int -> AnyPtr -> Int -> AnyPtr

%foreign "C:tensor_create_2d,libidrisml"
prim__create2d : Int -> Int -> AnyPtr -> Int -> AnyPtr

-- Tensor pointer array: stack scalar Variable tensorPtrs to create
-- a 1D/2D tensor that preserves the autograd graph.
%foreign "C:tensor_ptr_array_alloc,libidrisml"
prim__ptrArrayAlloc : Int -> AnyPtr

-- Returns the array for threading
%foreign "scheme:(lambda (arr idx t) ((foreign-procedure \"tensor_ptr_array_set\" (void* int void*) void) arr idx t) arr)"
prim__ptrArraySet : AnyPtr -> Int -> AnyPtr -> AnyPtr

%foreign "C:tensor_stack_from_array,libidrisml"
prim__stackFromArray : AnyPtr -> Int -> Int -> AnyPtr

%foreign "C:tensor_cat2,libidrisml"
export
prim__cat2 : AnyPtr -> AnyPtr -> AnyPtr

%foreign "C:tensor_narrow,libidrisml"
export
prim__narrow : AnyPtr -> Int -> Int -> Int -> AnyPtr

%foreign "C:tensor_reshape,libidrisml"
prim__reshape : AnyPtr -> AnyPtr -> Int -> AnyPtr

%foreign "scheme:(lambda (n) (foreign-alloc (* n 4)))"
prim__allocInts : Int -> AnyPtr

%foreign "scheme:(lambda (buf off val) (foreign-set! 'integer-32 buf (* off 4) val) buf)"
prim__setInt : AnyPtr -> Int -> Int -> AnyPtr

public export
record Variable where
  constructor Var
  tensorPtr : AnyPtr
  paramId : Maybe String
  value : Double

-- Pack scalar Variable values into a pre-allocated double buffer.
export
packScalarValues : AnyPtr -> Int -> Vect k (Scalar Variable) -> AnyPtr
packScalarValues buf _ [] = buf
packScalarValues buf off (STensor v :: rest) =
  let buf' = prim__setDouble buf off v.value
  in packScalarValues buf' (off + 1) rest

-- Pack all rows of a matrix into a flat double buffer (row-major).
export
packMatrixValues : AnyPtr -> Int -> {n : Nat} -> Vect m (Vector n Variable) -> AnyPtr
packMatrixValues buf _ {m=Z} [] = buf
packMatrixValues buf off {m=S k} {n} (VTensor row :: rows) =
  let buf' = packScalarValues buf off row
  in packMatrixValues buf' (off + cast {to=Int} n) rows

-- Create a 1D libtorch tensor from a Vect of scalar Variables.
-- The tensor inherits requires_grad if any input has it.
-- vecToTensor (value-based): pack scalar values into a C buffer.
-- Does NOT preserve autograd graph — use for non-differentiable contexts only.
vecToTensor : {n : Nat} -> Vect n (Scalar Variable) -> (requiresGrad : Int) -> AnyPtr
vecToTensor {n} elems rg =
  let nI = cast {to=Int} n
      buf = prim__allocDoubles nI
      buf' = packScalarValues buf 0 elems
  in prim__create1d nI buf' rg

-- matToTensor (value-based): pack matrix values into a C buffer.
-- Does NOT preserve autograd graph.
matToTensor : {m, n : Nat} -> Vect m (Vector n Variable) -> (requiresGrad : Int) -> AnyPtr
matToTensor {m} {n} rows rg =
  let mI = cast {to=Int} m
      nI = cast {to=Int} n
      buf = prim__allocDoubles (mI * nI)
      buf' = packMatrixValues buf 0 rows
  in prim__create2d mI nI buf' rg

-- Pack scalar Variable tensorPtrs into a C pointer array (for torch::stack).
packScalarPtrs : AnyPtr -> Int -> Vect k (Scalar Variable) -> AnyPtr
packScalarPtrs arr _ [] = arr
packScalarPtrs arr off (STensor v :: rest) =
  let arr' = prim__ptrArraySet arr off v.tensorPtr
  in packScalarPtrs arr' (off + 1) rest

-- vecStackTensor: stack scalar Variable tensorPtrs into a 1D tensor.
-- PRESERVES autograd graph — use for differentiable ops (dot, softmax, etc.)
export
vecStackTensor : {n : Nat} -> Vect n (Scalar Variable) -> AnyPtr
vecStackTensor {n} elems =
  let nI = cast {to=Int} n
      arr = prim__ptrArrayAlloc nI
      arr' = packScalarPtrs arr 0 elems
  in prim__stackFromArray arr' nI 0

-- Pack a matrix (Vect of Vectors) of tensorPtrs into a flat pointer array (row-major).
packMatrixPtrs : AnyPtr -> Int -> {n : Nat} -> Vect m (Vector n Variable) -> AnyPtr
packMatrixPtrs arr _ {m=Z} [] = arr
packMatrixPtrs arr off {m=S k} {n} (VTensor row :: rows) =
  let arr' = packScalarPtrs arr off row
  in packMatrixPtrs arr' (off + cast {to=Int} n) rows

%foreign "C:tensor_reshape_2d,libidrisml"
prim__reshape2d : AnyPtr -> Int -> Int -> AnyPtr

-- matStackTensor: stack matrix Variable tensorPtrs into a 2D tensor.
-- PRESERVES autograd graph.
matStackTensor : {m, n : Nat} -> Vect m (Vector n Variable) -> AnyPtr
matStackTensor {m} {n} rows =
  let mI = cast {to=Int} m
      nI = cast {to=Int} n
      -- Stack all m*n scalars into a flat 1D tensor, then reshape to [m, n]
      arr = prim__ptrArrayAlloc (mI * nI)
      arr' = packMatrixPtrs arr 0 rows
      flat = prim__stackFromArray arr' (mI * nI) 0  -- [m*n]
  in prim__reshape2d flat mI nI

-- Read k scalar values from a 1D libtorch tensor into a Vect.
export
tensorToScalars : AnyPtr -> Int -> (k : Nat) -> Vect k (Scalar Variable)
tensorToScalars _ _ Z = []
tensorToScalars t off (S k) =
  let elemPtr = prim__select t 0 off
      val = prim__item elemPtr
  in STensor (Var elemPtr Nothing val) :: tensorToScalars t (off + 1) k


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
    let ptr = prim__createScalar n 0
    in Var ptr Nothing n

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
-- Arithmetic Instances (libtorch autograd)
----------------------------------------------------------------------

public export
implementation Num Variable where
  v1 + v2 =
    let ptr = prim__add v1.tensorPtr v2.tensorPtr
        val = v1.value + v2.value
    in Var ptr Nothing val

  v1 * v2 =
    let ptr = prim__mul v1.tensorPtr v2.tensorPtr
        val = v1.value * v2.value
    in Var ptr Nothing val

  fromInteger v =
    let val = fromInteger v
    in fromDouble val

public export
implementation Neg Variable where
  v1 - v2 =
    let ptr = prim__sub v1.tensorPtr v2.tensorPtr
        val = v1.value - v2.value
    in Var ptr Nothing val

  negate v =
    let ptr = prim__neg v.tensorPtr
        val = negate v.value
    in Var ptr Nothing val

public export
implementation Abs Variable where
  abs v =
    let ptr = prim__abs v.tensorPtr
        val = abs v.value
    in Var ptr Nothing val

public export
implementation Fractional Variable where
  v1 / v2 =
    let ptr = prim__div v1.tensorPtr v2.tensorPtr
        val = v1.value / v2.value
    in Var ptr Nothing val

public export
implementation Floating Variable where
  exp v =
    let ptr = prim__exp v.tensorPtr
        val = exp v.value
    in Var ptr Nothing val

  log v =
    let ptr = prim__log v.tensorPtr
        val = log v.value
    in Var ptr Nothing val

  pow v1 v2 =
    let ptr = prim__pow v1.tensorPtr v2.tensorPtr
        val = pow v1.value v2.value
    in Var ptr Nothing val

  sqrt v =
    let ptr = prim__sqrt v.tensorPtr
        val = sqrt v.value
    in Var ptr Nothing val


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

||| Sigmoid with libtorch autograd.
export
sigmoidVar : Variable -> Variable
sigmoidVar v =
  let ptr = prim__sigmoid v.tensorPtr
      val = 1.0 / (1.0 + exp (negate v.value))
  in Var ptr Nothing val

||| Tanh with libtorch autograd.
export
tanhVar : Variable -> Variable
tanhVar v =
  let ptr = prim__tanh v.tensorPtr
      val = 2.0 / (1.0 + exp (negate (2.0 * v.value))) - 1.0
  in Var ptr Nothing val


----------------------------------------------------------------------
-- Parameter Naming
----------------------------------------------------------------------

export
setParamId : String -> Variable -> Variable
setParamId pid v =
  let ptr = prim__paramRegister pid v.tensorPtr
  in Var ptr (Just pid) v.value

export
param : String -> Double -> Variable
param pid val =
  let ptr = prim__createScalar val 1  -- requires_grad=true for parameters
  in setParamId pid (Var ptr Nothing val)

export
nameParam : String -> Nat -> Variable -> Variable
nameParam prefx i p = setParamId (prefx ++ show i) p


----------------------------------------------------------------------
-- Tensor-level Operations (libtorch compositions)
----------------------------------------------------------------------

||| Matrix-vector multiply using libtorch (autograd-tracked).
export
matrixVectorMultiplyVar : {m, n : Nat} -> Matrix m n Variable -> Vector n Variable -> Vector m Variable
matrixVectorMultiplyVar {m} {n} (VTensor rows) (VTensor xs) =
  let matTensor = matStackTensor rows
      vecTensor = vecStackTensor xs
      result = prim__mv matTensor vecTensor
  in VTensor $ tensorToScalars result 0 m


||| Dot product using libtorch (autograd-preserving).
export
dotProductVar : {n : Nat} -> Vector n Variable -> Vector n Variable -> Variable
dotProductVar {n} (VTensor as) (VTensor bs) =
  let aTensor = vecStackTensor as
      bTensor = prim__seq aTensor (vecStackTensor bs)
      result = prim__dot aTensor bTensor
      val = prim__item result
  in Var result Nothing val


----------------------------------------------------------------------
-- BCE with Logits
----------------------------------------------------------------------

||| Fused binary cross-entropy with logits loss using libtorch.
export
bceWithLogitsVar : {n : Nat} -> Vector n Variable -> Vector n Variable -> Variable
bceWithLogitsVar {n} (VTensor preds) (VTensor targets) =
  let predTensor = vecStackTensor preds
      targetTensor = vecStackTensor targets
      result = prim__bceWithLogits predTensor targetTensor
      val = prim__item result
  in Var result Nothing val


----------------------------------------------------------------------
-- Softmax / LogSoftmax
----------------------------------------------------------------------

||| Softmax using libtorch (autograd-tracked).
export
softmaxVar : {n : Nat} -> Vector n Variable -> Vector n Variable
softmaxVar {n} (VTensor xs) =
  let inTensor = vecStackTensor xs
      result = prim__softmax inTensor 0
  in VTensor $ tensorToScalars result 0 n

||| LogSoftmax using libtorch.
export
logSoftmaxVar : {n : Nat} -> Vector n Variable -> Vector n Variable
logSoftmaxVar {n} (VTensor xs) =
  let inTensor = vecStackTensor xs
      result = prim__logSoftmax inTensor 0
  in VTensor $ tensorToScalars result 0 n


----------------------------------------------------------------------
-- NTM Operations (libtorch compositions)
----------------------------------------------------------------------

||| Batch cosine similarity: beta * cosine_similarity(k, M).
||| k is a query vector [w], M is a memory matrix [n x w].
||| Returns scores [n] (not softmaxed — softmax applied downstream).
export
batchCosineSimilarityVar : {n, w : Nat} -> Variable -> Matrix n w Variable -> Vector w Variable -> Vector n Variable
batchCosineSimilarityVar {n} {w} beta (VTensor memRows) (VTensor keyElems) =
  let memTensor = matStackTensor memRows     -- [n, w], autograd-connected
      keyTensor = vecStackTensor keyElems    -- [w], autograd-connected
      keyExpanded = prim__unsqueeze keyTensor 0  -- [1, w]
      cosSim = prim__cosineSimilarity memTensor keyExpanded 1  -- [n]
      result = prim__mul beta.tensorPtr cosSim
  in VTensor $ tensorToScalars result 0 n

||| NTM read: weighted sum of memory rows.
||| read(w, M) = w^T * M (w is attention weights [n], M is memory [n x w_dim])
export
readOpVar : {n, w : Nat} -> Vector n Variable -> Matrix n w Variable -> Vector w Variable
readOpVar {n} {w} (VTensor wts) (VTensor memRows) =
  let wtTensor = vecStackTensor wts       -- [n], autograd-connected
      memTensor = matStackTensor memRows   -- [n, w], autograd-connected
      result = prim__matmul wtTensor memTensor  -- [w]
  in VTensor $ tensorToScalars result 0 w

||| NTM write: erase + add.
||| write(w, M, e, a) = M * (1 - outer(w, e)) + outer(w, a)
export
writeOpVar : {n, w : Nat} -> Vector n Variable -> Matrix n w Variable ->
             Vector w Variable -> Vector w Variable -> Matrix n w Variable
writeOpVar {n} {w} (VTensor wts) (VTensor memRows) (VTensor eraseElems) (VTensor addElems) =
  let wtTensor = vecStackTensor wts           -- [n]
      memTensor = matStackTensor memRows       -- [n, w]
      eraseTensor = vecStackTensor eraseElems  -- [w]
      addTensor = vecStackTensor addElems      -- [w]
      -- erase_gate = outer(w, e) -> [n, w]
      eraseGate = prim__outer wtTensor eraseTensor
      -- ones - erase_gate
      ones = prim__createScalar 1.0 0
      keepGate = prim__sub ones eraseGate
      -- erased = M * keep_gate
      erased = prim__mul memTensor keepGate
      -- add_gate = outer(w, a) -> [n, w]
      addGate = prim__outer wtTensor addTensor
      -- result = erased + add_gate
      result = prim__add erased addGate
  in -- Unpack result [n, w] back to Matrix
     VTensor $ buildMatrixFromTensor result 0 n w
  where
    buildRow : AnyPtr -> Int -> (k : Nat) -> Vect k (Scalar Variable)
    buildRow _ _ Z = []
    buildRow t col (S k) =
      let elemPtr = prim__select (prim__select t 0 0) 0 col  -- This won't work for 2D
          val = 0.0  -- placeholder
      in STensor (Var t Nothing val) :: buildRow t (col + 1) k

    buildMatrixFromTensor : AnyPtr -> Int -> (rows : Nat) -> (cols : Nat) -> Vect rows (Vector cols Variable)
    buildMatrixFromTensor _ _ Z _ = []
    buildMatrixFromTensor t row (S r) cols =
      let rowTensor = prim__select t 0 row  -- select row from [n, w] -> [w]
      in VTensor (tensorToScalars rowTensor 0 cols) :: buildMatrixFromTensor t (row + 1) r cols

||| NTM interpolation: g * new + (1-g) * old
export
interpolateVar : {n : Nat} -> Variable -> Vector n Variable -> Vector n Variable -> Vector n Variable
interpolateVar {n} g (VTensor newVec) (VTensor oldVec) =
  let newT = vecToTensor newVec 0
      oldT = vecToTensor oldVec 0
      gScaled = prim__mulScalar newT g.value
      oneMinusG = 1.0 - g.value
      oldScaled = prim__mulScalar oldT oneMinusG
      result = prim__add gScaled oldScaled
  in VTensor $ tensorToScalars result 0 n

||| NTM shift: circular convolution of weights with shift kernel.
||| Clamps output to [1e-10, ∞) to prevent negative weights feeding into focusVar.
export
shiftVar : {n : Nat} -> Vector n Variable -> Vector 3 Variable -> Vector n Variable
shiftVar {n} (VTensor wts) (VTensor shifts) =
  let wtTensor = vecStackTensor wts
      shiftTensor = vecStackTensor shifts
      shifted = prim__conv1dCircular wtTensor shiftTensor
      result = prim__clampMin shifted 1.0e-10
  in VTensor $ tensorToScalars result 0 n

||| NTM focus (sharpening): w^gamma / sum(w^gamma)
||| Clamps weights to [1e-10, ∞) before pow to prevent NaN from pow(0/negative, gamma).
export
focusVar : {n : Nat} -> Variable -> Vector n Variable -> Vector n Variable
focusVar {n} gamma (VTensor wts) =
  let wtTensor = vecStackTensor wts
      gammaT = gamma.tensorPtr
      -- Clamp to epsilon before pow (prevents NaN from pow(0, gamma) or pow(negative, gamma))
      clamped = prim__clampMin wtTensor 1.0e-10
      -- w^gamma
      powered = prim__pow clamped gammaT
      -- sum(w^gamma) + epsilon
      powSum = prim__addScalar (prim__sum powered) 1.0e-10
      -- normalize
      result = prim__div powered powSum
  in VTensor $ tensorToScalars result 0 n

||| NTM interpolation write: weights * outer(add) added to memory.
||| This is the simplified write used in the NTM layer (no erase gate).
export
interpolationWriteVar : {n, w : Nat} -> Vector n Variable -> Matrix n w Variable -> Vector w Variable -> Matrix n w Variable
interpolationWriteVar {n} {w} (VTensor wts) (VTensor memRows) (VTensor addElems) =
  let wtTensor = vecStackTensor wts
      memTensor = matStackTensor memRows
      addTensor = vecStackTensor addElems
      -- M' = M + outer(w, a)
      addGate = prim__outer wtTensor addTensor
      result = prim__add memTensor addGate
  in VTensor $ buildMatrixRows result 0 n w
  where
    buildMatrixRows : AnyPtr -> Int -> (rows : Nat) -> (cols : Nat) -> Vect rows (Vector cols Variable)
    buildMatrixRows _ _ Z _ = []
    buildMatrixRows t row (S r) cols =
      let rowTensor = prim__select t 0 row
      in VTensor (tensorToScalars rowTensor 0 cols) :: buildMatrixRows t (row + 1) r cols


----------------------------------------------------------------------
-- LSTM Cell
----------------------------------------------------------------------

||| LSTM cell implemented via Variable arithmetic (autograd-tracked).
||| combined = mulIW + mulRW + bias, then split into 4 gates of size o.
||| newCell = sigmoid(f) * prevCell + sigmoid(i) * tanh(g)
||| newHidden = sigmoid(o) * tanh(newCell)
export
lstmCellVar : {o : Nat} -> Vector (4 * o) Variable -> Vector (4 * o) Variable
           -> Vector (4 * o) Variable -> Vector o Variable
           -> (Vector o Variable, Vector o Variable)
lstmCellVar {o} mulIW mulRW bias prevCell =
  let combined = mulIW + mulRW + bias
      -- Split into 4 gates
      s1 = Tensor.splitAt o combined
      s2 = Tensor.splitAt o (snd s1)
      s3 = Tensor.splitAt o (snd s2)
      iGate = fst s1      -- input gate
      fGate = fst s2      -- forget gate
      gGate = fst s3      -- cell gate
      -- oGate needs coercion: (4*o) - o - o - o may not reduce to o
      oGate : Vector o Variable
      oGate = believe_me (snd s3)
      -- Apply activations and compute new cell/hidden
      newCell = map sigmoidVar fGate * prevCell + map sigmoidVar iGate * map tanhVar gGate
      newHidden = map sigmoidVar oGate * map tanhVar newCell
  in (newCell, newHidden)



----------------------------------------------------------------------
-- Backpropagation (libtorch autograd)
----------------------------------------------------------------------

-- Build gradient map by iterating over the parameter registry.
-- Reads each param's gradient and immediately zeros it (prevents accumulation).
buildGradMap : Int -> Int -> SortedMap String Double -> SortedMap String Double
buildGradMap n i acc = if i >= n then acc
  else let name = prim__paramName i
           grad = prim__paramGradItemAndZero i
       in buildGradMap n (i + 1) (insert name grad acc)

||| Collect gradients via libtorch backward pass.
||| Calls backward() on the loss tensor, then reads .grad() from all
||| registered parameters. Returns gradients keyed by parameter name.
export
-- Fused backward + param count: ensures backward() completes before
-- returning the count (Chez would drop a standalone backward call).
-- Fused backward + zero_grads + param count.
-- Ensures backward() runs, then zero_all_grads() runs, then returns count.
-- All in one Scheme lambda so Chez can't reorder.
%foreign "scheme:(lambda (t) (let ((rg ((foreign-procedure \"tensor_requires_grad\" (void*) int) t))) (when (= rg 1) ((foreign-procedure \"tensor_backward\" (void*) void) t))) (let ((n ((foreign-procedure \"param_count\" () int)))) n))"
prim__backwardAndCount : AnyPtr -> Int

-- Zero all parameter gradients. Returns 0 for threading.
%foreign "scheme:(lambda (dummy) ((foreign-procedure \"param_zero_all_grads\" () void)) 0)"
prim__zeroAllGrads : Int -> Int

export
collectGrads : Double -> Variable -> SortedMap String Double
collectGrads initGrad root =
  let n = prim__backwardAndCount root.tensorPtr
      grads = buildGradMap n 0 empty
  in grads


----------------------------------------------------------------------
-- Native Optimizer
----------------------------------------------------------------------

%foreign "C:optimizer_create_sgd,libidrisml"
prim__optimizerCreateSgd : Double -> AnyPtr

%foreign "C:optimizer_create_rmsprop,libidrisml"
prim__optimizerCreateRmsprop : Double -> Double -> Double -> Double -> Double -> AnyPtr

%foreign "C:optimizer_create_adam,libidrisml"
prim__optimizerCreateAdam : Double -> Double -> Double -> Double -> AnyPtr

-- optimizer_step and optimizer_zero_grad have void return — use Scheme wrappers
-- to return a threading value (prevents dead code elimination).
%foreign "scheme:(lambda (opt) ((foreign-procedure \"optimizer_zero_grad\" (void*) void) opt) opt)"
prim__optimizerZeroGrad : AnyPtr -> AnyPtr

%foreign "scheme:(lambda (opt) ((foreign-procedure \"optimizer_step\" (void*) void) opt) opt)"
prim__optimizerStep : AnyPtr -> AnyPtr

-- backward: same as prim__backward but returns the tensorPtr for threading
%foreign "scheme:(lambda (t) (let ((rg ((foreign-procedure \"tensor_requires_grad\" (void*) int) t))) (when (= rg 1) ((foreign-procedure \"tensor_backward\" (void*) void) t))) t)"
prim__backwardForNative : AnyPtr -> AnyPtr

%foreign "scheme:(lambda (maxVal) ((foreign-procedure \"optimizer_clip_grad_value\" (double) void) maxVal) 0)"
prim__clipGradValue : Double -> Int

%foreign "scheme:(lambda (maxNorm) ((foreign-procedure \"optimizer_clip_grad_norm\" (double) double) maxNorm))"
prim__clipGradNorm : Double -> Double

public export
data ClipMode = NoClip | ValueClip Double | NormClip Double

||| Native libtorch optimizer. Single step() call updates all parameters.
public export
record NativeOptimizer where
  constructor MkNativeOptimizer
  handle : AnyPtr
  clipMode : ClipMode

||| Create a native SGD optimizer.
export
nativeSgd : Double -> NativeOptimizer
nativeSgd lr = MkNativeOptimizer (prim__optimizerCreateSgd lr) NoClip

||| Create a native RMSprop optimizer (matches PyTorch defaults).
export
nativeRmsprop : (lr : Double) -> (alpha : Double) -> (eps : Double) ->
                (clipVal : Double) -> (momentum : Double) -> NativeOptimizer
nativeRmsprop lr alpha eps clipVal momentum =
  MkNativeOptimizer
    (prim__optimizerCreateRmsprop lr alpha eps 0.0 momentum)
    (ValueClip clipVal)

||| Create a native Adam optimizer with global norm clipping.
export
nativeAdamGlobalClip : (lr : Double) -> (beta1 : Double) -> (beta2 : Double) ->
                       (eps : Double) -> (maxNorm : Double) -> NativeOptimizer
nativeAdamGlobalClip lr beta1 beta2 eps maxNorm =
  MkNativeOptimizer
    (prim__optimizerCreateAdam lr beta1 beta2 eps)
    (NormClip maxNorm)

-- Fused native train step: zero_grad → backward → clip → step.
-- All in one Scheme lambda to ensure correct evaluation order.
-- Returns loss value (read before step, so not stale).
%foreign "scheme:(lambda (opt clip-mode clip-val loss-ptr loss-val) ((foreign-procedure \"optimizer_zero_grad\" (void*) void) opt) (let ((rg ((foreign-procedure \"tensor_requires_grad\" (void*) int) loss-ptr))) (when (= rg 1) ((foreign-procedure \"tensor_backward\" (void*) void) loss-ptr))) (cond ((= clip-mode 1) ((foreign-procedure \"optimizer_clip_grad_value\" (double) void) clip-val)) ((= clip-mode 2) ((foreign-procedure \"optimizer_clip_grad_norm\" (double) double) clip-val)) (else (void))) ((foreign-procedure \"optimizer_step\" (void*) void) opt) loss-val)"
prim__nativeTrainStep : AnyPtr -> Int -> Double -> AnyPtr -> Double -> Double

||| Run one native optimizer step: zero_grad → backward → clip → step.
||| Returns the loss value (read before step, so not stale).
export
nativeTrainStep : NativeOptimizer -> Variable -> Double
nativeTrainStep opt loss =
  let clipMode : Int
      clipMode = case opt.clipMode of NoClip => 0; ValueClip _ => 1; NormClip _ => 2
      clipVal : Double
      clipVal = case opt.clipMode of NoClip => 0.0; ValueClip v => v; NormClip v => v
  in prim__nativeTrainStep opt.handle clipMode clipVal loss.tensorPtr loss.value

||| Refresh cached Variable.value from the underlying tensorPtr.
||| Needed after native optimizer step since tensor values changed in-place.
export
refreshValue : Variable -> Variable
refreshValue v = case v.paramId of
  Just _ => { value := prim__item v.tensorPtr } v
  Nothing => v


----------------------------------------------------------------------
-- Parameter Count
----------------------------------------------------------------------

export
getNumPids : Int -> Int
getNumPids _ = prim__paramCount


----------------------------------------------------------------------
-- GC / RSS
----------------------------------------------------------------------

%foreign "C:backend_supports_tensor_params,libidrisml"
export
prim__backendSupportsTensorParams : Int

export
forceGC : IO ()
forceGC = pure ()

%foreign "C:get_rss_mb,libidrisml"
prim__getRssMB : Int

%foreign "C:get_current_rss_mb,libidrisml"
prim__getCurrentRssMB : Int

export
getRssMB : Nat -> Int
getRssMB _ = prim__getRssMB

export
getCurrentRssMB : Nat -> Int
getCurrentRssMB _ = prim__getCurrentRssMB
