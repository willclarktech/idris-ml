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

toTag : TapeOp -> Int
toTag ConstOp  = 0
toTag NegOp    = 1
toTag AbsOp    = 2
toTag ExpOp    = 3
toTag LogOp    = 4
toTag SqrtOp   = 5
toTag AddOp    = 6
toTag SubOp    = 7
toTag MulOp    = 8
toTag DivOp    = 9
toTag PowOp    = 10
toTag MatVecOp = 11
toTag DotOp    = 12

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
fromTag _  = ConstOp


----------------------------------------------------------------------
-- Tape FFI
----------------------------------------------------------------------

-- Each entry-point FFI function includes an init guard:
--   (when (not (top-level-bound? 'tape-gen)) ...)
-- The init also loads the C shared library and registers tape-ensure-cap!.

-- Init block (shared between prim__tapeGen and prim__tapeAppendConst).
-- Adds a tape-meta vector alongside the existing 5 tape vectors.
-- Loads build/libidrisml.dylib for C tensor ops.
--
-- tape-meta stores AnyPtr (C metadata pointers) for tensor op entries.
-- For scalar ops the slot is unused (#f).

-- Get current generation. Self-initializing.
%foreign "scheme:(lambda (dummy) (when (not (top-level-bound? 'tape-gen)) (begin (load-shared-object \"build/libidrisml.dylib\") (set-top-level-value! 'tape-tags (make-vector 4096 0)) (set-top-level-value! 'tape-arg1 (make-vector 4096 0)) (set-top-level-value! 'tape-arg2 (make-vector 4096 0)) (set-top-level-value! 'tape-vals (make-vector 4096 0.0)) (set-top-level-value! 'tape-pids (make-vector 4096 \"\")) (set-top-level-value! 'tape-meta (make-vector 4096 #f)) (set-top-level-value! 'tape-size 0) (set-top-level-value! 'tape-cap 4096) (set-top-level-value! 'tape-gen 0) (set-top-level-value! 'tape-ensure-cap! (lambda (idx) (when (>= idx (top-level-value 'tape-cap)) (let* ((old-cap (top-level-value 'tape-cap)) (new-cap (* 2 old-cap)) (ot (top-level-value 'tape-tags)) (oa (top-level-value 'tape-arg1)) (ob (top-level-value 'tape-arg2)) (ov (top-level-value 'tape-vals)) (op (top-level-value 'tape-pids)) (om (top-level-value 'tape-meta)) (nt (make-vector new-cap 0)) (na (make-vector new-cap 0)) (nb (make-vector new-cap 0)) (nv (make-vector new-cap 0.0)) (np (make-vector new-cap \"\")) (nm (make-vector new-cap #f))) (vector-copy! nt 0 ot 0 old-cap) (vector-copy! na 0 oa 0 old-cap) (vector-copy! nb 0 ob 0 old-cap) (vector-copy! nv 0 ov 0 old-cap) (vector-copy! np 0 op 0 old-cap) (vector-copy! nm 0 om 0 old-cap) (set-top-level-value! 'tape-tags nt) (set-top-level-value! 'tape-arg1 na) (set-top-level-value! 'tape-arg2 nb) (set-top-level-value! 'tape-vals nv) (set-top-level-value! 'tape-pids np) (set-top-level-value! 'tape-meta nm) (set-top-level-value! 'tape-cap new-cap))))))) (top-level-value 'tape-gen))"
prim__tapeGen : Int -> Int

-- Append a const entry. Self-initializing. Flat 2-arg lambda.
%foreign "scheme:(lambda (val pid) (when (not (top-level-bound? 'tape-gen)) (begin (load-shared-object \"build/libidrisml.dylib\") (set-top-level-value! 'tape-tags (make-vector 4096 0)) (set-top-level-value! 'tape-arg1 (make-vector 4096 0)) (set-top-level-value! 'tape-arg2 (make-vector 4096 0)) (set-top-level-value! 'tape-vals (make-vector 4096 0.0)) (set-top-level-value! 'tape-pids (make-vector 4096 \"\")) (set-top-level-value! 'tape-meta (make-vector 4096 #f)) (set-top-level-value! 'tape-size 0) (set-top-level-value! 'tape-cap 4096) (set-top-level-value! 'tape-gen 0) (set-top-level-value! 'tape-ensure-cap! (lambda (idx) (when (>= idx (top-level-value 'tape-cap)) (let* ((old-cap (top-level-value 'tape-cap)) (new-cap (* 2 old-cap)) (ot (top-level-value 'tape-tags)) (oa (top-level-value 'tape-arg1)) (ob (top-level-value 'tape-arg2)) (ov (top-level-value 'tape-vals)) (op (top-level-value 'tape-pids)) (om (top-level-value 'tape-meta)) (nt (make-vector new-cap 0)) (na (make-vector new-cap 0)) (nb (make-vector new-cap 0)) (nv (make-vector new-cap 0.0)) (np (make-vector new-cap \"\")) (nm (make-vector new-cap #f))) (vector-copy! nt 0 ot 0 old-cap) (vector-copy! na 0 oa 0 old-cap) (vector-copy! nb 0 ob 0 old-cap) (vector-copy! nv 0 ov 0 old-cap) (vector-copy! np 0 op 0 old-cap) (vector-copy! nm 0 om 0 old-cap) (set-top-level-value! 'tape-tags nt) (set-top-level-value! 'tape-arg1 na) (set-top-level-value! 'tape-arg2 nb) (set-top-level-value! 'tape-vals nv) (set-top-level-value! 'tape-pids np) (set-top-level-value! 'tape-meta nm) (set-top-level-value! 'tape-cap new-cap))))))) (let ((idx (top-level-value 'tape-size))) ((top-level-value 'tape-ensure-cap!) idx) (vector-set! (top-level-value 'tape-tags) idx 0) (vector-set! (top-level-value 'tape-vals) idx val) (vector-set! (top-level-value 'tape-pids) idx pid) (set-top-level-value! 'tape-size (+ idx 1)) idx))"
prim__tapeAppendConst : Double -> String -> Int

-- Append a unary op. Flat 3-arg lambda. Assumes tape is initialized.
%foreign "scheme:(lambda (tag a1 val) (let ((idx (top-level-value 'tape-size))) ((top-level-value 'tape-ensure-cap!) idx) (vector-set! (top-level-value 'tape-tags) idx tag) (vector-set! (top-level-value 'tape-arg1) idx a1) (vector-set! (top-level-value 'tape-vals) idx val) (vector-set! (top-level-value 'tape-pids) idx \"\") (set-top-level-value! 'tape-size (+ idx 1)) idx))"
prim__tapeAppendUnary : Int -> Int -> Double -> Int

-- Append a binary op. Flat 4-arg lambda. Assumes tape is initialized.
%foreign "scheme:(lambda (tag a1 a2 val) (let ((idx (top-level-value 'tape-size))) ((top-level-value 'tape-ensure-cap!) idx) (vector-set! (top-level-value 'tape-tags) idx tag) (vector-set! (top-level-value 'tape-arg1) idx a1) (vector-set! (top-level-value 'tape-arg2) idx a2) (vector-set! (top-level-value 'tape-vals) idx val) (vector-set! (top-level-value 'tape-pids) idx \"\") (set-top-level-value! 'tape-size (+ idx 1)) idx))"
prim__tapeAppendBinary : Int -> Int -> Int -> Double -> Int

-- Append a tensor op entry with metadata pointer. Stores tag, output count
-- in arg2, and meta pointer in tape-meta. Returns tape index.
%foreign "scheme:(lambda (tag count meta-ptr) (let ((idx (top-level-value 'tape-size))) ((top-level-value 'tape-ensure-cap!) idx) (vector-set! (top-level-value 'tape-tags) idx tag) (vector-set! (top-level-value 'tape-arg2) idx count) (vector-set! (top-level-value 'tape-meta) idx meta-ptr) (vector-set! (top-level-value 'tape-pids) idx \"\") (set-top-level-value! 'tape-size (+ idx 1)) idx))"
prim__tapeAppendTensorOp : Int -> Int -> AnyPtr -> Int

-- Update tape entry's paramId. Returns idx so result is used in Variable construction.
%foreign "scheme:(lambda (idx pid) (begin (vector-set! (top-level-value 'tape-pids) idx pid) idx))"
prim__tapeSetParamId : Int -> String -> Int

-- Read tape fields (only called during backward, after tape is initialized)
%foreign "scheme:(lambda (idx) (vector-ref (top-level-value 'tape-tags) idx))"
prim__tapeGetTag : Int -> Int

%foreign "scheme:(lambda (idx) (vector-ref (top-level-value 'tape-arg1) idx))"
prim__tapeGetArg1 : Int -> Int

%foreign "scheme:(lambda (idx) (vector-ref (top-level-value 'tape-arg2) idx))"
prim__tapeGetArg2 : Int -> Int

%foreign "scheme:(lambda (idx) (vector-ref (top-level-value 'tape-vals) idx))"
prim__tapeGetValue : Int -> Double

%foreign "scheme:(lambda (idx) (vector-ref (top-level-value 'tape-pids) idx))"
prim__tapeGetParamId : Int -> String

-- Read meta pointer for tensor op entries
%foreign "scheme:(lambda (idx) (vector-ref (top-level-value 'tape-meta) idx))"
prim__tapeGetMeta : Int -> AnyPtr

-- Mutable gradient array (C-backed for tensor backward compatibility)
%foreign "scheme:(lambda (size) (let ((ptr ((foreign-procedure \"grad_alloc\" (int) void*) size))) (when (top-level-bound? 'last-grad-ptr) ((foreign-procedure \"grad_free\" (void*) void) (top-level-value 'last-grad-ptr))) (set-top-level-value! 'last-grad-ptr ptr) ptr))"
prim__gradAlloc : Int -> AnyPtr

-- gradAdd returns the handle (same pointer) to enable threading
%foreign "scheme:(lambda (handle idx val) ((foreign-procedure \"grad_add\" (void* int double) void*) handle idx val))"
prim__gradAdd : AnyPtr -> Int -> Double -> AnyPtr

%foreign "scheme:(lambda (handle idx) ((foreign-procedure \"grad_get\" (void* int) double) handle idx))"
prim__gradGet : AnyPtr -> Int -> Double

-- Reset tape (size=0, gen++), reset arena, and return the given handle.
-- Arena reset is safe here: backward pass reads arena metadata AFTER this call,
-- but arena_reset only sets used=0 without freeing the buffer. Since backward
-- doesn't allocate from the arena, the metadata memory remains valid.
%foreign "scheme:(lambda (handle dummy) (begin (set-top-level-value! 'tape-size 0) (set-top-level-value! 'tape-gen (+ (top-level-value 'tape-gen) 1)) ((foreign-procedure \"arena_reset\" () void)) handle))"
prim__resetTapeReturn : AnyPtr -> Int -> AnyPtr


----------------------------------------------------------------------
-- Tensor C FFI
----------------------------------------------------------------------

-- Buffer allocation (calloc'd, must be freed)
%foreign "scheme:(lambda (n) ((foreign-procedure \"tensor_alloc\" (int) void*) n))"
prim__tensorAlloc : Int -> AnyPtr

%foreign "scheme:(lambda (ptr) (begin ((foreign-procedure \"tensor_free\" (void*) void) ptr) 0))"
prim__tensorFree : AnyPtr -> Int

%foreign "scheme:(lambda (ptr idx) ((foreign-procedure \"tensor_read\" (void* int) double) ptr idx))"
prim__tensorRead : AnyPtr -> Int -> Double

-- MatVec meta: alloc, pack, compute, backward
%foreign "scheme:(lambda (m n) ((foreign-procedure \"matvec_meta_alloc\" (int int) void*) m n))"
prim__matvecMetaAlloc : Int -> Int -> AnyPtr

%foreign "scheme:(lambda (meta idx val tidx) ((foreign-procedure \"matvec_meta_pack_w\" (void* int double int) void*) meta idx val tidx))"
prim__matvecPackW : AnyPtr -> Int -> Double -> Int -> AnyPtr

%foreign "scheme:(lambda (meta idx val tidx) ((foreign-procedure \"matvec_meta_pack_x\" (void* int double int) void*) meta idx val tidx))"
prim__matvecPackX : AnyPtr -> Int -> Double -> Int -> AnyPtr

%foreign "scheme:(lambda (meta start) ((foreign-procedure \"matvec_meta_set_out\" (void* int) void*) meta start))"
prim__matvecSetOut : AnyPtr -> Int -> AnyPtr

%foreign "scheme:(lambda (meta out) ((foreign-procedure \"matvec_meta_compute\" (void* void*) void*) meta out))"
prim__matvecCompute : AnyPtr -> AnyPtr -> AnyPtr

%foreign "scheme:(lambda (g meta) (begin ((foreign-procedure \"tensor_matvec_backward\" (void* void*) void) g meta) g))"
prim__matvecBackward : AnyPtr -> AnyPtr -> AnyPtr

-- Dot meta: alloc, pack, compute, backward
%foreign "scheme:(lambda (n) ((foreign-procedure \"dot_meta_alloc\" (int) void*) n))"
prim__dotMetaAlloc : Int -> AnyPtr

%foreign "scheme:(lambda (meta idx val tidx) ((foreign-procedure \"dot_meta_pack_a\" (void* int double int) void*) meta idx val tidx))"
prim__dotPackA : AnyPtr -> Int -> Double -> Int -> AnyPtr

%foreign "scheme:(lambda (meta idx val tidx) ((foreign-procedure \"dot_meta_pack_b\" (void* int double int) void*) meta idx val tidx))"
prim__dotPackB : AnyPtr -> Int -> Double -> Int -> AnyPtr

%foreign "scheme:(lambda (meta out) ((foreign-procedure \"dot_meta_set_out\" (void* int) void*) meta out))"
prim__dotSetOut : AnyPtr -> Int -> AnyPtr

%foreign "scheme:(lambda (meta) ((foreign-procedure \"dot_meta_compute\" (void*) double) meta))"
prim__dotCompute : AnyPtr -> Double

%foreign "scheme:(lambda (g meta) (begin ((foreign-procedure \"tensor_dot_backward\" (void* void*) void) g meta) g))"
prim__dotBackward : AnyPtr -> AnyPtr -> AnyPtr


----------------------------------------------------------------------
-- Idris Tape Wrappers
----------------------------------------------------------------------

-- Must pass a varying argument to prim__tapeGen to prevent CSE.
-- Zero-arg definitions and constant-body functions are evaluated once.
%noinline
tapeGeneration : Nat -> Nat
tapeGeneration dummy = cast (prim__tapeGen (cast dummy))

%noinline
tapeAppendConst : Double -> String -> Nat
tapeAppendConst val pid = cast (prim__tapeAppendConst val pid)

%noinline
tapeAppendUnary : TapeOp -> Nat -> Double -> Nat
tapeAppendUnary op a1 val = cast (prim__tapeAppendUnary (toTag op) (cast a1) val)

%noinline
tapeAppendBinary : TapeOp -> Nat -> Nat -> Double -> Nat
tapeAppendBinary op a1 a2 val = cast (prim__tapeAppendBinary (toTag op) (cast a1) (cast a2) val)

%noinline
tapeAppendTensorOp : TapeOp -> Int -> AnyPtr -> Nat
tapeAppendTensorOp op count meta = cast (prim__tapeAppendTensorOp (toTag op) count meta)


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

-- Pack a list of Variables into matvec meta weight slots, threading the
-- meta pointer to ensure evaluation ordering.
packWeights : AnyPtr -> Int -> List Variable -> AnyPtr
packWeights meta _ [] = meta
packWeights meta off (v :: vs) =
  let tIdx = ensureOnTape v
      meta' = prim__matvecPackW meta off v.value (cast tIdx)
  in packWeights meta' (off + 1) vs

-- Pack a list of Variables into matvec meta input slots.
packInputs : AnyPtr -> Int -> List Variable -> AnyPtr
packInputs meta _ [] = meta
packInputs meta off (v :: vs) =
  let tIdx = ensureOnTape v
      meta' = prim__matvecPackX meta off v.value (cast tIdx)
  in packInputs meta' (off + 1) vs

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
matrixVectorMultiplyVar {m} {n} weights input =
  let mI = cast {to=Int} m
      nI = cast {to=Int} n
      -- Flatten to lists for packing
      wList = toList (flatten weights)
      xList = toList (flatten input)
      -- Allocate meta (arena) and output buffer (heap)
      meta0 = prim__matvecMetaAlloc mI nI
      outBuf = prim__tensorAlloc mI
      -- Pack weight values and tape indices into meta
      meta1 = packWeights meta0 0 wList
      -- Pack input values and tape indices into meta
      meta2 = packInputs meta1 0 xList
      -- Compute forward: writes results into outBuf
      outBuf' = prim__matvecCompute meta2 outBuf
      -- Append MatVecOp entry to tape (count = m outputs)
      opIdx = tapeAppendTensorOp MatVecOp mI meta2
      -- Set meta's out_tape_start to the MatVecOp tape index
      meta3 = prim__matvecSetOut meta2 (cast opIdx)
      -- Append m ConstOp entries for outputs and build Variables
  in VTensor $ buildOutputScalars outBuf' 0 m

-- Pack variables into dot meta vector a slots.
packDotA : AnyPtr -> Int -> List Variable -> AnyPtr
packDotA meta _ [] = meta
packDotA meta off (v :: vs) =
  let tIdx = ensureOnTape v
      meta' = prim__dotPackA meta off v.value (cast tIdx)
  in packDotA meta' (off + 1) vs

-- Pack variables into dot meta vector b slots.
packDotB : AnyPtr -> Int -> List Variable -> AnyPtr
packDotB meta _ [] = meta
packDotB meta off (v :: vs) =
  let tIdx = ensureOnTape v
      meta' = prim__dotPackB meta off v.value (cast tIdx)
  in packDotB meta' (off + 1) vs

||| Dot product using C BLAS, recording a single DotOp tape entry.
export
dotProductVar : {n : Nat} -> Vector n Variable -> Vector n Variable -> Variable
dotProductVar {n} v1 v2 =
  let nI = cast {to=Int} n
      aList = toList (flatten v1)
      bList = toList (flatten v2)
      -- Allocate meta (arena)
      meta0 = prim__dotMetaAlloc nI
      -- Pack both vectors
      meta1 = packDotA meta0 0 aList
      meta2 = packDotB meta1 0 bList
      -- Compute forward
      val = prim__dotCompute meta2
      -- Append DotOp entry to tape (count = 0, scalar output)
      opIdx = tapeAppendTensorOp DotOp 0 meta2
      -- Append ConstOp for the output scalar
      outIdx = tapeAppendConst val ""
      gen = tapeGeneration outIdx
      -- Set meta's out_tape_idx
      meta3 = prim__dotSetOut meta2 (cast outIdx)
  in Var outIdx gen Nothing val


----------------------------------------------------------------------
-- Backpropagation (tape-based)
----------------------------------------------------------------------

-- Propagate gradient for a single tape entry. Returns updated handle.
propagateEntry : AnyPtr -> Int -> AnyPtr
propagateEntry g idx =
  let grad = prim__gradGet g idx
      tag = prim__tapeGetTag idx
      a1 = prim__tapeGetArg1 idx
      a2 = prim__tapeGetArg2 idx
  in case fromTag tag of
       ConstOp => g
       NegOp   => prim__gradAdd g a1 (negate grad)
       AbsOp   => prim__gradAdd g a1 (grad * signum (prim__tapeGetValue a1))
       ExpOp   => prim__gradAdd g a1 (grad * prim__tapeGetValue idx)
       LogOp   => prim__gradAdd g a1 (grad / prim__tapeGetValue a1)
       SqrtOp  => prim__gradAdd g a1 (grad / (2 * prim__tapeGetValue idx))
       AddOp   => prim__gradAdd (prim__gradAdd g a1 grad) a2 grad
       SubOp   => prim__gradAdd (prim__gradAdd g a1 grad) a2 (negate grad)
       MulOp   => prim__gradAdd (prim__gradAdd g a1 (grad * prim__tapeGetValue a2)) a2 (grad * prim__tapeGetValue a1)
       DivOp   => prim__gradAdd (prim__gradAdd g a1 (grad / prim__tapeGetValue a2)) a2 (negate grad * prim__tapeGetValue a1 / pow (prim__tapeGetValue a2) 2)
       PowOp   => let vx = prim__tapeGetValue a1
                      vy = prim__tapeGetValue a2
                  in prim__gradAdd (prim__gradAdd g a1 (grad * vy * pow vx (vy - 1))) a2 (if vx == 0 then 0 else grad * prim__tapeGetValue idx * log vx)
       MatVecOp => prim__matvecBackward g (prim__tapeGetMeta idx)
       DotOp    => prim__dotBackward g (prim__tapeGetMeta idx)

-- Collect param gradients. Handle g ensures correct eval ordering.
collectParamGrad : AnyPtr -> Int -> SortedMap String Double -> SortedMap String Double
collectParamGrad g idx acc =
  let pid = prim__tapeGetParamId idx
  in if pid == ""
       then acc
       else mergeWith (+) acc (singleton pid (prim__gradGet g idx))

walkBackward : AnyPtr -> Int -> SortedMap String Double -> SortedMap String Double
walkBackward g idx acc =
  if idx < 0 then acc
  else
    let g' = propagateEntry g idx
        acc' = collectParamGrad g' idx acc
    in walkBackward g' (idx - 1) acc'

export
collectGrads : Double -> Variable -> SortedMap String Double
collectGrads initGrad root =
  let size = cast {to=Int} root.tapeIdx + 1
      g = prim__gradAlloc size
      g' = prim__gradAdd g (cast root.tapeIdx) initGrad
      g'' = prim__resetTapeReturn g' size
  in walkBackward g'' (size - 1) empty
