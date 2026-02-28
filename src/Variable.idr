module Variable

import Data.List
import Data.Maybe
import Data.SortedMap
import System.Random

import Floating
import Util


----------------------------------------------------------------------
-- Operation Tags
----------------------------------------------------------------------

public export
data TapeOp = ConstOp
            | NegOp | AbsOp | ExpOp | LogOp | SqrtOp
            | AddOp | SubOp | MulOp | DivOp | PowOp

toTag : TapeOp -> Int
toTag ConstOp = 0
toTag NegOp   = 1
toTag AbsOp   = 2
toTag ExpOp   = 3
toTag LogOp   = 4
toTag SqrtOp  = 5
toTag AddOp   = 6
toTag SubOp   = 7
toTag MulOp   = 8
toTag DivOp   = 9
toTag PowOp   = 10

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
fromTag _  = ConstOp


----------------------------------------------------------------------
-- Tape FFI
----------------------------------------------------------------------

-- Each entry-point FFI function includes an init guard:
--   (when (not (top-level-bound? 'tape-gen)) ...)
-- The init also registers tape-ensure-cap! for use by append functions.

-- Get current generation. Self-initializing.
%foreign "scheme:(lambda (dummy) (when (not (top-level-bound? 'tape-gen)) (begin (set-top-level-value! 'tape-tags (make-vector 4096 0)) (set-top-level-value! 'tape-arg1 (make-vector 4096 0)) (set-top-level-value! 'tape-arg2 (make-vector 4096 0)) (set-top-level-value! 'tape-vals (make-vector 4096 0.0)) (set-top-level-value! 'tape-pids (make-vector 4096 \"\")) (set-top-level-value! 'tape-size 0) (set-top-level-value! 'tape-cap 4096) (set-top-level-value! 'tape-gen 0) (set-top-level-value! 'tape-ensure-cap! (lambda (idx) (when (>= idx (top-level-value 'tape-cap)) (let* ((old-cap (top-level-value 'tape-cap)) (new-cap (* 2 old-cap)) (ot (top-level-value 'tape-tags)) (oa (top-level-value 'tape-arg1)) (ob (top-level-value 'tape-arg2)) (ov (top-level-value 'tape-vals)) (op (top-level-value 'tape-pids)) (nt (make-vector new-cap 0)) (na (make-vector new-cap 0)) (nb (make-vector new-cap 0)) (nv (make-vector new-cap 0.0)) (np (make-vector new-cap \"\"))) (vector-copy! nt 0 ot 0 old-cap) (vector-copy! na 0 oa 0 old-cap) (vector-copy! nb 0 ob 0 old-cap) (vector-copy! nv 0 ov 0 old-cap) (vector-copy! np 0 op 0 old-cap) (set-top-level-value! 'tape-tags nt) (set-top-level-value! 'tape-arg1 na) (set-top-level-value! 'tape-arg2 nb) (set-top-level-value! 'tape-vals nv) (set-top-level-value! 'tape-pids np) (set-top-level-value! 'tape-cap new-cap))))))) (top-level-value 'tape-gen))"
prim__tapeGen : Int -> Int

-- Append a const entry. Self-initializing. Flat 2-arg lambda.
%foreign "scheme:(lambda (val pid) (when (not (top-level-bound? 'tape-gen)) (begin (set-top-level-value! 'tape-tags (make-vector 4096 0)) (set-top-level-value! 'tape-arg1 (make-vector 4096 0)) (set-top-level-value! 'tape-arg2 (make-vector 4096 0)) (set-top-level-value! 'tape-vals (make-vector 4096 0.0)) (set-top-level-value! 'tape-pids (make-vector 4096 \"\")) (set-top-level-value! 'tape-size 0) (set-top-level-value! 'tape-cap 4096) (set-top-level-value! 'tape-gen 0) (set-top-level-value! 'tape-ensure-cap! (lambda (idx) (when (>= idx (top-level-value 'tape-cap)) (let* ((old-cap (top-level-value 'tape-cap)) (new-cap (* 2 old-cap)) (ot (top-level-value 'tape-tags)) (oa (top-level-value 'tape-arg1)) (ob (top-level-value 'tape-arg2)) (ov (top-level-value 'tape-vals)) (op (top-level-value 'tape-pids)) (nt (make-vector new-cap 0)) (na (make-vector new-cap 0)) (nb (make-vector new-cap 0)) (nv (make-vector new-cap 0.0)) (np (make-vector new-cap \"\"))) (vector-copy! nt 0 ot 0 old-cap) (vector-copy! na 0 oa 0 old-cap) (vector-copy! nb 0 ob 0 old-cap) (vector-copy! nv 0 ov 0 old-cap) (vector-copy! np 0 op 0 old-cap) (set-top-level-value! 'tape-tags nt) (set-top-level-value! 'tape-arg1 na) (set-top-level-value! 'tape-arg2 nb) (set-top-level-value! 'tape-vals nv) (set-top-level-value! 'tape-pids np) (set-top-level-value! 'tape-cap new-cap))))))) (let ((idx (top-level-value 'tape-size))) ((top-level-value 'tape-ensure-cap!) idx) (vector-set! (top-level-value 'tape-tags) idx 0) (vector-set! (top-level-value 'tape-vals) idx val) (vector-set! (top-level-value 'tape-pids) idx pid) (set-top-level-value! 'tape-size (+ idx 1)) idx))"
prim__tapeAppendConst : Double -> String -> Int

-- Append a unary op. Flat 3-arg lambda. Assumes tape is initialized.
%foreign "scheme:(lambda (tag a1 val) (let ((idx (top-level-value 'tape-size))) ((top-level-value 'tape-ensure-cap!) idx) (vector-set! (top-level-value 'tape-tags) idx tag) (vector-set! (top-level-value 'tape-arg1) idx a1) (vector-set! (top-level-value 'tape-vals) idx val) (vector-set! (top-level-value 'tape-pids) idx \"\") (set-top-level-value! 'tape-size (+ idx 1)) idx))"
prim__tapeAppendUnary : Int -> Int -> Double -> Int

-- Append a binary op. Flat 4-arg lambda. Assumes tape is initialized.
%foreign "scheme:(lambda (tag a1 a2 val) (let ((idx (top-level-value 'tape-size))) ((top-level-value 'tape-ensure-cap!) idx) (vector-set! (top-level-value 'tape-tags) idx tag) (vector-set! (top-level-value 'tape-arg1) idx a1) (vector-set! (top-level-value 'tape-arg2) idx a2) (vector-set! (top-level-value 'tape-vals) idx val) (vector-set! (top-level-value 'tape-pids) idx \"\") (set-top-level-value! 'tape-size (+ idx 1)) idx))"
prim__tapeAppendBinary : Int -> Int -> Int -> Double -> Int

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

-- Mutable gradient array: returns the handle for threading
%foreign "scheme:(lambda (size) (make-vector size 0.0))"
prim__gradAlloc : Int -> AnyPtr

-- gradAdd returns the handle (same pointer) to enable threading
%foreign "scheme:(lambda (handle idx val) (begin (vector-set! handle idx (+ (vector-ref handle idx) val)) handle))"
prim__gradAdd : AnyPtr -> Int -> Double -> AnyPtr

%foreign "scheme:(lambda (handle idx) (vector-ref handle idx))"
prim__gradGet : AnyPtr -> Int -> Double

-- Reset tape (size=0, gen++) and return the given handle.
-- Threading through the handle forces evaluation and correct ordering.
-- Tape entries are still readable after reset (vectors not cleared).
%foreign "scheme:(lambda (handle dummy) (begin (set-top-level-value! 'tape-size 0) (set-top-level-value! 'tape-gen (+ (top-level-value 'tape-gen) 1)) handle))"
prim__resetTapeReturn : AnyPtr -> Int -> AnyPtr


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
