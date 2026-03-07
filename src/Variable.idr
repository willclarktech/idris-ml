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
-- Tape FFI (hybrid: foreign-alloc storage + C backward pass)
----------------------------------------------------------------------

-- Tape storage uses Scheme-allocated C-compatible arrays (foreign-alloc)
-- for tags/arg1/arg2/vals, Scheme vector for pids, and C ext_meta array
-- for meta pointers. Forward pass writes scalar fields via foreign-set!
-- (fast, no FFI boundary crossing) and meta via ext_meta_set C call.
-- Backward pass runs entirely in C via walk_backward_ext.
--
-- The init guard loads build/libidrisml.dylib and allocates tape arrays.
-- Embedded in prim__tapeGen and prim__tapeAppendConst (the two entry points).

%foreign "scheme:(lambda (dummy) (when (not (top-level-bound? 'tape-gen)) (begin (load-shared-object \"build/libidrisml.dylib\") (set-top-level-value! 'tape-tags-fp (foreign-alloc (* 4096 4))) (set-top-level-value! 'tape-arg1-fp (foreign-alloc (* 4096 4))) (set-top-level-value! 'tape-arg2-fp (foreign-alloc (* 4096 4))) (set-top-level-value! 'tape-vals-fp (foreign-alloc (* 4096 8))) (set-top-level-value! 'tape-pids (make-vector 4096 \"\")) (set-top-level-value! 'tape-size 0) (set-top-level-value! 'tape-cap 4096) (set-top-level-value! 'tape-gen 0) (set-top-level-value! 'tape-memcpy (foreign-procedure \"memcpy\" (void* void* size_t) void*)) (set-top-level-value! 'tape-ensure-cap! (lambda (need) (when (>= need (top-level-value 'tape-cap)) (let* ((old-cap (top-level-value 'tape-cap)) (new-cap (let lp ((nc (* 2 old-cap))) (if (> nc need) nc (lp (* 2 nc))))) (mc (top-level-value 'tape-memcpy)) (sz (top-level-value 'tape-size)) (nt (foreign-alloc (* new-cap 4))) (na1 (foreign-alloc (* new-cap 4))) (na2 (foreign-alloc (* new-cap 4))) (nv (foreign-alloc (* new-cap 8))) (np (make-vector new-cap \"\"))) (mc nt (top-level-value 'tape-tags-fp) (* sz 4)) (mc na1 (top-level-value 'tape-arg1-fp) (* sz 4)) (mc na2 (top-level-value 'tape-arg2-fp) (* sz 4)) (mc nv (top-level-value 'tape-vals-fp) (* sz 8)) (vector-copy! np 0 (top-level-value 'tape-pids) 0 sz) (foreign-free (top-level-value 'tape-tags-fp)) (foreign-free (top-level-value 'tape-arg1-fp)) (foreign-free (top-level-value 'tape-arg2-fp)) (foreign-free (top-level-value 'tape-vals-fp)) (set-top-level-value! 'tape-tags-fp nt) (set-top-level-value! 'tape-arg1-fp na1) (set-top-level-value! 'tape-arg2-fp na2) (set-top-level-value! 'tape-vals-fp nv) (set-top-level-value! 'tape-pids np) (set-top-level-value! 'tape-cap new-cap))))))) (top-level-value 'tape-gen))"
prim__tapeGen : Int -> Int

-- Append ConstOp. Self-initializing (same init guard as prim__tapeGen).
%foreign "scheme:(lambda (val pid) (when (not (top-level-bound? 'tape-gen)) (begin (load-shared-object \"build/libidrisml.dylib\") (set-top-level-value! 'tape-tags-fp (foreign-alloc (* 4096 4))) (set-top-level-value! 'tape-arg1-fp (foreign-alloc (* 4096 4))) (set-top-level-value! 'tape-arg2-fp (foreign-alloc (* 4096 4))) (set-top-level-value! 'tape-vals-fp (foreign-alloc (* 4096 8))) (set-top-level-value! 'tape-pids (make-vector 4096 \"\")) (set-top-level-value! 'tape-size 0) (set-top-level-value! 'tape-cap 4096) (set-top-level-value! 'tape-gen 0) (set-top-level-value! 'tape-memcpy (foreign-procedure \"memcpy\" (void* void* size_t) void*)) (set-top-level-value! 'tape-ensure-cap! (lambda (need) (when (>= need (top-level-value 'tape-cap)) (let* ((old-cap (top-level-value 'tape-cap)) (new-cap (let lp ((nc (* 2 old-cap))) (if (> nc need) nc (lp (* 2 nc))))) (mc (top-level-value 'tape-memcpy)) (sz (top-level-value 'tape-size)) (nt (foreign-alloc (* new-cap 4))) (na1 (foreign-alloc (* new-cap 4))) (na2 (foreign-alloc (* new-cap 4))) (nv (foreign-alloc (* new-cap 8))) (np (make-vector new-cap \"\"))) (mc nt (top-level-value 'tape-tags-fp) (* sz 4)) (mc na1 (top-level-value 'tape-arg1-fp) (* sz 4)) (mc na2 (top-level-value 'tape-arg2-fp) (* sz 4)) (mc nv (top-level-value 'tape-vals-fp) (* sz 8)) (vector-copy! np 0 (top-level-value 'tape-pids) 0 sz) (foreign-free (top-level-value 'tape-tags-fp)) (foreign-free (top-level-value 'tape-arg1-fp)) (foreign-free (top-level-value 'tape-arg2-fp)) (foreign-free (top-level-value 'tape-vals-fp)) (set-top-level-value! 'tape-tags-fp nt) (set-top-level-value! 'tape-arg1-fp na1) (set-top-level-value! 'tape-arg2-fp na2) (set-top-level-value! 'tape-vals-fp nv) (set-top-level-value! 'tape-pids np) (set-top-level-value! 'tape-cap new-cap))))))) (let ((idx (top-level-value 'tape-size))) ((top-level-value 'tape-ensure-cap!) idx) (foreign-set! 'integer-32 (top-level-value 'tape-tags-fp) (* idx 4) 0) (foreign-set! 'double (top-level-value 'tape-vals-fp) (* idx 8) val) (vector-set! (top-level-value 'tape-pids) idx pid) (set-top-level-value! 'tape-size (+ idx 1)) idx))"
prim__tapeAppendConst : Double -> String -> Int

-- Append unary op. Assumes tape initialized (prim__tapeGen or prim__tapeAppendConst called first).
%foreign "scheme:(lambda (tag a1 val) (let ((idx (top-level-value 'tape-size))) ((top-level-value 'tape-ensure-cap!) idx) (foreign-set! 'integer-32 (top-level-value 'tape-tags-fp) (* idx 4) tag) (foreign-set! 'integer-32 (top-level-value 'tape-arg1-fp) (* idx 4) a1) (foreign-set! 'double (top-level-value 'tape-vals-fp) (* idx 8) val) (vector-set! (top-level-value 'tape-pids) idx \"\") (set-top-level-value! 'tape-size (+ idx 1)) idx))"
prim__tapeAppendUnary : Int -> Int -> Double -> Int

-- Append binary op.
%foreign "scheme:(lambda (tag a1 a2 val) (let ((idx (top-level-value 'tape-size))) ((top-level-value 'tape-ensure-cap!) idx) (foreign-set! 'integer-32 (top-level-value 'tape-tags-fp) (* idx 4) tag) (foreign-set! 'integer-32 (top-level-value 'tape-arg1-fp) (* idx 4) a1) (foreign-set! 'integer-32 (top-level-value 'tape-arg2-fp) (* idx 4) a2) (foreign-set! 'double (top-level-value 'tape-vals-fp) (* idx 8) val) (vector-set! (top-level-value 'tape-pids) idx \"\") (set-top-level-value! 'tape-size (+ idx 1)) idx))"
prim__tapeAppendBinary : Int -> Int -> Int -> Double -> Int

-- Append tensor op entry. Meta stored in C ext_meta array via ext_meta_set.
-- set_out called via C to record output tape start in the meta struct.
%foreign "scheme:(lambda (tag count meta out) (let ((idx (top-level-value 'tape-size))) ((top-level-value 'tape-ensure-cap!) idx) (foreign-set! 'integer-32 (top-level-value 'tape-tags-fp) (* idx 4) tag) (foreign-set! 'integer-32 (top-level-value 'tape-arg2-fp) (* idx 4) count) ((foreign-procedure \"ext_meta_set\" (int void*) void) idx meta) (vector-set! (top-level-value 'tape-pids) idx \"\") ((foreign-procedure \"tensor_op_set_out\" (int void* int) void*) tag meta idx) (set-top-level-value! 'tape-size (+ idx 1)) out))"
prim__tapeAppendTensorOp : Int -> Int -> AnyPtr -> AnyPtr -> AnyPtr

-- Append DotOp + output ConstOp, set meta->out_tape_idx. Returns ConstOp tape index.
%foreign "scheme:(lambda (meta val) (let ((idx (top-level-value 'tape-size))) ((top-level-value 'tape-ensure-cap!) (+ idx 1)) (foreign-set! 'integer-32 (top-level-value 'tape-tags-fp) (* idx 4) 12) (foreign-set! 'integer-32 (top-level-value 'tape-arg2-fp) (* idx 4) 0) ((foreign-procedure \"ext_meta_set\" (int void*) void) idx meta) (vector-set! (top-level-value 'tape-pids) idx \"\") (let ((out-idx (+ idx 1))) (foreign-set! 'integer-32 (top-level-value 'tape-tags-fp) (* out-idx 4) 0) (foreign-set! 'double (top-level-value 'tape-vals-fp) (* out-idx 8) val) (vector-set! (top-level-value 'tape-pids) out-idx \"\") ((foreign-procedure \"dot_meta_set_out\" (void* int) void*) meta out-idx) (set-top-level-value! 'tape-size (+ out-idx 1)) out-idx)))"
prim__tapeAppendDotOp : AnyPtr -> Double -> Int

-- Update tape entry's paramId. PIDs stored in Scheme vector.
%foreign "scheme:(lambda (idx pid) (begin (vector-set! (top-level-value 'tape-pids) idx pid) idx))"
prim__tapeSetParamId : Int -> String -> Int

-- Gradient array allocation (C-allocated via grad_alloc)
%foreign "scheme:(lambda (n) ((foreign-procedure \"grad_alloc\" (int) void*) n))"
prim__gradAlloc : Int -> AnyPtr

-- gradAdd: Scheme-native read-modify-write on C array (no FFI crossing)
%foreign "scheme:(lambda (handle idx val) (let ((off (* idx 8))) (foreign-set! 'double handle off (+ (foreign-ref 'double handle off) val)) handle))"
prim__gradAdd : AnyPtr -> Int -> Double -> AnyPtr

-- C-backed backward pass. Meta pointers are already in C ext_meta array
-- (written during forward via ext_meta_set). Calls walk_backward_ext which
-- uses ext_meta directly. Resets ext_meta, tape, and gen after backward.
%foreign "scheme:(lambda (g sz) (let ((n ((foreign-procedure \"walk_backward_ext\" (void* int void* void* void* void*) int) g sz (top-level-value 'tape-tags-fp) (top-level-value 'tape-arg1-fp) (top-level-value 'tape-arg2-fp) (top-level-value 'tape-vals-fp)))) ((foreign-procedure \"ext_meta_reset\" () void)) ((foreign-procedure \"arena_reset\" () void)) (set-top-level-value! 'tape-size 0) (set-top-level-value! 'tape-gen (+ (top-level-value 'tape-gen) 1)) n))"
prim__walkBackwardExtAndReset : AnyPtr -> Int -> Int

-- Access collected results from walk_backward_ext: (tape_index, grad) pairs.
-- Caller looks up pid from Scheme tape-pids vector.
%foreign "scheme:(lambda (i) ((foreign-procedure \"result_get_idx\" (int) int) i))"
prim__resultGetIdx : Int -> Int

%foreign "scheme:(lambda (i) ((foreign-procedure \"result_get_val\" (int) double) i))"
prim__resultGetVal : Int -> Double

-- Look up pid from Scheme tape-pids vector by tape index.
%foreign "scheme:(lambda (idx) (vector-ref (top-level-value 'tape-pids) idx))"
prim__tapeGetPid : Int -> String


----------------------------------------------------------------------
-- Weight Buffer FFI (Scheme vector: [C double*, pid vector, cached-start, cached-gen])
----------------------------------------------------------------------

-- Allocate: Scheme 4-vector with C double buffer and pid vector.
export
%foreign "scheme:(lambda (count) (let ((cbuf ((foreign-procedure \"tensor_alloc\" (int) void*) count)) (pids (make-vector count \"\"))) (vector cbuf pids -1 -1)))"
prim__weightBufAlloc : Int -> AnyPtr

-- Get the C double* from a weight buffer.
%foreign "scheme:(lambda (wbuf) (vector-ref wbuf 0))"
prim__weightBufVals : AnyPtr -> AnyPtr

-- Write a double value to the C buffer at index.
%foreign "scheme:(lambda (wbuf idx val) (let ((ptr (vector-ref wbuf 0))) (foreign-set! 'double ptr (* idx 8) val) wbuf))"
prim__weightBufSetVal : AnyPtr -> Int -> Double -> AnyPtr

-- Write a pid string to the pid vector at index.
%foreign "scheme:(lambda (wbuf idx pid) (let ((pids (vector-ref wbuf 1))) (vector-set! pids idx pid) wbuf))"
prim__weightBufSetPid : AnyPtr -> Int -> String -> AnyPtr

-- Ensure weight buffer entries are on tape (epoch-cached).
-- Writes to foreign-alloc tape arrays using foreign-set!.
%foreign "scheme:(lambda (wbuf count) (if (= (vector-ref wbuf 3) (top-level-value 'tape-gen)) (vector-ref wbuf 2) (let* ((cbuf (vector-ref wbuf 0)) (pids (vector-ref wbuf 1)) (start (top-level-value 'tape-size)) (end (+ start count))) ((top-level-value 'tape-ensure-cap!) (- end 1)) (let ((tags-fp (top-level-value 'tape-tags-fp)) (vals-fp (top-level-value 'tape-vals-fp)) (pidv (top-level-value 'tape-pids))) (do ((k 0 (+ k 1))) ((= k count)) (let ((idx (+ start k))) (foreign-set! 'integer-32 tags-fp (* idx 4) 0) (foreign-set! 'double vals-fp (* idx 8) (foreign-ref 'double cbuf (* k 8))) (vector-set! pidv idx (vector-ref pids k))))) (set-top-level-value! 'tape-size end) (vector-set! wbuf 2 start) (vector-set! wbuf 3 (top-level-value 'tape-gen)) start)))"
prim__tapeEnsureBulkConst : AnyPtr -> Int -> Int


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
%foreign "scheme:(lambda (dummy) (top-level-value 'tape-size))"
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
tapeEnsureBulkConst wBuf count = prim__tapeEnsureBulkConst wBuf count

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
-- Backpropagation (hybrid: C backward + Scheme pid lookup)
----------------------------------------------------------------------

-- Build SortedMap from C result buffer (populated by walk_backward_ext).
-- walk_backward_ext collects (tape_index, grad) pairs. We look up the
-- pid from Scheme's tape-pids vector. Non-parameter entries (empty pid)
-- are skipped. Duplicate pids are accumulated with (+) via mergeWith.
buildGradMap : Int -> Int -> SortedMap String Double -> SortedMap String Double
buildGradMap n i acc = if i >= n then acc
  else let tapeIdx = prim__resultGetIdx i
           pid = prim__tapeGetPid tapeIdx
           val = prim__resultGetVal i
       in if pid == ""
            then buildGradMap n (i + 1) acc
            else buildGradMap n (i + 1) (mergeWith (+) acc (singleton pid val))

export
collectGrads : Double -> Variable -> SortedMap String Double
collectGrads initGrad root =
  let size = cast {to=Int} root.tapeIdx + 1
      g = prim__gradAlloc size
      g' = prim__gradAdd g (cast root.tapeIdx) initGrad
      nParams = prim__walkBackwardExtAndReset g' size
  in buildGradMap nParams 0 empty
