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
-- NTM Memory Buffer FFI (persistent C buffer for NTM memory matrix)
----------------------------------------------------------------------

-- NtmMemBuf wrapper: 6-vector [NtmMemBuf*, vals_ptr, pid_vector, count, cached_start, cached_gen]
-- vals_ptr is cached from ntm_mem_vals_ptr for direct Scheme-native reads/writes.

-- Allocate NtmMemBuf: C struct + Scheme pid vector + cached metadata.
export
%foreign "scheme:(lambda (n w) (let* ((count (* n w)) (cptr ((foreign-procedure \"ntm_mem_alloc\" (int int) void*) n w)) (vptr ((foreign-procedure \"ntm_mem_vals_ptr\" (void*) void*) cptr)) (pids (make-vector count \"\"))) (vector cptr vptr pids count -1 -1)))"
prim__ntmMemBufAlloc : Int -> Int -> AnyPtr

-- Set value at index (Scheme-native foreign-set! on cached vals_ptr, no FFI crossing).
%foreign "scheme:(lambda (mb idx val) (foreign-set! 'double (vector-ref mb 1) (* idx 8) val) mb)"
prim__ntmMemBufSetVal : AnyPtr -> Int -> Double -> AnyPtr

-- Set pid at index in Scheme pid vector.
%foreign "scheme:(lambda (mb idx pid) (vector-set! (vector-ref mb 2) idx pid) mb)"
prim__ntmMemBufSetPid : AnyPtr -> Int -> String -> AnyPtr

-- Ensure memory entries are on tape (epoch-cached). Creates ConstOps with pids,
-- updates C tape_idx. Returns the wrapper for threading (forces evaluation).
%foreign "scheme:(lambda (mb count) (if (= (vector-ref mb 5) (top-level-value 'tape-gen)) mb (let* ((cptr (vector-ref mb 0)) (vals-ptr (vector-ref mb 1)) (pids (vector-ref mb 2)) (start (top-level-value 'tape-size)) (end (+ start count))) ((top-level-value 'tape-ensure-cap!) (- end 1)) (let ((tags-fp (top-level-value 'tape-tags-fp)) (vals-fp (top-level-value 'tape-vals-fp)) (pidv (top-level-value 'tape-pids))) (do ((k 0 (+ k 1))) ((= k count)) (let ((idx (+ start k))) (foreign-set! 'integer-32 tags-fp (* idx 4) 0) (foreign-set! 'double vals-fp (* idx 8) (foreign-ref 'double vals-ptr (* k 8))) (vector-set! pidv idx (vector-ref pids k))))) (set-top-level-value! 'tape-size end) (vector-set! mb 4 start) (vector-set! mb 5 (top-level-value 'tape-gen)) ((foreign-procedure \"ntm_mem_update_tape_idx\" (void* int) void) cptr start) mb)))"
prim__ntmMemBufEnsure : AnyPtr -> Int -> AnyPtr

-- Pack memory from NtmMemBuf into BatchCosSim meta (C memcpy). Takes meta and wrapper.
%foreign "scheme:(lambda (meta mb) ((foreign-procedure \"batch_cossim_pack_mem_buf\" (void* void*) void*) meta (vector-ref mb 0)))"
prim__batchCosSimPackMemBuf : AnyPtr -> AnyPtr -> AnyPtr

-- Pack memory from NtmMemBuf into ReadOp meta (C memcpy).
%foreign "scheme:(lambda (meta mb) ((foreign-procedure \"readop_pack_mem_buf\" (void* void*) void*) meta (vector-ref mb 0)))"
prim__readOpPackMemBuf : AnyPtr -> AnyPtr -> AnyPtr

-- Pack memory from NtmMemBuf into InterpWrite meta (C memcpy).
%foreign "scheme:(lambda (meta mb) ((foreign-procedure \"interp_write_pack_mem_buf\" (void* void*) void*) meta (vector-ref mb 0)))"
prim__interpWritePackMemBuf : AnyPtr -> AnyPtr -> AnyPtr

-- Bulk append ConstOps from C output buffer (no pids). Returns start tape index.
%foreign "scheme:(lambda (outBuf count) (let* ((start (top-level-value 'tape-size)) (end (+ start count))) ((top-level-value 'tape-ensure-cap!) (- end 1)) (let ((tags-fp (top-level-value 'tape-tags-fp)) (vals-fp (top-level-value 'tape-vals-fp)) (pidv (top-level-value 'tape-pids))) (do ((k 0 (+ k 1))) ((= k count)) (let ((idx (+ start k))) (foreign-set! 'integer-32 tags-fp (* idx 4) 0) (foreign-set! 'double vals-fp (* idx 8) (foreign-ref 'double outBuf (* k 8))) (vector-set! pidv idx \"\")))) (set-top-level-value! 'tape-size end) start))"
prim__appendOutputConst : AnyPtr -> Int -> Int

-- Bulk append ConstOps from C output buffer with offset. Returns start tape index.
%foreign "scheme:(lambda (outBuf off count) (let* ((start (top-level-value 'tape-size)) (end (+ start count))) ((top-level-value 'tape-ensure-cap!) (- end 1)) (let ((tags-fp (top-level-value 'tape-tags-fp)) (vals-fp (top-level-value 'tape-vals-fp)) (pidv (top-level-value 'tape-pids))) (do ((k 0 (+ k 1))) ((= k count)) (let ((idx (+ start k))) (foreign-set! 'integer-32 tags-fp (* idx 4) 0) (foreign-set! 'double vals-fp (* idx 8) (foreign-ref 'double outBuf (* (+ off k) 8))) (vector-set! pidv idx \"\")))) (set-top-level-value! 'tape-size end) start))"
prim__appendOutputConstOff : AnyPtr -> Int -> Int -> Int

-- After InterpWrite: update buffer vals + tape_idx from output. Returns wrapper.
%foreign "scheme:(lambda (mb outBuf start) (let ((cptr (vector-ref mb 0))) ((foreign-procedure \"ntm_mem_update_vals\" (void* void*) void) cptr outBuf) ((foreign-procedure \"ntm_mem_update_tape_idx\" (void* int) void) cptr start) mb))"
prim__ntmMemBufUpdateAfterWrite : AnyPtr -> AnyPtr -> Int -> AnyPtr

-- Reset cache after applyDeltas + sync. Forces re-registration next epoch.
export
%foreign "scheme:(lambda (mb) (vector-set! mb 4 -1) (vector-set! mb 5 -1) mb)"
prim__ntmMemBufResetCache : AnyPtr -> AnyPtr


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

-- Allocate meta for persistent weight + bias buffer path (fused matvec+bias)
%foreign "scheme:(lambda (m n wptr wstart bptr bstart) ((foreign-procedure \"matvec_meta_alloc_buf_bias\" (int int void* int void* int) void*) m n wptr wstart bptr bstart))"
prim__matvecMetaAllocBufBias : Int -> Int -> AnyPtr -> Int -> AnyPtr -> Int -> AnyPtr

-- Buffer-passing helper: copy values + set contiguous tape indices (one C call)
%foreign "scheme:(lambda (dv dt sb ts n) ((foreign-procedure \"buf_to_meta\" (void* void* void* int int) void*) dv dt sb ts n))"
prim__bufToMeta : AnyPtr -> AnyPtr -> AnyPtr -> Int -> Int -> AnyPtr

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

-- InterpolateOp meta (tag 21): alloc, get internal pointers, set g, compute
%foreign "scheme:(lambda (n) ((foreign-procedure \"interpolate_meta_alloc\" (int) void*) n))"
prim__interpolateMetaAlloc : Int -> AnyPtr
%foreign "scheme:(lambda (meta) ((foreign-procedure \"interpolate_meta_content_vals\" (void*) void*) meta))"
prim__interpolateContentVals : AnyPtr -> AnyPtr
%foreign "scheme:(lambda (meta) ((foreign-procedure \"interpolate_meta_content_tape\" (void*) void*) meta))"
prim__interpolateContentTape : AnyPtr -> AnyPtr
%foreign "scheme:(lambda (meta) ((foreign-procedure \"interpolate_meta_prev_vals\" (void*) void*) meta))"
prim__interpolatePrevVals : AnyPtr -> AnyPtr
%foreign "scheme:(lambda (meta) ((foreign-procedure \"interpolate_meta_prev_tape\" (void*) void*) meta))"
prim__interpolatePrevTape : AnyPtr -> AnyPtr
%foreign "scheme:(lambda (meta val tidx) ((foreign-procedure \"interpolate_meta_set_g\" (void* double int) void*) meta val tidx))"
prim__interpolateSetG : AnyPtr -> Double -> Int -> AnyPtr
%foreign "scheme:(lambda (meta out) ((foreign-procedure \"interpolate_compute\" (void* void*) void*) meta out))"
prim__interpolateCompute : AnyPtr -> AnyPtr -> AnyPtr

-- ShiftOp meta (tag 22): alloc, get internal pointers, compute
%foreign "scheme:(lambda (n) ((foreign-procedure \"shift_meta_alloc\" (int) void*) n))"
prim__shiftMetaAlloc : Int -> AnyPtr
%foreign "scheme:(lambda (meta) ((foreign-procedure \"shift_meta_input_vals\" (void*) void*) meta))"
prim__shiftInputVals : AnyPtr -> AnyPtr
%foreign "scheme:(lambda (meta) ((foreign-procedure \"shift_meta_input_tape\" (void*) void*) meta))"
prim__shiftInputTape : AnyPtr -> AnyPtr
%foreign "scheme:(lambda (meta) ((foreign-procedure \"shift_meta_kernel_vals\" (void*) void*) meta))"
prim__shiftKernelVals : AnyPtr -> AnyPtr
%foreign "scheme:(lambda (meta) ((foreign-procedure \"shift_meta_kernel_tape\" (void*) void*) meta))"
prim__shiftKernelTape : AnyPtr -> AnyPtr
%foreign "scheme:(lambda (meta out) ((foreign-procedure \"shift_compute\" (void* void*) void*) meta out))"
prim__shiftCompute : AnyPtr -> AnyPtr -> AnyPtr

-- FocusOp meta (tag 23): alloc, get internal pointers, set gamma, compute
%foreign "scheme:(lambda (n) ((foreign-procedure \"focus_meta_alloc\" (int) void*) n))"
prim__focusMetaAlloc : Int -> AnyPtr
%foreign "scheme:(lambda (meta) ((foreign-procedure \"focus_meta_input_vals\" (void*) void*) meta))"
prim__focusInputVals : AnyPtr -> AnyPtr
%foreign "scheme:(lambda (meta) ((foreign-procedure \"focus_meta_input_tape\" (void*) void*) meta))"
prim__focusInputTape : AnyPtr -> AnyPtr
%foreign "scheme:(lambda (meta val tidx) ((foreign-procedure \"focus_meta_set_gamma\" (void* double int) void*) meta val tidx))"
prim__focusSetGamma : AnyPtr -> Double -> Int -> AnyPtr
%foreign "scheme:(lambda (meta out) ((foreign-procedure \"focus_compute\" (void* void*) void*) meta out))"
prim__focusCompute : AnyPtr -> AnyPtr -> AnyPtr

-- LstmCellOp meta (tag 24): alloc, get internal pointers, compute
%foreign "scheme:(lambda (o) ((foreign-procedure \"lstm_cell_meta_alloc\" (int) void*) o))"
prim__lstmCellMetaAlloc : Int -> AnyPtr
-- Allocate meta for bias WeightBuf path (no bias packing needed)
%foreign "scheme:(lambda (o bptr bstart) ((foreign-procedure \"lstm_cell_meta_alloc_buf\" (int void* int) void*) o bptr bstart))"
prim__lstmCellMetaAllocBuf : Int -> AnyPtr -> Int -> AnyPtr
%foreign "scheme:(lambda (meta) ((foreign-procedure \"lstm_cell_meta_muliw_vals\" (void*) void*) meta))"
prim__lstmCellMulIWVals : AnyPtr -> AnyPtr
%foreign "scheme:(lambda (meta) ((foreign-procedure \"lstm_cell_meta_muliw_tape\" (void*) void*) meta))"
prim__lstmCellMulIWTape : AnyPtr -> AnyPtr
%foreign "scheme:(lambda (meta) ((foreign-procedure \"lstm_cell_meta_mulrw_vals\" (void*) void*) meta))"
prim__lstmCellMulRWVals : AnyPtr -> AnyPtr
%foreign "scheme:(lambda (meta) ((foreign-procedure \"lstm_cell_meta_mulrw_tape\" (void*) void*) meta))"
prim__lstmCellMulRWTape : AnyPtr -> AnyPtr
%foreign "scheme:(lambda (meta) ((foreign-procedure \"lstm_cell_meta_bias_vals\" (void*) void*) meta))"
prim__lstmCellBiasVals : AnyPtr -> AnyPtr
%foreign "scheme:(lambda (meta) ((foreign-procedure \"lstm_cell_meta_bias_tape\" (void*) void*) meta))"
prim__lstmCellBiasTape : AnyPtr -> AnyPtr
%foreign "scheme:(lambda (meta) ((foreign-procedure \"lstm_cell_meta_prevcell_vals\" (void*) void*) meta))"
prim__lstmCellPrevCellVals : AnyPtr -> AnyPtr
%foreign "scheme:(lambda (meta) ((foreign-procedure \"lstm_cell_meta_prevcell_tape\" (void*) void*) meta))"
prim__lstmCellPrevCellTape : AnyPtr -> AnyPtr
%foreign "scheme:(lambda (meta out) ((foreign-procedure \"lstm_cell_compute\" (void* void*) void*) meta out))"
prim__lstmCellCompute : AnyPtr -> AnyPtr -> AnyPtr


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

%noinline
tapeAppendInterpolateOp : Int -> AnyPtr -> AnyPtr -> AnyPtr
tapeAppendInterpolateOp count meta outBuf = prim__tapeAppendTensorOp 21 count meta outBuf

%noinline
tapeAppendShiftOp : Int -> AnyPtr -> AnyPtr -> AnyPtr
tapeAppendShiftOp count meta outBuf = prim__tapeAppendTensorOp 22 count meta outBuf

%noinline
tapeAppendFocusOp : Int -> AnyPtr -> AnyPtr -> AnyPtr
tapeAppendFocusOp count meta outBuf = prim__tapeAppendTensorOp 23 count meta outBuf

%noinline
tapeAppendLstmCellOp : Int -> AnyPtr -> AnyPtr -> AnyPtr
tapeAppendLstmCellOp count meta outBuf = prim__tapeAppendTensorOp 24 count meta outBuf


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

-- Build k Variables from sequential tape indices and a C buffer.
-- constStart is the first tape index, off is the buffer offset.
buildVarsFromBuf : AnyPtr -> Int -> Nat -> Nat -> (k : Nat) -> Vect k (Scalar Variable)
buildVarsFromBuf outBuf off constStart gen Z = []
buildVarsFromBuf outBuf off constStart gen (S k) =
  let val = prim__tensorRead outBuf off
  in STensor (Var constStart gen Nothing val)
     :: buildVarsFromBuf outBuf (off + 1) (S constStart) gen k

-- Build k output Scalars by bulk-appending ConstOp entries in one FFI call,
-- then reading values from the C buffer. Much faster than per-element tapeAppendConst.
buildOutputScalars : AnyPtr -> Int -> (k : Nat) -> Vect k (Scalar Variable)
buildOutputScalars outBuf off Z = []
buildOutputScalars outBuf off (S k) =
  let count = cast {to=Int} (S k)
      constStart = prim__appendOutputConstOff outBuf off count
      gen = tapeGeneration (cast constStart)
  in buildVarsFromBuf outBuf off (cast constStart) (cast gen) (S k)

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


||| Matrix-vector multiply with fused bias using persistent weight + bias buffers.
||| Eliminates per-element bias AddOp entries by fusing bias into the C matmul kernel.
export
matrixVectorMultiplyVarBufBias : {m, n : Nat} -> AnyPtr -> AnyPtr -> Vector n Variable -> Vector m Variable
matrixVectorMultiplyVarBufBias {m} {n} wBuf bBuf (VTensor xs) =
  let mI = cast {to=Int} m
      nI = cast {to=Int} n
      -- Ensure weights and bias on tape (cached within epoch)
      wTapeStart = tapeEnsureBulkConst wBuf (mI * nI)
      bTapeStart = tapeEnsureBulkConst bBuf mI
      -- Allocate meta with fused bias
      wValsPtr = prim__weightBufVals wBuf
      bValsPtr = prim__weightBufVals bBuf
      meta = prim__matvecMetaAllocBufBias mI nI wValsPtr wTapeStart bValsPtr bTapeStart
      outBuf = prim__tensorAlloc mI
      -- Pack input values and tape indices
      xvPtr = prim__matvecXVals meta
      xtPtr = prim__matvecXTape meta
      xvPtr' = packVec xvPtr xtPtr 0 xs
      -- Compute forward (matmul + bias addition)
      outBuf' = prim__matvecCompute meta (prim__seq xvPtr' outBuf)
      -- Append MatVecOp entry
      outBuf'' = tapeAppendMatVecOp mI meta outBuf'
  in VTensor $ buildOutputScalars outBuf'' 0 m


||| Matrix-vector multiply returning raw buffer + tape start instead of Variables.
||| Used for buffer-passing to the next chained tensor op (e.g., LstmCell).
||| Returns (outBuf, constStart) where constStart is the first output ConstOp index.
export
matrixVectorMultiplyVarBufOut : {m, n : Nat} -> AnyPtr -> Vector n Variable -> (AnyPtr, Int)
matrixVectorMultiplyVarBufOut {m} {n} wBuf (VTensor xs) =
  let mI = cast {to=Int} m
      nI = cast {to=Int} n
      wTapeStart = tapeEnsureBulkConst wBuf (mI * nI)
      wValsPtr = prim__weightBufVals wBuf
      meta = prim__matvecMetaAllocBuf mI nI wValsPtr wTapeStart
      outBuf = prim__tensorAlloc mI
      xvPtr = prim__matvecXVals meta
      xtPtr = prim__matvecXTape meta
      xvPtr' = packVec xvPtr xtPtr 0 xs
      outBuf' = prim__matvecCompute meta (prim__seq xvPtr' outBuf)
      outBuf'' = tapeAppendMatVecOp mI meta outBuf'
      constStart = prim__appendOutputConst outBuf'' mI
  in (outBuf'', constStart)


----------------------------------------------------------------------
-- Weight Buffer Helpers
----------------------------------------------------------------------

-- Write initial values and pids from a matrix of Variables into a weight buffer.
-- Returns the buffer pointer for threading.
export
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
export
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


----------------------------------------------------------------------
-- NTM Memory Buffer Helpers
----------------------------------------------------------------------

-- Write initial values and pids from named memory Variables into NtmMemBuf.
initNtmMemBufRow : AnyPtr -> Int -> Vect k (Scalar Variable) -> AnyPtr
initNtmMemBufRow mb _ [] = mb
initNtmMemBufRow mb off (STensor v :: rest) =
  let mb' = prim__ntmMemBufSetVal mb off v.value
      mb'' = prim__ntmMemBufSetPid mb' off (fromMaybe "" v.paramId)
  in initNtmMemBufRow mb'' (off + 1) rest

export
initNtmMemBuf : AnyPtr -> Int -> {w : Nat} -> Vect n (Vector w Variable) -> AnyPtr
initNtmMemBuf mb _ {n=Z} [] = mb
initNtmMemBuf mb off {n=S k} {w} (VTensor row :: rows) =
  let mb' = initNtmMemBufRow mb off row
  in initNtmMemBuf mb' (off + cast {to=Int} w) rows

-- Sync updated Variable values into NtmMemBuf after applyDeltas.
syncNtmMemBufRow : AnyPtr -> Int -> Vect k (Scalar Variable) -> AnyPtr
syncNtmMemBufRow mb _ [] = mb
syncNtmMemBufRow mb off (STensor v :: rest) =
  let mb' = prim__ntmMemBufSetVal mb off v.value
  in syncNtmMemBufRow mb' (off + 1) rest

export
syncNtmMemBuf : AnyPtr -> Int -> {w : Nat} -> Vect n (Vector w Variable) -> AnyPtr
syncNtmMemBuf mb _ {n=Z} [] = mb
syncNtmMemBuf mb off {n=S k} {w} (VTensor row :: rows) =
  let mb' = syncNtmMemBufRow mb off row
  in syncNtmMemBuf mb' (off + cast {to=Int} w) rows


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

||| Softmax returning raw buffer + tape start instead of Variables.
||| Used for buffer-passing chains (e.g., addressing pipeline).
export
softmaxVarBufOut : {n : Nat} -> Vector n Variable -> (AnyPtr, Int)
softmaxVarBufOut {n} (VTensor xs) =
  let nI = cast {to=Int} n
      meta = prim__softmaxMetaAlloc nI
      outBuf = prim__tensorAlloc nI
      xvPtr = prim__softmaxXVals meta
      xtPtr = prim__softmaxXTape meta
      xvPtr' = packVec xvPtr xtPtr 0 xs
      outBuf' = prim__softmaxCompute meta (prim__seq xvPtr' outBuf)
      outBuf'' = tapeAppendSoftmaxOp (toTag SoftmaxOp) nI meta outBuf'
      constStart = prim__appendOutputConst outBuf'' nI
  in (outBuf'', constStart)

||| Softmax with buffer input and buffer output (full buffer-passing).
||| Input comes from a preceding buffer-passing op via (outBuf, constStart).
export
softmaxVarBufIO : {n : Nat} -> (AnyPtr, Int) -> (AnyPtr, Int)
softmaxVarBufIO {n} (srcBuf, srcStart) =
  let nI = cast {to=Int} n
      meta = prim__softmaxMetaAlloc nI
      outBuf = prim__tensorAlloc nI
      xvPtr = prim__softmaxXVals meta
      xtPtr = prim__softmaxXTape meta
      xvPtr' = prim__bufToMeta xvPtr xtPtr srcBuf srcStart nI
      outBuf' = prim__softmaxCompute meta (prim__seq xvPtr' outBuf)
      outBuf'' = tapeAppendSoftmaxOp (toTag SoftmaxOp) nI meta outBuf'
      constStart = prim__appendOutputConst outBuf'' nI
  in (outBuf'', constStart)

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
-- NTM Memory Operations (buffer-aware, using persistent NtmMemBuf)
----------------------------------------------------------------------

||| Batch cosine similarity using persistent memory buffer.
||| Memory is packed via C memcpy instead of iterating n*w Variables.
export
batchCosineSimilarityVarBuf : {n, w : Nat} -> Variable -> AnyPtr -> Vector w Variable -> Vector n Variable
batchCosineSimilarityVarBuf {n} {w} beta memBuf (VTensor keyElems) =
  let nI = cast {to=Int} n
      wI = cast {to=Int} w
      -- Ensure memory on tape (epoch-cached, returns wrapper for threading)
      mb' = prim__ntmMemBufEnsure memBuf (nI * wI)
      meta = prim__batchCosSimMetaAlloc nI wI
      outBuf = prim__tensorAlloc nI
      -- Pack memory from buffer (C memcpy, dependency on mb')
      meta' = prim__batchCosSimPackMemBuf meta mb'
      -- Pack key (Scheme-native)
      kvPtr = prim__batchCosSimKeyVals meta'
      ktPtr = prim__batchCosSimKeyTape meta'
      kvPtr' = packVec kvPtr ktPtr 0 keyElems
      -- Set beta
      betaIdx = ensureOnTape beta
      meta'' = prim__batchCosSimSetBeta (prim__seq kvPtr' meta') beta.value (cast betaIdx)
      -- Compute
      outBuf' = prim__batchCosSimCompute meta'' outBuf
      outBuf'' = tapeAppendBatchCosSimOp nI meta'' outBuf'
  in VTensor $ buildOutputScalars outBuf'' 0 n

||| Batch cosine similarity using persistent memory buffer, returning raw buffer.
||| Used for buffer-passing chains (addressing pipeline).
export
batchCosineSimilarityVarBufBufOut : {n, w : Nat} -> Variable -> AnyPtr -> Vector w Variable -> (AnyPtr, Int)
batchCosineSimilarityVarBufBufOut {n} {w} beta memBuf (VTensor keyElems) =
  let nI = cast {to=Int} n
      wI = cast {to=Int} w
      mb' = prim__ntmMemBufEnsure memBuf (nI * wI)
      meta = prim__batchCosSimMetaAlloc nI wI
      outBuf = prim__tensorAlloc nI
      meta' = prim__batchCosSimPackMemBuf meta mb'
      kvPtr = prim__batchCosSimKeyVals meta'
      ktPtr = prim__batchCosSimKeyTape meta'
      kvPtr' = packVec kvPtr ktPtr 0 keyElems
      betaIdx = ensureOnTape beta
      meta'' = prim__batchCosSimSetBeta (prim__seq kvPtr' meta') beta.value (cast betaIdx)
      outBuf' = prim__batchCosSimCompute meta'' outBuf
      outBuf'' = tapeAppendBatchCosSimOp nI meta'' outBuf'
      constStart = prim__appendOutputConst outBuf'' nI
  in (outBuf'', constStart)

||| Read operation using persistent memory buffer.
export
readOpVarBuf : {n, w : Nat} -> Vector n Variable -> AnyPtr -> Vector w Variable
readOpVarBuf {n} {w} (VTensor weightElems) memBuf =
  let nI = cast {to=Int} n
      wI = cast {to=Int} w
      -- Ensure memory on tape
      mb' = prim__ntmMemBufEnsure memBuf (nI * wI)
      meta = prim__readOpMetaAlloc nI wI
      outBuf = prim__tensorAlloc wI
      -- Pack memory from buffer (C memcpy)
      meta' = prim__readOpPackMemBuf meta mb'
      -- Pack weights (Scheme-native)
      wvPtr = prim__readOpWeightVals meta'
      wtPtr = prim__readOpWeightTape meta'
      wvPtr' = packVec wvPtr wtPtr 0 weightElems
      -- Compute
      outBuf' = prim__readOpCompute meta' (prim__seq wvPtr' outBuf)
      outBuf'' = tapeAppendReadOpOp wI meta' outBuf'
  in VTensor $ buildOutputScalars outBuf'' 0 w

||| Interpolation write using persistent memory buffer.
||| Mutates the buffer in-place (updates vals and tape_idx).
||| Returns the buffer wrapper for threading.
export
interpolationWriteVarBuf : {n, w : Nat} -> Vector n Variable -> AnyPtr -> Vector w Variable -> AnyPtr
interpolationWriteVarBuf {n} {w} (VTensor weightElems) memBuf (VTensor addElems) =
  let nI = cast {to=Int} n
      wI = cast {to=Int} w
      nw = nI * wI
      -- Ensure memory on tape
      mb' = prim__ntmMemBufEnsure memBuf nw
      meta = prim__interpWriteMetaAlloc nI wI
      outBuf = prim__tensorAlloc nw
      -- Pack memory from buffer (C memcpy)
      meta' = prim__interpWritePackMemBuf meta mb'
      -- Pack weight and add vectors (Scheme-native)
      wvPtr = prim__interpWriteWeightVals meta'
      wtPtr = prim__interpWriteWeightTape meta'
      avPtr = prim__interpWriteAddVals meta'
      atPtr = prim__interpWriteAddTape meta'
      wvPtr' = packVec wvPtr wtPtr 0 weightElems
      avPtr' = packVec (prim__seq wvPtr' avPtr) atPtr 0 addElems
      -- Compute
      outBuf' = prim__interpWriteCompute meta' (prim__seq avPtr' outBuf)
      -- Append InterpWriteOp tape entry
      outBuf'' = tapeAppendInterpWriteOp nw meta' outBuf'
      -- Append output ConstOps (bulk, no pids)
      outputStart = prim__appendOutputConst outBuf'' nw
      -- Update buffer vals and tape_idx
      mb'' = prim__ntmMemBufUpdateAfterWrite memBuf outBuf'' outputStart
  in mb''


----------------------------------------------------------------------
-- C-backed addressing operations
----------------------------------------------------------------------

||| C-backed interpolation: out[i] = g * content[i] + (1-g) * prev[i]
||| Records a single InterpolateOp tape entry (tag 21).
export
interpolateVar : {n : Nat} -> Variable -> Vector n Variable -> Vector n Variable -> Vector n Variable
interpolateVar {n} g (VTensor contentElems) (VTensor prevElems) =
  let nI = cast {to=Int} n
      meta = prim__interpolateMetaAlloc nI
      outBuf = prim__tensorAlloc nI
      -- Pack content vector
      cvPtr = prim__interpolateContentVals meta
      ctPtr = prim__interpolateContentTape meta
      cvPtr' = packVec cvPtr ctPtr 0 contentElems
      -- Pack prev vector
      pvPtr = prim__interpolatePrevVals (prim__seq cvPtr' meta)
      ptPtr = prim__interpolatePrevTape meta
      pvPtr' = packVec pvPtr ptPtr 0 prevElems
      -- Set scalar g
      gIdx = ensureOnTape g
      meta' = prim__interpolateSetG (prim__seq pvPtr' meta) g.value (cast gIdx)
      -- Compute
      outBuf' = prim__interpolateCompute meta' outBuf
      outBuf'' = tapeAppendInterpolateOp nI meta' outBuf'
  in VTensor $ buildOutputScalars outBuf'' 0 n

||| Interpolation with buffer content input and buffer output.
||| Content from buffer-passing chain, prev from Variables (previous state).
export
interpolateVarBufIO : {n : Nat} -> Variable -> (AnyPtr, Int) -> Vector n Variable -> (AnyPtr, Int)
interpolateVarBufIO {n} g (contentBuf, contentStart) (VTensor prevElems) =
  let nI = cast {to=Int} n
      meta = prim__interpolateMetaAlloc nI
      outBuf = prim__tensorAlloc nI
      -- Pack content from buffer (1 C call)
      cvPtr = prim__interpolateContentVals meta
      ctPtr = prim__interpolateContentTape meta
      cvPtr' = prim__bufToMeta cvPtr ctPtr contentBuf contentStart nI
      -- Pack prev from Variables
      pvPtr = prim__interpolatePrevVals (prim__seq cvPtr' meta)
      ptPtr = prim__interpolatePrevTape meta
      pvPtr' = packVec pvPtr ptPtr 0 prevElems
      -- Set scalar g
      gIdx = ensureOnTape g
      meta' = prim__interpolateSetG (prim__seq pvPtr' meta) g.value (cast gIdx)
      -- Compute
      outBuf' = prim__interpolateCompute meta' outBuf
      outBuf'' = tapeAppendInterpolateOp nI meta' outBuf'
      constStart = prim__appendOutputConst outBuf'' nI
  in (outBuf'', constStart)

||| C-backed circular shift: out[i] = k[0]*in[(i+1)%n] + k[1]*in[i] + k[2]*in[(i-1)%n]
||| Kernel must already be softmax'd. Records a single ShiftOp tape entry (tag 22).
export
shiftVar : {n : Nat} -> Vector n Variable -> Vector 3 Variable -> Vector n Variable
shiftVar {n} (VTensor inputElems) (VTensor kernelElems) =
  let nI = cast {to=Int} n
      meta = prim__shiftMetaAlloc nI
      outBuf = prim__tensorAlloc nI
      -- Pack input vector
      ivPtr = prim__shiftInputVals meta
      itPtr = prim__shiftInputTape meta
      ivPtr' = packVec ivPtr itPtr 0 inputElems
      -- Pack kernel (3 elements)
      kvPtr = prim__shiftKernelVals (prim__seq ivPtr' meta)
      ktPtr = prim__shiftKernelTape meta
      kvPtr' = packVec kvPtr ktPtr 0 kernelElems
      -- Compute
      outBuf' = prim__shiftCompute (prim__seq kvPtr' meta) outBuf
      outBuf'' = tapeAppendShiftOp nI meta outBuf'
  in VTensor $ buildOutputScalars outBuf'' 0 n

||| Circular shift with buffer inputs and buffer output (full buffer-passing).
||| Both input and kernel come from preceding buffer-passing ops.
export
shiftVarBufIO : {n : Nat} -> (AnyPtr, Int) -> (AnyPtr, Int) -> (AnyPtr, Int)
shiftVarBufIO {n} (inputBuf, inputStart) (kernelBuf, kernelStart) =
  let nI = cast {to=Int} n
      meta = prim__shiftMetaAlloc nI
      outBuf = prim__tensorAlloc nI
      -- Pack input from buffer
      ivPtr = prim__shiftInputVals meta
      itPtr = prim__shiftInputTape meta
      ivPtr' = prim__bufToMeta ivPtr itPtr inputBuf inputStart nI
      -- Pack kernel from buffer (3 elements)
      kvPtr = prim__shiftKernelVals (prim__seq ivPtr' meta)
      ktPtr = prim__shiftKernelTape meta
      kvPtr' = prim__bufToMeta kvPtr ktPtr kernelBuf kernelStart 3
      -- Compute
      outBuf' = prim__shiftCompute (prim__seq kvPtr' meta) outBuf
      outBuf'' = tapeAppendShiftOp nI meta outBuf'
      constStart = prim__appendOutputConst outBuf'' nI
  in (outBuf'', constStart)

||| C-backed focus/sharpening: out[i] = in[i]^gamma / sum(in[k]^gamma)
||| Records a single FocusOp tape entry (tag 23).
export
focusVar : {n : Nat} -> Variable -> Vector n Variable -> Vector n Variable
focusVar {n} gamma (VTensor inputElems) =
  let nI = cast {to=Int} n
      meta = prim__focusMetaAlloc nI
      outBuf = prim__tensorAlloc nI
      -- Pack input vector
      ivPtr = prim__focusInputVals meta
      itPtr = prim__focusInputTape meta
      ivPtr' = packVec ivPtr itPtr 0 inputElems
      -- Set scalar gamma
      gammaIdx = ensureOnTape gamma
      meta' = prim__focusSetGamma (prim__seq ivPtr' meta) gamma.value (cast gammaIdx)
      -- Compute
      outBuf' = prim__focusCompute meta' outBuf
      outBuf'' = tapeAppendFocusOp nI meta' outBuf'
  in VTensor $ buildOutputScalars outBuf'' 0 n

||| Focus/sharpening with buffer input, materializing output as Variables.
||| Used at the end of addressing chain where result is stored as state.
export
focusVarFromBuf : {n : Nat} -> Variable -> (AnyPtr, Int) -> Vector n Variable
focusVarFromBuf {n} gamma (inputBuf, inputStart) =
  let nI = cast {to=Int} n
      meta = prim__focusMetaAlloc nI
      outBuf = prim__tensorAlloc nI
      -- Pack input from buffer (1 C call)
      ivPtr = prim__focusInputVals meta
      itPtr = prim__focusInputTape meta
      ivPtr' = prim__bufToMeta ivPtr itPtr inputBuf inputStart nI
      -- Set scalar gamma
      gammaIdx = ensureOnTape gamma
      meta' = prim__focusSetGamma (prim__seq ivPtr' meta) gamma.value (cast gammaIdx)
      -- Compute
      outBuf' = prim__focusCompute meta' outBuf
      outBuf'' = tapeAppendFocusOp nI meta' outBuf'
  in VTensor $ buildOutputScalars outBuf'' 0 n


----------------------------------------------------------------------
-- C-backed LSTM cell operation
----------------------------------------------------------------------

||| Fused LSTM cell: combines bias add, gate activations, and cell/hidden
||| update into a single LstmCellOp tape entry (tag 24).
||| Inputs: mulIW (iW×x result), mulRW (rW×h result), bias, prevCell
||| Outputs: (newCell, newHidden) as separate vectors
export
lstmCellVar : {o : Nat} -> Vector (4 * o) Variable -> Vector (4 * o) Variable
           -> Vector (4 * o) Variable -> Vector o Variable
           -> (Vector o Variable, Vector o Variable)
lstmCellVar {o} (VTensor mulIWElems) (VTensor mulRWElems) (VTensor biasElems) (VTensor prevCellElems) =
  let oI = cast {to=Int} o
      fo = 4 * oI
      twoO = 2 * oI
      meta = prim__lstmCellMetaAlloc oI
      outBuf = prim__tensorAlloc twoO
      -- Pack mulIW
      iwvPtr = prim__lstmCellMulIWVals meta
      iwtPtr = prim__lstmCellMulIWTape meta
      iwvPtr' = packVec iwvPtr iwtPtr 0 mulIWElems
      -- Pack mulRW
      rwvPtr = prim__lstmCellMulRWVals (prim__seq iwvPtr' meta)
      rwtPtr = prim__lstmCellMulRWTape meta
      rwvPtr' = packVec rwvPtr rwtPtr 0 mulRWElems
      -- Pack bias
      bvPtr = prim__lstmCellBiasVals (prim__seq rwvPtr' meta)
      btPtr = prim__lstmCellBiasTape meta
      bvPtr' = packVec bvPtr btPtr 0 biasElems
      -- Pack prevCell
      pcvPtr = prim__lstmCellPrevCellVals (prim__seq bvPtr' meta)
      pctPtr = prim__lstmCellPrevCellTape meta
      pcvPtr' = packVec pcvPtr pctPtr 0 prevCellElems
      -- Compute
      outBuf' = prim__lstmCellCompute meta (prim__seq pcvPtr' outBuf)
      -- Append LstmCellOp tape entry (2*o outputs: cell + hidden)
      outBuf'' = tapeAppendLstmCellOp twoO meta outBuf'
      -- Build output Variables: first o = newCell, next o = newHidden
      cellScalars = buildOutputScalars outBuf'' 0 o
      hiddenScalars = buildOutputScalars outBuf'' oI o
  in (VTensor cellScalars, VTensor hiddenScalars)


||| Fused LSTM cell with bias from WeightBuf (no bias packing).
||| Same as lstmCellVar but bias values/indices come from persistent buffer.
export
lstmCellVarBuf : {o : Nat} -> Vector (4 * o) Variable -> Vector (4 * o) Variable
              -> AnyPtr -> Vector o Variable
              -> (Vector o Variable, Vector o Variable)
lstmCellVarBuf {o} (VTensor mulIWElems) (VTensor mulRWElems) bBuf (VTensor prevCellElems) =
  let oI = cast {to=Int} o
      fo = 4 * oI
      twoO = 2 * oI
      -- Ensure bias on tape (cached within epoch)
      bTapeStart = tapeEnsureBulkConst bBuf fo
      bValsPtr = prim__weightBufVals bBuf
      -- Allocate meta with bias buffer path (skips bias_vals/bias_tape_idx alloc)
      meta = prim__lstmCellMetaAllocBuf oI bValsPtr bTapeStart
      outBuf = prim__tensorAlloc twoO
      -- Pack mulIW
      iwvPtr = prim__lstmCellMulIWVals meta
      iwtPtr = prim__lstmCellMulIWTape meta
      iwvPtr' = packVec iwvPtr iwtPtr 0 mulIWElems
      -- Pack mulRW
      rwvPtr = prim__lstmCellMulRWVals (prim__seq iwvPtr' meta)
      rwtPtr = prim__lstmCellMulRWTape meta
      rwvPtr' = packVec rwvPtr rwtPtr 0 mulRWElems
      -- Pack prevCell (no bias packing needed!)
      pcvPtr = prim__lstmCellPrevCellVals (prim__seq rwvPtr' meta)
      pctPtr = prim__lstmCellPrevCellTape meta
      pcvPtr' = packVec pcvPtr pctPtr 0 prevCellElems
      -- Compute
      outBuf' = prim__lstmCellCompute meta (prim__seq pcvPtr' outBuf)
      -- Append LstmCellOp tape entry (2*o outputs: cell + hidden)
      outBuf'' = tapeAppendLstmCellOp twoO meta outBuf'
      -- Build output Variables: first o = newCell, next o = newHidden
      cellScalars = buildOutputScalars outBuf'' 0 o
      hiddenScalars = buildOutputScalars outBuf'' oI o
  in (VTensor cellScalars, VTensor hiddenScalars)


||| Fused LSTM cell with buffer-passing for mulIW, mulRW, and bias.
||| mulIW and mulRW come from matrixVectorMultiplyVarBufOut (C buffers + tape starts).
||| Eliminates all packVec overhead for the 3 largest inputs.
export
lstmCellVarFromBufs : {o : Nat}
                   -> AnyPtr -> Int    -- mulIW output buffer + const start
                   -> AnyPtr -> Int    -- mulRW output buffer + const start
                   -> AnyPtr           -- bias WeightBuf
                   -> Vector o Variable  -- prev cell state
                   -> (Vector o Variable, Vector o Variable)
lstmCellVarFromBufs {o} mulIWBuf mulIWStart mulRWBuf mulRWStart bBuf (VTensor prevCellElems) =
  let oI = cast {to=Int} o
      fo = 4 * oI
      twoO = 2 * oI
      -- Ensure bias on tape (cached within epoch)
      bTapeStart = tapeEnsureBulkConst bBuf fo
      bValsPtr = prim__weightBufVals bBuf
      -- Allocate meta with bias buffer path
      meta = prim__lstmCellMetaAllocBuf oI bValsPtr bTapeStart
      outBuf = prim__tensorAlloc twoO
      -- Bulk-copy mulIW from output buffer into meta (1 C call instead of 400 packVec iterations)
      iwvPtr = prim__lstmCellMulIWVals meta
      iwtPtr = prim__lstmCellMulIWTape meta
      iwvPtr' = prim__bufToMeta iwvPtr iwtPtr mulIWBuf mulIWStart fo
      -- Bulk-copy mulRW from output buffer into meta
      rwvPtr = prim__lstmCellMulRWVals (prim__seq iwvPtr' meta)
      rwtPtr = prim__lstmCellMulRWTape meta
      rwvPtr' = prim__bufToMeta rwvPtr rwtPtr mulRWBuf mulRWStart fo
      -- Pack prevCell (small: o elements, still worth packing individually)
      pcvPtr = prim__lstmCellPrevCellVals (prim__seq rwvPtr' meta)
      pctPtr = prim__lstmCellPrevCellTape meta
      pcvPtr' = packVec pcvPtr pctPtr 0 prevCellElems
      -- Compute
      outBuf' = prim__lstmCellCompute meta (prim__seq pcvPtr' outBuf)
      -- Append LstmCellOp tape entry (2*o outputs: cell + hidden)
      outBuf'' = tapeAppendLstmCellOp twoO meta outBuf'
      -- Build output Variables: first o = newCell, next o = newHidden
      cellScalars = buildOutputScalars outBuf'' 0 o
      hiddenScalars = buildOutputScalars outBuf'' oI o
  in (VTensor cellScalars, VTensor hiddenScalars)


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
