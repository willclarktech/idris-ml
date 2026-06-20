||| Linear-algebra instance slice (matmul, reductions, reshape,
||| indexing, sort/scan).
module Executor.Tape.Linear

import BackendLib
import DType.Core
import Executor.Core
import public Executor.Tape.Core
import Hardware
import Preset

----------------------------------------------------------------------
-- Linear-slice FFI bindings (tape-suffixed)
----------------------------------------------------------------------

%foreign "scheme:(lambda (a0 a1)  (when (not (top-level-bound? 'idris-ffi-tensor-mv-tape)) (set-top-level-value! 'idris-ffi-tensor-mv-tape (foreign-procedure \"tensor_mv_tape\" (void* void*) void*))) (when (not (top-level-bound? 'idris-ffi-tensor-retain-handle-tape)) (set-top-level-value! 'idris-ffi-tensor-retain-handle-tape (foreign-procedure \"tensor_retain_handle_tape\" (void*) void))) (let ((raw_r ((top-level-value 'idris-ffi-tensor-mv-tape) (vector-ref a0 2) (vector-ref a1 2)))) (let ((wr (vector 'tensor-handle-v2 \"tape\" raw_r))) ((top-level-value 'idris-tensor-guardian) wr) ((top-level-value 'idris-ffi-tensor-retain-handle-tape) raw_r) wr)))"
prim__mvTape : AnyPtr -> AnyPtr -> AnyPtr

%foreign "scheme:(lambda (a0 a1)  (when (not (top-level-bound? 'idris-ffi-tensor-mm-tape)) (set-top-level-value! 'idris-ffi-tensor-mm-tape (foreign-procedure \"tensor_mm_tape\" (void* void*) void*))) (when (not (top-level-bound? 'idris-ffi-tensor-retain-handle-tape)) (set-top-level-value! 'idris-ffi-tensor-retain-handle-tape (foreign-procedure \"tensor_retain_handle_tape\" (void*) void))) (let ((raw_r ((top-level-value 'idris-ffi-tensor-mm-tape) (vector-ref a0 2) (vector-ref a1 2)))) (let ((wr (vector 'tensor-handle-v2 \"tape\" raw_r))) ((top-level-value 'idris-tensor-guardian) wr) ((top-level-value 'idris-ffi-tensor-retain-handle-tape) raw_r) wr)))"
prim__mmTape : AnyPtr -> AnyPtr -> AnyPtr

%foreign "scheme:(lambda (a0 a1)  (when (not (top-level-bound? 'idris-ffi-tensor-matmul-tape)) (set-top-level-value! 'idris-ffi-tensor-matmul-tape (foreign-procedure \"tensor_matmul_tape\" (void* void*) void*))) (when (not (top-level-bound? 'idris-ffi-tensor-retain-handle-tape)) (set-top-level-value! 'idris-ffi-tensor-retain-handle-tape (foreign-procedure \"tensor_retain_handle_tape\" (void*) void))) (let ((raw_r ((top-level-value 'idris-ffi-tensor-matmul-tape) (vector-ref a0 2) (vector-ref a1 2)))) (let ((wr (vector 'tensor-handle-v2 \"tape\" raw_r))) ((top-level-value 'idris-tensor-guardian) wr) ((top-level-value 'idris-ffi-tensor-retain-handle-tape) raw_r) wr)))"
prim__matmulTape : AnyPtr -> AnyPtr -> AnyPtr

%foreign "scheme:(lambda (a0 a1 a2)  (when (not (top-level-bound? 'idris-ffi-tensor-linear-tape)) (set-top-level-value! 'idris-ffi-tensor-linear-tape (foreign-procedure \"tensor_linear_tape\" (void* void* void*) void*))) (when (not (top-level-bound? 'idris-ffi-tensor-retain-handle-tape)) (set-top-level-value! 'idris-ffi-tensor-retain-handle-tape (foreign-procedure \"tensor_retain_handle_tape\" (void*) void))) (let ((raw_r ((top-level-value 'idris-ffi-tensor-linear-tape) (vector-ref a0 2) (vector-ref a1 2) (vector-ref a2 2)))) (let ((wr (vector 'tensor-handle-v2 \"tape\" raw_r))) ((top-level-value 'idris-tensor-guardian) wr) ((top-level-value 'idris-ffi-tensor-retain-handle-tape) raw_r) wr)))"
prim__linearTape : AnyPtr -> AnyPtr -> AnyPtr -> AnyPtr

%foreign "scheme:(lambda (a0 a1)  (when (not (top-level-bound? 'idris-ffi-tensor-dot-tape)) (set-top-level-value! 'idris-ffi-tensor-dot-tape (foreign-procedure \"tensor_dot_tape\" (void* void*) void*))) (when (not (top-level-bound? 'idris-ffi-tensor-retain-handle-tape)) (set-top-level-value! 'idris-ffi-tensor-retain-handle-tape (foreign-procedure \"tensor_retain_handle_tape\" (void*) void))) (let ((raw_r ((top-level-value 'idris-ffi-tensor-dot-tape) (vector-ref a0 2) (vector-ref a1 2)))) (let ((wr (vector 'tensor-handle-v2 \"tape\" raw_r))) ((top-level-value 'idris-tensor-guardian) wr) ((top-level-value 'idris-ffi-tensor-retain-handle-tape) raw_r) wr)))"
prim__dotTape : AnyPtr -> AnyPtr -> AnyPtr

%foreign "scheme:(lambda (a0 a1)  (when (not (top-level-bound? 'idris-ffi-tensor-outer-tape)) (set-top-level-value! 'idris-ffi-tensor-outer-tape (foreign-procedure \"tensor_outer_tape\" (void* void*) void*))) (when (not (top-level-bound? 'idris-ffi-tensor-retain-handle-tape)) (set-top-level-value! 'idris-ffi-tensor-retain-handle-tape (foreign-procedure \"tensor_retain_handle_tape\" (void*) void))) (let ((raw_r ((top-level-value 'idris-ffi-tensor-outer-tape) (vector-ref a0 2) (vector-ref a1 2)))) (let ((wr (vector 'tensor-handle-v2 \"tape\" raw_r))) ((top-level-value 'idris-tensor-guardian) wr) ((top-level-value 'idris-ffi-tensor-retain-handle-tape) raw_r) wr)))"
prim__outerTape : AnyPtr -> AnyPtr -> AnyPtr

%foreign "scheme:(lambda (a0 a1)  (when (not (top-level-bound? 'idris-ffi-tensor-bmm-tape)) (set-top-level-value! 'idris-ffi-tensor-bmm-tape (foreign-procedure \"tensor_bmm_tape\" (void* void*) void*))) (when (not (top-level-bound? 'idris-ffi-tensor-retain-handle-tape)) (set-top-level-value! 'idris-ffi-tensor-retain-handle-tape (foreign-procedure \"tensor_retain_handle_tape\" (void*) void))) (let ((raw_r ((top-level-value 'idris-ffi-tensor-bmm-tape) (vector-ref a0 2) (vector-ref a1 2)))) (let ((wr (vector 'tensor-handle-v2 \"tape\" raw_r))) ((top-level-value 'idris-tensor-guardian) wr) ((top-level-value 'idris-ffi-tensor-retain-handle-tape) raw_r) wr)))"
prim__bmmTape : AnyPtr -> AnyPtr -> AnyPtr

%foreign "scheme:(lambda (a0 a1 a2)  (when (not (top-level-bound? 'idris-ffi-tensor-linear-2d-tape)) (set-top-level-value! 'idris-ffi-tensor-linear-2d-tape (foreign-procedure \"tensor_linear_2d_tape\" (void* void* void*) void*))) (when (not (top-level-bound? 'idris-ffi-tensor-retain-handle-tape)) (set-top-level-value! 'idris-ffi-tensor-retain-handle-tape (foreign-procedure \"tensor_retain_handle_tape\" (void*) void))) (let ((raw_r ((top-level-value 'idris-ffi-tensor-linear-2d-tape) (vector-ref a0 2) (vector-ref a1 2) (vector-ref a2 2)))) (let ((wr (vector 'tensor-handle-v2 \"tape\" raw_r))) ((top-level-value 'idris-tensor-guardian) wr) ((top-level-value 'idris-ffi-tensor-retain-handle-tape) raw_r) wr)))"
prim__linear2dTape : AnyPtr -> AnyPtr -> AnyPtr -> AnyPtr

%foreign "scheme:(lambda (a0)  (when (not (top-level-bound? 'idris-ffi-tensor-sum-tape)) (set-top-level-value! 'idris-ffi-tensor-sum-tape (foreign-procedure \"tensor_sum_tape\" (void*) void*))) (when (not (top-level-bound? 'idris-ffi-tensor-retain-handle-tape)) (set-top-level-value! 'idris-ffi-tensor-retain-handle-tape (foreign-procedure \"tensor_retain_handle_tape\" (void*) void))) (let ((raw_r ((top-level-value 'idris-ffi-tensor-sum-tape) (vector-ref a0 2)))) (let ((wr (vector 'tensor-handle-v2 \"tape\" raw_r))) ((top-level-value 'idris-tensor-guardian) wr) ((top-level-value 'idris-ffi-tensor-retain-handle-tape) raw_r) wr)))"
prim__sumTape : AnyPtr -> AnyPtr

%foreign "scheme:(lambda (a0)  (when (not (top-level-bound? 'idris-ffi-tensor-mean-tape)) (set-top-level-value! 'idris-ffi-tensor-mean-tape (foreign-procedure \"tensor_mean_tape\" (void*) void*))) (when (not (top-level-bound? 'idris-ffi-tensor-retain-handle-tape)) (set-top-level-value! 'idris-ffi-tensor-retain-handle-tape (foreign-procedure \"tensor_retain_handle_tape\" (void*) void))) (let ((raw_r ((top-level-value 'idris-ffi-tensor-mean-tape) (vector-ref a0 2)))) (let ((wr (vector 'tensor-handle-v2 \"tape\" raw_r))) ((top-level-value 'idris-tensor-guardian) wr) ((top-level-value 'idris-ffi-tensor-retain-handle-tape) raw_r) wr)))"
prim__meanTape : AnyPtr -> AnyPtr

%foreign "scheme:(lambda (a0)  (when (not (top-level-bound? 'idris-ffi-tensor-min-tape)) (set-top-level-value! 'idris-ffi-tensor-min-tape (foreign-procedure \"tensor_min_tape\" (void*) void*))) (when (not (top-level-bound? 'idris-ffi-tensor-retain-handle-tape)) (set-top-level-value! 'idris-ffi-tensor-retain-handle-tape (foreign-procedure \"tensor_retain_handle_tape\" (void*) void))) (let ((raw_r ((top-level-value 'idris-ffi-tensor-min-tape) (vector-ref a0 2)))) (let ((wr (vector 'tensor-handle-v2 \"tape\" raw_r))) ((top-level-value 'idris-tensor-guardian) wr) ((top-level-value 'idris-ffi-tensor-retain-handle-tape) raw_r) wr)))"
prim__tensorMinTape : AnyPtr -> AnyPtr

%foreign "scheme:(lambda (a0)  (when (not (top-level-bound? 'idris-ffi-tensor-max-tape)) (set-top-level-value! 'idris-ffi-tensor-max-tape (foreign-procedure \"tensor_max_tape\" (void*) void*))) (when (not (top-level-bound? 'idris-ffi-tensor-retain-handle-tape)) (set-top-level-value! 'idris-ffi-tensor-retain-handle-tape (foreign-procedure \"tensor_retain_handle_tape\" (void*) void))) (let ((raw_r ((top-level-value 'idris-ffi-tensor-max-tape) (vector-ref a0 2)))) (let ((wr (vector 'tensor-handle-v2 \"tape\" raw_r))) ((top-level-value 'idris-tensor-guardian) wr) ((top-level-value 'idris-ffi-tensor-retain-handle-tape) raw_r) wr)))"
prim__tensorMaxTape : AnyPtr -> AnyPtr

%foreign "scheme:(lambda (a0 a1 a2)  (when (not (top-level-bound? 'idris-ffi-tensor-sum-dim-tape)) (set-top-level-value! 'idris-ffi-tensor-sum-dim-tape (foreign-procedure \"tensor_sum_dim_tape\" (void* int int) void*))) (when (not (top-level-bound? 'idris-ffi-tensor-retain-handle-tape)) (set-top-level-value! 'idris-ffi-tensor-retain-handle-tape (foreign-procedure \"tensor_retain_handle_tape\" (void*) void))) (let ((raw_r ((top-level-value 'idris-ffi-tensor-sum-dim-tape) (vector-ref a0 2) a1 a2))) (let ((wr (vector 'tensor-handle-v2 \"tape\" raw_r))) ((top-level-value 'idris-tensor-guardian) wr) ((top-level-value 'idris-ffi-tensor-retain-handle-tape) raw_r) wr)))"
prim__sumDimTape : AnyPtr -> Int -> Int -> AnyPtr

%foreign "scheme:(lambda (a0 a1 a2)  (when (not (top-level-bound? 'idris-ffi-tensor-select-tape)) (set-top-level-value! 'idris-ffi-tensor-select-tape (foreign-procedure \"tensor_select_tape\" (void* int int) void*))) (when (not (top-level-bound? 'idris-ffi-tensor-retain-handle-tape)) (set-top-level-value! 'idris-ffi-tensor-retain-handle-tape (foreign-procedure \"tensor_retain_handle_tape\" (void*) void))) (let ((raw_r ((top-level-value 'idris-ffi-tensor-select-tape) (vector-ref a0 2) a1 a2))) (let ((wr (vector 'tensor-handle-v2 \"tape\" raw_r))) ((top-level-value 'idris-tensor-guardian) wr) ((top-level-value 'idris-ffi-tensor-retain-handle-tape) raw_r) wr)))"
prim__selectTape : AnyPtr -> Int -> Int -> AnyPtr

%foreign "scheme:(lambda (a0 a1)  (when (not (top-level-bound? 'idris-ffi-tensor-unsqueeze-tape)) (set-top-level-value! 'idris-ffi-tensor-unsqueeze-tape (foreign-procedure \"tensor_unsqueeze_tape\" (void* int) void*))) (when (not (top-level-bound? 'idris-ffi-tensor-retain-handle-tape)) (set-top-level-value! 'idris-ffi-tensor-retain-handle-tape (foreign-procedure \"tensor_retain_handle_tape\" (void*) void))) (let ((raw_r ((top-level-value 'idris-ffi-tensor-unsqueeze-tape) (vector-ref a0 2) a1))) (let ((wr (vector 'tensor-handle-v2 \"tape\" raw_r))) ((top-level-value 'idris-tensor-guardian) wr) ((top-level-value 'idris-ffi-tensor-retain-handle-tape) raw_r) wr)))"
prim__unsqueezeTape : AnyPtr -> Int -> AnyPtr

%foreign "scheme:(lambda (a0 a1)  (when (not (top-level-bound? 'idris-ffi-tensor-squeeze-tape)) (set-top-level-value! 'idris-ffi-tensor-squeeze-tape (foreign-procedure \"tensor_squeeze_tape\" (void* int) void*))) (when (not (top-level-bound? 'idris-ffi-tensor-retain-handle-tape)) (set-top-level-value! 'idris-ffi-tensor-retain-handle-tape (foreign-procedure \"tensor_retain_handle_tape\" (void*) void))) (let ((raw_r ((top-level-value 'idris-ffi-tensor-squeeze-tape) (vector-ref a0 2) a1))) (let ((wr (vector 'tensor-handle-v2 \"tape\" raw_r))) ((top-level-value 'idris-tensor-guardian) wr) ((top-level-value 'idris-ffi-tensor-retain-handle-tape) raw_r) wr)))"
prim__squeezeTape : AnyPtr -> Int -> AnyPtr

%foreign "scheme:(lambda (a0 a1 a2)  (when (not (top-level-bound? 'idris-ffi-tensor-stack-tape)) (set-top-level-value! 'idris-ffi-tensor-stack-tape (foreign-procedure \"tensor_stack_tape\" (void* int int) void*))) (when (not (top-level-bound? 'idris-ffi-tensor-retain-handle-tape)) (set-top-level-value! 'idris-ffi-tensor-retain-handle-tape (foreign-procedure \"tensor_retain_handle_tape\" (void*) void))) (let ((raw_r ((top-level-value 'idris-ffi-tensor-stack-tape) a0 a1 a2))) (let ((wr (vector 'tensor-handle-v2 \"tape\" raw_r))) ((top-level-value 'idris-tensor-guardian) wr) ((top-level-value 'idris-ffi-tensor-retain-handle-tape) raw_r) wr)))"
prim__stackTape : AnyPtr -> Int -> Int -> AnyPtr

%foreign "scheme:(lambda (a0 a1)  (when (not (top-level-bound? 'idris-ffi-tensor-batch-tape)) (set-top-level-value! 'idris-ffi-tensor-batch-tape (foreign-procedure \"tensor_batch_tape\" (void* int) void*))) (when (not (top-level-bound? 'idris-ffi-tensor-retain-handle-tape)) (set-top-level-value! 'idris-ffi-tensor-retain-handle-tape (foreign-procedure \"tensor_retain_handle_tape\" (void*) void))) (let ((raw_r ((top-level-value 'idris-ffi-tensor-batch-tape) a0 a1))) (let ((wr (vector 'tensor-handle-v2 \"tape\" raw_r))) ((top-level-value 'idris-tensor-guardian) wr) ((top-level-value 'idris-ffi-tensor-retain-handle-tape) raw_r) wr)))"
prim__batchTape : AnyPtr -> Int -> AnyPtr

%foreign "scheme:(lambda (a0 a1)  (when (not (top-level-bound? 'idris-ffi-tensor-view-1d-tape)) (set-top-level-value! 'idris-ffi-tensor-view-1d-tape (foreign-procedure \"tensor_view_1d_tape\" (void* int) void*))) (when (not (top-level-bound? 'idris-ffi-tensor-retain-handle-tape)) (set-top-level-value! 'idris-ffi-tensor-retain-handle-tape (foreign-procedure \"tensor_retain_handle_tape\" (void*) void))) (let ((raw_r ((top-level-value 'idris-ffi-tensor-view-1d-tape) (vector-ref a0 2) a1))) (let ((wr (vector 'tensor-handle-v2 \"tape\" raw_r))) ((top-level-value 'idris-tensor-guardian) wr) ((top-level-value 'idris-ffi-tensor-retain-handle-tape) raw_r) wr)))"
prim__view1dTape : AnyPtr -> Int -> AnyPtr

%foreign "scheme:(lambda (a0 a1 a2)  (when (not (top-level-bound? 'idris-ffi-tensor-view-2d-tape)) (set-top-level-value! 'idris-ffi-tensor-view-2d-tape (foreign-procedure \"tensor_view_2d_tape\" (void* int int) void*))) (when (not (top-level-bound? 'idris-ffi-tensor-retain-handle-tape)) (set-top-level-value! 'idris-ffi-tensor-retain-handle-tape (foreign-procedure \"tensor_retain_handle_tape\" (void*) void))) (let ((raw_r ((top-level-value 'idris-ffi-tensor-view-2d-tape) (vector-ref a0 2) a1 a2))) (let ((wr (vector 'tensor-handle-v2 \"tape\" raw_r))) ((top-level-value 'idris-tensor-guardian) wr) ((top-level-value 'idris-ffi-tensor-retain-handle-tape) raw_r) wr)))"
prim__view2dTape : AnyPtr -> Int -> Int -> AnyPtr

%foreign "scheme:(lambda (a0 a1)  (let ((raw_r ((foreign-procedure \"tensor_reshape_1d_tape\" (void* int) void*) (vector-ref a0 2) a1))) (let ((wr (vector 'tensor-handle-v2 \"tape\" raw_r))) ((top-level-value 'idris-tensor-guardian) wr) ((foreign-procedure \"tensor_retain_handle_tape\" (void*) void) raw_r) wr)))"
prim__reshape1dTape : AnyPtr -> Int -> AnyPtr

%foreign "scheme:(lambda (a0 a1 a2)  (when (not (top-level-bound? 'idris-ffi-tensor-reshape-2d-tape)) (set-top-level-value! 'idris-ffi-tensor-reshape-2d-tape (foreign-procedure \"tensor_reshape_2d_tape\" (void* int int) void*))) (when (not (top-level-bound? 'idris-ffi-tensor-retain-handle-tape)) (set-top-level-value! 'idris-ffi-tensor-retain-handle-tape (foreign-procedure \"tensor_retain_handle_tape\" (void*) void))) (let ((raw_r ((top-level-value 'idris-ffi-tensor-reshape-2d-tape) (vector-ref a0 2) a1 a2))) (let ((wr (vector 'tensor-handle-v2 \"tape\" raw_r))) ((top-level-value 'idris-tensor-guardian) wr) ((top-level-value 'idris-ffi-tensor-retain-handle-tape) raw_r) wr)))"
prim__reshape2dTape : AnyPtr -> Int -> Int -> AnyPtr

%foreign "scheme:(lambda (a0 a1 a2 a3)  (when (not (top-level-bound? 'idris-ffi-tensor-reshape-3d-tape)) (set-top-level-value! 'idris-ffi-tensor-reshape-3d-tape (foreign-procedure \"tensor_reshape_3d_tape\" (void* int int int) void*))) (when (not (top-level-bound? 'idris-ffi-tensor-retain-handle-tape)) (set-top-level-value! 'idris-ffi-tensor-retain-handle-tape (foreign-procedure \"tensor_retain_handle_tape\" (void*) void))) (let ((raw_r ((top-level-value 'idris-ffi-tensor-reshape-3d-tape) (vector-ref a0 2) a1 a2 a3))) (let ((wr (vector 'tensor-handle-v2 \"tape\" raw_r))) ((top-level-value 'idris-tensor-guardian) wr) ((top-level-value 'idris-ffi-tensor-retain-handle-tape) raw_r) wr)))"
prim__reshape3dTape : AnyPtr -> Int -> Int -> Int -> AnyPtr

%foreign "scheme:(lambda (a0 a1 a2 a3 a4)  (when (not (top-level-bound? 'idris-ffi-tensor-reshape-4d-tape)) (set-top-level-value! 'idris-ffi-tensor-reshape-4d-tape (foreign-procedure \"tensor_reshape_4d_tape\" (void* int int int int) void*))) (when (not (top-level-bound? 'idris-ffi-tensor-retain-handle-tape)) (set-top-level-value! 'idris-ffi-tensor-retain-handle-tape (foreign-procedure \"tensor_retain_handle_tape\" (void*) void))) (let ((raw_r ((top-level-value 'idris-ffi-tensor-reshape-4d-tape) (vector-ref a0 2) a1 a2 a3 a4))) (let ((wr (vector 'tensor-handle-v2 \"tape\" raw_r))) ((top-level-value 'idris-tensor-guardian) wr) ((top-level-value 'idris-ffi-tensor-retain-handle-tape) raw_r) wr)))"
prim__reshape4dTape : AnyPtr -> Int -> Int -> Int -> Int -> AnyPtr

%foreign "scheme:(lambda (a0 a1 a2)  (when (not (top-level-bound? 'idris-ffi-tensor-tile-2d-tape)) (set-top-level-value! 'idris-ffi-tensor-tile-2d-tape (foreign-procedure \"tensor_tile_2d_tape\" (void* int int) void*))) (when (not (top-level-bound? 'idris-ffi-tensor-retain-handle-tape)) (set-top-level-value! 'idris-ffi-tensor-retain-handle-tape (foreign-procedure \"tensor_retain_handle_tape\" (void*) void))) (let ((raw_r ((top-level-value 'idris-ffi-tensor-tile-2d-tape) (vector-ref a0 2) a1 a2))) (let ((wr (vector 'tensor-handle-v2 \"tape\" raw_r))) ((top-level-value 'idris-tensor-guardian) wr) ((top-level-value 'idris-ffi-tensor-retain-handle-tape) raw_r) wr)))"
export
prim__tile2dTape : AnyPtr -> Int -> Int -> AnyPtr

%foreign "scheme:(lambda (a0 a1 a2 a3)  (when (not (top-level-bound? 'idris-ffi-tensor-narrow-tape)) (set-top-level-value! 'idris-ffi-tensor-narrow-tape (foreign-procedure \"tensor_narrow_tape\" (void* int int int) void*))) (when (not (top-level-bound? 'idris-ffi-tensor-retain-handle-tape)) (set-top-level-value! 'idris-ffi-tensor-retain-handle-tape (foreign-procedure \"tensor_retain_handle_tape\" (void*) void))) (let ((raw_r ((top-level-value 'idris-ffi-tensor-narrow-tape) (vector-ref a0 2) a1 a2 a3))) (let ((wr (vector 'tensor-handle-v2 \"tape\" raw_r))) ((top-level-value 'idris-tensor-guardian) wr) ((top-level-value 'idris-ffi-tensor-retain-handle-tape) raw_r) wr)))"
prim__narrowTape : AnyPtr -> Int -> Int -> Int -> AnyPtr

%foreign "scheme:(lambda (a0)  (when (not (top-level-bound? 'idris-ffi-tensor-transpose-last2-tape)) (set-top-level-value! 'idris-ffi-tensor-transpose-last2-tape (foreign-procedure \"tensor_transpose_last2_tape\" (void*) void*))) (when (not (top-level-bound? 'idris-ffi-tensor-retain-handle-tape)) (set-top-level-value! 'idris-ffi-tensor-retain-handle-tape (foreign-procedure \"tensor_retain_handle_tape\" (void*) void))) (let ((raw_r ((top-level-value 'idris-ffi-tensor-transpose-last2-tape) (vector-ref a0 2)))) (let ((wr (vector 'tensor-handle-v2 \"tape\" raw_r))) ((top-level-value 'idris-tensor-guardian) wr) ((top-level-value 'idris-ffi-tensor-retain-handle-tape) raw_r) wr)))"
prim__transposeLast2Tape : AnyPtr -> AnyPtr

%foreign "scheme:(lambda (a0)  (when (not (top-level-bound? 'idris-ffi-tensor-transpose-2d-tape)) (set-top-level-value! 'idris-ffi-tensor-transpose-2d-tape (foreign-procedure \"tensor_transpose_2d_tape\" (void*) void*))) (when (not (top-level-bound? 'idris-ffi-tensor-retain-handle-tape)) (set-top-level-value! 'idris-ffi-tensor-retain-handle-tape (foreign-procedure \"tensor_retain_handle_tape\" (void*) void))) (let ((raw_r ((top-level-value 'idris-ffi-tensor-transpose-2d-tape) (vector-ref a0 2)))) (let ((wr (vector 'tensor-handle-v2 \"tape\" raw_r))) ((top-level-value 'idris-tensor-guardian) wr) ((top-level-value 'idris-ffi-tensor-retain-handle-tape) raw_r) wr)))"
prim__transpose2dTape : AnyPtr -> AnyPtr

%foreign "scheme:(lambda (a0 a1 a2)  (when (not (top-level-bound? 'idris-ffi-tensor-cat-tape)) (set-top-level-value! 'idris-ffi-tensor-cat-tape (foreign-procedure \"tensor_cat_tape\" (void* int int) void*))) (when (not (top-level-bound? 'idris-ffi-tensor-retain-handle-tape)) (set-top-level-value! 'idris-ffi-tensor-retain-handle-tape (foreign-procedure \"tensor_retain_handle_tape\" (void*) void))) (let ((raw_r ((top-level-value 'idris-ffi-tensor-cat-tape) a0 a1 a2))) (let ((wr (vector 'tensor-handle-v2 \"tape\" raw_r))) ((top-level-value 'idris-tensor-guardian) wr) ((top-level-value 'idris-ffi-tensor-retain-handle-tape) raw_r) wr)))"
prim__catTape : AnyPtr -> Int -> Int -> AnyPtr

%foreign "scheme:(lambda (a0 a1)  (when (not (top-level-bound? 'idris-ffi-tensor-cat2-tape)) (set-top-level-value! 'idris-ffi-tensor-cat2-tape (foreign-procedure \"tensor_cat2_tape\" (void* void*) void*))) (when (not (top-level-bound? 'idris-ffi-tensor-retain-handle-tape)) (set-top-level-value! 'idris-ffi-tensor-retain-handle-tape (foreign-procedure \"tensor_retain_handle_tape\" (void*) void))) (let ((raw_r ((top-level-value 'idris-ffi-tensor-cat2-tape) (vector-ref a0 2) (vector-ref a1 2)))) (let ((wr (vector 'tensor-handle-v2 \"tape\" raw_r))) ((top-level-value 'idris-tensor-guardian) wr) ((top-level-value 'idris-ffi-tensor-retain-handle-tape) raw_r) wr)))"
prim__cat2Tape : AnyPtr -> AnyPtr -> AnyPtr

%foreign "scheme:(lambda (a0 a1)  (when (not (top-level-bound? 'idris-ffi-tensor-concat-2d-axis1-tape)) (set-top-level-value! 'idris-ffi-tensor-concat-2d-axis1-tape (foreign-procedure \"tensor_concat_2d_axis1_tape\" (void* void*) void*))) (when (not (top-level-bound? 'idris-ffi-tensor-retain-handle-tape)) (set-top-level-value! 'idris-ffi-tensor-retain-handle-tape (foreign-procedure \"tensor_retain_handle_tape\" (void*) void))) (let ((raw_r ((top-level-value 'idris-ffi-tensor-concat-2d-axis1-tape) (vector-ref a0 2) (vector-ref a1 2)))) (let ((wr (vector 'tensor-handle-v2 \"tape\" raw_r))) ((top-level-value 'idris-tensor-guardian) wr) ((top-level-value 'idris-ffi-tensor-retain-handle-tape) raw_r) wr)))"
prim__concat2dAxis1Tape : AnyPtr -> AnyPtr -> AnyPtr

%foreign "scheme:(lambda (a0 a1 a2)  (when (not (top-level-bound? 'idris-ffi-tensor-gather-tape)) (set-top-level-value! 'idris-ffi-tensor-gather-tape (foreign-procedure \"tensor_gather_tape\" (void* void* int) void*))) (when (not (top-level-bound? 'idris-ffi-tensor-retain-handle-tape)) (set-top-level-value! 'idris-ffi-tensor-retain-handle-tape (foreign-procedure \"tensor_retain_handle_tape\" (void*) void))) (let ((raw_r ((top-level-value 'idris-ffi-tensor-gather-tape) (vector-ref a0 2) (vector-ref a1 2) a2))) (let ((wr (vector 'tensor-handle-v2 \"tape\" raw_r))) ((top-level-value 'idris-tensor-guardian) wr) ((top-level-value 'idris-ffi-tensor-retain-handle-tape) raw_r) wr)))"
prim__gatherTape : AnyPtr -> AnyPtr -> Int -> AnyPtr

%foreign "scheme:(lambda (a0 a1 a2 a3)  (when (not (top-level-bound? 'idris-ffi-tensor-gather-rows-tape)) (set-top-level-value! 'idris-ffi-tensor-gather-rows-tape (foreign-procedure \"tensor_gather_rows_tape\" (void* void* int int) void*))) (when (not (top-level-bound? 'idris-ffi-tensor-retain-handle-tape)) (set-top-level-value! 'idris-ffi-tensor-retain-handle-tape (foreign-procedure \"tensor_retain_handle_tape\" (void*) void))) (let ((raw_r ((top-level-value 'idris-ffi-tensor-gather-rows-tape) (vector-ref a0 2) (vector-ref a1 2) a2 a3))) (let ((wr (vector 'tensor-handle-v2 \"tape\" raw_r))) ((top-level-value 'idris-tensor-guardian) wr) ((top-level-value 'idris-ffi-tensor-retain-handle-tape) raw_r) wr)))"
prim__gatherRowsTape : AnyPtr -> AnyPtr -> Int -> Int -> AnyPtr

%foreign "scheme:(lambda (a0 a1 a2)  (when (not (top-level-bound? 'idris-ffi-tensor-max-rows-tape)) (set-top-level-value! 'idris-ffi-tensor-max-rows-tape (foreign-procedure \"tensor_max_rows_tape\" (void* int int) void*))) (when (not (top-level-bound? 'idris-ffi-tensor-retain-handle-tape)) (set-top-level-value! 'idris-ffi-tensor-retain-handle-tape (foreign-procedure \"tensor_retain_handle_tape\" (void*) void))) (let ((raw_r ((top-level-value 'idris-ffi-tensor-max-rows-tape) (vector-ref a0 2) a1 a2))) (let ((wr (vector 'tensor-handle-v2 \"tape\" raw_r))) ((top-level-value 'idris-tensor-guardian) wr) ((top-level-value 'idris-ffi-tensor-retain-handle-tape) raw_r) wr)))"
prim__maxRowsTape : AnyPtr -> Int -> Int -> AnyPtr

%foreign "scheme:(lambda (a0 a1 a2)  (when (not (top-level-bound? 'idris-ffi-tensor-scatter-add-tape)) (set-top-level-value! 'idris-ffi-tensor-scatter-add-tape (foreign-procedure \"tensor_scatter_add_tape\" (void* void* int) void*))) (when (not (top-level-bound? 'idris-ffi-tensor-retain-handle-tape)) (set-top-level-value! 'idris-ffi-tensor-retain-handle-tape (foreign-procedure \"tensor_retain_handle_tape\" (void*) void))) (let ((raw_r ((top-level-value 'idris-ffi-tensor-scatter-add-tape) (vector-ref a0 2) (vector-ref a1 2) a2))) (let ((wr (vector 'tensor-handle-v2 \"tape\" raw_r))) ((top-level-value 'idris-tensor-guardian) wr) ((top-level-value 'idris-ffi-tensor-retain-handle-tape) raw_r) wr)))"
prim__scatterAddTape : AnyPtr -> AnyPtr -> Int -> AnyPtr

%foreign "scheme:(lambda (a0 a1 a2)  (when (not (top-level-bound? 'idris-ffi-tensor-argsort-tape)) (set-top-level-value! 'idris-ffi-tensor-argsort-tape (foreign-procedure \"tensor_argsort_tape\" (void* int int) void*))) (when (not (top-level-bound? 'idris-ffi-tensor-retain-handle-tape)) (set-top-level-value! 'idris-ffi-tensor-retain-handle-tape (foreign-procedure \"tensor_retain_handle_tape\" (void*) void))) (let ((raw_r ((top-level-value 'idris-ffi-tensor-argsort-tape) (vector-ref a0 2) a1 a2))) (let ((wr (vector 'tensor-handle-v2 \"tape\" raw_r))) ((top-level-value 'idris-tensor-guardian) wr) ((top-level-value 'idris-ffi-tensor-retain-handle-tape) raw_r) wr)))"
prim__argsortTape : AnyPtr -> Int -> Int -> AnyPtr

%foreign "scheme:(lambda (a0 a1)  (when (not (top-level-bound? 'idris-ffi-tensor-cumprod-tape)) (set-top-level-value! 'idris-ffi-tensor-cumprod-tape (foreign-procedure \"tensor_cumprod_tape\" (void* int) void*))) (when (not (top-level-bound? 'idris-ffi-tensor-retain-handle-tape)) (set-top-level-value! 'idris-ffi-tensor-retain-handle-tape (foreign-procedure \"tensor_retain_handle_tape\" (void*) void))) (let ((raw_r ((top-level-value 'idris-ffi-tensor-cumprod-tape) (vector-ref a0 2) a1))) (let ((wr (vector 'tensor-handle-v2 \"tape\" raw_r))) ((top-level-value 'idris-tensor-guardian) wr) ((top-level-value 'idris-ffi-tensor-retain-handle-tape) raw_r) wr)))"
prim__cumprodTape : AnyPtr -> Int -> AnyPtr

public export
UserExecutorLinear TapeExecutor where
  -- >>> GENERATED FROM ffi_manifest.py — gen-executor-instances.py >>>
  primArgsort        = prim__argsortTape
  primBatch          = prim__batchTape
  primBmm            = prim__bmmTape
  primCat            = prim__catTape
  primCat2           = prim__cat2Tape
  primConcat2dAxis1  = prim__concat2dAxis1Tape
  primCumprod        = prim__cumprodTape
  primDot            = prim__dotTape
  primGather         = prim__gatherTape
  primGatherRows     = prim__gatherRowsTape
  primLinear         = prim__linearTape
  primLinear2d       = prim__linear2dTape
  primMatmul         = prim__matmulTape
  primMaxRows        = prim__maxRowsTape
  primMean           = prim__meanTape
  primMm             = prim__mmTape
  primMv             = prim__mvTape
  primNarrow         = prim__narrowTape
  primOuter          = prim__outerTape
  primReshape1d      = prim__reshape1dTape
  primReshape2d      = prim__reshape2dTape
  primReshape3d      = prim__reshape3dTape
  primReshape4d      = prim__reshape4dTape
  primScatterAdd     = prim__scatterAddTape
  primSelect         = prim__selectTape
  primSqueeze        = prim__squeezeTape
  primStack          = prim__stackTape
  primSum            = prim__sumTape
  primSumDim         = prim__sumDimTape
  primTensorMax      = prim__tensorMaxTape
  primTensorMin      = prim__tensorMinTape
  primTranspose2d    = prim__transpose2dTape
  primTransposeLast2 = prim__transposeLast2Tape
  primUnsqueeze      = prim__unsqueezeTape
  primView1d         = prim__view1dTape
  primView2d         = prim__view2dTape
  -- <<< END GENERATED <<<
