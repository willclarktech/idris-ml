||| Linear-algebra instance slice (matmul, reductions, reshape,
||| indexing, sort/scan).
module Executor.Torch.Linear

import BackendLib
import DType.Core
import Executor.Core
import public Executor.Torch.Core
import Hardware
import Preset

----------------------------------------------------------------------
-- Linear-slice FFI bindings (torch-suffixed)
----------------------------------------------------------------------

%foreign "scheme:(lambda (a0 a1)  (when (not (top-level-bound? 'idris-ffi-tensor-mv-torch)) (set-top-level-value! 'idris-ffi-tensor-mv-torch (foreign-procedure \"tensor_mv_torch\" (void* void*) void*))) (when (not (top-level-bound? 'idris-ffi-tensor-retain-handle-torch)) (set-top-level-value! 'idris-ffi-tensor-retain-handle-torch (foreign-procedure \"tensor_retain_handle_torch\" (void*) void))) (let ((raw_r ((top-level-value 'idris-ffi-tensor-mv-torch) (vector-ref a0 2) (vector-ref a1 2)))) (let ((wr (vector 'tensor-handle-v2 \"torch\" raw_r))) ((top-level-value 'idris-tensor-guardian) wr) ((top-level-value 'idris-ffi-tensor-retain-handle-torch) raw_r) wr)))"
prim__mvTorch : AnyPtr -> AnyPtr -> AnyPtr

%foreign "scheme:(lambda (a0 a1)  (when (not (top-level-bound? 'idris-ffi-tensor-mm-torch)) (set-top-level-value! 'idris-ffi-tensor-mm-torch (foreign-procedure \"tensor_mm_torch\" (void* void*) void*))) (when (not (top-level-bound? 'idris-ffi-tensor-retain-handle-torch)) (set-top-level-value! 'idris-ffi-tensor-retain-handle-torch (foreign-procedure \"tensor_retain_handle_torch\" (void*) void))) (let ((raw_r ((top-level-value 'idris-ffi-tensor-mm-torch) (vector-ref a0 2) (vector-ref a1 2)))) (let ((wr (vector 'tensor-handle-v2 \"torch\" raw_r))) ((top-level-value 'idris-tensor-guardian) wr) ((top-level-value 'idris-ffi-tensor-retain-handle-torch) raw_r) wr)))"
prim__mmTorch : AnyPtr -> AnyPtr -> AnyPtr

%foreign "scheme:(lambda (a0 a1)  (when (not (top-level-bound? 'idris-ffi-tensor-matmul-torch)) (set-top-level-value! 'idris-ffi-tensor-matmul-torch (foreign-procedure \"tensor_matmul_torch\" (void* void*) void*))) (when (not (top-level-bound? 'idris-ffi-tensor-retain-handle-torch)) (set-top-level-value! 'idris-ffi-tensor-retain-handle-torch (foreign-procedure \"tensor_retain_handle_torch\" (void*) void))) (let ((raw_r ((top-level-value 'idris-ffi-tensor-matmul-torch) (vector-ref a0 2) (vector-ref a1 2)))) (let ((wr (vector 'tensor-handle-v2 \"torch\" raw_r))) ((top-level-value 'idris-tensor-guardian) wr) ((top-level-value 'idris-ffi-tensor-retain-handle-torch) raw_r) wr)))"
prim__matmulTorch : AnyPtr -> AnyPtr -> AnyPtr

%foreign "scheme:(lambda (a0 a1 a2)  (when (not (top-level-bound? 'idris-ffi-tensor-linear-torch)) (set-top-level-value! 'idris-ffi-tensor-linear-torch (foreign-procedure \"tensor_linear_torch\" (void* void* void*) void*))) (when (not (top-level-bound? 'idris-ffi-tensor-retain-handle-torch)) (set-top-level-value! 'idris-ffi-tensor-retain-handle-torch (foreign-procedure \"tensor_retain_handle_torch\" (void*) void))) (let ((raw_r ((top-level-value 'idris-ffi-tensor-linear-torch) (vector-ref a0 2) (vector-ref a1 2) (vector-ref a2 2)))) (let ((wr (vector 'tensor-handle-v2 \"torch\" raw_r))) ((top-level-value 'idris-tensor-guardian) wr) ((top-level-value 'idris-ffi-tensor-retain-handle-torch) raw_r) wr)))"
prim__linearTorch : AnyPtr -> AnyPtr -> AnyPtr -> AnyPtr

%foreign "scheme:(lambda (a0 a1)  (when (not (top-level-bound? 'idris-ffi-tensor-dot-torch)) (set-top-level-value! 'idris-ffi-tensor-dot-torch (foreign-procedure \"tensor_dot_torch\" (void* void*) void*))) (when (not (top-level-bound? 'idris-ffi-tensor-retain-handle-torch)) (set-top-level-value! 'idris-ffi-tensor-retain-handle-torch (foreign-procedure \"tensor_retain_handle_torch\" (void*) void))) (let ((raw_r ((top-level-value 'idris-ffi-tensor-dot-torch) (vector-ref a0 2) (vector-ref a1 2)))) (let ((wr (vector 'tensor-handle-v2 \"torch\" raw_r))) ((top-level-value 'idris-tensor-guardian) wr) ((top-level-value 'idris-ffi-tensor-retain-handle-torch) raw_r) wr)))"
prim__dotTorch : AnyPtr -> AnyPtr -> AnyPtr

%foreign "scheme:(lambda (a0 a1)  (when (not (top-level-bound? 'idris-ffi-tensor-outer-torch)) (set-top-level-value! 'idris-ffi-tensor-outer-torch (foreign-procedure \"tensor_outer_torch\" (void* void*) void*))) (when (not (top-level-bound? 'idris-ffi-tensor-retain-handle-torch)) (set-top-level-value! 'idris-ffi-tensor-retain-handle-torch (foreign-procedure \"tensor_retain_handle_torch\" (void*) void))) (let ((raw_r ((top-level-value 'idris-ffi-tensor-outer-torch) (vector-ref a0 2) (vector-ref a1 2)))) (let ((wr (vector 'tensor-handle-v2 \"torch\" raw_r))) ((top-level-value 'idris-tensor-guardian) wr) ((top-level-value 'idris-ffi-tensor-retain-handle-torch) raw_r) wr)))"
prim__outerTorch : AnyPtr -> AnyPtr -> AnyPtr

%foreign "scheme:(lambda (a0 a1)  (when (not (top-level-bound? 'idris-ffi-tensor-bmm-torch)) (set-top-level-value! 'idris-ffi-tensor-bmm-torch (foreign-procedure \"tensor_bmm_torch\" (void* void*) void*))) (when (not (top-level-bound? 'idris-ffi-tensor-retain-handle-torch)) (set-top-level-value! 'idris-ffi-tensor-retain-handle-torch (foreign-procedure \"tensor_retain_handle_torch\" (void*) void))) (let ((raw_r ((top-level-value 'idris-ffi-tensor-bmm-torch) (vector-ref a0 2) (vector-ref a1 2)))) (let ((wr (vector 'tensor-handle-v2 \"torch\" raw_r))) ((top-level-value 'idris-tensor-guardian) wr) ((top-level-value 'idris-ffi-tensor-retain-handle-torch) raw_r) wr)))"
prim__bmmTorch : AnyPtr -> AnyPtr -> AnyPtr

%foreign "scheme:(lambda (a0 a1 a2)  (when (not (top-level-bound? 'idris-ffi-tensor-linear-2d-torch)) (set-top-level-value! 'idris-ffi-tensor-linear-2d-torch (foreign-procedure \"tensor_linear_2d_torch\" (void* void* void*) void*))) (when (not (top-level-bound? 'idris-ffi-tensor-retain-handle-torch)) (set-top-level-value! 'idris-ffi-tensor-retain-handle-torch (foreign-procedure \"tensor_retain_handle_torch\" (void*) void))) (let ((raw_r ((top-level-value 'idris-ffi-tensor-linear-2d-torch) (vector-ref a0 2) (vector-ref a1 2) (vector-ref a2 2)))) (let ((wr (vector 'tensor-handle-v2 \"torch\" raw_r))) ((top-level-value 'idris-tensor-guardian) wr) ((top-level-value 'idris-ffi-tensor-retain-handle-torch) raw_r) wr)))"
prim__linear2dTorch : AnyPtr -> AnyPtr -> AnyPtr -> AnyPtr

%foreign "scheme:(lambda (a0)  (when (not (top-level-bound? 'idris-ffi-tensor-sum-torch)) (set-top-level-value! 'idris-ffi-tensor-sum-torch (foreign-procedure \"tensor_sum_torch\" (void*) void*))) (when (not (top-level-bound? 'idris-ffi-tensor-retain-handle-torch)) (set-top-level-value! 'idris-ffi-tensor-retain-handle-torch (foreign-procedure \"tensor_retain_handle_torch\" (void*) void))) (let ((raw_r ((top-level-value 'idris-ffi-tensor-sum-torch) (vector-ref a0 2)))) (let ((wr (vector 'tensor-handle-v2 \"torch\" raw_r))) ((top-level-value 'idris-tensor-guardian) wr) ((top-level-value 'idris-ffi-tensor-retain-handle-torch) raw_r) wr)))"
prim__sumTorch : AnyPtr -> AnyPtr

%foreign "scheme:(lambda (a0)  (when (not (top-level-bound? 'idris-ffi-tensor-mean-torch)) (set-top-level-value! 'idris-ffi-tensor-mean-torch (foreign-procedure \"tensor_mean_torch\" (void*) void*))) (when (not (top-level-bound? 'idris-ffi-tensor-retain-handle-torch)) (set-top-level-value! 'idris-ffi-tensor-retain-handle-torch (foreign-procedure \"tensor_retain_handle_torch\" (void*) void))) (let ((raw_r ((top-level-value 'idris-ffi-tensor-mean-torch) (vector-ref a0 2)))) (let ((wr (vector 'tensor-handle-v2 \"torch\" raw_r))) ((top-level-value 'idris-tensor-guardian) wr) ((top-level-value 'idris-ffi-tensor-retain-handle-torch) raw_r) wr)))"
prim__meanTorch : AnyPtr -> AnyPtr

%foreign "scheme:(lambda (a0)  (when (not (top-level-bound? 'idris-ffi-tensor-min-torch)) (set-top-level-value! 'idris-ffi-tensor-min-torch (foreign-procedure \"tensor_min_torch\" (void*) void*))) (when (not (top-level-bound? 'idris-ffi-tensor-retain-handle-torch)) (set-top-level-value! 'idris-ffi-tensor-retain-handle-torch (foreign-procedure \"tensor_retain_handle_torch\" (void*) void))) (let ((raw_r ((top-level-value 'idris-ffi-tensor-min-torch) (vector-ref a0 2)))) (let ((wr (vector 'tensor-handle-v2 \"torch\" raw_r))) ((top-level-value 'idris-tensor-guardian) wr) ((top-level-value 'idris-ffi-tensor-retain-handle-torch) raw_r) wr)))"
prim__tensorMinTorch : AnyPtr -> AnyPtr

%foreign "scheme:(lambda (a0)  (when (not (top-level-bound? 'idris-ffi-tensor-max-torch)) (set-top-level-value! 'idris-ffi-tensor-max-torch (foreign-procedure \"tensor_max_torch\" (void*) void*))) (when (not (top-level-bound? 'idris-ffi-tensor-retain-handle-torch)) (set-top-level-value! 'idris-ffi-tensor-retain-handle-torch (foreign-procedure \"tensor_retain_handle_torch\" (void*) void))) (let ((raw_r ((top-level-value 'idris-ffi-tensor-max-torch) (vector-ref a0 2)))) (let ((wr (vector 'tensor-handle-v2 \"torch\" raw_r))) ((top-level-value 'idris-tensor-guardian) wr) ((top-level-value 'idris-ffi-tensor-retain-handle-torch) raw_r) wr)))"
prim__tensorMaxTorch : AnyPtr -> AnyPtr

%foreign "scheme:(lambda (a0 a1 a2)  (when (not (top-level-bound? 'idris-ffi-tensor-sum-dim-torch)) (set-top-level-value! 'idris-ffi-tensor-sum-dim-torch (foreign-procedure \"tensor_sum_dim_torch\" (void* int int) void*))) (when (not (top-level-bound? 'idris-ffi-tensor-retain-handle-torch)) (set-top-level-value! 'idris-ffi-tensor-retain-handle-torch (foreign-procedure \"tensor_retain_handle_torch\" (void*) void))) (let ((raw_r ((top-level-value 'idris-ffi-tensor-sum-dim-torch) (vector-ref a0 2) a1 a2))) (let ((wr (vector 'tensor-handle-v2 \"torch\" raw_r))) ((top-level-value 'idris-tensor-guardian) wr) ((top-level-value 'idris-ffi-tensor-retain-handle-torch) raw_r) wr)))"
prim__sumDimTorch : AnyPtr -> Int -> Int -> AnyPtr

%foreign "scheme:(lambda (a0 a1 a2)  (when (not (top-level-bound? 'idris-ffi-tensor-select-torch)) (set-top-level-value! 'idris-ffi-tensor-select-torch (foreign-procedure \"tensor_select_torch\" (void* int int) void*))) (when (not (top-level-bound? 'idris-ffi-tensor-retain-handle-torch)) (set-top-level-value! 'idris-ffi-tensor-retain-handle-torch (foreign-procedure \"tensor_retain_handle_torch\" (void*) void))) (let ((raw_r ((top-level-value 'idris-ffi-tensor-select-torch) (vector-ref a0 2) a1 a2))) (let ((wr (vector 'tensor-handle-v2 \"torch\" raw_r))) ((top-level-value 'idris-tensor-guardian) wr) ((top-level-value 'idris-ffi-tensor-retain-handle-torch) raw_r) wr)))"
prim__selectTorch : AnyPtr -> Int -> Int -> AnyPtr

%foreign "scheme:(lambda (a0 a1)  (when (not (top-level-bound? 'idris-ffi-tensor-unsqueeze-torch)) (set-top-level-value! 'idris-ffi-tensor-unsqueeze-torch (foreign-procedure \"tensor_unsqueeze_torch\" (void* int) void*))) (when (not (top-level-bound? 'idris-ffi-tensor-retain-handle-torch)) (set-top-level-value! 'idris-ffi-tensor-retain-handle-torch (foreign-procedure \"tensor_retain_handle_torch\" (void*) void))) (let ((raw_r ((top-level-value 'idris-ffi-tensor-unsqueeze-torch) (vector-ref a0 2) a1))) (let ((wr (vector 'tensor-handle-v2 \"torch\" raw_r))) ((top-level-value 'idris-tensor-guardian) wr) ((top-level-value 'idris-ffi-tensor-retain-handle-torch) raw_r) wr)))"
prim__unsqueezeTorch : AnyPtr -> Int -> AnyPtr

%foreign "scheme:(lambda (a0 a1)  (when (not (top-level-bound? 'idris-ffi-tensor-squeeze-torch)) (set-top-level-value! 'idris-ffi-tensor-squeeze-torch (foreign-procedure \"tensor_squeeze_torch\" (void* int) void*))) (when (not (top-level-bound? 'idris-ffi-tensor-retain-handle-torch)) (set-top-level-value! 'idris-ffi-tensor-retain-handle-torch (foreign-procedure \"tensor_retain_handle_torch\" (void*) void))) (let ((raw_r ((top-level-value 'idris-ffi-tensor-squeeze-torch) (vector-ref a0 2) a1))) (let ((wr (vector 'tensor-handle-v2 \"torch\" raw_r))) ((top-level-value 'idris-tensor-guardian) wr) ((top-level-value 'idris-ffi-tensor-retain-handle-torch) raw_r) wr)))"
prim__squeezeTorch : AnyPtr -> Int -> AnyPtr

%foreign "scheme:(lambda (a0 a1 a2)  (when (not (top-level-bound? 'idris-ffi-tensor-stack-torch)) (set-top-level-value! 'idris-ffi-tensor-stack-torch (foreign-procedure \"tensor_stack_torch\" (void* int int) void*))) (when (not (top-level-bound? 'idris-ffi-tensor-retain-handle-torch)) (set-top-level-value! 'idris-ffi-tensor-retain-handle-torch (foreign-procedure \"tensor_retain_handle_torch\" (void*) void))) (let ((raw_r ((top-level-value 'idris-ffi-tensor-stack-torch) a0 a1 a2))) (let ((wr (vector 'tensor-handle-v2 \"torch\" raw_r))) ((top-level-value 'idris-tensor-guardian) wr) ((top-level-value 'idris-ffi-tensor-retain-handle-torch) raw_r) wr)))"
prim__stackTorch : AnyPtr -> Int -> Int -> AnyPtr

%foreign "scheme:(lambda (a0 a1)  (when (not (top-level-bound? 'idris-ffi-tensor-batch-torch)) (set-top-level-value! 'idris-ffi-tensor-batch-torch (foreign-procedure \"tensor_batch_torch\" (void* int) void*))) (when (not (top-level-bound? 'idris-ffi-tensor-retain-handle-torch)) (set-top-level-value! 'idris-ffi-tensor-retain-handle-torch (foreign-procedure \"tensor_retain_handle_torch\" (void*) void))) (let ((raw_r ((top-level-value 'idris-ffi-tensor-batch-torch) a0 a1))) (let ((wr (vector 'tensor-handle-v2 \"torch\" raw_r))) ((top-level-value 'idris-tensor-guardian) wr) ((top-level-value 'idris-ffi-tensor-retain-handle-torch) raw_r) wr)))"
prim__batchTorch : AnyPtr -> Int -> AnyPtr

%foreign "scheme:(lambda (a0 a1)  (when (not (top-level-bound? 'idris-ffi-tensor-view-1d-torch)) (set-top-level-value! 'idris-ffi-tensor-view-1d-torch (foreign-procedure \"tensor_view_1d_torch\" (void* int) void*))) (when (not (top-level-bound? 'idris-ffi-tensor-retain-handle-torch)) (set-top-level-value! 'idris-ffi-tensor-retain-handle-torch (foreign-procedure \"tensor_retain_handle_torch\" (void*) void))) (let ((raw_r ((top-level-value 'idris-ffi-tensor-view-1d-torch) (vector-ref a0 2) a1))) (let ((wr (vector 'tensor-handle-v2 \"torch\" raw_r))) ((top-level-value 'idris-tensor-guardian) wr) ((top-level-value 'idris-ffi-tensor-retain-handle-torch) raw_r) wr)))"
prim__view1dTorch : AnyPtr -> Int -> AnyPtr

%foreign "scheme:(lambda (a0 a1 a2)  (when (not (top-level-bound? 'idris-ffi-tensor-view-2d-torch)) (set-top-level-value! 'idris-ffi-tensor-view-2d-torch (foreign-procedure \"tensor_view_2d_torch\" (void* int int) void*))) (when (not (top-level-bound? 'idris-ffi-tensor-retain-handle-torch)) (set-top-level-value! 'idris-ffi-tensor-retain-handle-torch (foreign-procedure \"tensor_retain_handle_torch\" (void*) void))) (let ((raw_r ((top-level-value 'idris-ffi-tensor-view-2d-torch) (vector-ref a0 2) a1 a2))) (let ((wr (vector 'tensor-handle-v2 \"torch\" raw_r))) ((top-level-value 'idris-tensor-guardian) wr) ((top-level-value 'idris-ffi-tensor-retain-handle-torch) raw_r) wr)))"
prim__view2dTorch : AnyPtr -> Int -> Int -> AnyPtr

%foreign "scheme:(lambda (a0 a1)  (let ((raw_r ((foreign-procedure \"tensor_reshape_1d_torch\" (void* int) void*) (vector-ref a0 2) a1))) (let ((wr (vector 'tensor-handle-v2 \"torch\" raw_r))) ((top-level-value 'idris-tensor-guardian) wr) ((foreign-procedure \"tensor_retain_handle_torch\" (void*) void) raw_r) wr)))"
prim__reshape1dTorch : AnyPtr -> Int -> AnyPtr

%foreign "scheme:(lambda (a0 a1 a2)  (when (not (top-level-bound? 'idris-ffi-tensor-reshape-2d-torch)) (set-top-level-value! 'idris-ffi-tensor-reshape-2d-torch (foreign-procedure \"tensor_reshape_2d_torch\" (void* int int) void*))) (when (not (top-level-bound? 'idris-ffi-tensor-retain-handle-torch)) (set-top-level-value! 'idris-ffi-tensor-retain-handle-torch (foreign-procedure \"tensor_retain_handle_torch\" (void*) void))) (let ((raw_r ((top-level-value 'idris-ffi-tensor-reshape-2d-torch) (vector-ref a0 2) a1 a2))) (let ((wr (vector 'tensor-handle-v2 \"torch\" raw_r))) ((top-level-value 'idris-tensor-guardian) wr) ((top-level-value 'idris-ffi-tensor-retain-handle-torch) raw_r) wr)))"
prim__reshape2dTorch : AnyPtr -> Int -> Int -> AnyPtr

%foreign "scheme:(lambda (a0 a1 a2 a3)  (when (not (top-level-bound? 'idris-ffi-tensor-reshape-3d-torch)) (set-top-level-value! 'idris-ffi-tensor-reshape-3d-torch (foreign-procedure \"tensor_reshape_3d_torch\" (void* int int int) void*))) (when (not (top-level-bound? 'idris-ffi-tensor-retain-handle-torch)) (set-top-level-value! 'idris-ffi-tensor-retain-handle-torch (foreign-procedure \"tensor_retain_handle_torch\" (void*) void))) (let ((raw_r ((top-level-value 'idris-ffi-tensor-reshape-3d-torch) (vector-ref a0 2) a1 a2 a3))) (let ((wr (vector 'tensor-handle-v2 \"torch\" raw_r))) ((top-level-value 'idris-tensor-guardian) wr) ((top-level-value 'idris-ffi-tensor-retain-handle-torch) raw_r) wr)))"
prim__reshape3dTorch : AnyPtr -> Int -> Int -> Int -> AnyPtr

%foreign "scheme:(lambda (a0 a1 a2 a3 a4)  (when (not (top-level-bound? 'idris-ffi-tensor-reshape-4d-torch)) (set-top-level-value! 'idris-ffi-tensor-reshape-4d-torch (foreign-procedure \"tensor_reshape_4d_torch\" (void* int int int int) void*))) (when (not (top-level-bound? 'idris-ffi-tensor-retain-handle-torch)) (set-top-level-value! 'idris-ffi-tensor-retain-handle-torch (foreign-procedure \"tensor_retain_handle_torch\" (void*) void))) (let ((raw_r ((top-level-value 'idris-ffi-tensor-reshape-4d-torch) (vector-ref a0 2) a1 a2 a3 a4))) (let ((wr (vector 'tensor-handle-v2 \"torch\" raw_r))) ((top-level-value 'idris-tensor-guardian) wr) ((top-level-value 'idris-ffi-tensor-retain-handle-torch) raw_r) wr)))"
prim__reshape4dTorch : AnyPtr -> Int -> Int -> Int -> Int -> AnyPtr

%foreign "scheme:(lambda (a0 a1 a2)  (when (not (top-level-bound? 'idris-ffi-tensor-tile-2d-torch)) (set-top-level-value! 'idris-ffi-tensor-tile-2d-torch (foreign-procedure \"tensor_tile_2d_torch\" (void* int int) void*))) (when (not (top-level-bound? 'idris-ffi-tensor-retain-handle-torch)) (set-top-level-value! 'idris-ffi-tensor-retain-handle-torch (foreign-procedure \"tensor_retain_handle_torch\" (void*) void))) (let ((raw_r ((top-level-value 'idris-ffi-tensor-tile-2d-torch) (vector-ref a0 2) a1 a2))) (let ((wr (vector 'tensor-handle-v2 \"torch\" raw_r))) ((top-level-value 'idris-tensor-guardian) wr) ((top-level-value 'idris-ffi-tensor-retain-handle-torch) raw_r) wr)))"
export
prim__tile2dTorch : AnyPtr -> Int -> Int -> AnyPtr

%foreign "scheme:(lambda (a0 a1 a2 a3)  (when (not (top-level-bound? 'idris-ffi-tensor-narrow-torch)) (set-top-level-value! 'idris-ffi-tensor-narrow-torch (foreign-procedure \"tensor_narrow_torch\" (void* int int int) void*))) (when (not (top-level-bound? 'idris-ffi-tensor-retain-handle-torch)) (set-top-level-value! 'idris-ffi-tensor-retain-handle-torch (foreign-procedure \"tensor_retain_handle_torch\" (void*) void))) (let ((raw_r ((top-level-value 'idris-ffi-tensor-narrow-torch) (vector-ref a0 2) a1 a2 a3))) (let ((wr (vector 'tensor-handle-v2 \"torch\" raw_r))) ((top-level-value 'idris-tensor-guardian) wr) ((top-level-value 'idris-ffi-tensor-retain-handle-torch) raw_r) wr)))"
prim__narrowTorch : AnyPtr -> Int -> Int -> Int -> AnyPtr

%foreign "scheme:(lambda (a0)  (when (not (top-level-bound? 'idris-ffi-tensor-transpose-last2-torch)) (set-top-level-value! 'idris-ffi-tensor-transpose-last2-torch (foreign-procedure \"tensor_transpose_last2_torch\" (void*) void*))) (when (not (top-level-bound? 'idris-ffi-tensor-retain-handle-torch)) (set-top-level-value! 'idris-ffi-tensor-retain-handle-torch (foreign-procedure \"tensor_retain_handle_torch\" (void*) void))) (let ((raw_r ((top-level-value 'idris-ffi-tensor-transpose-last2-torch) (vector-ref a0 2)))) (let ((wr (vector 'tensor-handle-v2 \"torch\" raw_r))) ((top-level-value 'idris-tensor-guardian) wr) ((top-level-value 'idris-ffi-tensor-retain-handle-torch) raw_r) wr)))"
prim__transposeLast2Torch : AnyPtr -> AnyPtr

%foreign "scheme:(lambda (a0)  (when (not (top-level-bound? 'idris-ffi-tensor-transpose-2d-torch)) (set-top-level-value! 'idris-ffi-tensor-transpose-2d-torch (foreign-procedure \"tensor_transpose_2d_torch\" (void*) void*))) (when (not (top-level-bound? 'idris-ffi-tensor-retain-handle-torch)) (set-top-level-value! 'idris-ffi-tensor-retain-handle-torch (foreign-procedure \"tensor_retain_handle_torch\" (void*) void))) (let ((raw_r ((top-level-value 'idris-ffi-tensor-transpose-2d-torch) (vector-ref a0 2)))) (let ((wr (vector 'tensor-handle-v2 \"torch\" raw_r))) ((top-level-value 'idris-tensor-guardian) wr) ((top-level-value 'idris-ffi-tensor-retain-handle-torch) raw_r) wr)))"
prim__transpose2dTorch : AnyPtr -> AnyPtr

%foreign "scheme:(lambda (a0 a1 a2)  (when (not (top-level-bound? 'idris-ffi-tensor-cat-torch)) (set-top-level-value! 'idris-ffi-tensor-cat-torch (foreign-procedure \"tensor_cat_torch\" (void* int int) void*))) (when (not (top-level-bound? 'idris-ffi-tensor-retain-handle-torch)) (set-top-level-value! 'idris-ffi-tensor-retain-handle-torch (foreign-procedure \"tensor_retain_handle_torch\" (void*) void))) (let ((raw_r ((top-level-value 'idris-ffi-tensor-cat-torch) a0 a1 a2))) (let ((wr (vector 'tensor-handle-v2 \"torch\" raw_r))) ((top-level-value 'idris-tensor-guardian) wr) ((top-level-value 'idris-ffi-tensor-retain-handle-torch) raw_r) wr)))"
prim__catTorch : AnyPtr -> Int -> Int -> AnyPtr

%foreign "scheme:(lambda (a0 a1)  (when (not (top-level-bound? 'idris-ffi-tensor-cat2-torch)) (set-top-level-value! 'idris-ffi-tensor-cat2-torch (foreign-procedure \"tensor_cat2_torch\" (void* void*) void*))) (when (not (top-level-bound? 'idris-ffi-tensor-retain-handle-torch)) (set-top-level-value! 'idris-ffi-tensor-retain-handle-torch (foreign-procedure \"tensor_retain_handle_torch\" (void*) void))) (let ((raw_r ((top-level-value 'idris-ffi-tensor-cat2-torch) (vector-ref a0 2) (vector-ref a1 2)))) (let ((wr (vector 'tensor-handle-v2 \"torch\" raw_r))) ((top-level-value 'idris-tensor-guardian) wr) ((top-level-value 'idris-ffi-tensor-retain-handle-torch) raw_r) wr)))"
prim__cat2Torch : AnyPtr -> AnyPtr -> AnyPtr

%foreign "scheme:(lambda (a0 a1)  (when (not (top-level-bound? 'idris-ffi-tensor-concat-2d-axis1-torch)) (set-top-level-value! 'idris-ffi-tensor-concat-2d-axis1-torch (foreign-procedure \"tensor_concat_2d_axis1_torch\" (void* void*) void*))) (when (not (top-level-bound? 'idris-ffi-tensor-retain-handle-torch)) (set-top-level-value! 'idris-ffi-tensor-retain-handle-torch (foreign-procedure \"tensor_retain_handle_torch\" (void*) void))) (let ((raw_r ((top-level-value 'idris-ffi-tensor-concat-2d-axis1-torch) (vector-ref a0 2) (vector-ref a1 2)))) (let ((wr (vector 'tensor-handle-v2 \"torch\" raw_r))) ((top-level-value 'idris-tensor-guardian) wr) ((top-level-value 'idris-ffi-tensor-retain-handle-torch) raw_r) wr)))"
prim__concat2dAxis1Torch : AnyPtr -> AnyPtr -> AnyPtr

%foreign "scheme:(lambda (a0 a1 a2)  (when (not (top-level-bound? 'idris-ffi-tensor-gather-torch)) (set-top-level-value! 'idris-ffi-tensor-gather-torch (foreign-procedure \"tensor_gather_torch\" (void* void* int) void*))) (when (not (top-level-bound? 'idris-ffi-tensor-retain-handle-torch)) (set-top-level-value! 'idris-ffi-tensor-retain-handle-torch (foreign-procedure \"tensor_retain_handle_torch\" (void*) void))) (let ((raw_r ((top-level-value 'idris-ffi-tensor-gather-torch) (vector-ref a0 2) (vector-ref a1 2) a2))) (let ((wr (vector 'tensor-handle-v2 \"torch\" raw_r))) ((top-level-value 'idris-tensor-guardian) wr) ((top-level-value 'idris-ffi-tensor-retain-handle-torch) raw_r) wr)))"
prim__gatherTorch : AnyPtr -> AnyPtr -> Int -> AnyPtr

%foreign "scheme:(lambda (a0 a1 a2 a3)  (when (not (top-level-bound? 'idris-ffi-tensor-gather-rows-torch)) (set-top-level-value! 'idris-ffi-tensor-gather-rows-torch (foreign-procedure \"tensor_gather_rows_torch\" (void* void* int int) void*))) (when (not (top-level-bound? 'idris-ffi-tensor-retain-handle-torch)) (set-top-level-value! 'idris-ffi-tensor-retain-handle-torch (foreign-procedure \"tensor_retain_handle_torch\" (void*) void))) (let ((raw_r ((top-level-value 'idris-ffi-tensor-gather-rows-torch) (vector-ref a0 2) (vector-ref a1 2) a2 a3))) (let ((wr (vector 'tensor-handle-v2 \"torch\" raw_r))) ((top-level-value 'idris-tensor-guardian) wr) ((top-level-value 'idris-ffi-tensor-retain-handle-torch) raw_r) wr)))"
prim__gatherRowsTorch : AnyPtr -> AnyPtr -> Int -> Int -> AnyPtr

%foreign "scheme:(lambda (a0 a1 a2)  (when (not (top-level-bound? 'idris-ffi-tensor-max-rows-torch)) (set-top-level-value! 'idris-ffi-tensor-max-rows-torch (foreign-procedure \"tensor_max_rows_torch\" (void* int int) void*))) (when (not (top-level-bound? 'idris-ffi-tensor-retain-handle-torch)) (set-top-level-value! 'idris-ffi-tensor-retain-handle-torch (foreign-procedure \"tensor_retain_handle_torch\" (void*) void))) (let ((raw_r ((top-level-value 'idris-ffi-tensor-max-rows-torch) (vector-ref a0 2) a1 a2))) (let ((wr (vector 'tensor-handle-v2 \"torch\" raw_r))) ((top-level-value 'idris-tensor-guardian) wr) ((top-level-value 'idris-ffi-tensor-retain-handle-torch) raw_r) wr)))"
prim__maxRowsTorch : AnyPtr -> Int -> Int -> AnyPtr

%foreign "scheme:(lambda (a0 a1 a2)  (when (not (top-level-bound? 'idris-ffi-tensor-scatter-add-torch)) (set-top-level-value! 'idris-ffi-tensor-scatter-add-torch (foreign-procedure \"tensor_scatter_add_torch\" (void* void* int) void*))) (when (not (top-level-bound? 'idris-ffi-tensor-retain-handle-torch)) (set-top-level-value! 'idris-ffi-tensor-retain-handle-torch (foreign-procedure \"tensor_retain_handle_torch\" (void*) void))) (let ((raw_r ((top-level-value 'idris-ffi-tensor-scatter-add-torch) (vector-ref a0 2) (vector-ref a1 2) a2))) (let ((wr (vector 'tensor-handle-v2 \"torch\" raw_r))) ((top-level-value 'idris-tensor-guardian) wr) ((top-level-value 'idris-ffi-tensor-retain-handle-torch) raw_r) wr)))"
prim__scatterAddTorch : AnyPtr -> AnyPtr -> Int -> AnyPtr

%foreign "scheme:(lambda (a0 a1 a2)  (when (not (top-level-bound? 'idris-ffi-tensor-argsort-torch)) (set-top-level-value! 'idris-ffi-tensor-argsort-torch (foreign-procedure \"tensor_argsort_torch\" (void* int int) void*))) (when (not (top-level-bound? 'idris-ffi-tensor-retain-handle-torch)) (set-top-level-value! 'idris-ffi-tensor-retain-handle-torch (foreign-procedure \"tensor_retain_handle_torch\" (void*) void))) (let ((raw_r ((top-level-value 'idris-ffi-tensor-argsort-torch) (vector-ref a0 2) a1 a2))) (let ((wr (vector 'tensor-handle-v2 \"torch\" raw_r))) ((top-level-value 'idris-tensor-guardian) wr) ((top-level-value 'idris-ffi-tensor-retain-handle-torch) raw_r) wr)))"
prim__argsortTorch : AnyPtr -> Int -> Int -> AnyPtr

%foreign "scheme:(lambda (a0 a1)  (when (not (top-level-bound? 'idris-ffi-tensor-cumprod-torch)) (set-top-level-value! 'idris-ffi-tensor-cumprod-torch (foreign-procedure \"tensor_cumprod_torch\" (void* int) void*))) (when (not (top-level-bound? 'idris-ffi-tensor-retain-handle-torch)) (set-top-level-value! 'idris-ffi-tensor-retain-handle-torch (foreign-procedure \"tensor_retain_handle_torch\" (void*) void))) (let ((raw_r ((top-level-value 'idris-ffi-tensor-cumprod-torch) (vector-ref a0 2) a1))) (let ((wr (vector 'tensor-handle-v2 \"torch\" raw_r))) ((top-level-value 'idris-tensor-guardian) wr) ((top-level-value 'idris-ffi-tensor-retain-handle-torch) raw_r) wr)))"
prim__cumprodTorch : AnyPtr -> Int -> AnyPtr

public export
{d : TorchHwDev} -> UserExecutorLinear (TorchExecutor d) where
  -- >>> GENERATED FROM ffi_manifest.py — gen-executor-instances.py >>>
  primArgsort        = prim__argsortTorch
  primBatch          = prim__batchTorch
  primBmm            = prim__bmmTorch
  primCat            = prim__catTorch
  primCat2           = prim__cat2Torch
  primConcat2dAxis1  = prim__concat2dAxis1Torch
  primCumprod        = prim__cumprodTorch
  primDot            = prim__dotTorch
  primGather         = prim__gatherTorch
  primGatherRows     = prim__gatherRowsTorch
  primLinear         = prim__linearTorch
  primLinear2d       = prim__linear2dTorch
  primMatmul         = prim__matmulTorch
  primMaxRows        = prim__maxRowsTorch
  primMean           = prim__meanTorch
  primMm             = prim__mmTorch
  primMv             = prim__mvTorch
  primNarrow         = prim__narrowTorch
  primOuter          = prim__outerTorch
  primReshape1d      = prim__reshape1dTorch
  primReshape2d      = prim__reshape2dTorch
  primReshape3d      = prim__reshape3dTorch
  primReshape4d      = prim__reshape4dTorch
  primScatterAdd     = prim__scatterAddTorch
  primSelect         = prim__selectTorch
  primSqueeze        = prim__squeezeTorch
  primStack          = prim__stackTorch
  primSum            = prim__sumTorch
  primSumDim         = prim__sumDimTorch
  primTensorMax      = prim__tensorMaxTorch
  primTensorMin      = prim__tensorMinTorch
  primTranspose2d    = prim__transpose2dTorch
  primTransposeLast2 = prim__transposeLast2Torch
  primUnsqueeze      = prim__unsqueezeTorch
  primView1d         = prim__view1dTorch
  primView2d         = prim__view2dTorch
  -- <<< END GENERATED <<<
