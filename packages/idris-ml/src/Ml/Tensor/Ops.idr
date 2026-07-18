||| Elementwise / linear-algebra ops, activations, recurrent cells,
||| losses, the scalar boundary, and the infix operator aliases.
module Ml.Tensor.Ops

import Data.Vect

import Ml.DType.Core
import Ml.Executor
import Ml.GradMode
import Ml.Tensor.Core
import Ml.Tensor.Internal

-- Arithmetic / linear algebra (autograd-tracked) ----------------------

||| Elementwise addition. Both operands share shape.
||| `%inline`: inlines to a direct `prim__add` + `MkTensor` allocation
||| at every call site. Critical for hot-path layers (LSTM/NTM/DNC
||| call this many times per timestep); without inlining, Idris2's
||| Chez codegen wraps each invocation in a closure dispatch that
||| adds ~20µs of Scheme-side overhead per call, accumulating to a
||| 2× regression on recurrent models.
export %inline
tadd : {0 ex : Executor} -> UserExecutorCore ex => Tensor dims ex dt g -> Tensor dims ex dt g -> IO (Tensor dims ex dt g)
tadd a b = ioRerun (\_ => MkTensor (primAdd {ex} a.tensorPtr b.tensorPtr) Nothing)

||| Matrix-vector multiply: [m, n] · [n] -> [m]. `%inline` for the
||| same reason as `tadd` (hot path in recurrent forward passes).
export %inline
tmv : {0 ex : Executor} -> UserExecutorTraining ex =>
      Tensor [m, n] ex dt g -> Tensor [n] ex dt g -> IO (Tensor [m] ex dt g)
tmv w x = ioRerun (\_ => MkTensor (primMv {ex} w.tensorPtr x.tensorPtr) Nothing)

||| Fused 1D linear: y = W[m,n] · x[n] + bias[m]. One C call instead
||| of `tadd (tmv W x) bias` — collapses two FFI hops into one and
||| eliminates the intermediate Idris-side glue. Used by `Nn.Linear`'s
||| forward and by NTM/DNC FCs.
export %inline
tlinear : {0 ex : Executor} -> UserExecutorTraining ex =>
          Tensor [o, i] ex dt g -> Tensor [i] ex dt g -> Tensor [o] ex dt g -> IO (Tensor [o] ex dt g)
tlinear w x bias = ioRerun (\_ =>
  MkTensor (primLinear {ex} w.tensorPtr x.tensorPtr bias.tensorPtr) Nothing)

||| Fused batched linear: W[o,i] · X^T[b,i] + bias[o] -> [b, o].
export %inline
tlinear2d : {0 ex : Executor} -> UserExecutorTraining ex =>
            Tensor [o, i] ex dt g -> Tensor [b, i] ex dt g -> Tensor [o] ex dt g -> IO (Tensor [b, o] ex dt g)
tlinear2d w x bias = ioRerun (\_ =>
  MkTensor (primLinear2d {ex} w.tensorPtr x.tensorPtr bias.tensorPtr) Nothing)

-- Per-sample extraction + scalar arithmetic (used by batched RL loss
-- builders: pluck a row from a [b, o] result, then a scalar from the
-- row, then build (q - target)^2 etc.) ---------------------------------

||| Row-wise gather: out[i] = t[i, indices[i]] — the typed
||| `q.gather(1, a.unsqueeze(1)).squeeze(1)` (PyTorch). Indices are a
||| [b] tensor of double-valued ints (the established index
||| convention; cf. the integer-dtyped 1-D `tgather` below, which is
||| torch-only); they carry no gradient, so any GradMode is accepted
||| on the index side. Backward scatters to the selected cells.
export %inline
tgatherRows : {0 ex : Executor} -> UserExecutorTraining ex => {b, n : Nat} ->
              Tensor [b, n] ex dt g -> Tensor [b] ex dt gi -> IO (Tensor [b] ex dt g)
tgatherRows t idx = ioRerun (\_ =>
  MkTensor (primGatherRows {ex} t.tensorPtr idx.tensorPtr (cast {to=Int} b) (cast {to=Int} n)) Nothing)

||| Row-wise max: out[i] = max_j t[i, j] — PyTorch's
||| `t.max(1).values`. Backward routes each row's gradient to its
||| argmax cell (tie-breaking unspecified across backends).
export %inline
tmaxRows : {0 ex : Executor} -> UserExecutorTraining ex => {b, n : Nat} ->
           Tensor [b, n] ex dt g -> IO (Tensor [b] ex dt g)
tmaxRows t = ioRerun (\_ =>
  MkTensor (primMaxRows {ex} t.tensorPtr (cast {to=Int} b) (cast {to=Int} n)) Nothing)

||| Select row `k` from a [b, n] Tensor, returning the n-vector slice.
||| Wraps `prim__select` on dim 0; preserves the autograd graph.
export
trowSelect : {0 ex : Executor} -> UserExecutorTraining ex => {b, n : Nat} ->
             Tensor [b, n] ex dt g -> Int -> IO (Tensor [n] ex dt g)
trowSelect t k = ioRerun (\_ => MkTensor (primSelect {ex} t.tensorPtr 0 k) Nothing)

||| Select element `i` from an n-vector, returning a scalar Tensor.
export
telemSelect : {0 ex : Executor} -> UserExecutorTraining ex => {n : Nat} ->
              Tensor [n] ex dt g -> Int -> IO (Tensor [] ex dt g)
telemSelect t i = ioRerun (\_ => MkTensor (primSelect {ex} t.tensorPtr 0 i) Nothing)

||| Subtract two equally-shaped Tensors (autograd-tracked).
export %inline
tsub : {0 ex : Executor} -> UserExecutorCore ex => Tensor dims ex dt g -> Tensor dims ex dt g -> IO (Tensor dims ex dt g)
tsub a b = ioRerun (\_ => MkTensor (primSub {ex} a.tensorPtr b.tensorPtr) Nothing)

||| Elementwise multiply two equally-shaped Tensors (autograd-tracked).
export %inline
tmul : {0 ex : Executor} -> UserExecutorCore ex => Tensor dims ex dt g -> Tensor dims ex dt g -> IO (Tensor dims ex dt g)
tmul a b = ioRerun (\_ => MkTensor (primMul {ex} a.tensorPtr b.tensorPtr) Nothing)

||| Negate a Tensor (autograd-tracked).
export %inline
tneg : {0 ex : Executor} -> UserExecutorCore ex => Tensor dims ex dt g -> IO (Tensor dims ex dt g)
tneg a = ioRerun (\_ => MkTensor (primNeg {ex} a.tensorPtr) Nothing)

||| Scale a Tensor by a Double (broadcasts the scalar; autograd-tracked).
||| Useful for mean-reduction (`tmulScalar loss (1.0 / cast n)`) and for
||| building per-sample loss expressions where one side of a product is
||| a runtime Double (e.g. DQN target value).
export %inline
tmulScalar : {0 ex : Executor} -> UserExecutorCore ex => Tensor dims ex dt g -> Double -> IO (Tensor dims ex dt g)
tmulScalar v s = ioRerun (\_ => MkTensor (primMulScalar {ex} v.tensorPtr s) Nothing)

----------------------------------------------------------------------
-- Expression operator aliases
----------------------------------------------------------------------
--
-- Infix spellings of the elementwise ops, on plain evaluated tensors,
-- returning IO — used with bang notation:
--
--   tgt <- r +. !(gamma *: !(tmaxRows qNext))
--
-- Deliberately NOT a Num instance and NOT on IO carriers (roadmap.md
-- decision 5): ops consume already-evaluated tensors, so nothing can
-- silently re-execute and there is no sharing combinator to teach.
-- Precedences mirror Prelude (+)/(*) so mixed expressions parse as
-- expected. `(*:)` is scalar-on-the-left, reading like PyTorch's
-- `0.99 * q`.

export infixl 6 +., -.
export infixl 7 *., *:

export %inline
(+.) : {0 ex : Executor} -> UserExecutorCore ex => Tensor dims ex dt g -> Tensor dims ex dt g -> IO (Tensor dims ex dt g)
(+.) = tadd

export %inline
(-.) : {0 ex : Executor} -> UserExecutorCore ex => Tensor dims ex dt g -> Tensor dims ex dt g -> IO (Tensor dims ex dt g)
(-.) = tsub

export %inline
(*.) : {0 ex : Executor} -> UserExecutorCore ex => Tensor dims ex dt g -> Tensor dims ex dt g -> IO (Tensor dims ex dt g)
(*.) = tmul

export %inline
(*:) : {0 ex : Executor} -> UserExecutorCore ex => Double -> Tensor dims ex dt g -> IO (Tensor dims ex dt g)
(*:) s v = tmulScalar v s

||| Elementwise exponential (autograd-tracked).
export %inline
texp : {0 ex : Executor} -> UserExecutorCore ex => Tensor dims ex dt g -> IO (Tensor dims ex dt g)
texp v = ioRerun (\_ => MkTensor (primExp {ex} v.tensorPtr) Nothing)

||| Elementwise natural log (autograd-tracked).
export %inline
tlog : {0 ex : Executor} -> UserExecutorCore ex => Tensor dims ex dt g -> IO (Tensor dims ex dt g)
tlog v = ioRerun (\_ => MkTensor (primLog {ex} v.tensorPtr) Nothing)

||| Concatenate two [b, m] / [b, n] TVars along axis 1, producing
||| [b, m + n]. Wraps `prim__concat2dAxis1`. Used by SAC's actor loss
||| to build a [B, ObsDim + ActDim] Q-input from obs + reparametrized
||| action while preserving the autograd path through the action.
export
tconcat2dAxis1 : {0 ex : Executor} -> UserExecutorTraining ex => {b, m, n : Nat} ->
                 Tensor [b, m] ex dt g -> Tensor [b, n] ex dt g ->
                 IO (Tensor [b, m + n] ex dt g)
tconcat2dAxis1 a b = ioRerun (\_ => MkTensor (primConcat2dAxis1 {ex} a.tensorPtr b.tensorPtr) Nothing)

||| Concatenate two [a, n] / [b, n] TVars along axis 0, producing
||| [a + b, n]. Wraps `primCat2` (which under the hood is
||| `torch::cat({a, b}, 0)` / `mx::concatenate({a, b}, 0)` /
||| tape's `tensor_cat2` — all rank-preserving). Distinct from
||| `tconcat2dAxis1` (axis-1 cat); pairs with it.
|||
||| Used by KV cache append: previous-step K (shape [len, kvOut])
||| concatenated with new-step K (shape [s, kvOut]) → [len + s,
||| kvOut]. The `kvOut` (trailing) dim must match on both inputs.
export
tconcat2dAxis0 : {0 ex : Executor} -> UserExecutorTraining ex => {a, b, n : Nat} ->
                 Tensor [a, n] ex dt g -> Tensor [b, n] ex dt g ->
                 IO (Tensor [a + b, n] ex dt g)
tconcat2dAxis0 x y = ioRerun (\_ => MkTensor (primCat2 {ex} x.tensorPtr y.tensorPtr) Nothing)

-- Activations (shape-preserving, pass-through autograd) ---------------
-- All `%inline` for hot-path performance — see `tadd` rationale.

export %inline
ttanh : {0 ex : Executor} -> UserExecutorCore ex => Tensor dims ex dt g -> IO (Tensor dims ex dt g)
ttanh v = ioRerun (\_ => MkTensor (primTanh {ex} v.tensorPtr) Nothing)

export %inline
tsigmoid : {0 ex : Executor} -> UserExecutorCore ex => Tensor dims ex dt g -> IO (Tensor dims ex dt g)
tsigmoid v = ioRerun (\_ => MkTensor (primSigmoid {ex} v.tensorPtr) Nothing)

export %inline
trelu : {0 ex : Executor} -> UserExecutorCore ex => Tensor dims ex dt g -> IO (Tensor dims ex dt g)
trelu v = ioRerun (\_ => MkTensor (primClampMin {ex} v.tensorPtr 0.0) Nothing)

||| Two-sided element-wise clamp: r[i] = min(max(t[i], lo), hi).
||| NoGrad output (the kernel is inference-only — file the
||| differentiable variant if a training path needs it). Bridges
||| straight to `tensor_clamp` on each backend.
export
tclamp : {0 ex : Executor} -> UserExecutorCore ex =>
         (lo, hi : Double) -> Tensor dims ex dt g -> IO (Tensor dims ex dt NoGrad)
tclamp lo hi v = ioRerun (\_ => MkTensor (primClamp {ex} v.tensorPtr lo hi) Nothing)

||| Element-wise round-to-nearest-even (banker's rounding — matches
||| `torch.round` and `mx::round`). NoGrad output. Used by the
||| BitNet activation quantization recipe.
export
tround : {0 ex : Executor} -> UserExecutorCore ex =>
         Tensor dims ex dt g -> IO (Tensor dims ex dt NoGrad)
tround v = ioRerun (\_ => MkTensor (primRound {ex} v.tensorPtr) Nothing)

||| Element-wise absolute value. Bridges to `primAbs` on each backend.
||| Inference-only — the autograd story for `abs` is sign-of-x times
||| upstream-grad which we don't yet thread; file the differentiable
||| variant if a training path needs it.
export
tabs : {0 ex : Executor} -> UserExecutorCore ex =>
       Tensor dims ex dt g -> IO (Tensor dims ex dt NoGrad)
tabs v = ioRerun (\_ => MkTensor (primAbs {ex} v.tensorPtr) Nothing)

||| Per-token symmetric int8 activation quantization (BitNet b1.58 /
||| HF microsoft/bitnet-b1.58-2B-4T recipe). Given a [n] activation:
|||
|||   input_scale = 127 / max(|x|).clamp(min=1e-5)        -- scalar
|||   x_quant     = round(x * input_scale).clamp(-128, 127)
|||
|||  Returns `(x_quant, input_scale)`. `x_quant`'s values lie in the
|||  int8 grid but the storage stays in the compute dtype — the BitNet
|||  forward composes this with `tBitlinearFwd` which dequants by
|||  `1 / (input_scale * weight_scale)` post-matmul.
|||
|||  The clamp-min of 1e-5 on `max(|x|)` matches HF transformers'
|||  `BitLinear.activation_quant` (`packages/pytorch/.venv/.../
|||  transformers/integrations/bitnet.py`). Pure Idris composition of
|||  `tabs` / `primTensorMax` / scalar arithmetic / `tmulScalar` /
|||  `tround` / `tclamp` — no new C kernel.
export
tActivationQuantInt8 : {0 ex : Executor} -> UserExecutorCore ex =>
                       UserExecutorLinear ex =>
                       {n : Nat} -> Tensor [n] ex dt g ->
                       IO (Tensor [n] ex dt NoGrad, Double)
tActivationQuantInt8 x = do
  xAbs <- tabs x
  let xMax = primItem {ex} (primTensorMax {ex} xAbs.tensorPtr)
      safeMax = if xMax > 1.0e-5 then xMax else 1.0e-5
      inScale = 127.0 / safeMax
  scaled <- tmulScalar x inScale
  rounded <- tround scaled
  clamped <- tclamp (-128.0) 127.0 rounded
  pure (clamped, inScale)

export %inline
tgelu : {0 ex : Executor} -> UserExecutorTraining ex => Tensor dims ex dt g -> IO (Tensor dims ex dt g)
tgelu v = ioRerun (\_ => MkTensor (primGelu {ex} v.tensorPtr) Nothing)

export %inline
tsilu : {0 ex : Executor} -> UserExecutorTraining ex => Tensor dims ex dt g -> IO (Tensor dims ex dt g)
tsilu v = ioRerun (\_ => MkTensor (primSilu {ex} v.tensorPtr) Nothing)

export %inline
tleakyRelu : {0 ex : Executor} -> UserExecutorTraining ex => Double -> Tensor dims ex dt g -> IO (Tensor dims ex dt g)
tleakyRelu slope v = ioRerun (\_ => MkTensor (primLeakyRelu {ex} v.tensorPtr slope) Nothing)

||| Softmax along axis 0 (1D vector).
export %inline
tsoftmax1d : {0 ex : Executor} -> UserExecutorTraining ex => {n : Nat} -> Tensor [n] ex dt g -> IO (Tensor [n] ex dt g)
tsoftmax1d v = ioRerun (\_ => MkTensor (primSoftmax {ex} v.tensorPtr 0) Nothing)

||| Log-softmax along axis 0 (1D vector).
export %inline
tlogSoftmax1d : {0 ex : Executor} -> UserExecutorTraining ex => {n : Nat} -> Tensor [n] ex dt g -> IO (Tensor [n] ex dt g)
tlogSoftmax1d v = ioRerun (\_ => MkTensor (primLogSoftmax {ex} v.tensorPtr 0) Nothing)

||| Fused LSTM gate computation: combined gates [4 * n] + previous cell [n]
||| → (new hidden [n], new cell [n]). Wraps `prim__lstmGatesPair`.
|||
||| The gate-vector size is encoded statically as `TVec (4 * n) d`
||| (alias for `Tensor [4 * n] ex`). Routing the `4 * n` through the
||| `TVec` alias avoids the type-checker hang that direct
||| `Tensor [4 * n] ex` triggers.
export
tlstmGatesPair : UserExecutorNN ex => {n : Nat} -> TVec (4 * n) ex dt g -> TVec n ex dt g ->
                 IO (TVec n ex dt g, TVec n ex dt g)
tlstmGatesPair {n} combined prevCell = ioRerun (\_ =>
  let nI = cast {to=Int} n
      pair = primLstmGatesPair {ex} combined.tensorPtr prevCell.tensorPtr nI
  in (MkTensor (primPairFirst {ex} pair) Nothing, MkTensor (primPairSecond {ex} pair) Nothing))

||| Allocate a zero-initialised persistent state Tensor of size [n].
||| Use for LSTM/RNN/GRU initial hidden + cell state. Persistent =
||| survives tape reset.
export
tzeroState1d : {0 ex : Executor} -> Backend ex dt => {n : Nat} -> IO (Tensor [n] ex dt g)
tzeroState1d {n} = ioRerun (\_ =>
  let nI = cast {to=Int} n
      buf = prim__allocDoubles nI
  in MkTensor (dtCreateState1d {ex} {t=dt} nI buf (deviceStreamTag {ex})) Nothing)

||| GRU cell — `nn.GRU` equation. Takes the two `[3 * n]` half-sums:
|||   ih = W_ih @ x + b_ih
|||   hh = W_hh @ h + b_hh
||| (computed by the caller via `tlinear`) plus the previous hidden
||| state. Internally:
|||   z = sigmoid(ih_z + hh_z),  r = sigmoid(ih_r + hh_r)
|||   n = tanh(ih_n + r * hh_n)
|||   h' = (1 - z) * n + z * prev
||| Pre-2026-05-09 this took a single fused `combined = ih + hh`
||| and ignored r (simplified GRU); aligned to the standard
||| `nn.GRU` equation so the example matches what library users
||| expect.
export
tgruCell : UserExecutorNN ex => {n : Nat} -> TVec (3 * n) ex dt g -> TVec (3 * n) ex dt g -> TVec n ex dt g -> IO (TVec n ex dt g)
tgruCell {n} ih hh prevH = ioRerun (\_ =>
  let nI = cast {to=Int} n
  in MkTensor (primGruCell {ex} ih.tensorPtr hh.tensorPtr prevH.tensorPtr nI) Nothing)

-- Scalar boundary --------------------------------------------------

||| Read the scalar value out of a `Tensor [] ex`.
export
tensorItem : UserExecutorCore ex => Tensor [] ex dt g -> Double
tensorItem v = primItem {ex} v.tensorPtr

||| Run backward on a loss tensor. The loss MUST be `WithGrad` —
||| a `NoGrad` scalar can't have come from a path the autograd tape
||| recorded, so backward would be a silent no-op at best and a
||| malformed-tape crash at worst. Rejecting at the type level
||| catches "loss computed inside `withNoGrad`, then fed to training"
||| — the bug class the entire `GradMode` refactor exists to prevent.
export
runBackward : UserExecutorTraining ex => IsFloating dt => Tensor [] ex dt WithGrad -> IO ()
runBackward t = primIO (primBackward {ex} t.tensorPtr)

-- Loss (vector targets → scalar loss) ---------------------------------

||| MSE loss over a 1D prediction/target pair. Sum-reduced.
export
tmseLoss : {0 ex : Executor} -> UserExecutorLinear ex => IsFloating dt => {n : Nat} ->
           Tensor [n] ex dt g -> Tensor [n] ex dt g -> IO (Tensor [] ex dt g)
tmseLoss p t = ioRerun (\_ =>
  let diff   = primSub {ex} p.tensorPtr t.tensorPtr in
  let sqDiff = primMul {ex} diff diff in
  MkTensor (primSum {ex} sqDiff) Nothing)

||| NLL loss against a one-hot target. Mirrors
||| `Example.Supervised.nllLossTensor` (divide by n to match the
||| reference's mean reduction).
export
tnllLoss : {0 ex : Executor} -> UserExecutorNN ex => IsFloating dt => {n : Nat} ->
           Tensor [n] ex dt g -> Tensor [n] ex dt g -> IO (Tensor [] ex dt g)
tnllLoss {n} p t = ioRerun (\_ =>
  let logP = primLogSoftmax {ex} p.tensorPtr 0 in
  let prod = primMul {ex} logP t.tensorPtr in
  let neg  = primNeg {ex} (primSum {ex} prod) in
  MkTensor (primMulScalar {ex} neg (1.0 / cast n)) Nothing)

||| Fused softmax cross-entropy with logits against soft/one-hot targets:
||| `-scale * (target * logSoftmax(pred, axis=1)).sum()` as ONE tape node
||| (replaces the decomposed logSoftmax → mul → sum → neg → mulScalar chain).
||| The caller picks the reduction `scale` (`1/(b*n)` for `tnllLossMean`,
||| `1/seqLen` for LM losses, `1/numMasked` for MLM). The tape F64 forward is
||| bit-identical to the decomposed chain.
export
tsoftmaxXent2d : {0 ex : Executor} -> UserExecutorOptimizations ex => IsFloating dt =>
                 {b, n : Nat} -> (scale : Double) ->
                 Tensor [b, n] ex dt g -> Tensor [b, n] ex dt g -> IO (Tensor [] ex dt g)
tsoftmaxXent2d scale p t = ioRerun (\_ =>
  MkTensor (primSoftmaxXent2d {ex} p.tensorPtr t.tensorPtr scale) Nothing)

||| Batched multiclass NLL against one-hot targets, mean-reduced over
||| `batch × classes`. The batched-first counterpart to `tnllLoss` for the
||| `Nn`/`fit` surface: `-(target * logSoftmax(pred, axis=1)).sum() / (b*n)`
||| — exactly the per-row `tnllLoss` (which carries the `1/n` classes factor)
||| meaned over the batch, and matching PyTorch's
||| `nll_loss(log_softmax(logits, -1), target)`. Since 2026-07-27 this is the
||| fused `tsoftmaxXent2d` at `scale = 1/(b*n)` — one tape node, bit-identical
||| F64 forward to the old decomposed chain.
export
tnllLossMean : {0 ex : Executor} -> UserExecutorOptimizations ex => IsFloating dt => {b, n : Nat} ->
               Tensor [b, n] ex dt g -> Tensor [b, n] ex dt g -> IO (Tensor [] ex dt g)
tnllLossMean {b} {n} p t = tsoftmaxXent2d (1.0 / cast (b * n)) p t

||| Binary cross-entropy with logits, mean-reduced. Numerically stable
||| (wraps `primBceWithLogits`). For multi-element predictions/targets
||| use `tbceLoss : Tensor [n] ex dt g-> Tensor [n] ex dt g-> Tensor [] ex dt g`;
||| the C op internally averages. Polymorphic in `g`: the loss's
||| grad-mode matches the predictions / targets, so a no-grad eval
||| `tbceLoss` (e.g. inside `withNoGrad`) returns a `NoGrad` scalar
||| that the type system will reject if accidentally fed to
||| `trainStep`.
export
tbceLoss : {0 ex : Executor} -> UserExecutorNN ex => IsFloating dt => {n : Nat} ->
           Tensor [n] ex dt g -> Tensor [n] ex dt g -> IO (Tensor [] ex dt g)
tbceLoss p t = ioRerun (\_ =>
  MkTensor (primBceWithLogits {ex} p.tensorPtr t.tensorPtr) Nothing)
