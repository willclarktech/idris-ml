||| Typed optimizer scoping by exact registry name — the v1 replacement for
||| string-prefix scoping (`adam {scope="q1_"}`, `OptimOpts.groups`, the old
||| `freezeByPrefix`/`Suffix`). You hand these the *exact* param names a
||| (sub)model owns; each sets the per-param LR override via `setParamLR`
||| (`-1` = the optimizer's base LR, `0` = frozen).
|||
||| Two leak-free sources of name lists:
|||   * `Nn.Group.groupOf` / `reflectNames` — structural, derived from a model's
|||     `Params` traversal (no string matching → no `q1_`-vs-`q1tgt_` leak).
|||     This is the source for multi-network ownership (`restrictTo`).
|||   * `namesMatching` — an explicit registry filter for the cases where the
|||     selection genuinely *is* a naming convention (HF checkpoint prefixes like
|||     `bert.*`, LoRA suffixes like `*lora_A`), opt-in and visible at the call
|||     site rather than hidden in a constructor string.
|||
||| Distinct from `Nn.Module`'s model-level `freeze`/`eval` (which flip C
||| `requires_grad` and retype the model): these only adjust optimizer LR
||| overrides on the flat registry, so they reach a backbone subset without
||| projecting a field out of a linear model.
module Train.Freeze

import Data.String

import Executor
import Tensor

||| Set the per-param LR override to `lr` for each named param. The names are
||| matched exactly (registry `setParamLR`), so a name not in the registry is a
||| silent no-op. `lr = 0` freezes; `lr = -1` restores the base LR.
export
setGroupLR : UserExecutorTraining ex => NativeOptimizer ex -> List String -> Double -> IO ()
setGroupLR opt names lr = traverse_ (\n => setParamLR {ex} opt n lr) names

||| Freeze the named params (LR override 0): subsequent steps leave them
||| unchanged. The fine-tune-backbone route — `freezeGroup opt =<<
||| namesMatching (isPrefixOf "bert.")`, or `freezeGroup opt (groupOf sub)`.
export
freezeGroup : UserExecutorTraining ex => NativeOptimizer ex -> List String -> IO ()
freezeGroup opt names = setGroupLR opt names 0.0

||| Clear the LR override on the named params (back to the optimizer's base LR).
||| Inverse of `freezeGroup`. Composes with it for the LoRA pattern: freeze the
||| backbone, then `unfreezeGroup opt =<< namesMatching (isSuffixOf "lora_A")`.
export
unfreezeGroup : UserExecutorTraining ex => NativeOptimizer ex -> List String -> IO ()
unfreezeGroup opt names = setGroupLR opt names (-1.0)

-- Walk registry indices k-1..0, applying `act name` for every non-buffer param.
walkNames : UserExecutorTraining ex => (String -> IO ()) -> Nat -> IO ()
walkNames act Z     = pure ()
walkNames act (S k) = do
  let i = cast {to=Int} k
  isBuf <- getParamIsBuffer {ex} i
  name  <- getParamName {ex} i
  when (not isBuf) (act name)
  walkNames {ex} act k

||| Scope an optimizer to exactly `keep`: every *other* registered (non-buffer)
||| param gets LR 0, so this optimizer's steps move only `keep`. The typed
||| replacement for `adam {scope="q1_"}` — pass `reflectNames net` (or
||| `groupOf net`) so ownership is the net's exact param set, eliminating the
||| substring-leak bug class. Run after the nets are registered (params added
||| later miss the walk — the same registry-order hazard the old scope had).
|||
||| Note on grad clipping: unlike the old C prefix-scope (which excluded
||| non-owned params from the optimizer entirely), restrictTo keeps them in the
||| set at LR 0, so a global `NormClip` aggregates their gradients into the norm.
||| This is inert in the intended multi-network pattern — each net's loss is
||| local, so the non-owned params carry exactly zero grad at this optimizer's
||| step and contribute nothing to the norm — but a loss that feeds grad to a
||| non-owned param would clip differently than the old scope did.
export
restrictTo : UserExecutorTraining ex => NativeOptimizer ex -> List String -> IO ()
restrictTo opt keep = do
  n <- getParamCount {ex}
  walkNames {ex} (\name => when (not (name `elem` keep)) (setParamLR {ex} opt name 0.0))
            (cast {to=Nat} n)

||| The registered (non-buffer) param names satisfying `pred`, in registry
||| order. The explicit escape hatch for name-pattern selection (HF prefixes,
||| LoRA suffixes) — `namesMatching (isPrefixOf "bert.")`. For structural model
||| subsets prefer `Nn.Group.groupOf` / `reflectNames`.
export
namesMatching : UserExecutorTraining ex => (String -> Bool) -> IO (List String)
namesMatching pred = do
  n <- getParamCount {ex}
  go (cast {to=Nat} n) []
  where
    go : Nat -> List String -> IO (List String)
    go Z     acc = pure acc
    go (S k) acc = do
      let i = cast {to=Int} k
      isBuf <- getParamIsBuffer {ex} i
      name  <- getParamName {ex} i
      go k (if not isBuf && pred name then name :: acc else acc)
