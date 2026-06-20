||| Executor-kind foundations: the `Executor` kind alias, the `Linked`
||| linkage-capability marker, and `HardwareClass`.
module Executor.Core.Kind

----------------------------------------------------------------------
-- `Executor` kind alias
--
-- `Executor` is a 0-quantity alias for `Type`. Tensor's `d` phantom is
-- declared as `(0 ex : Executor)`, which is exactly `(0 d : Type)`
-- underneath but reads as "d is a device tag" at every kind-binder
-- site. No type-system enforcement: nothing stops a caller writing
-- `Tensor [4] Bool`. But construction (`primCreate*`) and operations
-- (`tadd` etc.) both require `UserExecutorCore ex =>`, so non-device
-- `d`s can be declared but never inhabited or operated on.
--
-- See `docs/develop/design-decisions.md` "Open `d` kind: why
-- `Executor = Type` instead of a real sub-kind" for the alternatives
-- considered and why we kept it open.
----------------------------------------------------------------------

public export
0 Executor : Type
Executor = Type

----------------------------------------------------------------------
-- Linked — backend-linkage capability
--
-- Empty capability marker, sibling to `Compatible (device, dtype)`.
-- `Linked ex` declares "device `d`'s backend is compiled into this
-- `libidrisml`." Instances are NOT hardcoded here — they're emitted by
-- the generated `HwConfig` module from the build's `BACKEND` list, so a
-- torch-only build has no `Linked (MlxExecutor _)` instance and `MlxExecutor`
-- becomes unspellable at any constructor carrying the `Linked ex =>`
-- constraint. This is the compile-time *linkage* half of device
-- availability; the runtime *hardware-presence* half is EAFP (attempt
-- construction, catch the backend's exception). See
-- `docs/develop/device-availability-gating.md`.
--
-- Linkage is per-backend, not per-hardware-variant: a torch build admits
-- every `TorchExecutor hw` (TCpu / TMps / TCuda n) at the type level; whether
-- the MPS chip or `cuda:n` actually exists is the runtime question.
----------------------------------------------------------------------

public export
interface Linked (0 ex : Executor) where

----------------------------------------------------------------------
-- HardwareClass — physical-silicon classification (orthogonal to backend)
--
-- Backend-scoping (TorchExecutor TMps vs MlxExecutor MGpu) is correct: you can't
-- mix their tensor handles even though both live on the same Apple GPU.
-- But that scoping hides the hardware *commonality*. `HardwareClass`
-- recovers it as runtime data — for *reporting* / grouping during
-- discovery only. It never unifies tensor types: TMps and MGpu both map
-- to `AppleGpu`, yet their tensors still can't meet. See
-- `docs/develop/device-availability-gating.md`.
----------------------------------------------------------------------

public export
data HardwareClass = HostCpu | AppleGpu | Nvidia Nat | Other String

public export
Eq HardwareClass where
  HostCpu   == HostCpu  = True
  AppleGpu  == AppleGpu = True
  Nvidia m  == Nvidia n = m == n
  Other a   == Other b  = a == b
  _         == _        = False

public export
Show HardwareClass where
  show HostCpu    = "host-cpu"
  show AppleGpu   = "apple-gpu"
  show (Nvidia n) = "nvidia:" ++ show n
  show (Other s)  = s
