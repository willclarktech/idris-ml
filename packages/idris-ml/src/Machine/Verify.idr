||| Machine.Verify — runtime check that the host the binary is
||| executing on matches the host the build committed to.
|||
||| Three gates check backend / hardware availability against the
||| binary, at progressively-later points:
|||
|||   1. Compile-time `Linked` gate (`HwConfig.idr`). Generated per
|||      build; a tape-only build can't even spell `MlxExecutor _`.
|||   2. Startup `requireMachine` gate (this module). Resolves the
|||      build's `ExampleMachine` (or any caller-chosen Machine) and
|||      probes the host. Fail-fast on mismatch.
|||   3. First-use EAFP gate (`toExecutorChecked`). NULL handle → `Left
|||      DeviceError`. Catches sub-Machine-level failures, e.g. F64 on
|||      Metal at construction.
|||
||| Extensibility contract: every `Machine` MUST have a
||| `MachineRuntimeCheck` instance. A new contributor adding
||| `Machine.RaspberryPi5` with a `Preset` instance but no
||| `MachineRuntimeCheck` instance gets a clear `No implementation of
||| MachineRuntimeCheck RaspberryPi5` error at the first build that
||| references `requireMachine`. This is the enforcement mechanism —
||| typeclass resolution, not a hardcoded fingerprint table.
|||
||| Strictness: hard error by default; relax via env var
||| `IDRISML_MACHINE_CHECK=warn` (log + continue) or `=off` (skip).
module Machine.Verify

import System

import Machine
import Machine.Verify.Probes
import Util.Log

public export
data VerifyResult
  = HostMatches
  | HostMismatch String

||| Every `Machine` defines how to verify the live host matches it.
||| The check fires once at program startup via `requireMachine` and
||| should be cheap. Implementations compose primitives from
||| `Machine.Verify.Probes`.
public export
interface MachineRuntimeCheck (0 m : Machine) where
  verifyHost : IO VerifyResult

handleMismatch : String -> IO ()
handleMismatch diag = do
  mode <- getEnv "IDRISML_MACHINE_CHECK"
  case mode of
    Just "off"  => pure ()
    Just "OFF"  => pure ()
    Just "warn" => logWarn "machine check (warn): \{diag}"
    Just "WARN" => logWarn "machine check (warn): \{diag}"
    _ => do
      logError "idris-ml: machine check failed: \{diag}"
      logError "Set IDRISML_MACHINE_CHECK=warn to continue, =off to skip."
      exitFailure

||| Verify the host matches Machine `m`. Hard-errors via `exitFailure`
||| on mismatch unless `IDRISML_MACHINE_CHECK=warn|off` overrides.
|||
||| Typical usage at the top of an example's `main`:
|||
||| ```idris
||| main : IO ()
||| main = do
|||   requireMachine {m = ChosenMachine}
|||   ... rest of main ...
||| ```
export
requireMachine : MachineRuntimeCheck m => IO ()
requireMachine @{check} = do
  result <- verifyHost @{check}
  case result of
    HostMatches => pure ()
    HostMismatch diag => handleMismatch diag
