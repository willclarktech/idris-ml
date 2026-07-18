||| `MachineRuntimeCheck` instances for the built-in `Machine` tags
||| declared in `Machine`. Each composes primitives from
||| `Machine.Verify.Probes` into a per-Machine host check.
|||
||| Lives in its own module to keep `Machine.idr` clean of probe
||| dependencies (the kind alias module shouldn't pull C-side FFI) and
||| to break the import cycle that would otherwise arise between
||| `Machine` and `Machine.Verify`.
|||
||| Users adding a custom `Machine` tag in their own module declare
||| their `MachineRuntimeCheck` instance alongside the tag — they don't
||| extend this module.
module Ml.Machine.BuiltinChecks

import Ml.Machine
import Ml.Machine.Verify
import Ml.Machine.Verify.Probes

public export
MachineRuntimeCheck MacMSeries where
  verifyHost = do
    os    <- probeOS
    arch  <- probeArch
    metal <- probeMetalAvailable
    pure $ case (os, arch, metal) of
      (Darwin, Arm64, True) => HostMatches
      _                     => HostMismatch
        "expected darwin-arm64 with Metal framework; host is \{show os}-\{show arch} (metal=\{show metal})"

public export
MachineRuntimeCheck MacIntel where
  verifyHost = do
    os   <- probeOS
    arch <- probeArch
    pure $ case (os, arch) of
      (Darwin, X86_64) => HostMatches
      _                => HostMismatch
        "expected darwin-x86_64; host is \{show os}-\{show arch}"

public export
{n : Nat} -> MachineRuntimeCheck (IntelCuda n) where
  verifyHost = do
    os   <- probeOS
    arch <- probeArch
    cuda <- probeCudaAvailable
    pure $ case (os, arch, cuda) of
      (Linux, X86_64, True) => HostMatches
      _                     => HostMismatch
        "expected linux-x86_64 with libcuda.so.1; host is \{show os}-\{show arch} (cuda=\{show cuda})"

public export
MachineRuntimeCheck LinuxCpu where
  verifyHost = do
    os <- probeOS
    pure $ case os of
      Linux => HostMatches
      _     => HostMismatch "expected linux host; got \{show os}"

public export
{n : Nat} -> MachineRuntimeCheck (LinuxCuda n) where
  verifyHost = do
    os   <- probeOS
    cuda <- probeCudaAvailable
    pure $ case (os, cuda) of
      (Linux, True) => HostMatches
      _             => HostMismatch
        "expected linux host with libcuda.so.1; host is \{show os} (cuda=\{show cuda})"
