||| Host-fingerprint probe primitives that Machine instances compose
||| into per-Machine verification logic. Wrappers over the C-side
||| helpers in `packages/backends/probes.{c,h}`.
|||
||| Each probe is a one-shot syscall / filesystem stat / dlopen; cheap
||| enough that running all four at program startup is negligible.
||| Idris-side wrappers translate the C-side small-int enums into
||| structured `OS` / `Arch` values so Machine instances can pattern
||| match cleanly.
module Machine.Verify.Probes

import Data.IORef

%foreign "C:idrisml_probe_os,libidrisml"
prim__probeOS : PrimIO Int

%foreign "C:idrisml_probe_arch,libidrisml"
prim__probeArch : PrimIO Int

%foreign "C:idrisml_probe_metal_available,libidrisml"
prim__probeMetalAvailable : PrimIO Int

%foreign "C:idrisml_probe_cuda_available,libidrisml"
prim__probeCudaAvailable : PrimIO Int

public export
data OS = Darwin | Linux | Windows | UnknownOS

public export
Show OS where
  show Darwin    = "darwin"
  show Linux     = "linux"
  show Windows   = "windows"
  show UnknownOS = "unknown-os"

public export
Eq OS where
  Darwin    == Darwin    = True
  Linux     == Linux     = True
  Windows   == Windows   = True
  UnknownOS == UnknownOS = True
  _         == _         = False

public export
data Arch = Arm64 | X86_64 | UnknownArch

public export
Show Arch where
  show Arm64       = "arm64"
  show X86_64      = "x86_64"
  show UnknownArch = "unknown-arch"

public export
Eq Arch where
  Arm64       == Arm64       = True
  X86_64      == X86_64      = True
  UnknownArch == UnknownArch = True
  _           == _           = False

||| Probe the host's OS family. Reads `uname` once; safe to call
||| repeatedly but typically invoked once at startup via
||| `requireMachine`.
export
probeOS : IO OS
probeOS = do
  n <- primIO prim__probeOS
  pure $ case n of
    0 => Darwin
    1 => Linux
    2 => Windows
    _ => UnknownOS

||| Probe the host's CPU architecture.
export
probeArch : IO Arch
probeArch = do
  n <- primIO prim__probeArch
  pure $ case n of
    0 => Arm64
    1 => X86_64
    _ => UnknownArch

||| Returns True iff Apple's Metal framework is installed on the host.
||| Always False on non-Darwin hosts.
export
probeMetalAvailable : IO Bool
probeMetalAvailable = do
  n <- primIO prim__probeMetalAvailable
  pure (n /= 0)

||| Returns True iff `libcuda.so.1` can be dlopen'd on the host. Always
||| False on Darwin (no NVIDIA driver path).
export
probeCudaAvailable : IO Bool
probeCudaAvailable = do
  n <- primIO prim__probeCudaAvailable
  pure (n /= 0)
