||| Machine — open kind of physical compute environments.
|||
||| A Machine bundles a set of Hardware classes that a particular
||| physical box provides. Built-in presets cover common configurations
||| (M-series Mac, Intel Mac, Linux + CUDA, etc.). The user picks the
||| Machine matching their setup at build time; the build links code
||| for all provided hardware; then HARDWARE selects which one is the
||| example default. Other linked hardware stays reachable via explicit
||| `toExecutor` calls.
|||
||| Open by extension: a custom box (Jetson Orin, AMD ROCm workstation,
||| TPU pod) declares its own `Machine` tag and `Provides` instances in
||| its own module. The build flow accepts `MACHINE=<name>` for any
||| value that has a `Provides` instance.
|||
||| Phase E will introduce a low-priority TODO for distributed
||| multi-machine flows, where a program is parameterised over more
||| than one `Machine` (e.g. M-series host + CUDA workstation reachable
||| via RPC).
module Ml.Machine

import Ml.Hardware

public export 0 Machine : Type
Machine = Type

||| Apple Silicon Mac (M1, M2, M3, M4 family) — CPU + Apple GPU.
||| No CUDA on these machines (PCIe absent, no arm64 NVIDIA driver).
public export data MacMSeries : Type where MkMacMSeries : MacMSeries

||| Intel-era Mac — CPU only. (Some old Intel Macs had Thunderbolt
||| eGPU + NVIDIA driver support; that historical config can be
||| modelled with a custom Machine tag if needed.)
public export data MacIntel : Type where MkMacIntel : MacIntel

||| Intel workstation with N CUDA GPUs.
public export data IntelCuda : Nat -> Type where MkIntelCuda : (n : Nat) -> IntelCuda n

||| Linux box with CPU only.
public export data LinuxCpu : Type where MkLinuxCpu : LinuxCpu

||| Linux box with N CUDA GPUs.
public export data LinuxCuda : Nat -> Type where MkLinuxCuda : (n : Nat) -> LinuxCuda n

||| A Machine provides a Hardware class. Used by `Preset` (Phase A.4) to
||| verify at compile time that the requested Hardware is reachable on
||| the chosen Machine.
public export
interface Provides (0 m : Machine) (0 h : Hardware) where

public export Provides MacMSeries Cpu      where
public export Provides MacMSeries AppleGpu where
public export Provides MacIntel   Cpu      where
public export {n : Nat} -> Provides (IntelCuda n) Cpu where
public export {n : Nat} -> Provides (IntelCuda n) (Cuda 0) where
public export Provides LinuxCpu Cpu where
public export {n : Nat} -> Provides (LinuxCuda n) Cpu where
public export {n : Nat} -> Provides (LinuxCuda n) (Cuda 0) where
-- Multi-index Cuda enumeration (Cuda 1 .. Cuda n-1) is the High-Priority
-- CUDA TODO row's job; for now only Cuda 0 is admitted from each Cuda-
-- equipped Machine.

-- MachineRuntimeCheck instances for the built-in Machine tags live in
-- `Machine.BuiltinChecks` to avoid a circular import (Machine.Verify
-- depends on this module for the `Machine` kind alias). Users adding
-- their own Machine tag write their `MachineRuntimeCheck` instance
-- alongside the tag in their own module.
