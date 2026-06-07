# mk/genconfig.mk — generated Idris config sources. BuildConfig /
# TestConfig / HwConfig / HwDevices emitted per (BACKEND, MLX_DEVICE,
# TORCH_DEVICE) via sed, with stamps so no-op rebuilds skip TTC churn.

# Stamp + generated source for the example device/dtype selection.
# The Selection matrix lives in BuildConfig.idr.in's module docstring;
# this rule observes PRIMARY + MLX_DEVICE + TORCH_DEVICE and emits the
# right (ExampleExecutor, ExampleDType) into BuildConfig.idr via sed.
# The stamp records the active tuple so the generation step only
# writes the source file when the config actually changes (avoiding
# TTC churn on no-op rebuilds), mirror of the .backend-stamp pattern.
BUILDCONFIG_KEY := $(PRIMARY):$(MLX_DEVICE):$(TORCH_DEVICE)
BUILDCONFIG_IDR := packages/idris-ml-examples/src/BuildConfig.idr
BUILDCONFIG_IN  := packages/idris-ml-examples/src/BuildConfig.idr.in

# Generated `Linked` instances for the compiled-in backends. Unlike
# BuildConfig (one example device/dtype cell), HwConfig emits a variable
# number of instance blocks — one per backend in BACKEND_LIST — so the
# recipe appends them to the .in header rather than sed-substituting.
# Keyed on the whole BACKEND list. Lives in the core library (the Executor
# barrel re-exports it); git-ignored, regenerated each build.
HWCONFIG_KEY := $(BACKEND)
HWCONFIG_IDR := packages/idris-ml/src/HwConfig.idr
HWCONFIG_IN  := packages/idris-ml/src/HwConfig.idr.in

# Generated `builtinExecutors : List SomeExecutor` — the value-level mirror of
# HwConfig's `Linked` instances (one `someExecutor` candidate per linked
# backend's admissible (device, dtype) cells). Lives downstream of `Tensor`
# (where `someExecutor` is defined), unlike HwConfig which the Executor barrel
# re-exports upstream. Keyed on the BACKEND list; git-ignored, regenerated.
HWDEVICES_IDR := packages/idris-ml/src/HwExecutors.idr
HWDEVICES_IN  := packages/idris-ml/src/HwExecutors.idr.in

# Generated `TestExecutor` / `TestDType` for the Idris unit test suite. Same
# template trick as BuildConfig (one cell, sed-substituted from the active
# PRIMARY × hw-device envs); lives in the test sourcedir (now colocated
# under src/Test/ alongside the rest of the test files — dual-ipkg pattern,
# see docs/develop/testing.md). Keyed on the same tuple.
TESTCONFIG_IDR := packages/idris-ml/src/Test/Config.idr
TESTCONFIG_IN  := packages/idris-ml/src/Test/Config.idr.in

# Sibling Test.Config for the idris-transformers test ipkg — same template
# pattern. The two ipkgs can't share a generated module (different sourcedirs),
# so each gets its own resolved copy keyed on the same (MACHINE, PRIMARY,
# HARDWARE) tuple.
IDRIS_TRANSFORMERS_TESTCONFIG_IDR := packages/idris-transformers/src/Test/Config.idr
IDRIS_TRANSFORMERS_TESTCONFIG_IN  := packages/idris-transformers/src/Test/Config.idr.in

# Always-touch the stamp so it's at least as fresh as the current `make`
# invocation, even when content matches. Without the trailing touch, a
# *different* backend set rewriting BuildConfig.idr (mtime bumps) would
# leave this set's stamp older than the (now-other-content) BuildConfig.idr,
# Make would consider BuildConfig.idr "fresh" relative to its dep, and skip
# the recipe — leaving torch content on disk during a tape build. The
# recipe for the generated `.idr` is cmp-then-mv, so always-running is
# essentially free when content matches.
$(BUILD)/.buildconfig-stamp: FORCE | $(BUILD)
	@if [ "$$(cat $@ 2>/dev/null)" != "$(BUILDCONFIG_KEY)" ]; then echo "$(BUILDCONFIG_KEY)" > $@; else touch $@; fi

# Write-if-different: render to `.tmp`, then `cmp` against the existing
# generated file and `mv` only when content actually differs. Within a
# single set's reruns this keeps the .idr's mtime stable across re-makes;
# across sets it correctly bumps mtime when the (device, dtype) changes.
# The per-set ttc cache then absorbs the cascade — the .idr re-elabs but
# downstream modules with matching interface hashes don't.
# BuildConfig/TestConfig generation maps `(MACHINE, PRIMARY, HARDWARE)`
# tuples to type-level tags substituted into the .in templates. The
# Preset typeclass then resolves `ExampleExecutor` / `ExampleDType` from
# `(PrimaryBackend, ChosenHardware)`; the `Provides` typeclass enforces
# coherence between `ChosenMachine` and `ChosenHardware`. Mismatched
# combinations fail at example-compile, not silently at runtime.
#
# Sed substitutions:
#   @CHOSEN_MACHINE_TAG@   ← MACHINE_RESOLVED → Idris Machine type
#   @PRIMARY_BACKEND_TAG@  ← PRIMARY          → Idris Backend type
#   @CHOSEN_HARDWARE_TAG@  ← HARDWARE_RESOLVED → Idris Hardware type
#   @PRIMARY@              ← PRIMARY          → string (TestConfig only)
# Helper: turn (PRIMARY, HARDWARE_RESOLVED) into the concrete Executor
# type (TapeExecutor / TorchExecutor TCpu / MlxExecutor MGpu / ...) and
# DType (F64 / F32) per the Preset typeclass's canonical mapping. We
# emit the resolved type directly into BuildConfig.idr because Idris-2
# typeclass methods don't unfold during constraint search — using
# `presetExecutor {b=PrimaryBackend} {h=ChosenHardware}` would leave
# `UserExecutorTraining (presetExecutor ...)` unresolvable.
#
# These mirror the Preset instances in:
#   * packages/idris-ml/src/Executor/Tape.idr  (TapeBackend × Cpu)
#   * packages/idris-ml/src/Executor/Torch.idr (TorchBackend × Cpu/AppleGpu/Cuda)
#   * packages/idris-ml/src/Executor/Mlx.idr   (MlxBackend × Cpu/AppleGpu)
# Adding a new (Backend, Hardware) Preset means updating BOTH here and
# the Idris instance.

$(BUILDCONFIG_IDR): $(BUILDCONFIG_IN) $(BUILD)/.buildconfig-stamp
	@case "$(MACHINE_RESOLVED)" in \
		mac-m-series)   MTAG="MacMSeries" ;; \
		mac-intel)      MTAG="MacIntel" ;; \
		intel-cuda-*)   MTAG="IntelCuda $$(echo $(MACHINE_RESOLVED) | sed 's/intel-cuda-//')" ;; \
		linux-cpu)      MTAG="LinuxCpu" ;; \
		linux-cuda-*)   MTAG="LinuxCuda $$(echo $(MACHINE_RESOLVED) | sed 's/linux-cuda-//')" ;; \
		*)              MTAG="MacMSeries" ;; \
	esac; \
	case "$(PRIMARY)" in \
		tape)  BTAG="TapeBackend" ;;  \
		torch) BTAG="TorchBackend" ;; \
		mlx)   BTAG="MlxBackend" ;;   \
		*)     BTAG="TapeBackend" ;;  \
	esac; \
	case "$(HARDWARE_RESOLVED)" in \
		cpu)    HTAG="Cpu" ;;       \
		metal)  HTAG="AppleGpu" ;;  \
		cuda)   HTAG="Cuda 0" ;;    \
		*)      HTAG="Cpu" ;;       \
	esac; \
	case "$(PRIMARY)/$(HARDWARE_RESOLVED)" in \
		tape/cpu)     ETYPE="TapeExecutor";              DTYPE="F64" ;; \
		torch/cpu)    ETYPE="TorchExecutor TCpu";        DTYPE="F64" ;; \
		torch/metal)  ETYPE="TorchExecutor TMps";        DTYPE="F32" ;; \
		torch/cuda)   ETYPE="TorchExecutor (TCuda 0)";   DTYPE="F64" ;; \
		mlx/cpu)      ETYPE="MlxExecutor MCpu";          DTYPE="F64" ;; \
		mlx/metal)    ETYPE="MlxExecutor MGpu";          DTYPE="F32" ;; \
		*)            ETYPE="TapeExecutor";              DTYPE="F64" ;; \
	esac; \
	if [ -n "$(TORCH_DTYPE)" ] && [ "$(PRIMARY)" = "torch" ]; then DTYPE="$(TORCH_DTYPE)"; fi; \
	if [ -n "$(MLX_DTYPE)"   ] && [ "$(PRIMARY)" = "mlx"   ]; then DTYPE="$(MLX_DTYPE)";   fi; \
	if [ -n "$(TAPE_DTYPE)"  ] && [ "$(PRIMARY)" = "tape"  ]; then DTYPE="$(TAPE_DTYPE)";  fi; \
	sed "s|@CHOSEN_MACHINE_TAG@|$$MTAG|g; s|@PRIMARY_BACKEND_TAG@|$$BTAG|g; s|@CHOSEN_HARDWARE_TAG@|$$HTAG|g; s|@EXAMPLE_EXECUTOR_TYPE@|$$ETYPE|g; s|@EXAMPLE_DTYPE_TYPE@|$$DTYPE|g" $< > $@.tmp; \
	if cmp -s $@.tmp $@ 2>/dev/null; then rm $@.tmp; else mv $@.tmp $@; fi
	@echo "[BuildConfig] MACHINE=$(MACHINE_RESOLVED) PRIMARY=$(PRIMARY) HARDWARE=$(HARDWARE_RESOLVED) → ExampleExecutor=$$(awk -F' = ' '/^ExampleExecutor = / { print $$2; exit }' $@) / ExampleDType=$$(awk -F' = ' '/^ExampleDType = / { print $$2; exit }' $@)"

$(TESTCONFIG_IDR): $(TESTCONFIG_IN) $(BUILD)/.buildconfig-stamp
	@case "$(MACHINE_RESOLVED)" in \
		mac-m-series)   MTAG="MacMSeries" ;; \
		mac-intel)      MTAG="MacIntel" ;; \
		intel-cuda-*)   MTAG="IntelCuda $$(echo $(MACHINE_RESOLVED) | sed 's/intel-cuda-//')" ;; \
		linux-cpu)      MTAG="LinuxCpu" ;; \
		linux-cuda-*)   MTAG="LinuxCuda $$(echo $(MACHINE_RESOLVED) | sed 's/linux-cuda-//')" ;; \
		*)              MTAG="MacMSeries" ;; \
	esac; \
	case "$(PRIMARY)" in \
		tape)  BTAG="TapeBackend" ;;  \
		torch) BTAG="TorchBackend" ;; \
		mlx)   BTAG="MlxBackend" ;;   \
		*)     BTAG="TapeBackend" ;;  \
	esac; \
	case "$(HARDWARE_RESOLVED)" in \
		cpu)    HTAG="Cpu" ;;       \
		metal)  HTAG="AppleGpu" ;;  \
		cuda)   HTAG="Cuda 0" ;;    \
		*)      HTAG="Cpu" ;;       \
	esac; \
	case "$(PRIMARY)/$(HARDWARE_RESOLVED)" in \
		tape/cpu)     ETYPE="TapeExecutor";              DTYPE="F64" ;; \
		torch/cpu)    ETYPE="TorchExecutor TCpu";        DTYPE="F64" ;; \
		torch/metal)  ETYPE="TorchExecutor TMps";        DTYPE="F32" ;; \
		torch/cuda)   ETYPE="TorchExecutor (TCuda 0)";   DTYPE="F64" ;; \
		mlx/cpu)      ETYPE="MlxExecutor MCpu";          DTYPE="F64" ;; \
		mlx/metal)    ETYPE="MlxExecutor MGpu";          DTYPE="F32" ;; \
		*)            ETYPE="TapeExecutor";              DTYPE="F64" ;; \
	esac; \
	if [ -n "$(TORCH_DTYPE)" ] && [ "$(PRIMARY)" = "torch" ]; then DTYPE="$(TORCH_DTYPE)"; fi; \
	if [ -n "$(MLX_DTYPE)"   ] && [ "$(PRIMARY)" = "mlx"   ]; then DTYPE="$(MLX_DTYPE)";   fi; \
	if [ -n "$(TAPE_DTYPE)"  ] && [ "$(PRIMARY)" = "tape"  ]; then DTYPE="$(TAPE_DTYPE)";  fi; \
	sed "s|@CHOSEN_MACHINE_TAG@|$$MTAG|g; s|@PRIMARY_BACKEND_TAG@|$$BTAG|g; s|@CHOSEN_HARDWARE_TAG@|$$HTAG|g; s|@EXAMPLE_EXECUTOR_TYPE@|$$ETYPE|g; s|@EXAMPLE_DTYPE_TYPE@|$$DTYPE|g; s|@PRIMARY@|$(PRIMARY)|g" $< > $@.tmp; \
	if cmp -s $@.tmp $@ 2>/dev/null; then rm $@.tmp; else mv $@.tmp $@; fi
	@echo "[TestConfig] MACHINE=$(MACHINE_RESOLVED) PRIMARY=$(PRIMARY) HARDWARE=$(HARDWARE_RESOLVED) → TestExecutor=$$(awk -F' = ' '/^TestExecutor = / { print $$2; exit }' $@) / TestDType=$$(awk -F' = ' '/^TestDType = / { print $$2; exit }' $@)"

# The transformers test ipkg uses the same template as idris-ml's; both
# resolve to the same content (same active PRIMARY × hw envs). Generated
# by re-running the same sed substitution against the transformers
# template — the two .in files stay in sync (mirror copies).
$(IDRIS_TRANSFORMERS_TESTCONFIG_IDR): $(IDRIS_TRANSFORMERS_TESTCONFIG_IN) $(BUILD)/.buildconfig-stamp
	@case "$(MACHINE_RESOLVED)" in \
		mac-m-series)   MTAG="MacMSeries" ;; \
		mac-intel)      MTAG="MacIntel" ;; \
		intel-cuda-*)   MTAG="IntelCuda $$(echo $(MACHINE_RESOLVED) | sed 's/intel-cuda-//')" ;; \
		linux-cpu)      MTAG="LinuxCpu" ;; \
		linux-cuda-*)   MTAG="LinuxCuda $$(echo $(MACHINE_RESOLVED) | sed 's/linux-cuda-//')" ;; \
		*)              MTAG="MacMSeries" ;; \
	esac; \
	case "$(PRIMARY)" in \
		tape)  BTAG="TapeBackend" ;;  \
		torch) BTAG="TorchBackend" ;; \
		mlx)   BTAG="MlxBackend" ;;   \
		*)     BTAG="TapeBackend" ;;  \
	esac; \
	case "$(HARDWARE_RESOLVED)" in \
		cpu)    HTAG="Cpu" ;;       \
		metal)  HTAG="AppleGpu" ;;  \
		cuda)   HTAG="Cuda 0" ;;    \
		*)      HTAG="Cpu" ;;       \
	esac; \
	case "$(PRIMARY)/$(HARDWARE_RESOLVED)" in \
		tape/cpu)     ETYPE="TapeExecutor";              DTYPE="F64" ;; \
		torch/cpu)    ETYPE="TorchExecutor TCpu";        DTYPE="F64" ;; \
		torch/metal)  ETYPE="TorchExecutor TMps";        DTYPE="F32" ;; \
		torch/cuda)   ETYPE="TorchExecutor (TCuda 0)";   DTYPE="F64" ;; \
		mlx/cpu)      ETYPE="MlxExecutor MCpu";          DTYPE="F64" ;; \
		mlx/metal)    ETYPE="MlxExecutor MGpu";          DTYPE="F32" ;; \
		*)            ETYPE="TapeExecutor";              DTYPE="F64" ;; \
	esac; \
	if [ -n "$(TORCH_DTYPE)" ] && [ "$(PRIMARY)" = "torch" ]; then DTYPE="$(TORCH_DTYPE)"; fi; \
	if [ -n "$(MLX_DTYPE)"   ] && [ "$(PRIMARY)" = "mlx"   ]; then DTYPE="$(MLX_DTYPE)";   fi; \
	if [ -n "$(TAPE_DTYPE)"  ] && [ "$(PRIMARY)" = "tape"  ]; then DTYPE="$(TAPE_DTYPE)";  fi; \
	sed "s|@CHOSEN_MACHINE_TAG@|$$MTAG|g; s|@PRIMARY_BACKEND_TAG@|$$BTAG|g; s|@CHOSEN_HARDWARE_TAG@|$$HTAG|g; s|@EXAMPLE_EXECUTOR_TYPE@|$$ETYPE|g; s|@EXAMPLE_DTYPE_TYPE@|$$DTYPE|g; s|@PRIMARY@|$(PRIMARY)|g" $< > $@.tmp; \
	if cmp -s $@.tmp $@ 2>/dev/null; then rm $@.tmp; else mv $@.tmp $@; fi
	@echo "[TestConfig:transformers] MACHINE=$(MACHINE_RESOLVED) PRIMARY=$(PRIMARY) HARDWARE=$(HARDWARE_RESOLVED) → TestExecutor=$$(awk -F' = ' '/^TestExecutor = / { print $$2; exit }' $@) / TestDType=$$(awk -F' = ' '/^TestDType = / { print $$2; exit }' $@)"

# Same always-touch logic as the .buildconfig-stamp recipe above; see
# the comment there for why a content-equal stamp still needs an mtime
# bump when other backend sets share the generated `.idr` file paths.
$(BUILD)/.hwconfig-stamp: FORCE | $(BUILD)
	@if [ "$$(cat $@ 2>/dev/null)" != "$(HWCONFIG_KEY)" ]; then echo "$(HWCONFIG_KEY)" > $@; else touch $@; fi

# Emit one `Linked` instance per backend in BACKEND_LIST, appended to the
# .in header. Per-backend linkage admits every hardware variant of that
# backend (runtime presence is the EAFP concern, not this gate).
# Write-if-different (see BuildConfig rule above for rationale).
$(HWCONFIG_IDR): $(HWCONFIG_IN) $(BUILD)/.hwconfig-stamp
	@{ cat $(HWCONFIG_IN); \
	   for b in $(BACKEND_LIST); do \
	     case $$b in \
	       tape)  printf 'public export\nLinked TapeExecutor where\n\n' ;; \
	       torch) printf 'public export\n{hw : TorchHwDev} -> Linked (TorchExecutor hw) where\n\n' ;; \
	       mlx)   printf 'public export\n{s : MlxStream} -> Linked (MlxExecutor s) where\n\n' ;; \
	     esac; \
	   done; \
	 } > $@.tmp
	@if cmp -s $@.tmp $@ 2>/dev/null; then rm $@.tmp; else mv $@.tmp $@; fi
	@echo "[HwConfig] BACKEND=$(BACKEND) → Linked instances for: $(BACKEND_LIST)"

# Emit `builtinExecutors` as `[] ++ <per-backend candidate lists>`. Seeding
# with `[]` keeps every backend fragment a uniform `++ [...]`, so a
# tape-only build is `[] ++ [TapeExecutor/F64]` and the empty BACKEND case is a
# well-typed `[]`. Each `someExecutor {ex} {dt}` resolves its Linked /
# Compatible / HardwareClassed / UserExecutorTape constraints from the
# instances brought in via `import Executor` / `import Tensor`. torch lists
# all three hw variants (TCpu/TMps/TCuda 0) — EAFP filters to what's
# present (multi-GPU `TCuda n` enumeration via cuda_device_count is a
# separate follow-up).
$(HWDEVICES_IDR): $(HWDEVICES_IN) $(BUILD)/.hwconfig-stamp
	@{ cat $(HWDEVICES_IN); \
	   printf 'public export\nbuiltinExecutors : List SomeExecutor\nbuiltinExecutors = []\n'; \
	   for b in $(BACKEND_LIST); do \
	     case $$b in \
	       tape)  printf '  ++ [someExecutor {ex = TapeExecutor} {dt = F64}]\n' ;; \
	       torch) printf '  ++ [ someExecutor {ex = TorchExecutor TCpu} {dt = F64}\n     , someExecutor {ex = TorchExecutor TMps} {dt = F32}\n     , someExecutor {ex = TorchExecutor (TCuda 0)} {dt = F64} ]\n' ;; \
	       mlx)   printf '  ++ [ someExecutor {ex = MlxExecutor MCpu} {dt = F64}\n     , someExecutor {ex = MlxExecutor MGpu} {dt = F32} ]\n' ;; \
	     esac; \
	   done; \
	 } > $@.tmp
	@if cmp -s $@.tmp $@ 2>/dev/null; then rm $@.tmp; else mv $@.tmp $@; fi
	@echo "[HwExecutors] BACKEND=$(BACKEND) → builtinExecutors for: $(BACKEND_LIST)"
