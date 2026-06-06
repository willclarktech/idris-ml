||| Idris-side wrappers around the C-side `idrisml_log_*` helpers
||| defined in `packages/backends/log.{c,h}`. The 5-level scheme:
||| `SILENT < ERROR < WARN < INFO < DEBUG < TRACE`.
|||
||| Build-time ceiling: the `IDRISML_LOG=warn` Makefile knob threads
||| `-DIDRISML_LOG_LEVEL=IDRISML_LEVEL_WARN` to the C compiler, which
||| `#if`-elides INFO/DEBUG/TRACE call bodies in `log.c`. The wrappers
||| `idrisml_log_info` / `idrisml_log_debug` etc. still link as no-ops
||| in that build.
|||
||| Runtime override: env var `IDRISML_LOG_LEVEL=warn` (read once at
||| first call, cached in `idrisml_log_active_level`). Can only lower
||| below the build ceiling.
|||
||| Idris-2's FFI does not support varargs, so the C-side wrappers
||| take a single `const char *` — format on the Idris side and pass
||| the resulting `String` straight through.
module Util.Log

%foreign "C:idrisml_log_resolve_level,libidrisml"
prim__getLogLevel : PrimIO Int

%foreign "C:idrisml_log_error,libidrisml"
prim__logError : String -> PrimIO ()

%foreign "C:idrisml_log_warn,libidrisml"
prim__logWarn : String -> PrimIO ()

%foreign "C:idrisml_log_info,libidrisml"
prim__logInfo : String -> PrimIO ()

%foreign "C:idrisml_log_debug,libidrisml"
prim__logDebug : String -> PrimIO ()

%foreign "C:idrisml_log_trace,libidrisml"
prim__logTrace : String -> PrimIO ()

||| Emit at ERROR level (always visible). Use for aborts + crash
||| diagnostics that should reach the user regardless of build config.
public export
logError : String -> IO ()
logError msg = primIO (prim__logError msg)

||| Emit at WARN level. Use for non-fatal anomalies: NaN-diverge
||| messages, halved-scaler events, etc.
public export
logWarn : String -> IO ()
logWarn msg = primIO (prim__logWarn msg)

||| Emit at INFO level (the default). Use for user-facing training
||| output: epoch summaries, RSS/handle counts, timing reports.
public export
logInfo : String -> IO ()
logInfo msg = primIO (prim__logInfo msg)

||| Emit at DEBUG level. Use for opt-in diagnostics gated behind
||| feature env vars (DEBUG_PARAM_GRADS, DEBUG_LSTM_TRAJ, etc.). The
||| feature env var stays the inner gate; this is the outer level
||| gate.
public export
logDebug : String -> IO ()
logDebug msg = primIO (prim__logDebug msg)

||| Emit at TRACE level. Used by `forwardVarTraced`'s
||| activation-dump branch (see `Layer/Core.idr`); reserve for
||| per-op tracing.
public export
logTrace : String -> IO ()
logTrace msg = primIO (prim__logTrace msg)

||| Level constants matching `backends/log.h` IDRISML_LEVEL_*.
||| Use these for `getLogLevel >= levelTrace`-style branch gates
||| in callers (e.g. `forwardVarTraced`'s SafeTensors-dump branch).
public export
levelSilent : Int
levelSilent = 0

public export
levelError : Int
levelError = 1

public export
levelWarn : Int
levelWarn = 2

public export
levelInfo : Int
levelInfo = 3

public export
levelDebug : Int
levelDebug = 4

public export
levelTrace : Int
levelTrace = 5

||| Read the active log level. Resolves the runtime override on
||| first call (capped at the build ceiling) and caches; subsequent
||| calls are a single load. Returns one of the `level*` constants.
public export
getLogLevel : IO Int
getLogLevel = primIO prim__getLogLevel
