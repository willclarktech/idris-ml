/* log.h — 5-level log scheme for idris-ml backends.
 *
 * Levels: SILENT < ERROR < WARN < INFO < DEBUG < TRACE.
 *
 * Build-time ceiling: pass `-DIDRISML_LOG_LEVEL=N` at compile time. Sites
 * emitting at a level > the ceiling are `#if`-elided (no body, no fprintf
 * symbol). Default is INFO if unset.
 *
 * Runtime override: env var `IDRISML_LOG_LEVEL` (read once at first call,
 * cached). Can only LOWER the active level below the build ceiling — calls
 * at levels above the ceiling were elided at compile time so there's
 * nothing to enable.
 *
 * Mapping:
 *   ERROR — always visible; aborts and crash diagnostics
 *   WARN  — non-fatal anomalies (reserved)
 *   INFO  — default user-facing output: epoch summaries, RSS, NaN-diverge
 *   DEBUG — opt-in diagnostics: DEBUG_PARAM_GRADS / DEBUG_LSTM_TRAJ /
 *           DEBUG_NAN_TRAP fprintf bodies live behind this gate
 *   TRACE — per-op tracing; future home for forwardVarTraced telemetry
 *
 * Cross-language: the Idris-side Util.Log uses the same env var so an
 * `IDRISML_LOG_LEVEL=warn` suppresses both Idris-side `logInfo` calls and
 * C-side INFO-level fprintf bodies uniformly.
 */
#ifndef IDRISML_LOG_H
#define IDRISML_LOG_H

#include <stdio.h>
#include <stdarg.h>

#define IDRISML_LEVEL_SILENT 0
#define IDRISML_LEVEL_ERROR  1
#define IDRISML_LEVEL_WARN   2
#define IDRISML_LEVEL_INFO   3
#define IDRISML_LEVEL_DEBUG  4
#define IDRISML_LEVEL_TRACE  5

#ifndef IDRISML_LOG_LEVEL
#define IDRISML_LOG_LEVEL IDRISML_LEVEL_INFO
#endif

#ifdef __cplusplus
extern "C" {
#endif

/* The runtime-cached active level. -1 sentinel = "not yet probed".
 * Anything else = the IDRISML_LOG_LEVEL_* value derived from the env. */
extern int idrisml_log_active_level;

/* Probe the env var once on first call; cache. Returns the active level. */
int idrisml_log_resolve_level(void);

/* Internal: format a message at `level`. Honors both the build ceiling
 * (via #if at the macro call site) and the runtime cache (via a leading
 * branch here). */
void idrisml_log_impl(int level, const char *fmt, ...);

#ifdef __cplusplus
}
#endif

/* Macro form: short-circuits at compile time when level > build ceiling.
 * For levels passing the ceiling, runtime branch in idrisml_log_impl. */
#define IDRISML_LOG(level, ...) \
    do { if ((level) <= IDRISML_LOG_LEVEL) idrisml_log_impl((level), __VA_ARGS__); } while (0)

#endif /* IDRISML_LOG_H */
