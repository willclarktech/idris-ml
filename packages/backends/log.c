/* log.c — implementation of the 5-level log scheme.
 *
 * See log.h for level definitions and the build-time + runtime gating
 * model. This file holds the active-level cache and the impl function
 * that gets called when both gates allow.
 */
#include "log.h"
#include <stdlib.h>
#include <string.h>
#include <strings.h>

int idrisml_log_active_level = -1;

static int idrisml_parse_level(const char *s) {
    if (!s || *s == '\0') return IDRISML_LOG_LEVEL;
    if (strcasecmp(s, "silent") == 0) return IDRISML_LEVEL_SILENT;
    if (strcasecmp(s, "error")  == 0) return IDRISML_LEVEL_ERROR;
    if (strcasecmp(s, "warn")   == 0) return IDRISML_LEVEL_WARN;
    if (strcasecmp(s, "info")   == 0) return IDRISML_LEVEL_INFO;
    if (strcasecmp(s, "debug")  == 0) return IDRISML_LEVEL_DEBUG;
    if (strcasecmp(s, "trace")  == 0) return IDRISML_LEVEL_TRACE;
    return IDRISML_LOG_LEVEL;
}

int idrisml_log_resolve_level(void) {
    if (idrisml_log_active_level != -1) return idrisml_log_active_level;
    const char *env = getenv("IDRISML_LOG_LEVEL");
    int requested = idrisml_parse_level(env);
    /* Can only lower below build ceiling — higher levels are #if-elided. */
    int effective = (requested < IDRISML_LOG_LEVEL) ? requested : IDRISML_LOG_LEVEL;
    idrisml_log_active_level = effective;
    return effective;
}

void idrisml_log_impl(int level, const char *fmt, ...) {
    if (level > idrisml_log_resolve_level()) return;
    va_list args;
    va_start(args, fmt);
    vfprintf(stderr, fmt, args);
    va_end(args);
}
