/* backend_tape.c — Tape-based autograd backend implementing backend.h.
 *
 * Phase 1g closeout (Phase 1e.11): every implementation now lives under
 * `backend_tape/<slice>/`. This file is intentionally a skeleton — the
 * per-TU build picks up the actual sources from the subtree, and this
 * single TU exists only because `backend_tape` (no suffix) is the
 * primary-name handle the per-backend build rule expects.
 *
 * Each per-op file follows the convention documented in design-decisions.md
 * "Modular tape backend":
 *   - one op per file under backend_tape/<slice>/<op>.c
 *   - each op file with a backward arm registers via TAPE_REGISTER_OP
 *     in training/autograd/op_dispatch.{h,c}
 *   - the dispatch table fully drives the backward loop in
 *     training/autograd/backward.c — there is no monolith switch.
 *
 * Cross-cutting helpers (broadcast.{c,h}, training/host_io.c,
 * training/param_create.c, training/param_registry.c,
 * training/optimizer.c, training/profiling.c, training/dtype_streamed.c,
 * training/shims.c, training/per_dtype_legacy.c,
 * training/autograd/{helpers,backward,op_dispatch}.{c,h},
 * core/elementwise/{_dispatch.c,_helpers.h,_kernels.inc}) round out
 * the surface. The previous 7150-line monolith was decomposed in
 * Phases 1.0-1g; this skeleton landed at Phase 1e.11.
 */

#include "backend.h"

/* Phase 1g closeout — every implementation moved to backend_tape/<slice>/.
   Keep this TU non-empty so the per-backend compile rule remains happy. */
static const char tape_modular_marker[] = "tape-modular-1e11";
