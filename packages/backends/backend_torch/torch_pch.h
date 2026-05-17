/* Precompiled header for the torch backend.
 *
 * Pulls in `<torch/torch.h>` — the heavy libtorch C++ header that
 * every backend_torch/*.cpp TU includes (transitively via tensor.h).
 * Building this PCH once and reusing it across all ~90 TUs saves a
 * significant chunk of the cold compile wall (libtorch headers are
 * ~30K lines of templates and most of the per-TU cost is parsing +
 * instantiating them anew).
 *
 * The Makefile builds this into <BUILD>/torch_pch.gch using the
 * exact same CFLAGS as the per-TU compile so the PCH is valid for
 * every translation unit. The per-TU rule then passes
 * `-include-pch <BUILD>/torch_pch.gch` before any other -include.
 *
 * Coverage builds (EXTRA_CFLAGS adds -O0 -g -fcoverage-mapping
 * -fprofile-instr-generate) end up with their OWN PCH in $(COV_BUILD)
 * because $(BUILD) differs — clang would reject a PCH built with
 * different flags than the consuming TU, so each build tree gets its
 * own PCH stamped from the flags in effect.
 *
 * If you add a new commonly-included heavy header (e.g. ATen)
 * to most backend_torch TUs, include it here too.
 */
#ifndef IDRISML_BACKEND_TORCH_PCH_H
#define IDRISML_BACKEND_TORCH_PCH_H

#include <torch/torch.h>

#endif
