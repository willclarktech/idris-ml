/* cxx_abort.h — fatal precondition guard for the C++ backends (mlx, torch).
 *
 * C++ analogue of backend_tape/tensor.h's TAPE_ABORT_IF. Written as ONE
 * statement so gcov attributes the whole expansion (the std::abort()
 * included) to the single invocation line — which every valid-input test
 * already executes when it evaluates the condition. That makes the guard
 * line covered with NO `// GCOVR_EXCL` marker and NO SIGABRT death test
 * (gcov's same-line-guard rule; see docs/develop/coverage-policy.md
 * "Principled exclusions"). Use this for the C++ backends' loud
 * invalid-input guards instead of an own-line std::fprintf+std::abort.
 *
 * NOT for switch-`default:` / `else`-branch aborts or unconditional abort
 * helpers — no valid input evaluates a condition co-located with the
 * abort there, so those keep `// GCOVR_EXCL` with a reason.
 */

#ifndef IDRISML_BACKENDS_CXX_ABORT_H
#define IDRISML_BACKENDS_CXX_ABORT_H

#include <cstdio>  // std::fprintf
#include <cstdlib> // std::abort

#define CXX_ABORT_IF(cond, ...)                                                                    \
	do {                                                                                           \
		if (cond) {                                                                                \
			std::fprintf(stderr, __VA_ARGS__);                                                     \
			std::abort();                                                                          \
		}                                                                                          \
	} while (0)

#endif /* IDRISML_BACKENDS_CXX_ABORT_H */
