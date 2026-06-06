/* probes.h — host fingerprint primitives consumed by the Machine.Verify
 * typeclass on the Idris side.
 *
 * Each probe returns a small integer enum (OS / arch / boolean) so the
 * Idris-side wrappers can pattern-match cheaply. Probes are intentionally
 * minimal — uname for OS+arch, filesystem stat for Metal framework
 * presence on Darwin, dlopen for libcuda on Linux/Windows. No version
 * detection, no GPU enumeration; the goal is "is this binary even on
 * the right kind of host."
 *
 * See `Machine.Verify` (`packages/idris-ml/src/Machine/Verify.idr`) for
 * how Machine instances compose these into per-Machine host checks.
 */
#ifndef IDRISML_PROBES_H
#define IDRISML_PROBES_H

#ifdef __cplusplus
extern "C" {
#endif

/* Returns 0=Darwin, 1=Linux, 2=Windows, 3=Unknown. */
int idrisml_probe_os(void);

/* Returns 0=Arm64, 1=X86_64, 2=Unknown. */
int idrisml_probe_arch(void);

/* Returns 1 if Apple Metal framework appears installed (Darwin only),
 * else 0. Always 0 on non-Darwin hosts. */
int idrisml_probe_metal_available(void);

/* Returns 1 if libcuda.so.1 can be dlopen'd, else 0. Always 0 on Darwin
 * (no NVIDIA driver path there). The handle is closed immediately —
 * this is a probe, not a load. */
int idrisml_probe_cuda_available(void);

#ifdef __cplusplus
}
#endif

#endif /* IDRISML_PROBES_H */
