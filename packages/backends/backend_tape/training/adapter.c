/* backend_tape/training/adapter.c — tape's shared-port implementation.
 *
 * Stub adapter for the shared training port. Every method aborts at
 * runtime so any premature wiring (a shared/training/ TU calling into
 * the port before its real implementation lands) fails loudly rather
 * than silently no-op'ing. Subsequent commits in the shared-port lift
 * replace each stub with the real tape-side call exactly when the
 * corresponding shared/training/ TU is wired up to invoke it.
 *
 * Compile-only on initial introduction: shared/training/ has only
 * port.h, no consumers, so `g_active_port` is exported but unreferenced
 * from the dylib's symbol surface. The Criterion test reaches it via
 * the dylib export and verifies the abort fires.
 */

#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <stddef.h>
#include "../../shared/training/port.h"

#define TAPE_ADAPTER_STUB(name) \
  do { \
    fprintf(stderr, \
      "tape adapter stub: " #name \
      " invoked before its shared-port implementation landed — aborting.\n"); \
    abort(); \
  } while (0)

static int    stub_tensor_numel(void* t)                          { (void)t; TAPE_ADAPTER_STUB(tensor_numel); }
static int    stub_tensor_requires_grad(void* t)                  { (void)t; TAPE_ADAPTER_STUB(tensor_requires_grad); }
static int    stub_tensor_has_grad(void* t)                       { (void)t; TAPE_ADAPTER_STUB(tensor_has_grad); }
static double stub_data_read(void* t, int i)                      { (void)t; (void)i; TAPE_ADAPTER_STUB(data_read); }
static void   stub_data_write(void* t, int i, double v)           { (void)t; (void)i; (void)v; TAPE_ADAPTER_STUB(data_write); }
static double stub_grad_read(void* t, int i)                      { (void)t; (void)i; TAPE_ADAPTER_STUB(grad_read); }
static void   stub_grad_write(void* t, int i, double v)           { (void)t; (void)i; (void)v; TAPE_ADAPTER_STUB(grad_write); }
static void   stub_zero_grad(void* t)                             { (void)t; TAPE_ADAPTER_STUB(zero_grad); }
static void   stub_load_doubles(void* t, const double* s, int n)  { (void)t; (void)s; (void)n; TAPE_ADAPTER_STUB(load_doubles); }
static void   stub_load_int64(void* t, const int64_t* s, int n)   { (void)t; (void)s; (void)n; TAPE_ADAPTER_STUB(load_int64); }
static void   stub_backward(void* loss)                           { (void)loss; TAPE_ADAPTER_STUB(backward); }
static void   stub_epoch_boundary(void)                           { TAPE_ADAPTER_STUB(epoch_boundary); }
static double stub_wall_ms(void)                                  { TAPE_ADAPTER_STUB(wall_ms); }

const BackendPort g_active_port = {
  .tensor_numel         = stub_tensor_numel,
  .tensor_requires_grad = stub_tensor_requires_grad,
  .tensor_has_grad      = stub_tensor_has_grad,
  .data_read            = stub_data_read,
  .data_write           = stub_data_write,
  .grad_read            = stub_grad_read,
  .grad_write           = stub_grad_write,
  .zero_grad            = stub_zero_grad,
  .load_doubles         = stub_load_doubles,
  .load_int64           = stub_load_int64,
  .backward             = stub_backward,
  .epoch_boundary       = stub_epoch_boundary,
  .wall_ms              = stub_wall_ms,
};
