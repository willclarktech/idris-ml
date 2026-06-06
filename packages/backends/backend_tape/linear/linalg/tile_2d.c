/* linear/linalg/tile_2d.c — tile a 2D tensor along both axes.
 *
 * Forward: output[i, j] = input[i mod m, j mod n] over
 * shape [m*rep0, n*rep1]. Backward: sum grads from each tile into
 * the corresponding source element.
 *
 * Tile2dMeta lives here (TU-local) since this is the only op that
 * uses it. Subsequent commits may move per-op meta structs to per-op files
 * as a general rule; tile_2d is already there.
 */

#include <stdlib.h>
#include <string.h>
#include "../../tape.h"
#include "../../arena.h"
#include "../../tensor.h"
#include "../../training/autograd/op_dispatch.h"
#include "../../../backend.h"

typedef struct { int m, n, rep0, rep1; } Tile2dMeta;

TensorHandle tensor_tile_2d(TensorHandle h, int rep0, int rep1) {
    Tensor* t = (Tensor*)h;
    int m = t->shape[0], n = t->shape[1];
    int M = m * rep0, N = n * rep1;
    int shape[] = {M, N};
    Tensor* r;
    if (t->dtype_tag == DT_F32) {
        float* data = arena_alloc(M * N * sizeof(float));
        const float* td = (const float*)t->data;
        for (int i = 0; i < M; i++) {
            int si = i % m;
            for (int j = 0; j < N; j++) data[i * N + j] = td[si * n + (j % n)];
        }
        r = make_tensor_arena_f32(data, M * N, shape, 2, t->requires_grad);
    } else {
        double* data = malloc(M * N * sizeof(double));
        for (int i = 0; i < M; i++) {
            int si = i % m;
            for (int j = 0; j < N; j++) data[i * N + j] = ((double*)t->data)[si * n + (j % n)];
        }
        r = make_tensor(data, shape, 2, t->requires_grad);
        free(data);
    }
    if (t->requires_grad) {
        Tile2dMeta* meta = arena_alloc(sizeof(Tile2dMeta));
        meta->m = m; meta->n = n; meta->rep0 = rep0; meta->rep1 = rep1;
        TapeEntry* e = tape_append(OP_TILE_2D, r, t, NULL, 0);
        e->op_meta = meta;
    }
    return r;
}

static void tape_backward_tile_2d(TapeEntry* e) {
    Tensor* r = e->result;
    Tensor* a = e->arg1;
    if (a) {
        ensure_grad(a);
        Tile2dMeta* meta = (Tile2dMeta*)e->op_meta;
        int m = meta->m, n = meta->n;
        int rep0 = meta->rep0, rep1 = meta->rep1;
        int N = n * rep1;
        for (int si = 0; si < m; si++) {
            for (int sj = 0; sj < n; sj++) {
                double s = 0.0;
                for (int r0 = 0; r0 < rep0; r0++) {
                    for (int c0 = 0; c0 < rep1; c0++) {
                        s += tape_grad_load_d(r, (r0 * m + si) * N + (c0 * n + sj));
                    }
                }
                tape_grad_add_d(a, si * n + sj, s);
            }
        }
    }
}

TAPE_REGISTER_OP(OP_TILE_2D, tape_backward_tile_2d)
