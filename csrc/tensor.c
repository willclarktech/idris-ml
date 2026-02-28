#include <stdlib.h>
#include <string.h>
#include <math.h>

#ifdef __APPLE__
#define ACCELERATE_NEW_LAPACK
#include <Accelerate/Accelerate.h>
#endif


/* -------------------------------------------------------------------
   Arena allocator
   ------------------------------------------------------------------- */

typedef struct {
  char *buf;
  size_t used, cap;
} Arena;

static Arena arena = {NULL, 0, 0};

void arena_init(size_t cap) {
  if (arena.buf) return;
  arena.buf = (char *)malloc(cap);
  arena.used = 0;
  arena.cap = cap;
}

void *arena_alloc(size_t size) {
  /* align to 8 bytes */
  size = (size + 7) & ~(size_t)7;
  if (!arena.buf) arena_init(1 << 20); /* 1 MB default */
  if (arena.used + size > arena.cap) {
    size_t new_cap = arena.cap * 2;
    while (arena.used + size > new_cap) new_cap *= 2;
    arena.buf = (char *)realloc(arena.buf, new_cap);
    arena.cap = new_cap;
  }
  void *p = arena.buf + arena.used;
  arena.used += size;
  return p;
}

void arena_reset(void) {
  arena.used = 0;
}


/* -------------------------------------------------------------------
   Buffer management
   ------------------------------------------------------------------- */

double *tensor_alloc(int n) {
  return (double *)calloc(n, sizeof(double));
}

void tensor_free(double *ptr) {
  free(ptr);
}

void tensor_pack(double *ptr, int idx, double val) {
  ptr[idx] = val;
}

double tensor_read(double *ptr, int idx) {
  return ptr[idx];
}


/* -------------------------------------------------------------------
   Forward operations
   ------------------------------------------------------------------- */

/* out[i] = sum_j W[i*n + j] * x[j],  i in [0, m) */
void tensor_matvec(const double *w, const double *x, double *out,
                   int m, int n) {
#ifdef __APPLE__
  cblas_dgemv(CblasRowMajor, CblasNoTrans, m, n,
              1.0, w, n, x, 1, 0.0, out, 1);
#else
  for (int i = 0; i < m; i++) {
    double s = 0.0;
    for (int j = 0; j < n; j++) {
      s += w[i * n + j] * x[j];
    }
    out[i] = s;
  }
#endif
}

/* sum_i a[i] * b[i] */
double tensor_dot(const double *a, const double *b, int n) {
#ifdef __APPLE__
  return cblas_ddot(n, a, 1, b, 1);
#else
  double s = 0.0;
  for (int i = 0; i < n; i++) s += a[i] * b[i];
  return s;
#endif
}

/* out[i] = a[i] + b[i] */
void tensor_add_vec(const double *a, const double *b, double *out, int n) {
  for (int i = 0; i < n; i++) out[i] = a[i] + b[i];
}

/* out[i] = alpha * x[i] */
void tensor_scale_vec(double alpha, const double *x, double *out, int n) {
  for (int i = 0; i < n; i++) out[i] = alpha * x[i];
}


/* -------------------------------------------------------------------
   Backward metadata
   ------------------------------------------------------------------- */

typedef struct {
  int m, n;
  double *w_vals;     /* m*n weight values (copy from forward) */
  double *x_vals;     /* n input values (copy from forward) */
  int *w_tape_idx;    /* m*n tape indices of weight elements */
  int *x_tape_idx;    /* n tape indices of input elements */
  int out_tape_start; /* tape index of first output element */
} MatVecMeta;

typedef struct {
  int n;
  double *a_vals;     /* n values of first vector */
  double *b_vals;     /* n values of second vector */
  int *a_tape_idx;    /* n tape indices of first vector elements */
  int *b_tape_idx;    /* n tape indices of second vector elements */
  int out_tape_idx;   /* tape index of output scalar */
} DotMeta;


/* -------------------------------------------------------------------
   Metadata allocation (arena-backed)
   ------------------------------------------------------------------- */

/* Allocate MatVecMeta + arrays in one arena bump */
MatVecMeta *matvec_meta_alloc(int m, int n) {
  MatVecMeta *meta = (MatVecMeta *)arena_alloc(sizeof(MatVecMeta));
  meta->m = m;
  meta->n = n;
  meta->w_vals = (double *)arena_alloc(m * n * sizeof(double));
  meta->x_vals = (double *)arena_alloc(n * sizeof(double));
  meta->w_tape_idx = (int *)arena_alloc(m * n * sizeof(int));
  meta->x_tape_idx = (int *)arena_alloc(n * sizeof(int));
  meta->out_tape_start = 0;
  return meta;
}

DotMeta *dot_meta_alloc(int n) {
  DotMeta *meta = (DotMeta *)arena_alloc(sizeof(DotMeta));
  meta->n = n;
  meta->a_vals = (double *)arena_alloc(n * sizeof(double));
  meta->b_vals = (double *)arena_alloc(n * sizeof(double));
  meta->a_tape_idx = (int *)arena_alloc(n * sizeof(int));
  meta->b_tape_idx = (int *)arena_alloc(n * sizeof(int));
  meta->out_tape_idx = 0;
  return meta;
}


/* -------------------------------------------------------------------
   Gradient array (C-backed for use with tensor backward)
   ------------------------------------------------------------------- */

double *grad_alloc(int n) {
  return (double *)calloc(n, sizeof(double));
}

void grad_free(double *p) {
  free(p);
}

double grad_get(double *p, int i) {
  return p[i];
}

/* Returns p for handle threading in Idris FFI */
double *grad_add(double *p, int i, double v) {
  p[i] += v;
  return p;
}


/* -------------------------------------------------------------------
   Metadata packing (called from Idris during forward pass)
   ------------------------------------------------------------------- */

/* Pack one weight element into meta. Returns meta for threading. */
void *matvec_meta_pack_w(void *meta_ptr, int idx, double val, int tape_idx) {
  MatVecMeta *m = (MatVecMeta *)meta_ptr;
  m->w_vals[idx] = val;
  m->w_tape_idx[idx] = tape_idx;
  return meta_ptr;
}

/* Pack one input element into meta. Returns meta for threading. */
void *matvec_meta_pack_x(void *meta_ptr, int idx, double val, int tape_idx) {
  MatVecMeta *m = (MatVecMeta *)meta_ptr;
  m->x_vals[idx] = val;
  m->x_tape_idx[idx] = tape_idx;
  return meta_ptr;
}

/* --- Raw array accessors (for Scheme-side foreign-set! packing) --- */
double *matvec_meta_w_vals(void *p) { return ((MatVecMeta *)p)->w_vals; }
double *matvec_meta_x_vals(void *p) { return ((MatVecMeta *)p)->x_vals; }
int *matvec_meta_w_tape(void *p)    { return ((MatVecMeta *)p)->w_tape_idx; }
int *matvec_meta_x_tape(void *p)    { return ((MatVecMeta *)p)->x_tape_idx; }
double *dot_meta_a_vals(void *p)    { return ((DotMeta *)p)->a_vals; }
double *dot_meta_b_vals(void *p)    { return ((DotMeta *)p)->b_vals; }
int *dot_meta_a_tape(void *p)       { return ((DotMeta *)p)->a_tape_idx; }
int *dot_meta_b_tape(void *p)       { return ((DotMeta *)p)->b_tape_idx; }

/* Set the tape index where this op's output ConstOps start. */
void *matvec_meta_set_out(void *meta_ptr, int start) {
  ((MatVecMeta *)meta_ptr)->out_tape_start = start;
  return meta_ptr;
}

/* Run forward matmul using values packed in meta. */
void *matvec_meta_compute(void *meta_ptr, double *out) {
  MatVecMeta *m = (MatVecMeta *)meta_ptr;
  tensor_matvec(m->w_vals, m->x_vals, out, m->m, m->n);
  return out;
}

/* Pack one element of vector a into dot meta. */
void *dot_meta_pack_a(void *meta_ptr, int idx, double val, int tape_idx) {
  DotMeta *m = (DotMeta *)meta_ptr;
  m->a_vals[idx] = val;
  m->a_tape_idx[idx] = tape_idx;
  return meta_ptr;
}

/* Pack one element of vector b into dot meta. */
void *dot_meta_pack_b(void *meta_ptr, int idx, double val, int tape_idx) {
  DotMeta *m = (DotMeta *)meta_ptr;
  m->b_vals[idx] = val;
  m->b_tape_idx[idx] = tape_idx;
  return meta_ptr;
}

/* Set the output tape index for dot product. */
void *dot_meta_set_out(void *meta_ptr, int out_idx) {
  ((DotMeta *)meta_ptr)->out_tape_idx = out_idx;
  return meta_ptr;
}

/* Run forward dot product using values packed in meta. */
double dot_meta_compute(void *meta_ptr) {
  DotMeta *m = (DotMeta *)meta_ptr;
  return tensor_dot(m->a_vals, m->b_vals, m->n);
}


/* -------------------------------------------------------------------
   Backward operations
   ------------------------------------------------------------------- */

/*
 * MatVec backward: given dy[i] for i in [0, m),
 *   dW[i][j] += dy[i] * x[j]
 *   dx[j]    += sum_i dy[i] * W[i][j]
 *
 * grad_array is the Scheme gradient vector.
 * We read dy from grad_array at [out_tape_start+1 .. out_tape_start+m]
 * (the +1 is because the MatVecOp entry itself is at out_tape_start,
 *  and the ConstOp entries for outputs follow it)
 */
void tensor_matvec_backward(double *grad_array, MatVecMeta *meta) {
  int m = meta->m;
  int n = meta->n;
  int out_start = meta->out_tape_start + 1; /* skip the MatVecOp entry */

  for (int i = 0; i < m; i++) {
    double dy = grad_array[out_start + i];
    if (dy == 0.0) continue;

    /* dW[i][j] += dy * x[j] */
    for (int j = 0; j < n; j++) {
      int w_idx = meta->w_tape_idx[i * n + j];
      grad_array[w_idx] += dy * meta->x_vals[j];
    }

    /* dx[j] += dy * W[i][j] */
    for (int j = 0; j < n; j++) {
      int x_idx = meta->x_tape_idx[j];
      grad_array[x_idx] += dy * meta->w_vals[i * n + j];
    }
  }
}

/*
 * Dot backward: given dy (scalar at out_tape_idx),
 *   da[i] += dy * b[i]
 *   db[i] += dy * a[i]
 */
void tensor_dot_backward(double *grad_array, DotMeta *meta) {
  int n = meta->n;
  double dy = grad_array[meta->out_tape_idx];
  if (dy == 0.0) return;

  for (int i = 0; i < n; i++) {
    grad_array[meta->a_tape_idx[i]] += dy * meta->b_vals[i];
    grad_array[meta->b_tape_idx[i]] += dy * meta->a_vals[i];
  }
}
