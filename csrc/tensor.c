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
  double *w_vals;     /* m*n weight values (copy or persistent buffer) */
  double *x_vals;     /* n input values (copy from forward) */
  int *w_tape_idx;    /* m*n tape indices of weight elements (NULL if bulk) */
  int *x_tape_idx;    /* n tape indices of input elements */
  int out_tape_start; /* tape index of first output element */
  int w_tape_start;   /* tape start index for bulk-registered weights */
} MatVecMeta;

typedef struct {
  int n;
  double *a_vals;     /* n values of first vector */
  double *b_vals;     /* n values of second vector */
  int *a_tape_idx;    /* n tape indices of first vector elements */
  int *b_tape_idx;    /* n tape indices of second vector elements */
  int out_tape_idx;   /* tape index of output scalar */
} DotMeta;

typedef struct {
  int n;
  double *x_vals;        /* n input values (arena) */
  double *out_vals;      /* n output values, saved for backward (arena) */
  int *x_tape_idx;       /* n input tape indices (arena) */
  int out_tape_start;    /* tape index of the SoftmaxOp/LogSoftmaxOp entry */
} SoftmaxMeta;

typedef struct {
  int n, w;
  double *mem_vals;      /* n*w memory values (arena) */
  double *key_vals;      /* w key values (arena) */
  double beta_val;       /* scalar beta value */
  double key_norm;       /* |key|, saved for backward */
  double *row_norms;     /* n row norms, saved for backward (arena) */
  double *dots;          /* n dot products, saved for backward (arena) */
  int *mem_tape_idx;     /* n*w tape indices (arena) */
  int *key_tape_idx;     /* w tape indices (arena) */
  int beta_tape_idx;     /* tape index for beta */
  int out_tape_start;    /* set by tape append */
} BatchCosSimMeta;

typedef struct {
  int n, w;
  double *mem_vals;       /* n*w memory values (arena) */
  double *weight_vals;    /* n weight values (arena) */
  int *mem_tape_idx;      /* n*w tape indices (arena) */
  int *weight_tape_idx;   /* n tape indices (arena) */
  int out_tape_start;     /* set by tape append */
} ReadOpMeta;

typedef struct {
  int n, w;
  double *mem_vals;       /* n*w input memory values (arena) */
  double *weight_vals;    /* n weight values (arena) */
  double *erase_vals;     /* w erase values (arena) */
  double *add_vals;       /* w add values (arena) */
  int *mem_tape_idx;      /* n*w tape indices (arena) */
  int *weight_tape_idx;   /* n tape indices (arena) */
  int *erase_tape_idx;    /* w tape indices (arena) */
  int *add_tape_idx;      /* w tape indices (arena) */
  int out_tape_start;     /* set by tape append */
} WriteOpMeta;

typedef struct {
  int n, w;
  double *mem_vals;       /* n*w input memory values (arena) */
  double *weight_vals;    /* n weight values (arena) */
  double *add_vals;       /* w add values (arena) */
  int *mem_tape_idx;      /* n*w tape indices (arena) */
  int *weight_tape_idx;   /* n tape indices (arena) */
  int *add_tape_idx;      /* w tape indices (arena) */
  double *out_vals;       /* n*w tanh-bounded output values (arena) */
  int out_tape_start;     /* set by tape append */
} InterpWriteMeta;


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
  meta->w_tape_start = 0;
  return meta;
}

/* Allocate MatVecMeta for persistent weight buffer path.
 * w_vals points to the persistent C buffer (no arena copy).
 * w_tape_idx is NULL — backward computes indices from w_tape_start. */
MatVecMeta *matvec_meta_alloc_buf(int m, int n, double *w_vals_ptr,
                                  int w_tape_start) {
  MatVecMeta *meta = (MatVecMeta *)arena_alloc(sizeof(MatVecMeta));
  meta->m = m;
  meta->n = n;
  meta->w_vals = w_vals_ptr;
  meta->x_vals = (double *)arena_alloc(n * sizeof(double));
  meta->w_tape_idx = NULL;
  meta->x_tape_idx = (int *)arena_alloc(n * sizeof(int));
  meta->out_tape_start = 0;
  meta->w_tape_start = w_tape_start;
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

SoftmaxMeta *softmax_meta_alloc(int n) {
  SoftmaxMeta *meta = (SoftmaxMeta *)arena_alloc(sizeof(SoftmaxMeta));
  meta->n = n;
  meta->x_vals = (double *)arena_alloc(n * sizeof(double));
  meta->out_vals = (double *)arena_alloc(n * sizeof(double));
  meta->x_tape_idx = (int *)arena_alloc(n * sizeof(int));
  meta->out_tape_start = 0;
  return meta;
}

BatchCosSimMeta *batch_cossim_meta_alloc(int n, int w) {
  BatchCosSimMeta *m = (BatchCosSimMeta *)arena_alloc(sizeof(BatchCosSimMeta));
  m->n = n; m->w = w;
  m->mem_vals = (double *)arena_alloc(n * w * sizeof(double));
  m->key_vals = (double *)arena_alloc(w * sizeof(double));
  m->row_norms = (double *)arena_alloc(n * sizeof(double));
  m->dots = (double *)arena_alloc(n * sizeof(double));
  m->mem_tape_idx = (int *)arena_alloc(n * w * sizeof(int));
  m->key_tape_idx = (int *)arena_alloc(w * sizeof(int));
  m->beta_val = 0; m->key_norm = 0; m->beta_tape_idx = 0; m->out_tape_start = 0;
  return m;
}

ReadOpMeta *readop_meta_alloc(int n, int w) {
  ReadOpMeta *m = (ReadOpMeta *)arena_alloc(sizeof(ReadOpMeta));
  m->n = n; m->w = w;
  m->mem_vals = (double *)arena_alloc(n * w * sizeof(double));
  m->weight_vals = (double *)arena_alloc(n * sizeof(double));
  m->mem_tape_idx = (int *)arena_alloc(n * w * sizeof(int));
  m->weight_tape_idx = (int *)arena_alloc(n * sizeof(int));
  m->out_tape_start = 0;
  return m;
}

WriteOpMeta *writeop_meta_alloc(int n, int w) {
  WriteOpMeta *m = (WriteOpMeta *)arena_alloc(sizeof(WriteOpMeta));
  m->n = n; m->w = w;
  m->mem_vals = (double *)arena_alloc(n * w * sizeof(double));
  m->weight_vals = (double *)arena_alloc(n * sizeof(double));
  m->erase_vals = (double *)arena_alloc(w * sizeof(double));
  m->add_vals = (double *)arena_alloc(w * sizeof(double));
  m->mem_tape_idx = (int *)arena_alloc(n * w * sizeof(int));
  m->weight_tape_idx = (int *)arena_alloc(n * sizeof(int));
  m->erase_tape_idx = (int *)arena_alloc(w * sizeof(int));
  m->add_tape_idx = (int *)arena_alloc(w * sizeof(int));
  m->out_tape_start = 0;
  return m;
}

InterpWriteMeta *interp_write_meta_alloc(int n, int w) {
  InterpWriteMeta *m = (InterpWriteMeta *)arena_alloc(sizeof(InterpWriteMeta));
  m->n = n; m->w = w;
  m->mem_vals = (double *)arena_alloc(n * w * sizeof(double));
  m->weight_vals = (double *)arena_alloc(n * sizeof(double));
  m->add_vals = (double *)arena_alloc(w * sizeof(double));
  m->mem_tape_idx = (int *)arena_alloc(n * w * sizeof(int));
  m->weight_tape_idx = (int *)arena_alloc(n * sizeof(int));
  m->add_tape_idx = (int *)arena_alloc(w * sizeof(int));
  m->out_vals = (double *)arena_alloc(n * w * sizeof(double));
  m->out_tape_start = 0;
  return m;
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

/* --- Softmax meta accessors --- */
double *softmax_meta_x_vals(void *p) { return ((SoftmaxMeta *)p)->x_vals; }
int *softmax_meta_x_tape(void *p)    { return ((SoftmaxMeta *)p)->x_tape_idx; }

void *softmax_meta_set_out(void *meta_ptr, int start) {
  ((SoftmaxMeta *)meta_ptr)->out_tape_start = start;
  return meta_ptr;
}

/* --- BatchCosSim meta accessors --- */
double *batch_cossim_meta_mem_vals(void *p) { return ((BatchCosSimMeta *)p)->mem_vals; }
int *batch_cossim_meta_mem_tape(void *p)    { return ((BatchCosSimMeta *)p)->mem_tape_idx; }
double *batch_cossim_meta_key_vals(void *p) { return ((BatchCosSimMeta *)p)->key_vals; }
int *batch_cossim_meta_key_tape(void *p)    { return ((BatchCosSimMeta *)p)->key_tape_idx; }

void *batch_cossim_meta_set_beta(void *p, double val, int tape_idx) {
  BatchCosSimMeta *m = (BatchCosSimMeta *)p;
  m->beta_val = val;
  m->beta_tape_idx = tape_idx;
  return p;
}

void *batch_cossim_meta_set_out(void *p, int start) {
  ((BatchCosSimMeta *)p)->out_tape_start = start;
  return p;
}

/* --- ReadOp meta accessors --- */
double *readop_meta_mem_vals(void *p)     { return ((ReadOpMeta *)p)->mem_vals; }
int *readop_meta_mem_tape(void *p)        { return ((ReadOpMeta *)p)->mem_tape_idx; }
double *readop_meta_weight_vals(void *p)  { return ((ReadOpMeta *)p)->weight_vals; }
int *readop_meta_weight_tape(void *p)     { return ((ReadOpMeta *)p)->weight_tape_idx; }

void *readop_meta_set_out(void *p, int start) {
  ((ReadOpMeta *)p)->out_tape_start = start;
  return p;
}

/* --- WriteOp meta accessors --- */
double *writeop_meta_mem_vals(void *p)     { return ((WriteOpMeta *)p)->mem_vals; }
int *writeop_meta_mem_tape(void *p)        { return ((WriteOpMeta *)p)->mem_tape_idx; }
double *writeop_meta_weight_vals(void *p)  { return ((WriteOpMeta *)p)->weight_vals; }
int *writeop_meta_weight_tape(void *p)     { return ((WriteOpMeta *)p)->weight_tape_idx; }
double *writeop_meta_erase_vals(void *p)   { return ((WriteOpMeta *)p)->erase_vals; }
int *writeop_meta_erase_tape(void *p)      { return ((WriteOpMeta *)p)->erase_tape_idx; }
double *writeop_meta_add_vals(void *p)     { return ((WriteOpMeta *)p)->add_vals; }
int *writeop_meta_add_tape(void *p)        { return ((WriteOpMeta *)p)->add_tape_idx; }

double *interp_write_meta_mem_vals(void *p)     { return ((InterpWriteMeta *)p)->mem_vals; }
int *interp_write_meta_mem_tape(void *p)        { return ((InterpWriteMeta *)p)->mem_tape_idx; }
double *interp_write_meta_weight_vals(void *p)  { return ((InterpWriteMeta *)p)->weight_vals; }
int *interp_write_meta_weight_tape(void *p)     { return ((InterpWriteMeta *)p)->weight_tape_idx; }
double *interp_write_meta_add_vals(void *p)     { return ((InterpWriteMeta *)p)->add_vals; }
int *interp_write_meta_add_tape(void *p)        { return ((InterpWriteMeta *)p)->add_tape_idx; }

void *interp_write_meta_set_out(void *p, int start) {
  ((InterpWriteMeta *)p)->out_tape_start = start;
  return p;
}

void *writeop_meta_set_out(void *p, int start) {
  ((WriteOpMeta *)p)->out_tape_start = start;
  return p;
}

/* Softmax forward: out[i] = exp(x[i] - max) / sum(exp(x - max))
 * Saves output to meta->out_vals for backward. Writes to out buffer. */
void *softmax_meta_compute(void *meta_ptr, double *out) {
  SoftmaxMeta *m = (SoftmaxMeta *)meta_ptr;
  int n = m->n;
  double mx = m->x_vals[0];
  for (int i = 1; i < n; i++)
    if (m->x_vals[i] > mx) mx = m->x_vals[i];
  double sum = 0.0;
  for (int i = 0; i < n; i++) {
    m->out_vals[i] = exp(m->x_vals[i] - mx);
    sum += m->out_vals[i];
  }
  for (int i = 0; i < n; i++) {
    m->out_vals[i] /= sum;
    out[i] = m->out_vals[i];
  }
  return out;
}

/* LogSoftmax forward: out[i] = x[i] - max - log(sum(exp(x - max)))
 * Saves output to meta->out_vals for backward. Writes to out buffer. */
void *logsoftmax_meta_compute(void *meta_ptr, double *out) {
  SoftmaxMeta *m = (SoftmaxMeta *)meta_ptr;
  int n = m->n;
  double mx = m->x_vals[0];
  for (int i = 1; i < n; i++)
    if (m->x_vals[i] > mx) mx = m->x_vals[i];
  double sum = 0.0;
  for (int i = 0; i < n; i++)
    sum += exp(m->x_vals[i] - mx);
  double logSumExp = mx + log(sum);
  for (int i = 0; i < n; i++) {
    m->out_vals[i] = m->x_vals[i] - logSumExp;
    out[i] = m->out_vals[i];
  }
  return out;
}

/* Batch cosine similarity: out[i] = beta * cos_sim(key, mem[i])
 * Saves intermediate values (norms, dots) for backward. */
void *batch_cossim_compute(void *meta_ptr, double *out) {
  BatchCosSimMeta *m = (BatchCosSimMeta *)meta_ptr;
  int n = m->n, w = m->w;

  double kn_sq = 0.0;
  for (int j = 0; j < w; j++) kn_sq += m->key_vals[j] * m->key_vals[j];
  m->key_norm = sqrt(kn_sq);

  for (int i = 0; i < n; i++) {
    double dot = 0.0, rn_sq = 0.0;
    for (int j = 0; j < w; j++) {
      dot += m->mem_vals[i * w + j] * m->key_vals[j];
      rn_sq += m->mem_vals[i * w + j] * m->mem_vals[i * w + j];
    }
    m->dots[i] = dot;
    m->row_norms[i] = sqrt(rn_sq);

    double denom = m->key_norm * m->row_norms[i];
    double sim = (denom > 1e-12) ? dot / denom : 0.0;
    out[i] = m->beta_val * sim;
  }
  return out;
}

/* Read operation: out[j] = sum_i weight[i] * mem[i*w + j] */
void *readop_compute(void *meta_ptr, double *out) {
  ReadOpMeta *m = (ReadOpMeta *)meta_ptr;
  int n = m->n, w = m->w;

  for (int j = 0; j < w; j++) out[j] = 0.0;
  for (int i = 0; i < n; i++) {
    for (int j = 0; j < w; j++) {
      out[j] += m->weight_vals[i] * m->mem_vals[i * w + j];
    }
  }
  return out;
}

/* Interpolation write with fused tanh:
 * raw = (1 - w[i]) * mem[i*w+j] + w[i] * add[j]
 * out[i*w+j] = tanh(raw)
 * Saves out_vals for backward (tanh derivative needs output values). */
void *interp_write_compute(void *meta_ptr, double *out) {
  InterpWriteMeta *m = (InterpWriteMeta *)meta_ptr;
  int n = m->n, w = m->w;

  for (int i = 0; i < n; i++) {
    double wi = m->weight_vals[i];
    for (int j = 0; j < w; j++) {
      double raw = (1.0 - wi) * m->mem_vals[i * w + j] + wi * m->add_vals[j];
      double t = tanh(raw);
      out[i * w + j] = t;
      m->out_vals[i * w + j] = t;
    }
  }
  return out;
}

/* Write operation: out[i*w+j] = mem[i*w+j]*(1 - w[i]*e[j]) + w[i]*a[j] */
void *writeop_compute(void *meta_ptr, double *out) {
  WriteOpMeta *m = (WriteOpMeta *)meta_ptr;
  int n = m->n, w = m->w;

  for (int i = 0; i < n; i++) {
    for (int j = 0; j < w; j++) {
      double we = m->weight_vals[i] * m->erase_vals[j];
      out[i * w + j] = m->mem_vals[i * w + j] * (1.0 - we)
                      + m->weight_vals[i] * m->add_vals[j];
    }
  }
  return out;
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

  if (meta->w_tape_idx != NULL) {
    /* Original path: per-element tape indices */
    for (int i = 0; i < m; i++) {
      double dy = grad_array[out_start + i];
      if (dy == 0.0) continue;

      for (int j = 0; j < n; j++) {
        int w_idx = meta->w_tape_idx[i * n + j];
        grad_array[w_idx] += dy * meta->x_vals[j];
      }

      for (int j = 0; j < n; j++) {
        int x_idx = meta->x_tape_idx[j];
        grad_array[x_idx] += dy * meta->w_vals[i * n + j];
      }
    }
  } else {
    /* Bulk path: consecutive indices from w_tape_start */
    int ws = meta->w_tape_start;
    for (int i = 0; i < m; i++) {
      double dy = grad_array[out_start + i];
      if (dy == 0.0) continue;

      for (int j = 0; j < n; j++) {
        grad_array[ws + i * n + j] += dy * meta->x_vals[j];
      }

      for (int j = 0; j < n; j++) {
        int x_idx = meta->x_tape_idx[j];
        grad_array[x_idx] += dy * meta->w_vals[i * n + j];
      }
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

/*
 * Softmax backward: given dy[i] for i in [0, n),
 *   dx[j] = s[j] * (dy[j] - dot(dy, s))
 * where s = softmax output (stored in meta->out_vals).
 */
void tensor_softmax_backward(double *grad_array, SoftmaxMeta *meta) {
  int n = meta->n;
  int out_start = meta->out_tape_start + 1; /* skip the op entry */

  double dot = 0.0;
  for (int i = 0; i < n; i++)
    dot += grad_array[out_start + i] * meta->out_vals[i];

  for (int i = 0; i < n; i++) {
    double dy = grad_array[out_start + i];
    grad_array[meta->x_tape_idx[i]] += meta->out_vals[i] * (dy - dot);
  }
}

/*
 * LogSoftmax backward: given dy[i] for i in [0, n),
 *   dx[j] = dy[j] - exp(logS[j]) * sum(dy)
 * where logS = logsoftmax output (stored in meta->out_vals).
 */
void tensor_logsoftmax_backward(double *grad_array, SoftmaxMeta *meta) {
  int n = meta->n;
  int out_start = meta->out_tape_start + 1; /* skip the op entry */

  double sum_dy = 0.0;
  for (int i = 0; i < n; i++)
    sum_dy += grad_array[out_start + i];

  for (int i = 0; i < n; i++) {
    double dy = grad_array[out_start + i];
    grad_array[meta->x_tape_idx[i]] += dy - exp(meta->out_vals[i]) * sum_dy;
  }
}

/*
 * BatchCosSim backward: given dy[i] for i in [0, n),
 * where out[i] = beta * cos_sim(key, mem[i]):
 *   d_beta     = sum_i dy[i] * sim[i]
 *   d_key[j]  += sum_i dy[i] * beta * (mem[i][j] - dot_i/|key|^2 * key[j]) / (|key|*|mem_i|)
 *   d_mem[i][j] = dy[i] * beta * (key[j] - dot_i/|mem_i|^2 * mem[i][j]) / (|key|*|mem_i|)
 */
void tensor_batch_cossim_backward(double *grad_array, BatchCosSimMeta *m) {
  int n = m->n, w = m->w;
  int out_start = m->out_tape_start + 1;
  double kn = m->key_norm;
  double kn_sq = kn * kn;
  double d_beta = 0.0;

  for (int i = 0; i < n; i++) {
    double dy = grad_array[out_start + i];
    if (dy == 0.0) continue;

    double rn = m->row_norms[i];
    double denom = kn * rn;
    if (denom <= 1e-12) continue;

    double sim = m->dots[i] / denom;
    d_beta += dy * sim;

    double rn_sq = rn * rn;
    double coeff = dy * m->beta_val / denom;
    double dot_over_kn2 = m->dots[i] / kn_sq;
    double dot_over_rn2 = m->dots[i] / rn_sq;

    for (int j = 0; j < w; j++) {
      grad_array[m->mem_tape_idx[i * w + j]] +=
        coeff * (m->key_vals[j] - dot_over_rn2 * m->mem_vals[i * w + j]);
      grad_array[m->key_tape_idx[j]] +=
        coeff * (m->mem_vals[i * w + j] - dot_over_kn2 * m->key_vals[j]);
    }
  }

  grad_array[m->beta_tape_idx] += d_beta;
}

/*
 * ReadOp backward: given dy[j] for j in [0, w),
 * where out[j] = sum_i weight[i] * mem[i*w+j]:
 *   d_weight[i] = sum_j dy[j] * mem[i*w+j]
 *   d_mem[i][j] = dy[j] * weight[i]
 */
void tensor_readop_backward(double *grad_array, ReadOpMeta *m) {
  int n = m->n, w = m->w;
  int out_start = m->out_tape_start + 1;

  for (int i = 0; i < n; i++) {
    double d_weight = 0.0;
    for (int j = 0; j < w; j++) {
      double dy = grad_array[out_start + j];
      grad_array[m->mem_tape_idx[i * w + j]] += dy * m->weight_vals[i];
      d_weight += dy * m->mem_vals[i * w + j];
    }
    grad_array[m->weight_tape_idx[i]] += d_weight;
  }
}

/*
 * WriteOp backward: given dy[i*w+j] for output matrix,
 * where out[i][j] = mem[i][j]*(1-w[i]*e[j]) + w[i]*a[j]:
 *   d_mem[i][j] = dy[i][j] * (1 - w[i]*e[j])
 *   d_weight[i] = sum_j dy[i][j] * (-mem[i][j]*e[j] + a[j])
 *   d_erase[j] += sum_i dy[i][j] * (-mem[i][j]*w[i])
 *   d_add[j]   += sum_i dy[i][j] * w[i]
 */
void tensor_writeop_backward(double *grad_array, WriteOpMeta *m) {
  int n = m->n, w = m->w;
  int out_start = m->out_tape_start + 1;

  for (int i = 0; i < n; i++) {
    double d_weight = 0.0;
    for (int j = 0; j < w; j++) {
      double dy = grad_array[out_start + i * w + j];
      double we = m->weight_vals[i] * m->erase_vals[j];

      grad_array[m->mem_tape_idx[i * w + j]] += dy * (1.0 - we);
      d_weight += dy * (-m->mem_vals[i * w + j] * m->erase_vals[j]
                        + m->add_vals[j]);
      grad_array[m->erase_tape_idx[j]] +=
        dy * (-m->mem_vals[i * w + j] * m->weight_vals[i]);
      grad_array[m->add_tape_idx[j]] += dy * m->weight_vals[i];
    }
    grad_array[m->weight_tape_idx[i]] += d_weight;
  }
}

/*
 * Interpolation write backward (with fused tanh):
 *   raw[i][j] = (1 - w[i]) * mem[i][j] + w[i] * add[j]
 *   out[i][j] = tanh(raw[i][j])
 *
 *   d_raw = d_out * (1 - out^2)       (tanh derivative)
 *   d_mem[i][j] += d_raw * (1 - w[i])
 *   d_weight[i] += sum_j d_raw * (add[j] - mem[i][j])
 *   d_add[j]    += sum_i d_raw * w[i]
 */
void tensor_interp_write_backward(double *grad_array, InterpWriteMeta *m) {
  int n = m->n, w = m->w;
  int out_start = m->out_tape_start + 1;

  for (int i = 0; i < n; i++) {
    double d_weight = 0.0;
    double wi = m->weight_vals[i];
    for (int j = 0; j < w; j++) {
      double d_out = grad_array[out_start + i * w + j];
      double oval = m->out_vals[i * w + j];
      double d_raw = d_out * (1.0 - oval * oval);   /* tanh derivative */
      grad_array[m->mem_tape_idx[i * w + j]] += d_raw * (1.0 - wi);
      d_weight += d_raw * (m->add_vals[j] - m->mem_vals[i * w + j]);
      grad_array[m->add_tape_idx[j]] += d_raw * wi;
    }
    grad_array[m->weight_tape_idx[i]] += d_weight;
  }
}
