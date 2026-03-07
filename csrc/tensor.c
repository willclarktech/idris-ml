#include <stdlib.h>
#include <string.h>
#include <math.h>

#ifdef __APPLE__
#define ACCELERATE_NEW_LAPACK
#include <Accelerate/Accelerate.h>
#endif


/* -------------------------------------------------------------------
   Arena allocator (chunked: never moves previous allocations)
   ------------------------------------------------------------------- */

typedef struct ArenaChunk {
  char *buf;
  size_t used, cap;
  struct ArenaChunk *next; /* older chunks */
} ArenaChunk;

static ArenaChunk *arena_head = NULL;

static ArenaChunk *arena_new_chunk(size_t cap) {
  ArenaChunk *c = (ArenaChunk *)malloc(sizeof(ArenaChunk));
  c->buf = (char *)malloc(cap);
  c->used = 0;
  c->cap = cap;
  c->next = NULL;
  return c;
}

void arena_init(size_t cap) {
  if (arena_head) return;
  arena_head = arena_new_chunk(cap);
}

void *arena_alloc(size_t size) {
  /* align to 8 bytes */
  size = (size + 7) & ~(size_t)7;
  if (!arena_head) arena_init(1 << 20); /* 1 MB default */
  if (arena_head->used + size > arena_head->cap) {
    /* Allocate a new chunk at least 2x the current and big enough for this alloc */
    size_t new_cap = arena_head->cap * 2;
    while (size > new_cap) new_cap *= 2;
    ArenaChunk *c = arena_new_chunk(new_cap);
    c->next = arena_head;
    arena_head = c;
  }
  void *p = arena_head->buf + arena_head->used;
  arena_head->used += size;
  return p;
}

void arena_reset(void) {
  /* Free all chunks except the largest (head), reset its usage */
  while (arena_head && arena_head->next) {
    ArenaChunk *old = arena_head->next;
    arena_head->next = old->next;
    free(old->buf);
    free(old);
  }
  if (arena_head) arena_head->used = 0;
}


/* Forward declarations (defined after meta types) */
void *dot_meta_set_out(void *meta_ptr, int out_idx);
void *matvec_meta_set_out(void *meta_ptr, int start);
void *softmax_meta_set_out(void *meta_ptr, int start);
void *batch_cossim_meta_set_out(void *p, int start);
void *readop_meta_set_out(void *p, int start);
void *writeop_meta_set_out(void *p, int start);
void *interp_write_meta_set_out(void *p, int start);

/* -------------------------------------------------------------------
   String interning (for parameter IDs)
   ------------------------------------------------------------------- */

#define MAX_INTERN 4096
static char *intern_table[MAX_INTERN];
static int intern_count = 0;

/* Intern a C string: returns a persistent pointer valid for the process
 * lifetime. Returns NULL for NULL or empty strings. */
static char *intern_str(const char *s) {
  if (!s || !s[0]) return NULL;
  for (int i = 0; i < intern_count; i++) {
    if (strcmp(intern_table[i], s) == 0) return intern_table[i];
  }
  if (intern_count >= MAX_INTERN) {
    /* Fallback: strdup without interning (should not happen) */
    return strdup(s);
  }
  intern_table[intern_count] = strdup(s);
  return intern_table[intern_count++];
}


/* -------------------------------------------------------------------
   Tape storage (C-backed autograd tape)
   ------------------------------------------------------------------- */

static int *tape_tags  = NULL;
static int *tape_arg1  = NULL;
static int *tape_arg2  = NULL;
static double *tape_vals = NULL;
static void **tape_meta  = NULL;
static char **tape_pids  = NULL;
static int tape_size = 0;
static int tape_cap  = 0;
static int tape_gen  = 0;

/* Result buffer for walk_backward: collected parameter results.
 * walk_backward uses (pid, grad), walk_backward_ext uses (index, grad). */
static char **result_pids = NULL;
static int   *result_idxs = NULL;
static double *result_vals = NULL;
static int result_cap = 0;

void tape_init(void) {
  if (tape_tags) return;
  tape_cap = 4096;
  tape_tags = (int *)calloc(tape_cap, sizeof(int));
  tape_arg1 = (int *)calloc(tape_cap, sizeof(int));
  tape_arg2 = (int *)calloc(tape_cap, sizeof(int));
  tape_vals = (double *)calloc(tape_cap, sizeof(double));
  tape_meta = (void **)calloc(tape_cap, sizeof(void *));
  tape_pids = (char **)calloc(tape_cap, sizeof(char *));
  tape_size = 0;
  tape_gen = 0;
  result_cap = 256;
  result_pids = (char **)malloc(result_cap * sizeof(char *));
  result_idxs = (int *)malloc(result_cap * sizeof(int));
  result_vals = (double *)malloc(result_cap * sizeof(double));
}

static void tape_ensure_cap(int needed) {
  if (needed < tape_cap) return;
  int new_cap = tape_cap * 2;
  while (needed >= new_cap) new_cap *= 2;
  tape_tags = (int *)realloc(tape_tags, new_cap * sizeof(int));
  tape_arg1 = (int *)realloc(tape_arg1, new_cap * sizeof(int));
  tape_arg2 = (int *)realloc(tape_arg2, new_cap * sizeof(int));
  tape_vals = (double *)realloc(tape_vals, new_cap * sizeof(double));
  tape_meta = (void **)realloc(tape_meta, new_cap * sizeof(void *));
  tape_pids = (char **)realloc(tape_pids, new_cap * sizeof(char *));
  /* Zero new slots */
  memset(tape_tags + tape_cap, 0, (new_cap - tape_cap) * sizeof(int));
  memset(tape_arg1 + tape_cap, 0, (new_cap - tape_cap) * sizeof(int));
  memset(tape_arg2 + tape_cap, 0, (new_cap - tape_cap) * sizeof(int));
  memset(tape_vals + tape_cap, 0, (new_cap - tape_cap) * sizeof(double));
  memset(tape_meta + tape_cap, 0, (new_cap - tape_cap) * sizeof(void *));
  memset(tape_pids + tape_cap, 0, (new_cap - tape_cap) * sizeof(char *));
  tape_cap = new_cap;
}

int tape_get_gen(int dummy) { tape_init(); return tape_gen; }
int tape_get_size(int dummy) { return tape_size; }

int tape_append_const(double val, char *pid) {
  tape_init();
  int idx = tape_size;
  tape_ensure_cap(idx);
  tape_tags[idx] = 0;
  tape_vals[idx] = val;
  tape_pids[idx] = intern_str(pid);
  tape_size = idx + 1;
  return idx;
}

int tape_append_unary(int tag, int a1, double val) {
  int idx = tape_size;
  tape_ensure_cap(idx);
  tape_tags[idx] = tag;
  tape_arg1[idx] = a1;
  tape_vals[idx] = val;
  tape_pids[idx] = NULL;
  tape_size = idx + 1;
  return idx;
}

int tape_append_binary(int tag, int a1, int a2, double val) {
  int idx = tape_size;
  tape_ensure_cap(idx);
  tape_tags[idx] = tag;
  tape_arg1[idx] = a1;
  tape_arg2[idx] = a2;
  tape_vals[idx] = val;
  tape_pids[idx] = NULL;
  tape_size = idx + 1;
  return idx;
}

/* Last tape index written by tape_append_tensor_op (for set_out calls) */
static int tape_last_op_idx = 0;

/* Append tensor op (MatVec, Softmax, BatchCosSim, etc.)
 * tag: op tag, count: number of outputs, meta: op-specific metadata
 * Also calls the appropriate meta_set_out to record the tape index. */
void *tape_append_tensor_op(int tag, int count, void *meta, void *out_buf) {
  int idx = tape_size;
  tape_ensure_cap(idx);
  tape_tags[idx] = tag;
  tape_arg2[idx] = count;
  tape_meta[idx] = meta;
  tape_pids[idx] = NULL;
  tape_size = idx + 1;
  /* Call the appropriate meta_set_out based on tag */
  switch (tag) {
    case 11: matvec_meta_set_out(meta, idx); break;
    case 13: /* fall through */
    case 14: softmax_meta_set_out(meta, idx); break;
    case 15: batch_cossim_meta_set_out(meta, idx); break;
    case 16: readop_meta_set_out(meta, idx); break;
    case 17: writeop_meta_set_out(meta, idx); break;
    case 18: interp_write_meta_set_out(meta, idx); break;
  }
  return out_buf;
}

/* Append DotOp + output ConstOp, set meta->out_tape_idx */
int tape_append_dot_op(void *meta, double val) {
  int idx = tape_size;
  tape_ensure_cap(idx + 1);
  tape_tags[idx] = 12; /* DotOp */
  tape_arg2[idx] = 0;
  tape_meta[idx] = meta;
  tape_pids[idx] = NULL;
  int out_idx = idx + 1;
  tape_tags[out_idx] = 0; /* ConstOp */
  tape_vals[out_idx] = val;
  tape_pids[out_idx] = NULL;
  dot_meta_set_out(meta, out_idx);
  tape_size = out_idx + 1;
  return out_idx;
}

/* Set paramId on existing tape entry. Interns the string. */
int tape_set_pid(int idx, char *pid) {
  tape_pids[idx] = intern_str(pid);
  return idx;
}

/* Reset tape: size=0, gen++, arena reset */
void tape_reset(void) {
  tape_size = 0;
  tape_gen++;
  arena_reset();
}

/* Bulk-append const entries from a weight buffer.
 * Returns start index. */
int tape_bulk_const(double *vals, char **pids, int count) {
  int start = tape_size;
  int end = start + count;
  tape_ensure_cap(end - 1);
  for (int k = 0; k < count; k++) {
    int idx = start + k;
    tape_tags[idx] = 0;
    tape_vals[idx] = vals[k];
    tape_pids[idx] = pids[k]; /* already interned by weight_buf_set_pid */
  }
  tape_size = end;
  return start;
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
   External meta array (for hybrid Scheme/C tape)
   Stores meta pointers written from Scheme via C function call,
   avoiding both foreign-set! 'void* corruption and Scheme vector GC issues.
   ------------------------------------------------------------------- */

static void **ext_meta = NULL;
static int ext_meta_cap = 0;

void ext_meta_set(int idx, void *ptr) {
  if (idx >= ext_meta_cap) {
    int new_cap = ext_meta_cap == 0 ? 4096 : ext_meta_cap * 2;
    while (idx >= new_cap) new_cap *= 2;
    ext_meta = (void **)realloc(ext_meta, new_cap * sizeof(void *));
    memset(ext_meta + ext_meta_cap, 0, (new_cap - ext_meta_cap) * sizeof(void *));
    ext_meta_cap = new_cap;
  }
  ext_meta[idx] = ptr;
}

void *ext_meta_get_arr(void) {
  return ext_meta;
}

void ext_meta_reset(void) {
  if (ext_meta) memset(ext_meta, 0, ext_meta_cap * sizeof(void *));
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


/* -------------------------------------------------------------------
   C-backed backward pass
   ------------------------------------------------------------------- */

int walk_backward(double *grad, int tape_sz) {
  int n_collected = 0;

  for (int idx = tape_sz - 1; idx >= 0; idx--) {
    int tag = tape_tags[idx];
    double g = grad[idx];

    if (tag == 0) {
      /* ConstOp: collect gradient if named parameter */
      if (tape_pids[idx] && g != 0.0) {
        if (n_collected >= result_cap) {
          result_cap *= 2;
          result_pids = (char **)realloc(result_pids, result_cap * sizeof(char *));
          result_vals = (double *)realloc(result_vals, result_cap * sizeof(double));
        }
        result_pids[n_collected] = tape_pids[idx];
        result_vals[n_collected] = g;
        n_collected++;
      }
      continue;
    }

    /* Scalar ops (1-10, 19-20) store gradient at grad[idx]; safe to skip.
     * Tensor ops (11-18) store gradient at output ConstOps; never skip. */
    if (g == 0.0 && (tag <= 10 || tag >= 19)) continue;

    int a1 = tape_arg1[idx];
    int a2 = tape_arg2[idx];

    switch (tag) {
      case 1:  /* NegOp */
        grad[a1] += -g;
        break;
      case 2:  /* AbsOp */
        grad[a1] += g * (tape_vals[a1] > 0 ? 1.0 : (tape_vals[a1] < 0 ? -1.0 : 0.0));
        break;
      case 3:  /* ExpOp */
        grad[a1] += g * tape_vals[idx];
        break;
      case 4:  /* LogOp */
        grad[a1] += g / tape_vals[a1];
        break;
      case 5:  /* SqrtOp */
        grad[a1] += g / (2.0 * tape_vals[idx]);
        break;
      case 6:  /* AddOp */
        grad[a1] += g;
        grad[a2] += g;
        break;
      case 7:  /* SubOp */
        grad[a1] += g;
        grad[a2] += -g;
        break;
      case 8:  /* MulOp */
        grad[a1] += g * tape_vals[a2];
        grad[a2] += g * tape_vals[a1];
        break;
      case 9:  /* DivOp */
        grad[a1] += g / tape_vals[a2];
        grad[a2] += -g * tape_vals[a1] / (tape_vals[a2] * tape_vals[a2]);
        break;
      case 10: { /* PowOp */
        double vx = tape_vals[a1];
        double vy = tape_vals[a2];
        grad[a1] += g * vy * pow(vx, vy - 1.0);
        if (vx != 0.0)
          grad[a2] += g * tape_vals[idx] * log(vx);
        break;
      }
      case 11: /* MatVecOp */
        tensor_matvec_backward(grad, (MatVecMeta *)tape_meta[idx]);
        break;
      case 12: /* DotOp */
        tensor_dot_backward(grad, (DotMeta *)tape_meta[idx]);
        break;
      case 13: /* SoftmaxOp */
        tensor_softmax_backward(grad, (SoftmaxMeta *)tape_meta[idx]);
        break;
      case 14: /* LogSoftmaxOp */
        tensor_logsoftmax_backward(grad, (SoftmaxMeta *)tape_meta[idx]);
        break;
      case 15: /* BatchCosSimOp */
        tensor_batch_cossim_backward(grad, (BatchCosSimMeta *)tape_meta[idx]);
        break;
      case 16: /* ReadOpOp */
        tensor_readop_backward(grad, (ReadOpMeta *)tape_meta[idx]);
        break;
      case 17: /* WriteOpOp */
        tensor_writeop_backward(grad, (WriteOpMeta *)tape_meta[idx]);
        break;
      case 18: /* InterpWriteOp */
        tensor_interp_write_backward(grad, (InterpWriteMeta *)tape_meta[idx]);
        break;
      case 19: { /* SigmoidOp */
        double val = tape_vals[idx];
        grad[a1] += g * val * (1.0 - val);
        break;
      }
      case 20: { /* TanhOp */
        double val = tape_vals[idx];
        grad[a1] += g * (1.0 - val * val);
        break;
      }
      default:
        break;
    }
  }

  return n_collected;
}

/* Walk backward, collect results, reset tape, free grad array.
 * Combined operation ensures proper ordering: backward completes before
 * arena reset invalidates metadata, and result strings are interned
 * (valid forever). */
int walk_backward_and_reset(double *grad, int tape_sz) {
  int n = walk_backward(grad, tape_sz);
  tape_size = 0;
  tape_gen++;
  arena_reset();
  free(grad);
  return n;
}

/* Access collected results */
char *result_get_pid(int i) { return result_pids[i]; }
int result_get_idx(int i) { return result_idxs[i]; }
double result_get_val(int i) { return result_vals[i]; }

/* Walk backward with external arrays (Scheme-allocated foreign memory).
 * Same logic as walk_backward but reads from passed-in arrays instead
 * of the global C tape. Collects (index, grad) pairs instead of
 * (pid, grad) pairs — caller looks up pid from Scheme vector.
 * Also frees the grad array after use. */
int walk_backward_ext(double *grad, int tape_sz,
                      int *tags, int *arg1, int *arg2,
                      double *vals) {
  void **meta = ext_meta;
  /* Ensure result arrays are allocated (tape_init may not have been called) */
  if (!result_idxs) {
    result_cap = 256;
    result_idxs = (int *)malloc(result_cap * sizeof(int));
    result_vals = (double *)malloc(result_cap * sizeof(double));
  }
  int n_collected = 0;

  for (int idx = tape_sz - 1; idx >= 0; idx--) {
    int tag = tags[idx];
    double g = grad[idx];

    if (tag == 0) {
      /* ConstOp: always collect — caller filters by pid in Scheme.
       * We can't check pid here since pids are in Scheme vector. */
      if (g != 0.0) {
        if (n_collected >= result_cap) {
          result_cap *= 2;
          result_idxs = (int *)realloc(result_idxs, result_cap * sizeof(int));
          result_vals = (double *)realloc(result_vals, result_cap * sizeof(double));
        }
        result_idxs[n_collected] = idx;
        result_vals[n_collected] = g;
        n_collected++;
      }
      continue;
    }

    /* Scalar ops (1-10, 19-20) store gradient at grad[idx]; safe to skip.
     * Tensor ops (11-18) store gradient at output ConstOps; never skip. */
    if (g == 0.0 && (tag <= 10 || tag >= 19)) continue;

    int a1 = arg1[idx];
    int a2 = arg2[idx];

    switch (tag) {
      case 1:  grad[a1] += -g; break;
      case 2:  grad[a1] += g * (vals[a1] > 0 ? 1.0 : (vals[a1] < 0 ? -1.0 : 0.0)); break;
      case 3:  grad[a1] += g * vals[idx]; break;
      case 4:  grad[a1] += g / vals[a1]; break;
      case 5:  grad[a1] += g / (2.0 * vals[idx]); break;
      case 6:  grad[a1] += g; grad[a2] += g; break;
      case 7:  grad[a1] += g; grad[a2] += -g; break;
      case 8:  grad[a1] += g * vals[a2]; grad[a2] += g * vals[a1]; break;
      case 9:  grad[a1] += g / vals[a2]; grad[a2] += -g * vals[a1] / (vals[a2] * vals[a2]); break;
      case 10: {
        double vx = vals[a1], vy = vals[a2];
        grad[a1] += g * vy * pow(vx, vy - 1.0);
        if (vx != 0.0) grad[a2] += g * vals[idx] * log(vx);
        break;
      }
      case 11: tensor_matvec_backward(grad, (MatVecMeta *)meta[idx]); break;
      case 12: tensor_dot_backward(grad, (DotMeta *)meta[idx]); break;
      case 13: tensor_softmax_backward(grad, (SoftmaxMeta *)meta[idx]); break;
      case 14: tensor_logsoftmax_backward(grad, (SoftmaxMeta *)meta[idx]); break;
      case 15: tensor_batch_cossim_backward(grad, (BatchCosSimMeta *)meta[idx]); break;
      case 16: tensor_readop_backward(grad, (ReadOpMeta *)meta[idx]); break;
      case 17: tensor_writeop_backward(grad, (WriteOpMeta *)meta[idx]); break;
      case 18: tensor_interp_write_backward(grad, (InterpWriteMeta *)meta[idx]); break;
      case 19: { double val = vals[idx]; grad[a1] += g * val * (1.0 - val); break; }
      case 20: { double val = vals[idx]; grad[a1] += g * (1.0 - val * val); break; }
      default: break;
    }
  }

  free(grad);
  /* arena_reset and ext_meta_reset called by Scheme after this returns */
  return n_collected;
}

/* Route tensor op set_out based on tag. Called from Scheme after
 * writing tape entry with foreign-set!. */
void *tensor_op_set_out(int tag, void *meta, int idx) {
  switch (tag) {
    case 11: return matvec_meta_set_out(meta, idx);
    case 13: case 14: return softmax_meta_set_out(meta, idx);
    case 15: return batch_cossim_meta_set_out(meta, idx);
    case 16: return readop_meta_set_out(meta, idx);
    case 17: return writeop_meta_set_out(meta, idx);
    case 18: return interp_write_meta_set_out(meta, idx);
    default: return meta;
  }
}


/* -------------------------------------------------------------------
   Weight buffer (C-backed)
   ------------------------------------------------------------------- */

typedef struct {
  double *vals;     /* C double array */
  char **pids;      /* pid string pointers */
  int count;
  int cached_start; /* tape start index */
  int cached_gen;   /* tape generation (-1 = uncached) */
} WeightBuf;

WeightBuf *weight_buf_alloc(int count) {
  WeightBuf *wb = (WeightBuf *)malloc(sizeof(WeightBuf));
  wb->vals = (double *)calloc(count, sizeof(double));
  wb->pids = (char **)calloc(count, sizeof(char *));
  wb->count = count;
  wb->cached_start = -1;
  wb->cached_gen = -1;
  return wb;
}

WeightBuf *weight_buf_set_val(WeightBuf *wb, int idx, double val) {
  wb->vals[idx] = val;
  return wb;
}

WeightBuf *weight_buf_set_pid(WeightBuf *wb, int idx, char *pid) {
  wb->pids[idx] = intern_str(pid);
  return wb;
}

double *weight_buf_vals(WeightBuf *wb) { return wb->vals; }

/* Ensure weight buffer entries are on tape (epoch-cached).
 * Returns tape start index. */
int weight_buf_ensure(WeightBuf *wb) {
  if (wb->cached_gen == tape_gen) return wb->cached_start;
  int start = tape_bulk_const(wb->vals, wb->pids, wb->count);
  wb->cached_start = start;
  wb->cached_gen = tape_gen;
  return start;
}
