/* backend_tape_kernels.inc — element-wise kernel bodies, parameterized by
   the storage scalar type so the same source compiles for F64 and F32.

   Included once per scalar from backend_tape.c with these macros set:

     SCALAR        : double (F64 stamping) or float (F32 stamping)
     SFX(name)     : suffix macro, e.g. `name##_f64` / `name##_f32`
     VV_FN(stem)   : vForce element-wise unary, e.g. `stem##_d` -> vvexp,
                     `stem##_f` -> vvexpf (mapped per stamping)
     VDSP_VADD/SUB/MUL/DIV/NEG : vDSP element-wise primitives
       (vDSP_vaddD vs vDSP_vadd, ...)

   The bodies use only SCALAR locals + ((SCALAR*)t->data) casts — they DO
   NOT touch dtype_tag. Today only the F64 stamping is invoked (step 3
   single include); step 6 adds the F32 stamping as a compile-cleanliness
   probe (dead code); step 7 wires tag-aware dispatch from the public
   tensor_* surface.

   Scope: element-wise binary (`binop_elementwise_inner`) + element-wise
   unary (`unop_elementwise`). Backward switch, linalg/conv, and the
   optimizer per-param loops move in by plan steps 4 and 5. */

static TensorHandle SFX(binop_elementwise_inner)(TensorHandle ha, TensorHandle hb, int op_tag,
                                                 SCALAR (*scalar_fn)(SCALAR, SCALAR)) {
    Tensor* a = (Tensor*)ha; Tensor* b = (Tensor*)hb;
    int rg = a->requires_grad || b->requires_grad;
    SCALAR* a_d = (SCALAR*)a->data;
    SCALAR* b_d = (SCALAR*)b->data;

    /* Both scalar */
    if (a->numel == 1 && b->numel == 1) {
        Tensor* r = SFX(make_scalar)((double)scalar_fn(a_d[0], b_d[0]), rg);
        if (rg) tape_append(op_tag, r, a, b, 0);
        return r;
    }

    /* Same shape — vDSP fast path */
    if (a->numel == b->numel && a->rank == b->rank) {
        int same = 1;
        for (int k = 0; k < a->rank; k++) {
            if (a->shape[k] != b->shape[k]) { same = 0; break; }
        }
        if (same) {
            int n = a->numel;
            extern int prof_binop_path_count[];
            prof_binop_path_count[0]++;
            SCALAR* data = arena_alloc(n * sizeof(SCALAR));
            /* Direct kernel-time probe (diagnostic): measure the actual
               vDSP call separately from the tape_append attribution. */
            extern double prof_kernel_per_op[];
            extern int prof_kernel_count_per_op[];
            double _k0 = _wall_ms();
#ifdef __APPLE__
            vDSP_Length vn = (vDSP_Length)n;
            switch (op_tag) {
                case OP_ADD: VDSP_VADD(a_d, 1, b_d, 1, data, 1, vn); break;
                case OP_SUB: VDSP_VSUB(b_d, 1, a_d, 1, data, 1, vn); break;
                case OP_MUL: VDSP_VMUL(a_d, 1, b_d, 1, data, 1, vn); break;
                case OP_DIV: VDSP_VDIV(b_d, 1, a_d, 1, data, 1, vn); break;
                default:
                    for (int i = 0; i < n; i++) data[i] = scalar_fn(a_d[i], b_d[i]);
                    break;
            }
#else
            for (int i = 0; i < n; i++) data[i] = scalar_fn(a_d[i], b_d[i]);
#endif
            if (op_tag >= 0 && op_tag < OP_COUNT) {
                prof_kernel_per_op[op_tag] += _wall_ms() - _k0;
                prof_kernel_count_per_op[op_tag]++;
            }
            Tensor* r = SFX(make_tensor_arena)(data, n, a->shape, a->rank, rg);
            if (rg) tape_append(op_tag, r, a, b, 0);
            return r;
        }
    }

    /* Scalar broadcast (one side is rank-0 / numel=1) */
    if (a->numel == 1 || b->numel == 1) {
        Tensor* big = (a->numel == 1) ? b : a;
        SCALAR* big_d = (SCALAR*)big->data;
        SCALAR sv = (a->numel == 1) ? a_d[0] : b_d[0];
        int n = big->numel;
        extern int prof_binop_path_count[];
        prof_binop_path_count[1]++;
        SCALAR* data = arena_alloc(n * sizeof(SCALAR));
        if (a->numel == 1) {
            for (int i = 0; i < n; i++) data[i] = scalar_fn(sv, big_d[i]);
        } else {
            for (int i = 0; i < n; i++) data[i] = scalar_fn(big_d[i], sv);
        }
        Tensor* r = SFX(make_tensor_arena)(data, n, big->shape, big->rank, rg);
        if (rg) tape_append(op_tag, r, a, b, 0);
        return r;
    }

    /* General broadcast */
    {
        extern int prof_binop_path_count[];
        prof_binop_path_count[2]++;
        /* Log shapes once per op_tag (production default; off by
           default if IDRISML_BCAST_QUIET=1 is set). Useful to spot
           call sites taking the slow path. Set
           IDRISML_BCAST_TRACE=1 to log every call (debug-only). */
        static int env_checked = 0;
        static int trace_enabled = 0;
        static int quiet_enabled = 0;
        if (!env_checked) {
            const char* t = getenv("IDRISML_BCAST_TRACE");
            const char* q = getenv("IDRISML_BCAST_QUIET");
            trace_enabled = (t && t[0] == '1');
            quiet_enabled = (q && q[0] == '1');
            env_checked = 1;
        }
        static int logged[OP_COUNT] = {0};
        int should_log = trace_enabled
                       || (!quiet_enabled && op_tag >= 0
                           && op_tag < OP_COUNT && !logged[op_tag]);
        if (should_log) {
            if (op_tag >= 0 && op_tag < OP_COUNT) logged[op_tag] = 1;
            fprintf(stderr, "[tape diag] op=%d general_bcast shapes: a=[", op_tag);
            for (int k = 0; k < a->rank; k++)
                fprintf(stderr, "%d%s", a->shape[k], k+1<a->rank?",":"");
            fprintf(stderr, "] b=[");
            for (int k = 0; k < b->rank; k++)
                fprintf(stderr, "%d%s", b->shape[k], k+1<b->rank?",":"");
            fprintf(stderr, "]\n");
        }
    }
    double _gb0 = _wall_ms();
    int r_shape[MAX_BCAST_RANK], r_rank, r_numel;
    if (!compute_bcast_shape(a, b, r_shape, &r_rank, &r_numel)) {
        // GCOVR_EXCL_START — incompatible-shape abort; the shape-printing loops can't
        // collapse to a same-line guard, and abort() skips the gcov flush. Reachable
        // only on a shape-mismatched binop (the typed surface forbids it upstream).
        fprintf(stderr, "binop_elementwise: incompatible shapes op=%d a=[", op_tag);
        for (int k = 0; k < a->rank; k++)
            fprintf(stderr, "%d%s", a->shape[k], k+1<a->rank?",":"");
        fprintf(stderr, "] b=[");
        for (int k = 0; k < b->rank; k++)
            fprintf(stderr, "%d%s", b->shape[k], k+1<b->rank?",":"");
        fprintf(stderr, "]\n");
        abort();
        // GCOVR_EXCL_STOP
    }
    int a_strides[MAX_BCAST_RANK], b_strides[MAX_BCAST_RANK];
    compute_bcast_strides(a, r_rank, r_shape, a_strides);
    compute_bcast_strides(b, r_rank, r_shape, b_strides);

    SCALAR* data = arena_alloc(r_numel * sizeof(SCALAR));
    int idx[MAX_BCAST_RANK] = {0};
    for (int i = 0; i < r_numel; i++) {
        int ai = 0, bi = 0;
        for (int k = 0; k < r_rank; k++) {
            ai += idx[k] * a_strides[k];
            bi += idx[k] * b_strides[k];
        }
        data[i] = scalar_fn(a_d[ai], b_d[bi]);
        for (int k = r_rank - 1; k >= 0; k--) {
            if (++idx[k] < r_shape[k]) break;
            idx[k] = 0;
        }
    }
    Tensor* r = SFX(make_tensor_arena)(data, r_numel, r_shape, r_rank, rg);
    extern double prof_binop_general_ms;
    prof_binop_general_ms += _wall_ms() - _gb0;
    if (rg) tape_append(op_tag, r, a, b, 0);
    return r;
}

static TensorHandle SFX(unop_elementwise)(TensorHandle ha, int op, SCALAR (*fn)(SCALAR)) {
    Tensor* a = (Tensor*)ha;
    SCALAR* a_d = (SCALAR*)a->data;
    if (a->numel == 1) {
        /* Scalar path (backward rules use [0] indexing) */
        Tensor* r = SFX(make_scalar)((double)fn(a_d[0]), a->requires_grad);
        if (a->requires_grad) tape_append(op, r, a, NULL, 0);
        return r;
    }
    /* Multi-element: apply fn element-wise, preserve shape */
    int n = a->numel;
    SCALAR* data = arena_alloc(n * sizeof(SCALAR));
#ifdef __APPLE__
    {
        int vn = n;
        int used_vdsp = 1;
        switch (op) {
            case OP_NEG:  VDSP_VNEG(a_d, 1, data, 1, (vDSP_Length)n); break;
            case OP_EXP:  VV_EXP(data, a_d, &vn); break;
            case OP_LOG:  VV_LOG(data, a_d, &vn); break;
            case OP_SQRT: VV_SQRT(data, a_d, &vn); break;
            case OP_TANH: VV_TANH(data, a_d, &vn); break;
            case OP_ABS:  VV_FABS(data, a_d, &vn); break;
            default: used_vdsp = 0; break;
        }
        if (!used_vdsp)
            for (int i = 0; i < n; i++) data[i] = fn(a_d[i]);
    }
#else
    for (int i = 0; i < n; i++) data[i] = fn(a_d[i]);
#endif
    Tensor* r = SFX(make_tensor_arena)(data, n, a->shape, a->rank, a->requires_grad);
    if (a->requires_grad) tape_append(op, r, a, NULL, 0);
    return r;
}
