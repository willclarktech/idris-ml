/*
 * SafeTensors serialization for idris-ml.
 * Backend-agnostic: uses only backend.h public API.
 *
 * Format: [8-byte LE u64 header_size][JSON header][tensor data]
 *
 * Per-tensor dtype is read from the param's actual runtime dtype via
 * `tensor_dtype_name()`; bytes are written in the matching width
 * (F64 -> 8, F32 -> 4, BF16/F16 -> 2, I8/U8/BOOL -> 1, I16 -> 2,
 * I32 -> 4, I64 -> 8). This matches the SafeTensors convention so
 * `safetensors.torch.load_file()` round-trips correctly when the model
 * is later loaded from Python.
 *
 * Everything moves through a `double` lingua franca: save pulls each
 * param into doubles (`tensor_to_doubles`), then packs them into the
 * on-disk element type; load reads the on-disk bytes into doubles, then
 * `param_load_data` narrows back to the param's storage dtype. This keeps
 * all dtype knowledge here — no backend-interface changes — and is
 * byte-exact for bf16/f16 round-trips (bf16 -> f64 -> bf16 is identity)
 * and for every integer type except I64 above 2^53 (a double can't hold
 * those exactly; torch's .to(kFloat64) rounds before we ever see the
 * bytes). I64 weights are otherwise rare; the exact path would need a new
 * backend int64 extractor.
 *
 * Loading enforces a dtype gate by default (param_load): mismatch
 * between the on-disk dtype and the destination param's dtype is an
 * error. Callers wanting silent precision conversion at load time
 * pass `allow_cast=1` to `param_load_with_policy()` — file bytes are
 * read into doubles regardless of source dtype, then loaded into the
 * destination param via `param_load_data` (which narrows back to the
 * param's actual storage dtype as needed).
 */

#include "backend.h"
#include "cJSON.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdint.h>

_Static_assert(__BYTE_ORDER__ == __ORDER_LITTLE_ENDIAN__,
               "SafeTensors requires little-endian platform");

/* Map a SafeTensors dtype name to bytes-per-element. Unknown -> 0. */
static size_t dtype_byte_width(const char* name) {
    if (!name) return 0;
    if (strcmp(name, "F64")  == 0) return 8;
    if (strcmp(name, "F32")  == 0) return 4;
    if (strcmp(name, "BF16") == 0) return 2;
    if (strcmp(name, "F16")  == 0) return 2;
    if (strcmp(name, "I8")   == 0) return 1;
    if (strcmp(name, "I16")  == 0) return 2;
    if (strcmp(name, "I32")  == 0) return 4;
    if (strcmp(name, "I64")  == 0) return 8;
    if (strcmp(name, "U8")   == 0) return 1;
    if (strcmp(name, "BOOL") == 0) return 1;
    return 0;
}

/* ----------------------------------------------------------------------
   bf16 / f16 <-> double bit conversions

   bf16 is the high 16 bits of an IEEE-754 binary32. f16 is IEEE-754
   binary16. Both go through `float` then widen/narrow to `double`. These
   are the only dtypes that aren't a plain integral cast; everything moves
   through the `double` lingua franca above.
   ---------------------------------------------------------------------- */

static double bf16_bits_to_double(uint16_t h) {
    uint32_t bits = (uint32_t)h << 16;  /* bf16 occupies the f32 high half */
    float f;
    memcpy(&f, &bits, sizeof(f));
    return (double)f;
}

static uint16_t double_to_bf16_bits(double d) {
    float f = (float)d;
    uint32_t bits;
    memcpy(&bits, &f, sizeof(bits));
    /* NaN: keep it quiet and non-zero so it survives the round-trip. */
    if ((bits & 0x7f800000u) == 0x7f800000u && (bits & 0x007fffffu) != 0u)
        return (uint16_t)((bits >> 16) | 0x0040u);
    /* Round to nearest, ties to even on the dropped low 16 bits. */
    uint32_t rounding_bias = 0x00007fffu + ((bits >> 16) & 1u);
    bits += rounding_bias;
    return (uint16_t)(bits >> 16);
}

static double f16_bits_to_double(uint16_t h) {
    uint32_t sign = (uint32_t)(h & 0x8000u) << 16;
    uint32_t exp  = (h >> 10) & 0x1fu;
    uint32_t mant = h & 0x3ffu;
    uint32_t bits;
    if (exp == 0) {
        if (mant == 0) {
            bits = sign;                       /* +/- zero */
        } else {
            /* Subnormal: normalize into f32. */
            exp = 1;
            while ((mant & 0x400u) == 0) { mant <<= 1; exp--; }
            mant &= 0x3ffu;
            bits = sign | ((exp + (127 - 15)) << 23) | (mant << 13);
        }
    } else if (exp == 0x1fu) {
        bits = sign | 0x7f800000u | (mant << 13);  /* Inf / NaN */
    } else {
        bits = sign | ((exp + (127 - 15)) << 23) | (mant << 13);
    }
    float f;
    memcpy(&f, &bits, sizeof(f));
    return (double)f;
}

static uint16_t double_to_f16_bits(double d) {
    float f = (float)d;
    uint32_t bits;
    memcpy(&bits, &f, sizeof(bits));
    uint32_t sign = (bits >> 16) & 0x8000u;
    int32_t  exp  = (int32_t)((bits >> 23) & 0xffu) - 127 + 15;  /* rebias */
    uint32_t mant = bits & 0x7fffffu;

    if (((bits >> 23) & 0xffu) == 0xffu) {            /* Inf / NaN */
        if (mant) return (uint16_t)(sign | 0x7e00u);  /* quiet NaN */
        return (uint16_t)(sign | 0x7c00u);            /* Inf */
    }
    if (exp >= 0x1f) return (uint16_t)(sign | 0x7c00u);   /* overflow -> Inf */
    if (exp <= 0) {
        if (exp < -10) return (uint16_t)sign;             /* underflow -> 0 */
        /* Subnormal: add implicit leading 1, shift, round to nearest even. */
        mant |= 0x800000u;
        uint32_t shift = (uint32_t)(14 - exp);
        uint32_t halfm = mant >> shift;
        uint32_t rem   = mant & ((1u << shift) - 1u);
        uint32_t half  = 1u << (shift - 1);
        if (rem > half || (rem == half && (halfm & 1u))) halfm++;
        return (uint16_t)(sign | halfm);
    }
    /* Normal: round mantissa to 10 bits, nearest, ties to even. */
    uint32_t halfm = mant >> 13;
    uint32_t rem   = mant & 0x1fffu;
    if (rem > 0x1000u || (rem == 0x1000u && (halfm & 1u))) {
        halfm++;
        if (halfm == 0x400u) { halfm = 0; exp++; }     /* mantissa carry */
        if (exp >= 0x1f) return (uint16_t)(sign | 0x7c00u);
    }
    return (uint16_t)(sign | ((uint32_t)exp << 10) | halfm);
}

/* ================================================================
   Save
   ================================================================ */

int param_save(const char* path) {
    int n = param_count();
    if (n == 0) {
        fprintf(stderr, "param_save: no parameters registered\n");
        return -1;
    }

    /* Build JSON header */
    cJSON* root = cJSON_CreateObject();
    if (!root) return -1;

    /* First pass: compute per-tensor byte size (depends on dtype) +
       running data offsets. */
    size_t data_offset = 0;
    size_t* offsets = (size_t*)calloc(n, sizeof(size_t));
    size_t* sizes = (size_t*)calloc(n, sizeof(size_t));
    const char** dtypes = (const char**)calloc(n, sizeof(const char*));
    if (!offsets || !sizes || !dtypes) {
        cJSON_Delete(root); free(offsets); free(sizes); free(dtypes); return -1;
    }

    for (int i = 0; i < n; i++) {
        offsets[i] = data_offset;
        TensorHandle t = param_tensor(i);
        dtypes[i] = tensor_dtype_name(t);
        size_t width = dtype_byte_width(dtypes[i]);
        if (width == 0) {
            fprintf(stderr, "param_save: unsupported dtype '%s' for '%s'\n",
                    dtypes[i] ? dtypes[i] : "(null)", param_name(i));
            cJSON_Delete(root); free(offsets); free(sizes); free(dtypes);
            return -1;
        }
        sizes[i] = (size_t)tensor_numel(t) * width;
        data_offset += sizes[i];
    }

    /* Build JSON entries */
    for (int i = 0; i < n; i++) {
        const char* name = param_name(i);
        TensorHandle t = param_tensor(i);
        int rank = tensor_dim(t);

        cJSON* entry = cJSON_CreateObject();
        cJSON_AddStringToObject(entry, "dtype", dtypes[i]);

        cJSON* shape = cJSON_CreateArray();
        for (int d = 0; d < rank; d++) {
            cJSON_AddItemToArray(shape, cJSON_CreateNumber(tensor_size(t, d)));
        }
        cJSON_AddItemToObject(entry, "shape", shape);

        cJSON* data_off = cJSON_CreateArray();
        cJSON_AddItemToArray(data_off, cJSON_CreateNumber((double)offsets[i]));
        cJSON_AddItemToArray(data_off, cJSON_CreateNumber((double)(offsets[i] + sizes[i])));
        cJSON_AddItemToObject(entry, "data_offsets", data_off);

        cJSON_AddItemToObject(root, name, entry);
    }

    char* json_str = cJSON_PrintUnformatted(root);
    cJSON_Delete(root);
    if (!json_str) { free(offsets); free(sizes); free(dtypes); return -1; }

    size_t json_len = strlen(json_str);
    /* Pad to 8-byte alignment */
    size_t padded_len = (json_len + 7) & ~(size_t)7;

    /* Write file */
    FILE* f = fopen(path, "wb");
    if (!f) {
        fprintf(stderr, "param_save: cannot open '%s' for writing\n", path);
        free(json_str); free(offsets); free(sizes); free(dtypes);
        return -1;
    }

    /* 8-byte LE header size */
    uint64_t header_size = (uint64_t)padded_len;
    fwrite(&header_size, sizeof(uint64_t), 1, f);

    /* JSON header (padded with spaces) */
    fwrite(json_str, 1, json_len, f);
    for (size_t p = json_len; p < padded_len; p++) {
        fputc(' ', f);
    }
    free(json_str);

    /* Tensor data — write in the param's actual dtype width. Everything
       is pulled into a double buffer first (the lingua franca), then packed
       into the on-disk element type. */
    for (int i = 0; i < n; i++) {
        TensorHandle t = param_tensor(i);
        int numel = tensor_numel(t);
        const char* dt = dtypes[i];
        if (strcmp(dt, "F32") == 0) {
            float* buf = (float*)malloc((size_t)numel * sizeof(float));
            if (!buf) { fclose(f); free(offsets); free(sizes); free(dtypes); return -1; }
            tensor_to_floats(t, buf);
            fwrite(buf, sizeof(float), numel, f);
            free(buf);
            continue;
        }
        double* dbuf = (double*)malloc((size_t)numel * sizeof(double));
        if (!dbuf) { fclose(f); free(offsets); free(sizes); free(dtypes); return -1; }
        tensor_to_doubles(t, dbuf);
        if (strcmp(dt, "F64") == 0) {
            fwrite(dbuf, sizeof(double), numel, f);
        } else if (strcmp(dt, "BF16") == 0) {
            for (int e = 0; e < numel; e++) {
                uint16_t v = double_to_bf16_bits(dbuf[e]);
                fwrite(&v, sizeof(v), 1, f);
            }
        } else if (strcmp(dt, "F16") == 0) {
            for (int e = 0; e < numel; e++) {
                uint16_t v = double_to_f16_bits(dbuf[e]);
                fwrite(&v, sizeof(v), 1, f);
            }
        } else if (strcmp(dt, "I8") == 0) {
            for (int e = 0; e < numel; e++) {
                int8_t v = (int8_t)dbuf[e];
                fwrite(&v, sizeof(v), 1, f);
            }
        } else if (strcmp(dt, "I16") == 0) {
            for (int e = 0; e < numel; e++) {
                int16_t v = (int16_t)dbuf[e];
                fwrite(&v, sizeof(v), 1, f);
            }
        } else if (strcmp(dt, "I32") == 0) {
            for (int e = 0; e < numel; e++) {
                int32_t v = (int32_t)dbuf[e];
                fwrite(&v, sizeof(v), 1, f);
            }
        } else if (strcmp(dt, "I64") == 0) {
            for (int e = 0; e < numel; e++) {
                int64_t v = (int64_t)dbuf[e];
                fwrite(&v, sizeof(v), 1, f);
            }
        } else if (strcmp(dt, "U8") == 0) {
            for (int e = 0; e < numel; e++) {
                uint8_t v = (uint8_t)dbuf[e];
                fwrite(&v, sizeof(v), 1, f);
            }
        } else { /* BOOL */
            for (int e = 0; e < numel; e++) {
                uint8_t v = (dbuf[e] != 0.0) ? 1 : 0;
                fwrite(&v, sizeof(v), 1, f);
            }
        }
        free(dbuf);
    }

    fclose(f);
    free(offsets);
    free(sizes);
    free(dtypes);
    return 0;
}

/* ================================================================
   Load
   ================================================================ */

int param_load_with_policy(const char* path, int allow_cast) {
    FILE* f = fopen(path, "rb");
    if (!f) {
        fprintf(stderr, "param_load: cannot open '%s'\n", path);
        return -1;
    }

    /* Read header size */
    uint64_t header_size;
    if (fread(&header_size, sizeof(uint64_t), 1, f) != 1) {
        fprintf(stderr, "param_load: failed to read header size\n");
        fclose(f);
        return -1;
    }

    /* Read JSON header */
    char* json_str = (char*)malloc(header_size + 1);
    if (!json_str) { fclose(f); return -1; }
    if (fread(json_str, 1, header_size, f) != header_size) {
        fprintf(stderr, "param_load: failed to read header\n");
        free(json_str); fclose(f);
        return -1;
    }
    json_str[header_size] = '\0';

    /* Data section starts at offset 8 + header_size */
    long data_start = 8 + (long)header_size;

    cJSON* root = cJSON_Parse(json_str);
    free(json_str);
    if (!root) {
        fprintf(stderr, "param_load: failed to parse JSON header\n");
        fclose(f);
        return -1;
    }

    int n = param_count();
    int loaded = 0;
    int rc = 0;

    /* For each tensor in the file, find matching param */
    cJSON* entry = NULL;
    cJSON_ArrayForEach(entry, root) {
        const char* name = entry->string;
        if (!name) continue;

        /* Look up in param registry */
        int pidx = -1;
        for (int i = 0; i < n; i++) {
            if (strcmp(param_name(i), name) == 0) {
                pidx = i;
                break;
            }
        }
        if (pidx < 0) {
            fprintf(stderr, "param_load: skipping '%s' (not in registry)\n", name);
            continue;
        }

        /* Read dtype tag — driver for the rest of the load. */
        cJSON* dtype_node = cJSON_GetObjectItem(entry, "dtype");
        const char* src_dtype = (dtype_node && cJSON_IsString(dtype_node)) ? dtype_node->valuestring : "F64";
        size_t src_width = dtype_byte_width(src_dtype);
        if (src_width == 0) {
            fprintf(stderr, "param_load: unsupported on-disk dtype '%s' for '%s'\n",
                    src_dtype, name);
            rc = -1;
            continue;
        }

        TensorHandle t = param_tensor(pidx);
        const char* dst_dtype = tensor_dtype_name(t);
        int dtypes_match = strcmp(src_dtype, dst_dtype) == 0;
        if (!dtypes_match && !allow_cast) {
            fprintf(stderr,
                    "param_load: dtype mismatch for '%s' — on disk %s, destination %s. "
                    "Pass allow_cast=1 to convert at load time.\n",
                    name, src_dtype, dst_dtype);
            rc = -1;
            continue;
        }

        /* Read data_offsets */
        cJSON* offsets = cJSON_GetObjectItem(entry, "data_offsets");
        if (!offsets || cJSON_GetArraySize(offsets) != 2) {
            fprintf(stderr, "param_load: bad data_offsets for '%s'\n", name);
            rc = -1;
            continue;
        }
        size_t start = (size_t)cJSON_GetArrayItem(offsets, 0)->valuedouble;
        size_t end = (size_t)cJSON_GetArrayItem(offsets, 1)->valuedouble;
        size_t byte_len = end - start;
        int numel = (int)(byte_len / src_width);

        /* Validate element count */
        int expected_numel = tensor_numel(t);
        if (numel != expected_numel) {
            fprintf(stderr, "param_load: size mismatch for '%s': file has %d, registry has %d\n",
                    name, numel, expected_numel);
            rc = -1;
            continue;
        }

        /* Read raw bytes, then convert to doubles (the lingua franca of
           param_load_data — destination dtype conversion happens C-side). */
        void* raw_buf = malloc(byte_len);
        if (!raw_buf) { rc = -1; continue; }
        fseek(f, data_start + (long)start, SEEK_SET);
        if (fread(raw_buf, 1, byte_len, f) != byte_len) {
            fprintf(stderr, "param_load: failed to read data for '%s'\n", name);
            free(raw_buf);
            rc = -1;
            continue;
        }

        double* dbuf;
        int owns_dbuf = 0;
        if (strcmp(src_dtype, "F64") == 0) {  /* already lingua franca */
            dbuf = (double*)raw_buf;
        } else {
            dbuf = (double*)malloc((size_t)numel * sizeof(double));
            if (!dbuf) { free(raw_buf); rc = -1; continue; }
            owns_dbuf = 1;
            if (strcmp(src_dtype, "F32") == 0) {
                const float* s = (const float*)raw_buf;
                for (int i = 0; i < numel; i++) dbuf[i] = (double)s[i];
            } else if (strcmp(src_dtype, "BF16") == 0) {
                const uint16_t* s = (const uint16_t*)raw_buf;
                for (int i = 0; i < numel; i++) dbuf[i] = bf16_bits_to_double(s[i]);
            } else if (strcmp(src_dtype, "F16") == 0) {
                const uint16_t* s = (const uint16_t*)raw_buf;
                for (int i = 0; i < numel; i++) dbuf[i] = f16_bits_to_double(s[i]);
            } else if (strcmp(src_dtype, "I8") == 0) {
                const int8_t* s = (const int8_t*)raw_buf;
                for (int i = 0; i < numel; i++) dbuf[i] = (double)s[i];
            } else if (strcmp(src_dtype, "I16") == 0) {
                const int16_t* s = (const int16_t*)raw_buf;
                for (int i = 0; i < numel; i++) dbuf[i] = (double)s[i];
            } else if (strcmp(src_dtype, "I32") == 0) {
                const int32_t* s = (const int32_t*)raw_buf;
                for (int i = 0; i < numel; i++) dbuf[i] = (double)s[i];
            } else if (strcmp(src_dtype, "I64") == 0) {
                const int64_t* s = (const int64_t*)raw_buf;
                for (int i = 0; i < numel; i++) dbuf[i] = (double)s[i];
            } else if (strcmp(src_dtype, "U8") == 0) {
                const uint8_t* s = (const uint8_t*)raw_buf;
                for (int i = 0; i < numel; i++) dbuf[i] = (double)s[i];
            } else {  /* BOOL */
                const uint8_t* s = (const uint8_t*)raw_buf;
                for (int i = 0; i < numel; i++) dbuf[i] = (s[i] != 0) ? 1.0 : 0.0;
            }
        }

        param_load_data(pidx, dbuf, numel);
        if (owns_dbuf) { free(raw_buf); free(dbuf); }
        else { free(raw_buf); }
        loaded++;
    }

    cJSON_Delete(root);
    fclose(f);

    fprintf(stderr, "param_load: loaded %d/%d parameters from '%s'%s\n",
            loaded, n, path,
            (rc != 0) ? " (with errors — see above)" : "");
    return rc;
}

int param_load(const char* path) {
    return param_load_with_policy(path, /*allow_cast=*/0);
}

/* ================================================================
   Optimizer state save/load
   ================================================================ */

/* Helper: write a named tensor entry to a safetensors-style save context.
   We reuse the same safetensors format but with prefixed names. */

int optimizer_save(OptimizerHandle opt, const char* path) {
    int n = optimizer_buf_count(opt);
    if (n == 0) {
        fprintf(stderr, "optimizer_save: no parameters registered\n");
        return -1;
    }

    cJSON* root = cJSON_CreateObject();
    if (!root) return -1;

    /* Collect all tensor data: meta(9) + m_bufs(n) + v_bufs(n) */
    int total_tensors = 1 + 2 * n;  /* meta + m + v */
    size_t* offsets = (size_t*)calloc(total_tensors, sizeof(size_t));
    size_t* byte_sizes = (size_t*)calloc(total_tensors, sizeof(size_t));
    double** bufs = (double**)calloc(total_tensors, sizeof(double*));
    if (!offsets || !byte_sizes || !bufs) {
        cJSON_Delete(root); free(offsets); free(byte_sizes); free(bufs);
        return -1;
    }

    int tidx = 0;
    size_t data_offset = 0;

    /* Meta tensor: 9 doubles encoding optimizer scalar state */
    {
        double meta[9];
        optimizer_get_meta(opt, meta);
        bufs[tidx] = (double*)malloc(9 * sizeof(double));
        memcpy(bufs[tidx], meta, 9 * sizeof(double));
        offsets[tidx] = data_offset;
        byte_sizes[tidx] = 9 * sizeof(double);
        data_offset += byte_sizes[tidx];

        cJSON* entry = cJSON_CreateObject();
        cJSON_AddStringToObject(entry, "dtype", "F64");
        cJSON* shape = cJSON_CreateArray();
        cJSON_AddItemToArray(shape, cJSON_CreateNumber(9));
        cJSON_AddItemToObject(entry, "shape", shape);
        cJSON* d_off = cJSON_CreateArray();
        cJSON_AddItemToArray(d_off, cJSON_CreateNumber((double)offsets[tidx]));
        cJSON_AddItemToArray(d_off, cJSON_CreateNumber((double)(offsets[tidx] + byte_sizes[tidx])));
        cJSON_AddItemToObject(entry, "data_offsets", d_off);
        cJSON_AddItemToObject(root, "__opt_meta__", entry);
        tidx++;
    }

    /* Per-param m and v buffers */
    for (int i = 0; i < n; i++) {
        TensorHandle t = param_tensor(i);
        int numel = tensor_numel(t);
        const char* pname = param_name(i);

        /* m buffer */
        bufs[tidx] = (double*)malloc(numel * sizeof(double));
        optimizer_get_m(opt, i, bufs[tidx]);
        offsets[tidx] = data_offset;
        byte_sizes[tidx] = numel * sizeof(double);
        data_offset += byte_sizes[tidx];

        char name_m[512];
        snprintf(name_m, sizeof(name_m), "opt_m__%s", pname);

        cJSON* entry_m = cJSON_CreateObject();
        cJSON_AddStringToObject(entry_m, "dtype", "F64");
        cJSON* shape_m = cJSON_CreateArray();
        int rank = tensor_dim(t);
        for (int d = 0; d < rank; d++)
            cJSON_AddItemToArray(shape_m, cJSON_CreateNumber(tensor_size(t, d)));
        cJSON_AddItemToObject(entry_m, "shape", shape_m);
        cJSON* d_off_m = cJSON_CreateArray();
        cJSON_AddItemToArray(d_off_m, cJSON_CreateNumber((double)offsets[tidx]));
        cJSON_AddItemToArray(d_off_m, cJSON_CreateNumber((double)(offsets[tidx] + byte_sizes[tidx])));
        cJSON_AddItemToObject(entry_m, "data_offsets", d_off_m);
        cJSON_AddItemToObject(root, name_m, entry_m);
        tidx++;

        /* v buffer */
        bufs[tidx] = (double*)malloc(numel * sizeof(double));
        optimizer_get_v(opt, i, bufs[tidx]);
        offsets[tidx] = data_offset;
        byte_sizes[tidx] = numel * sizeof(double);
        data_offset += byte_sizes[tidx];

        char name_v[512];
        snprintf(name_v, sizeof(name_v), "opt_v__%s", pname);

        cJSON* entry_v = cJSON_CreateObject();
        cJSON_AddStringToObject(entry_v, "dtype", "F64");
        cJSON* shape_v = cJSON_CreateArray();
        for (int d = 0; d < rank; d++)
            cJSON_AddItemToArray(shape_v, cJSON_CreateNumber(tensor_size(t, d)));
        cJSON_AddItemToObject(entry_v, "shape", shape_v);
        cJSON* d_off_v = cJSON_CreateArray();
        cJSON_AddItemToArray(d_off_v, cJSON_CreateNumber((double)offsets[tidx]));
        cJSON_AddItemToArray(d_off_v, cJSON_CreateNumber((double)(offsets[tidx] + byte_sizes[tidx])));
        cJSON_AddItemToObject(entry_v, "data_offsets", d_off_v);
        cJSON_AddItemToObject(root, name_v, entry_v);
        tidx++;
    }

    /* Write file */
    char* json_str = cJSON_PrintUnformatted(root);
    cJSON_Delete(root);
    if (!json_str) goto cleanup_fail;

    {
        size_t json_len = strlen(json_str);
        size_t padded_len = (json_len + 7) & ~(size_t)7;

        FILE* f = fopen(path, "wb");
        if (!f) {
            fprintf(stderr, "optimizer_save: cannot open '%s'\n", path);
            free(json_str);
            goto cleanup_fail;
        }

        uint64_t header_size = (uint64_t)padded_len;
        fwrite(&header_size, sizeof(uint64_t), 1, f);
        fwrite(json_str, 1, json_len, f);
        for (size_t p = json_len; p < padded_len; p++) fputc(' ', f);
        free(json_str);

        for (int i = 0; i < tidx; i++) {
            fwrite(bufs[i], 1, byte_sizes[i], f);
        }
        fclose(f);
    }

    for (int i = 0; i < tidx; i++) free(bufs[i]);
    free(offsets); free(byte_sizes); free(bufs);
    return 0;

cleanup_fail:
    for (int i = 0; i < tidx; i++) free(bufs[i]);
    free(offsets); free(byte_sizes); free(bufs);
    return -1;
}

int optimizer_load(OptimizerHandle opt, const char* path) {
    FILE* f = fopen(path, "rb");
    if (!f) {
        fprintf(stderr, "optimizer_load: cannot open '%s'\n", path);
        return -1;
    }

    uint64_t header_size;
    if (fread(&header_size, sizeof(uint64_t), 1, f) != 1) {
        fclose(f); return -1;
    }

    char* json_str = (char*)malloc(header_size + 1);
    if (!json_str) { fclose(f); return -1; }
    if (fread(json_str, 1, header_size, f) != header_size) {
        free(json_str); fclose(f); return -1;
    }
    json_str[header_size] = '\0';

    long data_start = 8 + (long)header_size;

    cJSON* root = cJSON_Parse(json_str);
    free(json_str);
    if (!root) { fclose(f); return -1; }

    int n = param_count();

    /* Load meta */
    cJSON* meta_entry = cJSON_GetObjectItem(root, "__opt_meta__");
    if (meta_entry) {
        cJSON* meta_off = cJSON_GetObjectItem(meta_entry, "data_offsets");
        if (meta_off && cJSON_GetArraySize(meta_off) == 2) {
            size_t start = (size_t)cJSON_GetArrayItem(meta_off, 0)->valuedouble;
            double meta[9];
            fseek(f, data_start + (long)start, SEEK_SET);
            if (fread(meta, sizeof(double), 9, f) == 9) {
                optimizer_set_meta(opt, meta);
            }
        }
    }

    /* Load per-param m and v buffers */
    for (int i = 0; i < n; i++) {
        const char* pname = param_name(i);
        int numel = tensor_numel(param_tensor(i));

        char name_m[512], name_v[512];
        snprintf(name_m, sizeof(name_m), "opt_m__%s", pname);
        snprintf(name_v, sizeof(name_v), "opt_v__%s", pname);

        cJSON* m_entry = cJSON_GetObjectItem(root, name_m);
        if (m_entry) {
            cJSON* m_off = cJSON_GetObjectItem(m_entry, "data_offsets");
            if (m_off && cJSON_GetArraySize(m_off) == 2) {
                size_t start = (size_t)cJSON_GetArrayItem(m_off, 0)->valuedouble;
                double* buf = (double*)malloc(numel * sizeof(double));
                if (buf) {
                    fseek(f, data_start + (long)start, SEEK_SET);
                    if (fread(buf, sizeof(double), numel, f) == (size_t)numel) {
                        optimizer_set_m(opt, i, buf);
                    }
                    free(buf);
                }
            }
        }

        cJSON* v_entry = cJSON_GetObjectItem(root, name_v);
        if (v_entry) {
            cJSON* v_off = cJSON_GetObjectItem(v_entry, "data_offsets");
            if (v_off && cJSON_GetArraySize(v_off) == 2) {
                size_t start = (size_t)cJSON_GetArrayItem(v_off, 0)->valuedouble;
                double* buf = (double*)malloc(numel * sizeof(double));
                if (buf) {
                    fseek(f, data_start + (long)start, SEEK_SET);
                    if (fread(buf, sizeof(double), numel, f) == (size_t)numel) {
                        optimizer_set_v(opt, i, buf);
                    }
                    free(buf);
                }
            }
        }
    }

    cJSON_Delete(root);
    fclose(f);
    fprintf(stderr, "optimizer_load: loaded optimizer state from '%s'\n", path);
    return 0;
}
