/*
 * SafeTensors serialization for idris-ml.
 * Backend-agnostic: uses only backend.h public API.
 *
 * Format: [8-byte LE u64 header_size][JSON header][tensor data]
 * All tensors stored as F64 (double), row-major.
 */

#include "backend.h"
#include "cJSON.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdint.h>

_Static_assert(__BYTE_ORDER__ == __ORDER_LITTLE_ENDIAN__,
               "SafeTensors requires little-endian platform");

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

    /* First pass: compute data offsets */
    size_t data_offset = 0;
    size_t* offsets = (size_t*)calloc(n, sizeof(size_t));
    size_t* sizes = (size_t*)calloc(n, sizeof(size_t));
    if (!offsets || !sizes) { cJSON_Delete(root); free(offsets); free(sizes); return -1; }

    for (int i = 0; i < n; i++) {
        offsets[i] = data_offset;
        TensorHandle t = param_tensor(i);
        sizes[i] = (size_t)tensor_numel(t) * sizeof(double);
        data_offset += sizes[i];
    }

    /* Build JSON entries */
    for (int i = 0; i < n; i++) {
        const char* name = param_name(i);
        TensorHandle t = param_tensor(i);
        int rank = tensor_dim(t);

        cJSON* entry = cJSON_CreateObject();
        cJSON_AddStringToObject(entry, "dtype", "F64");

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
    if (!json_str) { free(offsets); free(sizes); return -1; }

    size_t json_len = strlen(json_str);
    /* Pad to 8-byte alignment */
    size_t padded_len = (json_len + 7) & ~(size_t)7;

    /* Write file */
    FILE* f = fopen(path, "wb");
    if (!f) {
        fprintf(stderr, "param_save: cannot open '%s' for writing\n", path);
        free(json_str); free(offsets); free(sizes);
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

    /* Tensor data */
    for (int i = 0; i < n; i++) {
        TensorHandle t = param_tensor(i);
        int numel = tensor_numel(t);
        double* buf = (double*)malloc(numel * sizeof(double));
        if (!buf) { fclose(f); free(offsets); free(sizes); return -1; }
        tensor_to_doubles(t, buf);
        fwrite(buf, sizeof(double), numel, f);
        free(buf);
    }

    fclose(f);
    free(offsets);
    free(sizes);
    return 0;
}

/* ================================================================
   Load
   ================================================================ */

int param_load(const char* path) {
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

        /* Read data_offsets */
        cJSON* offsets = cJSON_GetObjectItem(entry, "data_offsets");
        if (!offsets || cJSON_GetArraySize(offsets) != 2) {
            fprintf(stderr, "param_load: bad data_offsets for '%s'\n", name);
            continue;
        }
        size_t start = (size_t)cJSON_GetArrayItem(offsets, 0)->valuedouble;
        size_t end = (size_t)cJSON_GetArrayItem(offsets, 1)->valuedouble;
        size_t byte_len = end - start;
        int numel = (int)(byte_len / sizeof(double));

        /* Validate shape */
        TensorHandle t = param_tensor(pidx);
        int expected_numel = tensor_numel(t);
        if (numel != expected_numel) {
            fprintf(stderr, "param_load: size mismatch for '%s': file has %d, registry has %d\n",
                    name, numel, expected_numel);
            continue;
        }

        /* Read tensor data from file */
        double* buf = (double*)malloc(byte_len);
        if (!buf) continue;
        fseek(f, data_start + (long)start, SEEK_SET);
        if (fread(buf, 1, byte_len, f) != byte_len) {
            fprintf(stderr, "param_load: failed to read data for '%s'\n", name);
            free(buf);
            continue;
        }

        /* Load into param */
        param_load_data(pidx, buf, numel);
        free(buf);
        loaded++;
    }

    cJSON_Delete(root);
    fclose(f);

    fprintf(stderr, "param_load: loaded %d/%d parameters from '%s'\n", loaded, n, path);
    return 0;
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
