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
