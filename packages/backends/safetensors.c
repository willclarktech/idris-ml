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
#include "shared_utils.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdint.h>

_Static_assert(__BYTE_ORDER__ == __ORDER_LITTLE_ENDIAN__,
               "SafeTensors requires little-endian platform");

/* Map a SafeTensors dtype name to bytes-per-element. Unknown -> 0. */
static size_t dtype_byte_width(const char* name) {
	if (!name) return 0;
	if (strcmp(name, "F64") == 0) return 8;
	if (strcmp(name, "F32") == 0) return 4;
	if (strcmp(name, "BF16") == 0) return 2;
	if (strcmp(name, "F16") == 0) return 2;
	if (strcmp(name, "I8") == 0) return 1;
	if (strcmp(name, "I16") == 0) return 2;
	if (strcmp(name, "I32") == 0) return 4;
	if (strcmp(name, "I64") == 0) return 8;
	if (strcmp(name, "U8") == 0) return 1;
	if (strcmp(name, "BOOL") == 0) return 1;
	return 0;
}

/* bf16/f16 <-> double helpers live in shared_utils (lifted there for
   Phase 4 — backend_tape.c's tape_round_to_dtype now shares the same
   precision-accurate round-trip). Declarations come from
   "shared_utils.h" included above. */

/* ================================================================
   Save
   ================================================================ */

/* Shared core. When filter_indices is non-NULL, only those n_filter
   indices into the registry are saved (in caller-given order); when
   NULL, every registered param is saved.

   `override_names` is an optional array of length n_filter giving
   the on-disk JSON-header name for each saved tensor (in lockstep
   with filter_indices). NULL = use the registry name verbatim.
   Used by the LoRA/peft adapter-IO path to wrap registry names
   (`bert.[...].lora_A`) in peft's on-disk decorations
   (`base_model.model.bert.[...].lora_A.default.weight`).
*/
static int param_save_core(const char* path, int n_filter, const int* filter_indices,
                           const char* const* override_names) {
	int n_all = param_count();
	int n;
	int* identity = NULL;
	const int* idx;
	if (filter_indices == NULL) {
		n = n_all;
		if (n == 0) {
			fprintf(stderr, "param_save: no parameters registered\n");
			return -1;
		}
		identity = (int*)malloc((size_t)n * sizeof(int));
		if (!identity) return -1;
		for (int k = 0; k < n; k++)
			identity[k] = k;
		idx = identity;
	} else {
		n = n_filter;
		if (n == 0) {
			fprintf(stderr, "param_save_by_name: empty filter list\n");
			return -1;
		}
		idx = filter_indices;
	}

	/* Build JSON header */
	cJSON* root = cJSON_CreateObject();
	if (!root) {
		free(identity);
		return -1;
	}

	/* First pass: compute per-tensor byte size (depends on dtype) +
	   running data offsets. */
	size_t data_offset = 0;
	size_t* offsets = (size_t*)calloc(n, sizeof(size_t));
	size_t* sizes = (size_t*)calloc(n, sizeof(size_t));
	const char** dtypes = (const char**)calloc(n, sizeof(const char*));
	if (!offsets || !sizes || !dtypes) {
		cJSON_Delete(root);
		free(offsets);
		free(sizes);
		free(dtypes);
		free(identity);
		return -1;
	}

	for (int k = 0; k < n; k++) {
		int i = idx[k];
		offsets[k] = data_offset;
		TensorHandle t = param_tensor(i);
		dtypes[k] = tensor_dtype_name(t);
		size_t width = dtype_byte_width(dtypes[k]);
		if (width == 0) {
			fprintf(stderr, "param_save: unsupported dtype '%s' for '%s'\n",
			        dtypes[k] ? dtypes[k] : "(null)", param_name(i));
			cJSON_Delete(root);
			free(offsets);
			free(sizes);
			free(dtypes);
			free(identity);
			return -1;
		}
		sizes[k] = (size_t)tensor_numel(t) * width;
		data_offset += sizes[k];
	}

	/* Build JSON entries */
	for (int k = 0; k < n; k++) {
		int i = idx[k];
		const char* name = override_names ? override_names[k] : param_name(i);
		TensorHandle t = param_tensor(i);
		int rank = tensor_dim(t);

		cJSON* entry = cJSON_CreateObject();
		cJSON_AddStringToObject(entry, "dtype", dtypes[k]);

		cJSON* shape = cJSON_CreateArray();
		for (int d = 0; d < rank; d++) {
			cJSON_AddItemToArray(shape, cJSON_CreateNumber(tensor_size(t, d)));
		}
		cJSON_AddItemToObject(entry, "shape", shape);

		cJSON* data_off = cJSON_CreateArray();
		cJSON_AddItemToArray(data_off, cJSON_CreateNumber((double)offsets[k]));
		cJSON_AddItemToArray(data_off, cJSON_CreateNumber((double)(offsets[k] + sizes[k])));
		cJSON_AddItemToObject(entry, "data_offsets", data_off);

		cJSON_AddItemToObject(root, name, entry);
	}

	char* json_str = cJSON_PrintUnformatted(root);
	cJSON_Delete(root);
	if (!json_str) {
		free(offsets);
		free(sizes);
		free(dtypes);
		free(identity);
		return -1;
	}

	size_t json_len = strlen(json_str);
	/* Pad to 8-byte alignment */
	size_t padded_len = (json_len + 7) & ~(size_t)7;

	/* Write file */
	FILE* f = fopen(path, "wb");
	if (!f) {
		fprintf(stderr, "param_save: cannot open '%s' for writing\n", path);
		free(json_str);
		free(offsets);
		free(sizes);
		free(dtypes);
		free(identity);
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

	/* Tensor data — write in the param's actual dtype width. F32 and I64
	   go directly via their byte-exact extractors so values that don't
	   round-trip through `double` (F32 outside the f32 grid; I64 above
	   2^53) survive the file write. Every other dtype uses the
	   `double` lingua franca — bf16/f16 via the shared bit helpers,
	   the small ints via plain casts (exact in their ranges). */
	for (int k = 0; k < n; k++) {
		int i = idx[k];
		TensorHandle t = param_tensor(i);
		int numel = tensor_numel(t);
		const char* dt = dtypes[k];
		if (strcmp(dt, "F32") == 0) {
			float* buf = (float*)malloc((size_t)numel * sizeof(float));
			if (!buf) {
				fclose(f);
				free(offsets);
				free(sizes);
				free(dtypes);
				free(identity);
				return -1;
			}
			tensor_to_floats(t, buf);
			fwrite(buf, sizeof(float), numel, f);
			free(buf);
			continue;
		}
		if (strcmp(dt, "I64") == 0) {
			/* Byte-exact path — bypasses the double pivot. Honest on
			   torch (native int64 storage); on tape/mlx the extractor
			   still rounds through the lingua-franca buffer, matching
			   the prior behaviour (no regression). */
			int64_t* ibuf = (int64_t*)malloc((size_t)numel * sizeof(int64_t));
			if (!ibuf) {
				fclose(f);
				free(offsets);
				free(sizes);
				free(dtypes);
				free(identity);
				return -1;
			}
			tensor_to_int64(t, ibuf);
			fwrite(ibuf, sizeof(int64_t), numel, f);
			free(ibuf);
			continue;
		}
		double* dbuf = (double*)malloc((size_t)numel * sizeof(double));
		if (!dbuf) {
			fclose(f);
			free(offsets);
			free(sizes);
			free(dtypes);
			free(identity);
			return -1;
		}
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
	free(identity);
	return 0;
}

int param_save(const char* path) {
	return param_save_core(path, 0, NULL, NULL);
}

/* Save only the named subset. `names_nl` is a newline-separated list
   of exact paramId names (no trailing newline required); `count` is
   the number of names. Each name is looked up in the registry by
   linear strcmp; any miss fails with stderr diagnostic.
   The on-disk order matches the order of names in `names_nl` (NOT
   the registry order), so callers can control the layout. */
int param_save_by_name(const char* path, const char* names_nl, int count) {
	if (!names_nl || count <= 0) {
		fprintf(stderr, "param_save_by_name: empty name list (count=%d)\n", count);
		return -1;
	}

	int n_reg = param_count();
	int* indices = (int*)malloc((size_t)count * sizeof(int));
	if (!indices) return -1;

	const char* p = names_nl;
	int parsed = 0;
	while (*p && parsed < count) {
		const char* end = p;
		while (*end && *end != '\n')
			end++;
		size_t len = (size_t)(end - p);

		int found = -1;
		for (int i = 0; i < n_reg; i++) {
			const char* nm = param_name(i);
			if (nm && strlen(nm) == len && memcmp(nm, p, len) == 0) {
				found = i;
				break;
			}
		}
		if (found < 0) {
			fprintf(stderr, "param_save_by_name: '%.*s' not in registry\n", (int)len, p);
			free(indices);
			return -1;
		}
		indices[parsed++] = found;

		p = end;
		if (*p == '\n') p++;
	}

	if (parsed != count) {
		fprintf(stderr, "param_save_by_name: expected %d names, got %d\n", count, parsed);
		free(indices);
		return -1;
	}

	int rc = param_save_core(path, count, indices, NULL);
	free(indices);
	return rc;
}

/* Save the named subset, but write each tensor under an OVERRIDE
   name (in lockstep with names_nl). Used by the LoRA/peft adapter
   export path to wrap registry names like
   `bert.[...].lora_A` in peft's on-disk decorations
   `base_model.model.bert.[...].lora_A.default.weight`. Both
   name-lists are newline-separated and must have exactly `count`
   entries each. Returns 0 on success. */
int param_save_by_name_renamed(const char* path, const char* lookup_names_nl,
                               const char* ondisk_names_nl, int count) {
	if (!lookup_names_nl || !ondisk_names_nl || count <= 0) {
		fprintf(stderr, "param_save_by_name_renamed: invalid arg (count=%d)\n", count);
		return -1;
	}

	int n_reg = param_count();
	int* indices = (int*)malloc((size_t)count * sizeof(int));
	char** overrides = (char**)calloc((size_t)count, sizeof(char*));
	if (!indices || !overrides) {
		free(indices);
		free(overrides);
		return -1;
	}

	/* Parse lookup_names_nl + look up each in registry. */
	const char* p = lookup_names_nl;
	int parsed = 0;
	while (*p && parsed < count) {
		const char* end = p;
		while (*end && *end != '\n')
			end++;
		size_t len = (size_t)(end - p);

		int found = -1;
		for (int i = 0; i < n_reg; i++) {
			const char* nm = param_name(i);
			if (nm && strlen(nm) == len && memcmp(nm, p, len) == 0) {
				found = i;
				break;
			}
		}
		if (found < 0) {
			fprintf(stderr, "param_save_by_name_renamed: '%.*s' not in registry\n", (int)len, p);
			free(indices);
			for (int j = 0; j < parsed; j++)
				free(overrides[j]);
			free(overrides);
			return -1;
		}
		indices[parsed++] = found;
		p = end;
		if (*p == '\n') p++;
	}

	if (parsed != count) {
		fprintf(stderr, "param_save_by_name_renamed: expected %d lookup names, got %d\n", count,
		        parsed);
		free(indices);
		for (int j = 0; j < parsed; j++)
			free(overrides[j]);
		free(overrides);
		return -1;
	}

	/* Parse ondisk_names_nl into the overrides buffer. */
	p = ondisk_names_nl;
	int oi = 0;
	while (*p && oi < count) {
		const char* end = p;
		while (*end && *end != '\n')
			end++;
		size_t len = (size_t)(end - p);
		overrides[oi] = (char*)malloc(len + 1);
		if (!overrides[oi]) {
			free(indices);
			for (int j = 0; j < oi; j++)
				free(overrides[j]);
			free(overrides);
			return -1;
		}
		memcpy(overrides[oi], p, len);
		overrides[oi][len] = '\0';
		oi++;
		p = end;
		if (*p == '\n') p++;
	}

	if (oi != count) {
		fprintf(stderr, "param_save_by_name_renamed: expected %d on-disk names, got %d\n", count,
		        oi);
		free(indices);
		for (int j = 0; j < oi; j++)
			free(overrides[j]);
		free(overrides);
		return -1;
	}

	int rc = param_save_core(path, count, indices, (const char* const*)overrides);

	for (int j = 0; j < count; j++)
		free(overrides[j]);
	free(overrides);
	free(indices);
	return rc;
}

/* ================================================================
   Load
   ================================================================ */

/* Load one file `entry`'s tensor data into registry param `pidx`.
   `name` is the on-disk key (used only for diagnostics). Returns 1 if
   the tensor was loaded, 0 if it was skipped or errored; first-error
   wins via *rc (set-if-zero), so callers accumulate `loaded += ...`
   and the loop continues past a bad entry. Shared by the by-key loop
   (`param_load_core`) and the renamed by-pair loop
   (`param_load_renamed`). */
static int load_entry_into(FILE* f, cJSON* entry, long data_start, int pidx, int allow_cast,
                           const char* name, int* rc) {
	/* Read dtype tag — driver for the rest of the load. */
	cJSON* dtype_node = cJSON_GetObjectItem(entry, "dtype");
	const char* src_dtype =
	    (dtype_node && cJSON_IsString(dtype_node)) ? dtype_node->valuestring : "F64";
	size_t src_width = dtype_byte_width(src_dtype);
	if (src_width == 0) {
		fprintf(stderr, "param_load: unsupported on-disk dtype '%s' for '%s'\n", src_dtype, name);
		if (*rc == 0) *rc = -5;
		return 0;
	}

	TensorHandle t = param_tensor(pidx);
	const char* dst_dtype = tensor_dtype_name(t);
	int dtypes_match = strcmp(src_dtype, dst_dtype) == 0;
	if (!dtypes_match && !allow_cast) {
		fprintf(stderr,
		        "param_load: dtype mismatch for '%s' — on disk %s, destination %s. "
		        "Pass allow_cast=1 to convert at load time.\n",
		        name, src_dtype, dst_dtype);
		if (*rc == 0) *rc = -3;
		return 0;
	}

	/* Read data_offsets */
	cJSON* offsets = cJSON_GetObjectItem(entry, "data_offsets");
	if (!offsets || cJSON_GetArraySize(offsets) != 2) {
		fprintf(stderr, "param_load: bad data_offsets for '%s'\n", name);
		if (*rc == 0) *rc = -2;
		return 0;
	}
	size_t start = (size_t)cJSON_GetArrayItem(offsets, 0)->valuedouble;
	size_t end = (size_t)cJSON_GetArrayItem(offsets, 1)->valuedouble;
	size_t byte_len = end - start;
	int numel = (int)(byte_len / src_width);

	/* Validate element count */
	int expected_numel = tensor_numel(t);
	if (numel != expected_numel) {
		fprintf(stderr, "param_load: size mismatch for '%s': file has %d, registry has %d\n", name,
		        numel, expected_numel);
		if (*rc == 0) *rc = -4;
		return 0;
	}

	/* Read raw bytes, then convert to doubles (the lingua franca of
	   param_load_data — destination dtype conversion happens C-side). */
	void* raw_buf = malloc(byte_len);
	if (!raw_buf) {
		if (*rc == 0) *rc = -6;
		return 0;
	}
	fseek(f, data_start + (long)start, SEEK_SET);
	if (fread(raw_buf, 1, byte_len, f) != byte_len) {
		fprintf(stderr, "param_load: failed to read data for '%s'\n", name);
		free(raw_buf);
		if (*rc == 0) *rc = -6;
		return 0;
	}

	/* Byte-exact I64 path — bypasses the double pivot so the
	   bits read off disk reach the destination tensor without
	   rounding. Only valid when src==dst==I64; an allow_cast=1
	   load that narrows I64 → some other dtype still goes
	   through the double lingua franca below (the destination
	   dtype can't preserve >2^53 anyway). */
	if (dtypes_match && strcmp(src_dtype, "I64") == 0) {
		param_load_data_int64(pidx, (const int64_t*)raw_buf, numel);
		free(raw_buf);
		return 1;
	}

	double* dbuf;
	int owns_dbuf = 0;
	if (strcmp(src_dtype, "F64") == 0) { /* already lingua franca */
		dbuf = (double*)raw_buf;
	} else {
		dbuf = (double*)malloc((size_t)numel * sizeof(double));
		if (!dbuf) {
			free(raw_buf);
			if (*rc == 0) *rc = -6;
			return 0;
		}
		owns_dbuf = 1;
		if (strcmp(src_dtype, "F32") == 0) {
			const float* s = (const float*)raw_buf;
			for (int i = 0; i < numel; i++)
				dbuf[i] = (double)s[i];
		} else if (strcmp(src_dtype, "BF16") == 0) {
			const uint16_t* s = (const uint16_t*)raw_buf;
			for (int i = 0; i < numel; i++)
				dbuf[i] = bf16_bits_to_double(s[i]);
		} else if (strcmp(src_dtype, "F16") == 0) {
			const uint16_t* s = (const uint16_t*)raw_buf;
			for (int i = 0; i < numel; i++)
				dbuf[i] = f16_bits_to_double(s[i]);
		} else if (strcmp(src_dtype, "I8") == 0) {
			const int8_t* s = (const int8_t*)raw_buf;
			for (int i = 0; i < numel; i++)
				dbuf[i] = (double)s[i];
		} else if (strcmp(src_dtype, "I16") == 0) {
			const int16_t* s = (const int16_t*)raw_buf;
			for (int i = 0; i < numel; i++)
				dbuf[i] = (double)s[i];
		} else if (strcmp(src_dtype, "I32") == 0) {
			const int32_t* s = (const int32_t*)raw_buf;
			for (int i = 0; i < numel; i++)
				dbuf[i] = (double)s[i];
		} else if (strcmp(src_dtype, "I64") == 0) {
			const int64_t* s = (const int64_t*)raw_buf;
			for (int i = 0; i < numel; i++)
				dbuf[i] = (double)s[i];
		} else if (strcmp(src_dtype, "U8") == 0) {
			const uint8_t* s = (const uint8_t*)raw_buf;
			for (int i = 0; i < numel; i++)
				dbuf[i] = (double)s[i];
		} else { /* BOOL */
			const uint8_t* s = (const uint8_t*)raw_buf;
			for (int i = 0; i < numel; i++)
				dbuf[i] = (s[i] != 0) ? 1.0 : 0.0;
		}
	}

	param_load_data(pidx, dbuf, numel);
	if (owns_dbuf) {
		free(raw_buf);
		free(dbuf);
	} else {
		free(raw_buf);
	}
	return 1;
}

/* Core loader. `prefix == NULL` loads every key (legacy behaviour);
   `prefix != NULL` loads only safetensors keys whose name starts with
   `prefix` (used by Idris-side `loadModelPrefix` to warm-start a
   pretrained backbone while leaving a fresh head at its init).

   Returns 0 on success, otherwise a typed error code (consumed by the
   Idris-side Checkpoint.load to build a LoadError):
     -1 cannot open file       -2 malformed file (size prelude, JSON
                                  header, or a bad data_offsets entry)
     -3 dtype mismatch (gate)  -4 element-count mismatch
     -5 unsupported on-disk dtype
     -6 data read / alloc failure
   Per-entry errors are first-error-wins (set-if-zero) and the loop
   continues, so one bad entry doesn't abort the rest of the load.
   A file key missing from the registry is a SKIP, not an error —
   the warm-start path depends on that. */
static int param_load_core(const char* path, int allow_cast, const char* prefix) {
	FILE* f = fopen(path, "rb");
	if (!f) {
		fprintf(stderr, "param_load: cannot open '%s'\n", path);
		return -1;
	}
	size_t prefix_len = prefix ? strlen(prefix) : 0;

	/* Read header size */
	uint64_t header_size;
	if (fread(&header_size, sizeof(uint64_t), 1, f) != 1) {
		fprintf(stderr, "param_load: failed to read header size\n");
		fclose(f);
		return -2;
	}

	/* Read JSON header */
	char* json_str = (char*)malloc(header_size + 1);
	if (!json_str) {
		fclose(f);
		return -6;
	}
	if (fread(json_str, 1, header_size, f) != header_size) {
		fprintf(stderr, "param_load: failed to read header\n");
		free(json_str);
		fclose(f);
		return -2;
	}
	json_str[header_size] = '\0';

	/* Data section starts at offset 8 + header_size */
	long data_start = 8 + (long)header_size;

	cJSON* root = cJSON_Parse(json_str);
	free(json_str);
	if (!root) {
		fprintf(stderr, "param_load: failed to parse JSON header\n");
		fclose(f);
		return -2;
	}

	int n = param_count();
	int loaded = 0;
	int rc = 0;

	/* For each tensor in the file, find matching param */
	cJSON* entry = NULL;
	int considered = 0;
	cJSON_ArrayForEach(entry, root) {
		const char* name = entry->string;
		if (!name) continue;

		/* Prefix filter — silently skip non-matching keys when set.
		   Distinct from the registry-miss "skipping" log below: a prefix
		   skip is an intended hit, not a missing param. */
		if (prefix && strncmp(name, prefix, prefix_len) != 0) continue;
		considered++;

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

		loaded += load_entry_into(f, entry, data_start, pidx, allow_cast, name, &rc);
	}

	cJSON_Delete(root);
	fclose(f);

	if (prefix) {
		fprintf(stderr, "param_load: loaded %d/%d (prefix-matched %d) parameters from '%s'%s\n",
		        loaded, n, considered, path, (rc != 0) ? " (with errors — see above)" : "");
	} else {
		fprintf(stderr, "param_load: loaded %d/%d parameters from '%s'%s\n", loaded, n, path,
		        (rc != 0) ? " (with errors — see above)" : "");
	}
	return rc;
}

int param_load_with_policy(const char* path, int allow_cast) {
	return param_load_core(path, allow_cast, /*prefix=*/NULL);
}

int param_load_with_prefix(const char* path, int allow_cast, const char* prefix) {
	return param_load_core(path, allow_cast, prefix);
}

int param_load(const char* path) {
	return param_load_core(path, /*allow_cast=*/0, /*prefix=*/NULL);
}

/* Renamed loader — the symmetric inverse of param_save_by_name_renamed.
   For each (registry_name, ondisk_name) pair the tensor stored on disk
   under `ondisk_name` is loaded into the registry param `registry_name`.
   Used by Idris-side `Checkpoint.load` with a `remap` set, e.g. to read
   a peft-saved adapter whose JSON keys are
   `base_model.model.[...].lora_A.default.weight` into registry params
   named `[...].lora_A`. Both name lists are newline-joined, `count`
   long, in lockstep (built by `Checkpoint.collectRenamedNames`).

   A pair whose `ondisk_name` is absent from the file is SKIPPED (not an
   error) — same warm-start semantics as the by-key loader's
   registry-miss skip. Per-entry errors are first-error-wins; return
   codes match param_load_core. */
int param_load_renamed(const char* path, int allow_cast, const char* registry_names_nl,
                       const char* ondisk_names_nl, int count) {
	if (!registry_names_nl || !ondisk_names_nl || count <= 0) {
		fprintf(stderr, "param_load_renamed: invalid arg (count=%d)\n", count);
		return -1;
	}

	int n_reg = param_count();
	int* indices = (int*)malloc((size_t)count * sizeof(int));
	char** ondisks = (char**)calloc((size_t)count, sizeof(char*));
	if (!indices || !ondisks) {
		free(indices);
		free(ondisks);
		return -1;
	}

	/* Parse registry_names_nl + resolve each to a registry index. The
	   names come from an Idris-side registry walk, so a miss is an
	   internal inconsistency — error out rather than silently skip. */
	const char* p = registry_names_nl;
	int parsed = 0;
	int rc = 0;
	while (*p && parsed < count) {
		const char* end = p;
		while (*end && *end != '\n')
			end++;
		size_t len = (size_t)(end - p);

		int found = -1;
		for (int i = 0; i < n_reg; i++) {
			const char* nm = param_name(i);
			if (nm && strlen(nm) == len && memcmp(nm, p, len) == 0) {
				found = i;
				break;
			}
		}
		if (found < 0) {
			fprintf(stderr, "param_load_renamed: '%.*s' not in registry\n", (int)len, p);
			free(indices);
			for (int j = 0; j < parsed; j++)
				free(ondisks[j]);
			free(ondisks);
			return -1;
		}
		indices[parsed++] = found;
		p = end;
		if (*p == '\n') p++;
	}
	if (parsed != count) {
		fprintf(stderr, "param_load_renamed: expected %d registry names, got %d\n", count, parsed);
		free(indices);
		free(ondisks);
		return -1;
	}

	/* Parse ondisk_names_nl into the lookup-key buffer. */
	p = ondisk_names_nl;
	int oi = 0;
	while (*p && oi < count) {
		const char* end = p;
		while (*end && *end != '\n')
			end++;
		size_t len = (size_t)(end - p);
		ondisks[oi] = (char*)malloc(len + 1);
		if (!ondisks[oi]) {
			free(indices);
			for (int j = 0; j < oi; j++)
				free(ondisks[j]);
			free(ondisks);
			return -1;
		}
		memcpy(ondisks[oi], p, len);
		ondisks[oi][len] = '\0';
		oi++;
		p = end;
		if (*p == '\n') p++;
	}
	if (oi != count) {
		fprintf(stderr, "param_load_renamed: expected %d on-disk names, got %d\n", count, oi);
		free(indices);
		for (int j = 0; j < oi; j++)
			free(ondisks[j]);
		free(ondisks);
		return -1;
	}

	/* Open + parse the safetensors header (mirrors param_load_core). */
	FILE* f = fopen(path, "rb");
	if (!f) {
		fprintf(stderr, "param_load_renamed: cannot open '%s'\n", path);
		for (int j = 0; j < count; j++)
			free(ondisks[j]);
		free(ondisks);
		free(indices);
		return -1;
	}

	uint64_t header_size;
	if (fread(&header_size, sizeof(uint64_t), 1, f) != 1) {
		fprintf(stderr, "param_load_renamed: failed to read header size\n");
		fclose(f);
		for (int j = 0; j < count; j++)
			free(ondisks[j]);
		free(ondisks);
		free(indices);
		return -2;
	}
	char* json_str = (char*)malloc(header_size + 1);
	if (!json_str) {
		fclose(f);
		for (int j = 0; j < count; j++)
			free(ondisks[j]);
		free(ondisks);
		free(indices);
		return -6;
	}
	if (fread(json_str, 1, header_size, f) != header_size) {
		fprintf(stderr, "param_load_renamed: failed to read header\n");
		free(json_str);
		fclose(f);
		for (int j = 0; j < count; j++)
			free(ondisks[j]);
		free(ondisks);
		free(indices);
		return -2;
	}
	json_str[header_size] = '\0';
	long data_start = 8 + (long)header_size;

	cJSON* root = cJSON_Parse(json_str);
	free(json_str);
	if (!root) {
		fprintf(stderr, "param_load_renamed: failed to parse JSON header\n");
		fclose(f);
		for (int j = 0; j < count; j++)
			free(ondisks[j]);
		free(ondisks);
		free(indices);
		return -2;
	}

	int loaded = 0;
	for (int i = 0; i < count; i++) {
		cJSON* entry = cJSON_GetObjectItem(root, ondisks[i]);
		if (!entry) {
			fprintf(stderr, "param_load_renamed: skipping '%s' (not in file)\n", ondisks[i]);
			continue;
		}
		loaded += load_entry_into(f, entry, data_start, indices[i], allow_cast, ondisks[i], &rc);
	}

	cJSON_Delete(root);
	fclose(f);
	for (int j = 0; j < count; j++)
		free(ondisks[j]);
	free(ondisks);
	free(indices);

	fprintf(stderr, "param_load: loaded %d/%d (remapped) parameters from '%s'%s\n", loaded, count,
	        path, (rc != 0) ? " (with errors — see above)" : "");
	return rc;
}

/* Reads the raw on-disk bytes of a named tensor from a safetensors
   file without dtype interpretation. See backend.h for the contract. */
int64_t safetensors_read_raw_bytes(const char* path, const char* tensor_name, uint8_t* out_buf,
                                   size_t out_cap) {
	if (!path || !tensor_name || !out_buf) return -1;

	FILE* f = fopen(path, "rb");
	if (!f) return -1;

	uint64_t header_size;
	if (fread(&header_size, sizeof(uint64_t), 1, f) != 1) {
		fclose(f);
		return -1;
	}

	char* json_str = (char*)malloc(header_size + 1);
	if (!json_str) {
		fclose(f);
		return -1;
	}
	if (fread(json_str, 1, header_size, f) != header_size) {
		free(json_str);
		fclose(f);
		return -1;
	}
	json_str[header_size] = '\0';

	long data_start = 8 + (long)header_size;

	cJSON* root = cJSON_Parse(json_str);
	free(json_str);
	if (!root) {
		fclose(f);
		return -1;
	}

	cJSON* entry = cJSON_GetObjectItem(root, tensor_name);
	if (!entry) {
		cJSON_Delete(root);
		fclose(f);
		return -1;
	}

	cJSON* offsets = cJSON_GetObjectItem(entry, "data_offsets");
	if (!offsets || cJSON_GetArraySize(offsets) != 2) {
		cJSON_Delete(root);
		fclose(f);
		return -1;
	}
	size_t start = (size_t)cJSON_GetArrayItem(offsets, 0)->valuedouble;
	size_t end = (size_t)cJSON_GetArrayItem(offsets, 1)->valuedouble;
	if (end < start) {
		cJSON_Delete(root);
		fclose(f);
		return -1;
	}
	size_t byte_len = end - start;
	cJSON_Delete(root);

	if (byte_len > out_cap) {
		fclose(f);
		return -1;
	}

	if (fseek(f, data_start + (long)start, SEEK_SET) != 0) {
		fclose(f);
		return -1;
	}
	if (fread(out_buf, 1, byte_len, f) != byte_len) {
		fclose(f);
		return -1;
	}
	fclose(f);
	return (int64_t)byte_len;
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
	int total_tensors = 1 + 2 * n; /* meta + m + v */
	size_t* offsets = (size_t*)calloc(total_tensors, sizeof(size_t));
	size_t* byte_sizes = (size_t*)calloc(total_tensors, sizeof(size_t));
	double** bufs = (double**)calloc(total_tensors, sizeof(double*));
	if (!offsets || !byte_sizes || !bufs) {
		cJSON_Delete(root);
		free(offsets);
		free(byte_sizes);
		free(bufs);
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
		cJSON_AddItemToArray(d_off_m,
		                     cJSON_CreateNumber((double)(offsets[tidx] + byte_sizes[tidx])));
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
		cJSON_AddItemToArray(d_off_v,
		                     cJSON_CreateNumber((double)(offsets[tidx] + byte_sizes[tidx])));
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
		for (size_t p = json_len; p < padded_len; p++)
			fputc(' ', f);
		free(json_str);

		for (int i = 0; i < tidx; i++) {
			fwrite(bufs[i], 1, byte_sizes[i], f);
		}
		fclose(f);
	}

	for (int i = 0; i < tidx; i++)
		free(bufs[i]);
	free(offsets);
	free(byte_sizes);
	free(bufs);
	return 0;

cleanup_fail:
	for (int i = 0; i < tidx; i++)
		free(bufs[i]);
	free(offsets);
	free(byte_sizes);
	free(bufs);
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
		fclose(f);
		return -1;
	}

	char* json_str = (char*)malloc(header_size + 1);
	if (!json_str) {
		fclose(f);
		return -1;
	}
	if (fread(json_str, 1, header_size, f) != header_size) {
		free(json_str);
		fclose(f);
		return -1;
	}
	json_str[header_size] = '\0';

	long data_start = 8 + (long)header_size;

	cJSON* root = cJSON_Parse(json_str);
	free(json_str);
	if (!root) {
		fclose(f);
		return -1;
	}

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
