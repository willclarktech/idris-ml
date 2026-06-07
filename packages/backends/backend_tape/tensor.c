/* backend_tape/tensor.c — element-size + ABI<->internal dtype-tag
 * translation + lingua-franca rounding helpers.
 *
 * Standalone TU compiled into backend_tape_tensor.o.
 */

#include <stdio.h>
#include <stdlib.h>
#include "tensor.h"
#include "../shared_utils.h" /* bf16/f16 bit-conv helpers */

size_t tape_elem_size(int tag) {
	switch (tag) {
	case DT_F64:
		return sizeof(double);
	case DT_F32:
		return sizeof(float);
	case DT_BF16:
	case DT_F16:
		return sizeof(uint16_t);
	case DT_I8:
	case DT_U8:
	case DT_BOOL:
		return sizeof(int8_t);
	case DT_I16:
		return sizeof(int16_t);
	case DT_I32:
		return sizeof(int32_t);
	case DT_I64:
		return sizeof(int64_t);
	case DT_BINARY:
	case DT_TERNARY:
		return 0; /* sub-byte;
consult tape_packed_bytes() (#411 B3) for buffer sizing. */
	default:
		return sizeof(double);
	}
}

int tape_tag_from_dtag(int dtag) {
	switch (dtag) {
	case 1:
		return DT_BOOL;
	case 4:
		return DT_U8;
	case 8:
		return DT_I8;
	case 9:
		return DT_I16;
	case 10:
		return DT_I32;
	case 11:
		return DT_I64;
	case 13:
		return DT_F16;
	case 14:
		return DT_F32;
	case 15:
		return DT_F64;
	case 17:
		return DT_BF16;
	case 24:
		return DT_BINARY;
	case 25:
		return DT_TERNARY;
	default:
		fprintf(stderr,
		        "[tape backend] invalid dtag=%d (expected one of "
		        "{1=Bool, 4=U8, 8-11=I8/I16/I32/I64, 13-15=F16/F32/F64, 17=BF16, "
		        "24=Binary, 25=Ternary})\n",
		        dtag);
		abort();
	}
}

double tape_round_to_dtype(double v, int tag) {
	switch (tag) {
	case DT_F32:
		return (double)(float)v;
	case DT_BF16:
		return bf16_bits_to_double(double_to_bf16_bits(v));
	case DT_F16:
		return f16_bits_to_double(double_to_f16_bits(v));
	case DT_I8:
		return (double)(signed char)(long long)v;
	case DT_I16:
		return (double)(short)(long long)v;
	case DT_I32:
		return (double)(int)(long long)v;
	case DT_I64:
		return (double)(long long)v;
	case DT_U8:
		return (double)(unsigned char)(long long)v;
	case DT_BOOL:
		return v != 0.0 ? 1.0 : 0.0;
	default:
		return v; /* DT_F64 */
	}
}
