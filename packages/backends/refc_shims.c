/* Shims for RefC runtime functions missing from Idris 0.8.0.
 * These exist in Idris2 main branch but are absent from 0.8.0's
 * libidris2_refc.a. Remove once Idris 2 updates.
 *
 * Uses the exact struct layouts from _datatypes.h (Idris 0.8.0).
 */

#include <stdlib.h>
#include <string.h>
#include <gmp.h>

/* Match _datatypes.h exactly */
typedef struct {
	int tag;
	int refCounter;
} Value_header;
typedef struct {
	Value_header header;
	double d;
} Value_Double;
typedef struct {
	Value_header header;
	mpz_t i;
} Value_Integer;
typedef struct {
	Value_header header;
	char* str;
} Value_String;

#define DOUBLE_TAG 10
#define INTEGER_TAG 9

extern void* idris2_newValue(size_t size);

void* idris2_negate_Double(void* x) {
	Value_Double* v = (Value_Double*)idris2_newValue(sizeof(Value_Double));
	v->header.tag = DOUBLE_TAG;
	v->d = -(((Value_Double*)x)->d);
	return v;
}

void* idris2_cast_string_to_Double(void* x) {
	Value_Double* v = (Value_Double*)idris2_newValue(sizeof(Value_Double));
	v->header.tag = DOUBLE_TAG;
	v->d = atof(((Value_String*)x)->str);
	return v;
}

void* idris2_cast_string_to_Integer(void* x) {
	Value_Integer* v = (Value_Integer*)idris2_newValue(sizeof(Value_Integer));
	v->header.tag = INTEGER_TAG;
	mpz_init(v->i);
	if (mpz_set_str(v->i, ((Value_String*)x)->str, 10) != 0) mpz_set_si(v->i, 0);
	return v;
}
