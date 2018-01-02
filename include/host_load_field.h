#ifndef _HOST_LOAD_FIELD_H
#define _HOST_LOAD_FIELD_H
#include "mex.h"
int load_int_field(mxArray const *pm, const char *fieldname);
float load_float_field(mxArray const *pm, const char *fieldname);
void load_int_pointer_field(int *ptr, mxArray const *pm, const char *fieldname);
void load_float_pointer_field(float *ptr, mxArray const *pm, const char *fieldname);
#endif // _HOST_LOAD_FIELD_H
