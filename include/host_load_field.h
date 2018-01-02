#ifndef _HOST_LOAD_FIELD_H
#define _HOST_LOAD_FIELD_H
<<<<<<< HEAD
#include "universal.h"
=======
#include "mex.h"
>>>>>>> 7f9f4366d3c0653b395c1a7c8b31341946826fbd
int load_int_field(mxArray const *pm, const char *fieldname);
float load_float_field(mxArray const *pm, const char *fieldname);
void load_int_pointer_field(int *ptr, mxArray const *pm, const char *fieldname);
void load_float_pointer_field(float *ptr, mxArray const *pm, const char *fieldname);
#endif // _HOST_LOAD_FIELD_H
