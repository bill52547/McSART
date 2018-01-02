#include "host_load_field.h"

int load_int_field(mxArray const *pm, const char *fieldname)
{
    int n;
    if (mxGetField(pm, 0, fieldname) != NULL)
        n = (int)mxGetScalar(mxGetField(pm, 0, fieldname));
    else
        mexErrMsgIdAndTxt("MATLAB:badInput","Can't found valid volume resolution %s.\n", fieldname);
    return n;
}


float load_float_field(mxArray const *pm, const char *fieldname)
{
    float x;
    if (mxGetField(pm, 0, fieldname) != NULL)
        x = (float)mxGetScalar(mxGetField(pm, 0, fieldname));
    else
        mexErrMsgIdAndTxt("MATLAB:badInput","Can't found valid volume resolution %s.\n", fieldname);
    return x;
}

void load_int_pointer_field(int *ptr, mxArray const *pm, const char *fieldname)
{        
    if (mxGetField(pm, 0, fieldname) != NULL)
        ptr = (int*)mxGetData(mxGetField(pm, 0, fieldname));
    else
        mexErrMsgIdAndTxt("MATLAB:badInput","Can't found valid volume resolution %s.\n", fieldname);
}


void load_float_pointer_field(float *ptr, mxArray const *pm, const char *fieldname)
{        
    if (mxGetField(pm, 0, fieldname) != NULL)
        ptr = (float*)mxGetData(mxGetField(pm, 0, fieldname));
    else
        mexErrMsgIdAndTxt("MATLAB:badInput","Can't found valid volume resolution %s.\n", fieldname);

}
