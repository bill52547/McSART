#ifndef _CU_INITIAL_H
#define _CU_INITIAL_H
#include "universal.h"

__host__ void host_initial(float *img, int nx, int ny, int nz, float value);
__global__ void kernel_initial(float *img, int nx, int ny, int nz, float value);

#endif //CU_INITIAL_H