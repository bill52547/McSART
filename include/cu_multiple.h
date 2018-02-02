#ifndef _CU_MULTIPLE_H
#define _CU_MULTIPLE_H
#include "universal.h"
__host__ void host_multiple(float *img1, int nx, int ny, int nz, float *img0, float weight, int ind);
__global__ void kernel_multiple(float *img1, int nx, int ny, int nz, float *img0, float weight, int ind);
#endif // #ifndef _CU_MULTIPLE_H