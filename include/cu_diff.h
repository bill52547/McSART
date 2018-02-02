#ifndef _CU_DIFF_H
#define _CU_DIFF_H
#include "universal.h"
__device__ float diff_x(float *img, int ix, int iy, int iz, int nx, int ny, int nz);
__device__ float diff_y(float *img, int ix, int iy, int iz, int nx, int ny, int nz);
__device__ float diff_z(float *img, int ix, int iy, int iz, int nx, int ny, int nz);
#endif // _CU_DIFF_H