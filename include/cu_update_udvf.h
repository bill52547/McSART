#ifndef _CU_UPDATE_UDVF_H
#define _CU_UPDATE_UDVF_H
#include "universal.h"
// #include "cu_diff.h"
// __device__ float diff_x(float *img, int ix, int iy, int iz, int nx, int ny, int nz);
// __device__ float diff_y(float *img, int ix, int iy, int iz, int nx, int ny, int nz);
// __device__ float diff_z(float *img, int ix, int iy, int iz, int nx, int ny, int nz);
__host__ void host_update_udvf(float *alpha_x, float* alpha_y, float *alpha_z, float *beta_x, float *beta_y, float *beta_z, float *img, float *img0, float volume_diff, float flow_diff, int nx, int ny, int nz, int iter);
__global__ void kernel_update_udvf(float *alpha_x, float* alpha_y, float *alpha_z, float *beta_x, float *beta_y, float *beta_z, float *img, float *img0, float volume_diff, float flow_diff, int nx, int ny, int nz, int iter);

#endif // _CU_UPDATE_UDVF_H