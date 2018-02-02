#ifndef _CU_DEFORM_H
#define _CU_DEFORM_H
#include "universal.h"
// #include "cu_diff.h"
// __device__ float diff_x2(float *img, int ix, int iy, int iz, int nx, int ny, int nz);
// __device__ float diff_y2(float *img, int ix, int iy, int iz, int nx, int ny, int nz);
// __device__ float diff_z2(float *img, int ix, int iy, int iz, int nx, int ny, int nz);
__host__ void host_deform(float *d_img1, float *d_img, int nx, int ny, int nz, float volume_diff, float flow_diff, float *alpha_x, float *alpha_y, float *alpha_z, float *beta_x, float *beta_y, float *beta_z);
__global__ void kernel_forwardDVF(float *mx, float *my, float *mz, float *alpha_x, float *alpha_y, float *alpha_z, float *beta_x, float *beta_y, float *beta_z, float volume, float flow, int nx, int ny, int nz);
__global__ void kernel_deformation(float *img1, cudaTextureObject_t tex_img, float *mx2, float *my2, float *mz2, int nx, int ny, int nz);
#endif // _CU_DEFORM_H