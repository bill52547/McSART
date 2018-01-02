#ifndef _KERNEL_FORWARDDVF_H
#define _KERNEL_FORWARDDVF_H
__global__ void kernel_forwardDVF(float *mx, float *my, float *mz, cudaTextureObject_t alpha_x, cudaTextureObject_t alpha_y, cudaTextureObject_t alpha_z, cudaTextureObject_t beta_x, cudaTextureObject_t beta_y, cudaTextureObject_t beta_z, float volume, float flow, int nx, int ny, int nz);
__global__ void kernel_forwardDVF(float *mx, float *my, float *mz, float* alpha_x, float* alpha_y, float* alpha_z, float* beta_x, float* beta_y, float* beta_z, float* const_x, float* const_y, float* const_z, float volume, float flow, int isConst, int nx, int ny, int nz);
#endif // _KERNEL_FORWARDDVF_H
