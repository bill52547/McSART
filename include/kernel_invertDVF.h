#ifndef _KERNEL_INVERTDVF_H
#define _KERNEL_INVERTDVF_H
__global__ void kernel_invertDVF(float *mx2, float *my2, float *mz2, cudaTextureObject_t mx, cudaTextureObject_t my, cudaTextureObject_t mz, int nx, int ny, int nz, int niter);
__host__ void host_invertDVF(float *mx2, float *my2, float *mz2, float *mx, float *my, float *mz, int nx, int ny, int nz, int niter);
#endif // _KERNEL_INVERTDVF_H