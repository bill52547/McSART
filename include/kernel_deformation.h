#ifndef _KERNEL_DEFORMATION_H
#define _KERNEL_DEFORMATION_H
__global__ void kernel_deformation(float *img1, cudaTextureObject_t img, float *mx2, float *my2, float *mz2, int nx, int ny, int nz);
#endif // _KERNEL_DEFORMATION_H