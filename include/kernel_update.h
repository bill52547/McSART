#ifndef _KERNEL_UPDATE_H
#define KERNEL_UPDATE_H
__global__ void kernel_update(float *img1, float *img, int nx, int ny, int nz, float lambda);
#endif // KERNEL_UPDATE_H