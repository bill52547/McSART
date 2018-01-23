#include "cu_division.h"
__host__ void host_division(float *img1, float *img, int nx, int ny, int nz)
{
    const dim3 gridSize((nx + BLOCKSIZE_X - 1) / BLOCKSIZE_X, (ny + BLOCKSIZE_Y - 1) / BLOCKSIZE_Y, (nz + BLOCKSIZE_Z - 1) / BLOCKSIZE_Z);
    const dim3 blockSize(BLOCKSIZE_X, BLOCKSIZE_Y, BLOCKSIZE_Z);
    kernel_division<<<gridSize, blockSize>>>(img1, img, nx, ny, nz);
    cudaDeviceSynchronize();
}


__global__ void kernel_division(float *img1, float *img, int nx, int ny, int nz)
{
    int ix = BLOCKSIZE_X * blockIdx.x + threadIdx.x;
    int iy = BLOCKSIZE_Y * blockIdx.y + threadIdx.y;
    int iz = BLOCKSIZE_Z * blockIdx.z + threadIdx.z;
    if (ix >= nx || iy >= ny || iz >= nz)
        return;
    int id = ix + iy * nx + iz * nx * ny;

    img1[id] /= img[id];
    if (isnan(img1[id]))
        img1[id] = 0.0f;
    if (isinf(img1[id]))
        img1[id] = 0.0f;
}
