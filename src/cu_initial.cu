#include "cu_initial.h"
__host__ void host_initial(float *img, int nx, int ny, int nz, float value){
    const dim3 gridSize((nx + BLOCKSIZE_X - 1) / BLOCKSIZE_X, (ny + BLOCKSIZE_Y - 1) / BLOCKSIZE_Y, (nz + BLOCKSIZE_Z - 1) / BLOCKSIZE_Z);
    const dim3 blockSize(BLOCKSIZE_X, BLOCKSIZE_Y, BLOCKSIZE_Z);
    kernel_initial<<<gridSize, blockSize>>>(img, nx, ny, nz, value);
    cudaDeviceSynchronize();

}

__global__ void kernel_initial(float *img, int nx, int ny, int nz, float value){
    int ix = BLOCKSIZE_X * blockIdx.x + threadIdx.x;
    int iy = BLOCKSIZE_Y * blockIdx.y + threadIdx.y;
    int iz = BLOCKSIZE_Z * blockIdx.z + threadIdx.z;
    if (ix >= nx || iy >= ny || iz >= nz)
        return;
    img[ix + iy * nx + iz * nx * ny] = value;
}