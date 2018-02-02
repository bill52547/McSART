#include "cu_multiple.h"
__host__ void host_multiple(float *img1, int nx, int ny, int nz, float *img0, float weight, int ind)
{
    const dim3 gridSize((nx + BLOCKSIZE_X - 1) / BLOCKSIZE_X, (ny + BLOCKSIZE_Y - 1) / BLOCKSIZE_Y, (nz + BLOCKSIZE_Z - 1) / BLOCKSIZE_Z);
    const dim3 blockSize(BLOCKSIZE_X, BLOCKSIZE_Y, BLOCKSIZE_Z);
    kernel_multiple<<<gridSize, blockSize>>>(img1, nx, ny, nz, img0, weight, ind);
    cudaDeviceSynchronize();
}


__global__ void kernel_multiple(float *img1, int nx, int ny, int nz, float *img0, float weight, int ind)
{
    int ix = BLOCKSIZE_X * blockIdx.x + threadIdx.x;
    int iy = BLOCKSIZE_Y * blockIdx.y + threadIdx.y;
    int iz = BLOCKSIZE_Z * blockIdx.z + threadIdx.z;
    if (ix >= nx || iy >= ny || iz >= nz)
        return;
    int id = ix + iy * nx + iz * nx * ny;

    float df;
    switch (ind)
    {
        case 1:
            if (ix == nx - 1)
                df = 0.0f;
            else
                df = img0[id + 1] - img0[id];
            break;
        case 2:
            if (iy == ny - 1)
                df = 0.0f;
            else
                df = img0[id + nx] - img0[id];    
            break;
        case 3:
            if (iz == nz - 1)
                df = 0.0f;
            else
                df = img0[id + nx * ny] - img0[id];
            break;
    }
    img1[id] *= weight * df;
}
