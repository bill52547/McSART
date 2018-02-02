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


__host__ void host_initial2(float *img, int nx, int ny, int nz, float *img0, float volume, float flow)
{
    const dim3 gridSize((nx + BLOCKSIZE_X - 1) / BLOCKSIZE_X, (ny + BLOCKSIZE_Y - 1) / BLOCKSIZE_Y, (nz + BLOCKSIZE_Z - 1) / BLOCKSIZE_Z);
    const dim3 blockSize(BLOCKSIZE_X, BLOCKSIZE_Y, BLOCKSIZE_Z);
    kernel_initial2<<<gridSize, blockSize>>>(img, nx, ny, nz, img0, volume, flow);
    cudaDeviceSynchronize();
}
__global__ void kernel_initial2(float *img, int nx, int ny, int nz, float *img0, float volume, float flow)
{
    int ix = BLOCKSIZE_X * blockIdx.x + threadIdx.x;
    int iy = BLOCKSIZE_Y * blockIdx.y + threadIdx.y;
    int iz = BLOCKSIZE_Z * blockIdx.z + threadIdx.z;
    if (ix >= nx || iy >= ny || iz >= nz)
        return;
    int id = ix + iy * nx + iz * nx * ny;
    float dfx, dfy, dfz;
    if (ix == nx - 1)
        dfx = 0.0f;
    else
        dfx = img0[id + 1] - img0[id];
    if (iy == ny - 1)
        dfy = 0.0f;
    else
        dfy = img0[id + nx] - img0[id];
    if (iz == nz - 1)
        dfz = 0.0f;
    else
        dfz = img0[id + nx * ny] - img0[id];
    img[ix + iy * nx + iz * nx * ny] = (dfx + dfy + dfz) * (volume + flow);
}