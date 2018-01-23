#include "cu_deform.h"

__host__ void host_deform(float *d_img1, float *d_img, int nx, int ny, int nz, float volume_diff, float flow_diff, float *alpha_x, float *alpha_y, float *alpha_z, float *beta_x, float *beta_y, float *beta_z)
{
    const dim3 gridSize((nx + BLOCKSIZE_X - 1) / BLOCKSIZE_X, (ny + BLOCKSIZE_Y - 1) / BLOCKSIZE_Y, (nz + BLOCKSIZE_Z - 1) / BLOCKSIZE_Z);
    const dim3 blockSize(BLOCKSIZE_X, BLOCKSIZE_Y, BLOCKSIZE_Z);
    kernel_deform<<<gridSize, blockSize>>>(d_img1, d_img, nx, ny, nz, volume_diff, flow_diff, alpha_x, alpha_y, alpha_z, beta_x, beta_y, beta_z);
    cudaDeviceSynchronize();
}

__global__ void kernel_deform(float *d_img1, float *d_img, int nx, int ny, int nz, float volume_diff, float flow_diff, float *alpha_x, float *alpha_y, float *alpha_z, float *beta_x, float *beta_y, float *beta_z)
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
        dfx = d_img[id + 1] - d_img[id];
    if (iy == ny - 1)
        dfy = 0.0f;
    else
        dfy = d_img[id + nx] - d_img[id];
    if (iz == nz - 1)
        dfz = 0.0f;
    else
        dfz = d_img[id + nx * ny] - d_img[id];
    d_img1[id] = d_img[id] 
    - dfx * (alpha_x[id] * volume_diff + beta_x[id] * flow_diff)
    - dfy * (alpha_y[id] * volume_diff + beta_y[id] * flow_diff)
    - dfz * (alpha_z[id] * volume_diff + beta_z[id] * flow_diff); 
    
}

__device__ float diff_x2(float *img, int ix, int iy, int iz, int nx, int ny, int nz){
    if (ix == nx - 1)
        return 0.0f;
    else
    {
        int id = ix + iy * nx + iz * nx * ny;
        int id1 = id + 1;
        return img[id1] - img[id];
    }
}

__device__ float diff_y2(float *img, int ix, int iy, int iz, int nx, int ny, int nz){
    if (iy == ny - 1)
        return 0.0f;
    else
    {
        int id = ix + iy * nx + iz * nx * ny;
        int id1 = id + nx;
        return img[id1] - img[id];
    }
}

__device__ float diff_z2(float *img, int ix, int iy, int iz, int nx, int ny, int nz){
    if (iz == nz - 1)
        return 0.0f;
    else
    {
        int id = ix + iy * nx + iz * nx * ny;
        int id1 = id + nx * ny;
        return img[id1] - img[id];
    }
}