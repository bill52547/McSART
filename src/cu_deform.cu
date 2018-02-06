#include "cu_deform.h"

__host__ void host_deform(float *d_img1, float *d_img, int nx, int ny, int nz, float volume, float flow, float *alpha_x, float *alpha_y, float *alpha_z, float *beta_x, float *beta_y, float *beta_z)
{
    const dim3 gridSize((nx + BLOCKSIZE_X - 1) / BLOCKSIZE_X, (ny + BLOCKSIZE_Y - 1) / BLOCKSIZE_Y, (nz + BLOCKSIZE_Z - 1) / BLOCKSIZE_Z);
    const dim3 blockSize(BLOCKSIZE_X, BLOCKSIZE_Y, BLOCKSIZE_Z);
    float *mx, *my, *mz;
    cudaMalloc((void**)&mx, nx * ny * nz * sizeof(float));
    cudaMalloc((void**)&my, nx * ny * nz * sizeof(float));
    cudaMalloc((void**)&mz, nx * ny * nz * sizeof(float));
    kernel_forwardDVF<<<gridSize, blockSize>>>(mx, my, mz, alpha_x, alpha_y, alpha_z, beta_x, beta_y, beta_z, volume, flow, nx, ny, nz);
    cudaDeviceSynchronize();
    cudaChannelFormatDesc channelDesc = cudaCreateChannelDesc(32, 0, 0, 0, cudaChannelFormatKindFloat);
    cudaPitchedPtr dp_img = make_cudaPitchedPtr((void*) d_img, nx * sizeof(float), nx, ny);
    cudaMemcpy3DParms copyParams = {0};
    struct cudaExtent extent_img = make_cudaExtent(nx, ny, nz);
    copyParams.extent = extent_img;
    copyParams.kind = cudaMemcpyDeviceToDevice;
    copyParams.srcPtr = dp_img;
    cudaArray *array_img;
    cudaMalloc3DArray(&array_img, &channelDesc, extent_img);
    copyParams.dstArray = array_img;
    cudaMemcpy3D(&copyParams);   

    cudaResourceDesc resDesc;
    cudaTextureDesc texDesc;
    memset(&resDesc, 0, sizeof(resDesc));
    resDesc.resType = cudaResourceTypeArray;
    memset(&texDesc, 0, sizeof(texDesc));
    texDesc.addressMode[0] = cudaAddressModeClamp;
    texDesc.addressMode[1] = cudaAddressModeClamp;
    texDesc.addressMode[2] = cudaAddressModeClamp;
    texDesc.filterMode = cudaFilterModeLinear;
    texDesc.readMode = cudaReadModeElementType;
    texDesc.normalizedCoords = 0;
    resDesc.res.array.array = array_img;
    cudaTextureObject_t tex_img = 0;
    cudaCreateTextureObject(&tex_img, &resDesc, &texDesc, NULL);
    kernel_deformation<<<gridSize, blockSize>>>(d_img1, tex_img, mx, my, mz, nx, ny, nz);
    cudaDeviceSynchronize();
    cudaFree(mx);   
    cudaFree(my);   
    cudaFree(mz);   
    cudaDestroyTextureObject(tex_img);
    cudaFreeArray(array_img);

}

__host__ void host_deform2(float *d_img1, float *d_img, int nx, int ny, int nz, float volume, float flow, float *alpha_x, float *alpha_y, float *alpha_z, float *beta_x, float *beta_y, float *beta_z)
{
    const dim3 gridSize((nx + BLOCKSIZE_X - 1) / BLOCKSIZE_X, (ny + BLOCKSIZE_Y - 1) / BLOCKSIZE_Y, (nz + BLOCKSIZE_Z - 1) / BLOCKSIZE_Z);
    const dim3 blockSize(BLOCKSIZE_X, BLOCKSIZE_Y, BLOCKSIZE_Z);
    float *mx, *my, *mz;
    cudaMalloc((void**)&mx, nx * ny * nz * sizeof(float));
    cudaMalloc((void**)&my, nx * ny * nz * sizeof(float));
    cudaMalloc((void**)&mz, nx * ny * nz * sizeof(float));
    kernel_forwardDVF<<<gridSize, blockSize>>>(mx, my, mz, alpha_x, alpha_y, alpha_z, beta_x, beta_y, beta_z, volume, flow, nx, ny, nz);
    cudaDeviceSynchronize();
    kernel_deformation2<<<gridSize, blockSize>>>(d_img1, d_img, mx, my, mz, nx, ny, nz);
    cudaDeviceSynchronize();
    cudaFree(mx);
    cudaFree(my);
    cudaFree(mz);
}

__global__ void kernel_forwardDVF(float *mx, float *my, float *mz, float *alpha_x, float *alpha_y, float *alpha_z, float *beta_x, float *beta_y, float *beta_z, float volume, float flow, int nx, int ny, int nz)
{
    int ix = BLOCKSIZE_X * blockIdx.x + threadIdx.x;
    int iy = BLOCKSIZE_Y * blockIdx.y + threadIdx.y;
    int iz = BLOCKSIZE_Z * blockIdx.z + threadIdx.z;
    if (ix >= nx || iy >= ny || iz >= nz)
        return;
    int id = ix + iy * nx + iz * nx * ny;    
    mx[id] = alpha_x[id] * volume + beta_x[id] * flow;
    my[id] = alpha_y[id] * volume + beta_y[id] * flow;
    mz[id] = alpha_z[id] * volume + beta_z[id] * flow;
}

__global__ void kernel_deformation(float *img1, cudaTextureObject_t tex_img, float *mx, float *my, float *mz, int nx, int ny, int nz){
    int ix = BLOCKSIZE_X * blockIdx.x + threadIdx.x;
    int iy = BLOCKSIZE_Y * blockIdx.y + threadIdx.y;
    int iz = BLOCKSIZE_Z * blockIdx.z + threadIdx.z;
    if (ix >= nx || iy >= ny || iz >= nz)
        return;
    int id = iy + ix * ny + iz * nx * ny;
    float xi = iy + 1.0f + my[id];
    float yi = ix + 1.0f + mx[id];
    float zi = iz + 1.0f + mz[id];
    img1[id] = tex3D<float>(tex_img, xi - 0.5f, yi - 0.5f, zi - 0.5f);
}

__global__ void kernel_deformation2(float *img1, float *img, float *mx, float *my, float *mz, int nx, int ny, int nz){
    int ix = BLOCKSIZE_X * blockIdx.x + threadIdx.x;
    int iy = BLOCKSIZE_Y * blockIdx.y + threadIdx.y;
    int iz = BLOCKSIZE_Z * blockIdx.z + threadIdx.z;
    if (ix >= nx || iy >= ny || iz >= nz)
        return;
    int id = iy + ix * ny + iz * nx * ny;
    float dx, dy, dz;
    if (ix == nx - 1)
        dx = 0;
    else
        dx = img[id + 1] - img[id];
    if (iy == ny - 1)
        dy = 0;
    else
        dy = img[id + nx] - img[id];
    if (iz == nz - 1)
        dz = 0;
    else
        dz = img[id + nx * ny] - img[id];
    img1[id] = img[id] + dy * mx[id] + dx * my[id] + dz * mz[id];
}