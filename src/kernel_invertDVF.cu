__global__ void kernel_invertDVF(float *mx2, float *my2, float *mz2, cudaTextureObject_t mx, cudaTextureObject_t my, cudaTextureObject_t mz, int nx, int ny, int nz, int niter);
__host__ void host_invertDVF(float *mx2, float *my2, float *mz2, float *mx, float *my, float *mz, int nx, int ny, int nz, int niter)
{
    cudaChannelFormatDesc channelDesc = cudaCreateChannelDesc(32, 0, 0, 0, cudaChannelFormatKindFloat);
    struct cudaExtent extent_img = make_cudaExtent(nx, ny, nz);

    cudaPitchedPtr dp_mx = make_cudaPitchedPtr((void*) mx, nx * sizeof(float), nx, ny);
    cudaPitchedPtr dp_my = make_cudaPitchedPtr((void*) my, nx * sizeof(float), nx, ny);
    cudaPitchedPtr dp_mz = make_cudaPitchedPtr((void*) mz, nx * sizeof(float), nx, ny);
    cudaMemcpy3DParms copyParams = {0};
    copyParams.extent = extent_img;
    copyParams.kind = cudaMemcpyHostToDevice;
    cudaArray *array_mx, *array_my, *array_mz;

    cudaMalloc3DArray(&array_mx, &channelDesc, extent_img);
    cudaMalloc3DArray(&array_my, &channelDesc, extent_img);
    cudaMalloc3DArray(&array_mz, &channelDesc, extent_img);

    copyParams.srcPtr = dp_mx;
    copyParams.dstArray = array_mx;
    cudaMemcpy3D(&copyParams);  

    copyParams.srcPtr = dp_my;
    copyParams.dstArray = array_my;
    cudaMemcpy3D(&copyParams);  

    copyParams.srcPtr = dp_mz;
    copyParams.dstArray = array_mz;
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
        resDesc.res.array.array = array_mx;
    cudaTextureObject_t tex_mx = 0;
    cudaCreateTextureObject(&tex_mx, &resDesc, &texDesc, NULL);
        resDesc.res.array.array = array_my;
    cudaTextureObject_t tex_my = 0;
    cudaCreateTextureObject(&tex_my, &resDesc, &texDesc, NULL);
        resDesc.res.array.array = array_mz;
    cudaTextureObject_t tex_mz = 0;
    cudaCreateTextureObject(&tex_mz, &resDesc, &texDesc, NULL);

    const dim3 gridSize_img((nx + 16 - 1) / 16, (ny + 16 - 1) / 16, (nz + 4 - 1) / 4);
    const dim3 blockSize(16, 16, 4);
    kernel_invertDVF<<<gridSize_img, blockSize>>>(mx2, my2, mz2, tex_mx, tex_my, tex_mz, nx, ny, nz, niter);
cudaDeviceSynchronize();


cudaDestroyTextureObject(tex_mx);
cudaDestroyTextureObject(tex_my);
cudaDestroyTextureObject(tex_mz);

cudaFreeArray(array_mx);
cudaFreeArray(array_my);
cudaFreeArray(array_mz);

return;
}
__global__ void kernel_invertDVF(float *mx2, float *my2, float *mz2, cudaTextureObject_t mx, cudaTextureObject_t my, cudaTextureObject_t mz, int nx, int ny, int nz, int niter)
{
    int ix = 16 * blockIdx.x + threadIdx.x;
    int iy = 16 * blockIdx.y + threadIdx.y;
    int iz = 4 * blockIdx.z + threadIdx.z;
    if (ix >= nx || iy >= ny || iz >= nz)
        return;
    int id = ix + iy * nx + iz * nx * ny;
    float x = 0, y = 0, z = 0;
    for (int iter = 0; iter < niter; iter ++){
        x = - tex3D<float>(mx, (x + ix + 0.5f), (y + iy + 0.5f), (z + iz + 0.5f));
        y = - tex3D<float>(my, (x + ix + 0.5f), (y + iy + 0.5f), (z + iz + 0.5f));
        z = - tex3D<float>(mz, (x + ix + 0.5f), (y + iy + 0.5f), (z + iz + 0.5f));
    }
    mx2[id] = x;
    my2[id] = y;
    mz2[id] = z;
}