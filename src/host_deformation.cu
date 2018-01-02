#include "host_deformation.h"
void host_deformation(float *img1, float *img, float *mx, float *my, float *mz, int nx, int ny, int nz)
{
    cudaTextureObject_t tex_img = 0;
    cudaChannelFormatDesc channelDesc = cudaCreateChannelDesc(32, 0, 0, 0, cudaChannelFormatKindFloat);
    cudaError_t cudaStat;

    // create texture object alpha and beta
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

    struct cudaExtent extent_img = make_cudaExtent(nx, ny, nz);
    cudaMemcpy3DParms copyParams = {0};
    copyParams.extent = extent_img;
    copyParams.kind = cudaMemcpyDeviceToDevice;
    cudaArray *array_img;
    cudaStat = cudaMalloc3DArray(&array_img, &channelDesc, extent_img);

    cudaPitchedPtr dp_img = make_cudaPitchedPtr((void*) img, nx * sizeof(float), nx, ny);
    copyParams.srcPtr = dp_img;
    copyParams.dstArray = array_img;
    cudaStat = cudaMemcpy3D(&copyParams);   
    if (cudaStat != cudaSuccess) {
        mexPrintf("Failed to copy dp_img to array memory array_img.\n");
        mexPrintf("Error code %d: %s\n",cudaStat,cudaGetErrorString(cudaStat));
            mexErrMsgIdAndTxt("MATLAB:cudaFail","SART failed.\n");
    }
    resDesc.res.array.array = array_img;
    cudaCreateTextureObject(&tex_img, &resDesc, &texDesc, NULL);
    const dim3 gridSize_img((nx + 16 - 1) / 16, (ny + 16 - 1) / 16, (nz + 4 - 1) / 4);
    const dim3 blockSize(16,16, 4);
    kernel_deformation<<<gridSize_img, blockSize>>>(img1, tex_img, mx, my, mz, nx, ny, nz);
    cudaDeviceSynchronize();

    cudaDestroyTextureObject(tex_img);
    cudaFreeArray(array_img);
}
