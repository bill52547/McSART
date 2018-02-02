#include "mex.h"
#define BLOCKSIZE_X 16
#define BLOCKSIZE_Y 16 
#define BLOCKSIZE_Z 4
__host__ void host_diff(float *img1, float *img, int nx, int ny, int nz, int ind);
__global__ void kernel_diff(float *img1, float *img, int nx, int ny, int nz, int ind);
void mexFunction(int nlhs, mxArray *plhs[], int nrhs, mxArray const *prhs[])
{
// Macro for input and output
#define IN_IMG prhs[0]
#define GEO_PARA prhs[1]
#define OUT_IMG plhs[0]

int nx, ny, nz, ind;

// resolutions of volumes 
if (mxGetField(GEO_PARA, 0, "nx") != NULL)
    nx = (int)mxGetScalar(mxGetField(GEO_PARA, 0, "nx"));
else
	mexErrMsgIdAndTxt("MATLAB:badInput","Can't found valid volume resolution nx.\n");

if (mxGetField(GEO_PARA, 0, "ny") != NULL)
    ny = (int)mxGetScalar(mxGetField(GEO_PARA, 0, "ny"));
else
	mexErrMsgIdAndTxt("MATLAB:badInput","Can't found valid volume resolution ny.\n");

if (mxGetField(GEO_PARA, 0, "nz") != NULL)
    nz = (int)mxGetScalar(mxGetField(GEO_PARA, 0, "nz"));
else
	mexErrMsgIdAndTxt("MATLAB:badInput","Can't found valid volume resolution nz.\n");

int numImg = nx * ny * nz; // size of image
int numBytesImg = numImg * sizeof(float); // number of bytes in image
if (mxGetField(GEO_PARA, 0, "ind") != NULL)
    ind = (int)mxGetScalar(mxGetField(GEO_PARA, 0, "ind"));
else
	mexErrMsgIdAndTxt("MATLAB:badInput","Can't found ind.\n");
float *h_img;
h_img = (float*)mxGetData(IN_IMG);

float *d_img, *d_img1;
cudaMalloc((void**)&d_img, numBytesImg);
cudaMalloc((void**)&d_img1, numBytesImg);

cudaMemcpy(d_img, h_img, numBytesImg, cudaMemcpyHostToDevice);

host_diff(d_img1, d_img, nx, ny, nz, ind);


OUT_IMG = mxCreateNumericMatrix(0, 0, mxSINGLE_CLASS, mxREAL);
const mwSize outDim[3] = {(mwSize)nx, (mwSize)ny, (mwSize)nz};

mxSetDimensions(OUT_IMG, outDim, 3);
mxSetData(OUT_IMG, mxMalloc(numBytesImg));
float *h_outimg = (float*)mxGetData(OUT_IMG);

cudaMemcpy(h_outimg, d_img1, numBytesImg, cudaMemcpyDeviceToHost);

cudaFree(d_img1);
cudaFree(d_img);
cudaDeviceReset();
return;
}



__host__ void host_diff(float *img1, float *img, int nx, int ny, int nz, int ind)
{
    const dim3 gridSize((nx + BLOCKSIZE_X - 1) / BLOCKSIZE_X, (ny + BLOCKSIZE_Y - 1) / BLOCKSIZE_Y, (nz + BLOCKSIZE_Z - 1) / BLOCKSIZE_Z);
    const dim3 blockSize(BLOCKSIZE_X, BLOCKSIZE_Y, BLOCKSIZE_Z);
    kernel_diff<<<gridSize, blockSize>>>(img1, img, nx, ny, nz, ind);
    cudaDeviceSynchronize();
}

__global__ void kernel_diff(float *img1, float *img, int nx, int ny, int nz, int ind)
{
    int ix = BLOCKSIZE_X * blockIdx.x + threadIdx.x;
    int iy = BLOCKSIZE_Y * blockIdx.y + threadIdx.y;
    int iz = BLOCKSIZE_Z * blockIdx.z + threadIdx.z;
    
    if (ix >= nx || iy >= ny || iz >= nz)
        return;
    int id = ix + iy * nx + iz * nx * ny;
    switch (ind)
    {    
    case 1:
        if (ix == nx - 1)
            img1[id] = 0.0f;
        else if (ix == 0)
            img1[id] = 0.0f;
        else
            img1[id] = img[id] - img[id - 1];
        break;
    case 2:
        if (iy == ny - 1)
            img1[id] = 0.0f;
        else if (iy == 0)
            img1[id] = 0.0f;
        else
            img1[id] = img[id] - img[id - nx];
        break;
    case 3:
        if (iz == nz - 1)
            img1[id] = 0.0f;
        else if (iz == 0)
            img1[id] = 0.0f;
        else
            img1[id] = img[id] - img[id - nx * ny];
        break;

    }

}