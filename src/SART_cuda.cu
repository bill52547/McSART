#include "SART_cuda.h" // consists all required package and functions

void mexFunction(int nlhs, mxArray *plhs[], int nrhs, mxArray const *prhs[])
{
// Macro for input and output
#define IN_IMG prhs[0]
#define PROJ prhs[1]
#define GEO_PARA prhs[2]
#define ITER_PARA prhs[3]
#define OUT_IMG plhs[0]
// #define OUT_ERR plhs[1]

int nx, ny, nz, na, nb, numImg, numBytesImg, numSingleProj, numBytesSingleProj;
float da, db, ai, bi, SO, SD, dx;

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

numImg = nx * ny * nz; // size of image
numBytesImg = numImg * sizeof(float); // number of bytes in image

// detector plane resolutions
if (mxGetField(GEO_PARA, 0, "na") != NULL)
    na = (int)mxGetScalar(mxGetField(GEO_PARA, 0, "na"));
else if (mxGetField(GEO_PARA, 0, "nv") != NULL)
    na = (int)mxGetScalar(mxGetField(GEO_PARA, 0, "nv"));
else
	mexErrMsgIdAndTxt("MATLAB:badInput","Can't found valid number of detector in plane, which is denoted as na or nu.\n");

if (mxGetField(GEO_PARA, 0, "nb") != NULL)
    nb = (int)mxGetScalar(mxGetField(GEO_PARA, 0, "nb"));
else if (mxGetField(GEO_PARA, 0, "nu") != NULL)
    nb = (int)mxGetScalar(mxGetField(GEO_PARA, 0, "nu"));
else
	mexErrMsgIdAndTxt("MATLAB:badInput","Can't found valid number of detector across plane, which is denoted as nb or nv.\n");

numSingleProj = na * nb;
numBytesSingleProj = numSingleProj * sizeof(float);

// voxel resolution dx, which is also the scaling factor of the whole system
if (mxGetField(GEO_PARA, 0, "dx") != NULL)
    dx = (float)mxGetScalar(mxGetField(GEO_PARA, 0, "dx"));
else{
    dx = 1.0f;
    mexPrintf("Automatically set voxel size dx to 1. \n");
    mexPrintf("If don't want that default value, please set para.dx manually.\n");
}

// detector resolution
if (mxGetField(GEO_PARA, 0, "da") != NULL)
    da = (float)mxGetScalar(mxGetField(GEO_PARA, 0, "da"));
else{
    da = 1.0f;
    mexPrintf("Automatically set detector cell size da to 1. \n");
    mexPrintf("If don't want that default value, please set para.da manually.\n");
}

if (mxGetField(GEO_PARA, 0, "db") != NULL)
    db = (float)mxGetScalar(mxGetField(GEO_PARA, 0, "db"));
else{
    db = 1.0f;
    mexPrintf("Automatically set detectof cell size db to 1. \n");
    mexPrintf("If don't want that default value, please set para.db manually.\n");
}


// detector plane offset from centered calibrations
if (mxGetField(GEO_PARA, 0, "ai") != NULL){
    ai = (float)mxGetScalar(mxGetField(GEO_PARA, 0, "ai"));
    ai -= ((float)na / 2 - 0.5f);
}
else{
    mexPrintf("Automatically set detector offset ai to 0. \n");
    mexPrintf("If don't want that default value, please set para.ai manually.\n");
    ai = - ((float)na / 2 - 0.5f);
}

if (mxGetField(GEO_PARA, 0, "bi") != NULL){
    bi = (float)mxGetScalar(mxGetField(GEO_PARA, 0, "bi"));
    bi -= ((float)nb / 2 - 0.5f);
}
else{
    mexPrintf("Automatically set detector offset bi to 0. \n");
    mexPrintf("If don't want that default value, please set para.bi manually.\n");
    bi = - ((float)nb / 2 - 0.5f);
}


if (mxGetField(GEO_PARA, 0, "SO") != NULL)
    SO = (float)mxGetScalar(mxGetField(GEO_PARA, 0, "SO"));
else if (mxGetField(GEO_PARA, 0, "SI") != NULL)
    SO = (float)mxGetScalar(mxGetField(GEO_PARA, 0, "SI"));
else
    mexErrMsgIdAndTxt("MATLAB:badInput","Can't found valid distance between source and isocenter, which is denoted with para.SO or para.DI.\n");

if (mxGetField(GEO_PARA, 0, "SD") != NULL)
    SD = (float)mxGetScalar(mxGetField(GEO_PARA, 0, "SD"));
else if (mxGetField(GEO_PARA, 0, "DI") != NULL)
    SD = (float)mxGetScalar(mxGetField(GEO_PARA, 0, "DI")) + SO;
else
    mexErrMsgIdAndTxt("MATLAB:badInput","Can't found valid distance between source and detector plane, which is denoted with para.SD or para.SI + para.DI.\n");


// load iterating parameters, for the whole bin
int n_iter, n_iter_invertDVF;
if (mxGetField(ITER_PARA, 0, "n_iter") != NULL)
    n_iter = (int)mxGetScalar(mxGetField(ITER_PARA, 0, "n_iter")); // number of views in this bin
else{
    n_iter = 1;
    mexPrintf("Automatically set number of iterations to 1. \n");
    mexPrintf("If don't want that default value, please set iter_para.n_iter manually.\n");
}

if (mxGetField(ITER_PARA, 0, "n_iter_invertDVF") != NULL)
    n_iter_invertDVF = (int)mxGetScalar(mxGetField(ITER_PARA, 0, "n_iter_invertDVF"));
else{
    n_iter_invertDVF = 10;
    mexPrintf("Automatically set number of iterations for inverting DVF to 10. \n");
    mexPrintf("If don't want that default value, please set iter_para.n_iter_invertDVF manually.\n");
}

int n_bin, *n_views, numProj, numBytesProj, N_view; // number of bins, numbers of views of bins, and the index view of each bin.
// e.g. we have 3 bins here with 10 total views. For these 3 bins, they holds 1,3,6 views. Then we will set n_views as {0, 1, 4, 10}, which is the starting view indices of each bin. Moreover, we need to well arrange the volumes and flows.

if (mxGetField(ITER_PARA, 0, "n_bin") != NULL)
    n_bin = (int)mxGetScalar(mxGetField(ITER_PARA, 0, "n_bin"));
else{
    n_bin = 8;
    mexPrintf("Automatically set number of bins to 8. \n");
    mexPrintf("If don't want that default value, please set iter_para.n_bin manually.\n");
}

if (mxGetField(ITER_PARA, 0, "n_views") != NULL)
    n_views = (int*)mxGetData(mxGetField(ITER_PARA, 0, "n_views"));
else{
    mexErrMsgIdAndTxt("MATLAB:badInput","Can't found valid number bins, which is denoted as iter_para.n_views.\n");
}
N_view = n_views[n_bin];

// 5D models
float *h_alpha_x, *h_alpha_y, *h_alpha_z, *h_beta_x, *h_beta_y, *h_beta_z, *angles, lambda;

// load 5DCT alpha and beta
if (mxGetField(ITER_PARA, 0, "alpha_x") != NULL)
    h_alpha_x = (float*)mxGetData(mxGetField(ITER_PARA, 0, "alpha_x")); 
else
    mexErrMsgIdAndTxt("MATLAB:badInput","Can't found valid iter_para.alpha_x.\n");    

if (mxGetField(ITER_PARA, 0, "alpha_y") != NULL)
    h_alpha_y = (float*)mxGetData(mxGetField(ITER_PARA, 0, "alpha_y")); 
else
    mexErrMsgIdAndTxt("MATLAB:badInput","Can't found valid iter_para.alpha_y.\n");

if (mxGetField(ITER_PARA, 0, "alpha_z") != NULL)
    h_alpha_z = (float*)mxGetData(mxGetField(ITER_PARA, 0, "alpha_z"));
else
    mexErrMsgIdAndTxt("MATLAB:badInput","Can't found valid iter_para.alpha_z.\n");

if (mxGetField(ITER_PARA, 0, "beta_x") != NULL)
    h_beta_x = (float*)mxGetData(mxGetField(ITER_PARA, 0, "beta_x"));
else
    mexErrMsgIdAndTxt("MATLAB:badInput","Can't found valid iter_para.beta_x.\n");

if (mxGetField(ITER_PARA, 0, "beta_y") != NULL)
    h_beta_y = (float*)mxGetData(mxGetField(ITER_PARA, 0, "beta_y")); 
else
    mexErrMsgIdAndTxt("MATLAB:badInput","Can't found valid iter_para.beta_y.\n");

if (mxGetField(ITER_PARA, 0, "beta_z") != NULL)
    h_beta_z = (float*)mxGetData(mxGetField(ITER_PARA, 0, "beta_z"));
else
    mexErrMsgIdAndTxt("MATLAB:badInput","Can't found valid iter_para.beta_z.\n");

// load 5DCT parameters volume (v) and flow (f)
float *volumes, *flows, *ref_volumes, *ref_flows;
if (mxGetField(ITER_PARA, 0, "volumes") != NULL)
    volumes= (float*)mxGetData(mxGetField(ITER_PARA, 0, "volumes"));
else
    mexErrMsgIdAndTxt("MATLAB:badInput","Can't found valid volume in iter_para.volumes.\n");  

if (mxGetField(ITER_PARA, 0, "flows") != NULL)
    flows = (float*)mxGetData(mxGetField(ITER_PARA, 0, "flows"));
else
    mexErrMsgIdAndTxt("MATLAB:badInput","Can't found valid flow in iter_para.flows.\n");    

if (mxGetField(ITER_PARA, 0, "volume0") != NULL)
    ref_volumes = (float*)mxGetData(mxGetField(ITER_PARA, 0, "volume0"));
else
    mexErrMsgIdAndTxt("MATLAB:badInput","Can't found valid referenced volume in iter_para.volume0.\n");    

if (mxGetField(ITER_PARA, 0, "flow0") != NULL)
    ref_flows = (float*)mxGetData(mxGetField(ITER_PARA, 0, "flow0"));
else
    mexErrMsgIdAndTxt("MATLAB:badInput","Can't found valid referenced flow in iter_para.flow0.\n");    

if (mxGetField(ITER_PARA, 0, "angles") != NULL)
    angles = (float*)mxGetData(mxGetField(ITER_PARA, 0, "angles"));
else
    mexErrMsgIdAndTxt("MATLAB:badInput","Can't found valid angles iter_para.angles.\n");
if (mxGetField(ITER_PARA, 0, "lambda") != NULL)
    lambda = (float)mxGetScalar(mxGetField(ITER_PARA, 0, "lambda"));
else
    mexErrMsgIdAndTxt("MATLAB:badInput","Can't found valid coefficience iter_para.lambda.\n");

numProj = numSingleProj * N_view;
numBytesProj = numProj * sizeof(float);

// load initial guess of image
float *h_img;
h_img = (float*)mxGetData(IN_IMG);

// load true projection value
float *h_proj;
h_proj = (float*)mxGetData(PROJ);

// define thread distributions
const dim3 gridSize_img((nx + BLOCKWIDTH - 1) / BLOCKWIDTH, (ny + BLOCKHEIGHT - 1) / BLOCKHEIGHT, (nz + BLOCKDEPTH - 1) / BLOCKDEPTH);
const dim3 gridSize_singleProj((na + BLOCKWIDTH - 1) / BLOCKWIDTH, (nb + BLOCKHEIGHT - 1) / BLOCKHEIGHT, 1);
const dim3 blockSize(BLOCKWIDTH,BLOCKHEIGHT, BLOCKDEPTH);

// CUDA 3DArray Malloc parameters
struct cudaExtent extent_img = make_cudaExtent(nx, ny, nz);
struct cudaExtent extent_singleProj = make_cudaExtent(na, nb, 1);

//Allocate CUDA array in device memory of 5DCT matrices: alpha and beta
cudaArray *d_alpha_x, *d_alpha_y, *d_alpha_z, *d_beta_x, *d_beta_y, *d_beta_z;
cudaChannelFormatDesc channelDesc = cudaCreateChannelDesc(32, 0, 0, 0, cudaChannelFormatKindFloat);

cudaError_t cudaStat;
// alpha_x
cudaStat = cudaMalloc3DArray(&d_alpha_x, &channelDesc, extent_img);
if (cudaStat != cudaSuccess) {
	mexPrintf("Array memory allocation for alpha_x failed.\n");
	mexPrintf("Error code %d: %s\n",cudaStat,cudaGetErrorString(cudaStat));
        mexErrMsgIdAndTxt("MATLAB:cudaFail","SART failed.\n");
}

// alpha_y
cudaStat = cudaMalloc3DArray(&d_alpha_y, &channelDesc, extent_img);
if (cudaStat != cudaSuccess) {
	mexPrintf("Array memory allocation for alpha_y failed.\n");
	mexPrintf("Error code %d: %s\n",cudaStat,cudaGetErrorString(cudaStat));
        mexErrMsgIdAndTxt("MATLAB:cudaFail","SART failed.\n");
}

// alpha_z
cudaStat = cudaMalloc3DArray(&d_alpha_z, &channelDesc, extent_img);
if (cudaStat != cudaSuccess) {
	mexPrintf("Array memory allocation for alpha_z failed.\n");
	mexPrintf("Error code %d: %s\n",cudaStat,cudaGetErrorString(cudaStat));
        mexErrMsgIdAndTxt("MATLAB:cudaFail","SART failed.\n");
}

// beta_x
cudaStat = cudaMalloc3DArray(&d_beta_x, &channelDesc, extent_img);
if (cudaStat != cudaSuccess) {
	mexPrintf("Array memory allocation for beta_x failed.\n");
	mexPrintf("Error code %d: %s\n",cudaStat,cudaGetErrorString(cudaStat));
        mexErrMsgIdAndTxt("MATLAB:cudaFail","SART failed.\n");
}
// beta_y
cudaStat = cudaMalloc3DArray(&d_beta_y, &channelDesc, extent_img);
if (cudaStat != cudaSuccess) {
	mexPrintf("Array memory allocation for beta_y failed.\n");
	mexPrintf("Error code %d: %s\n",cudaStat,cudaGetErrorString(cudaStat));
        mexErrMsgIdAndTxt("MATLAB:cudaFail","SART failed.\n");
}
// beta_z
cudaStat = cudaMalloc3DArray(&d_beta_z, &channelDesc, extent_img);
if (cudaStat != cudaSuccess) {
	mexPrintf("Array memory allocation for beta_z failed.\n");
	mexPrintf("Error code %d: %s\n",cudaStat,cudaGetErrorString(cudaStat));
        mexErrMsgIdAndTxt("MATLAB:cudaFail","SART failed.\n");
}


// Get pitched pointer to alpha and beta in host memory
cudaPitchedPtr hp_alpha_x = make_cudaPitchedPtr((void*) h_alpha_x, nx * sizeof(float), nx, ny);
cudaPitchedPtr hp_alpha_y = make_cudaPitchedPtr((void*) h_alpha_y, nx * sizeof(float), nx, ny);
cudaPitchedPtr hp_alpha_z = make_cudaPitchedPtr((void*) h_alpha_z, nx * sizeof(float), nx, ny);
cudaPitchedPtr hp_beta_x = make_cudaPitchedPtr((void*) h_beta_x, nx * sizeof(float), nx, ny);
cudaPitchedPtr hp_beta_y = make_cudaPitchedPtr((void*) h_beta_y, nx * sizeof(float), nx, ny);
cudaPitchedPtr hp_beta_z = make_cudaPitchedPtr((void*) h_beta_z, nx * sizeof(float), nx, ny);

// Copy alpha and beta to texture memory from pitched pointer
cudaMemcpy3DParms copyParams = {0};
copyParams.extent = extent_img;
copyParams.kind = cudaMemcpyHostToDevice;

//alpha_x
copyParams.srcPtr = hp_alpha_x;
copyParams.dstArray = d_alpha_x;
cudaStat = cudaMemcpy3D(&copyParams);
if (cudaStat != cudaSuccess) {
	mexPrintf("Failed to copy alpha_x to device memory.\n");
	mexPrintf("Error code %d: %s\n",cudaStat,cudaGetErrorString(cudaStat));
        mexErrMsgIdAndTxt("MATLAB:cudaFail","SART failed.\n");
}

//alpha_y
copyParams.srcPtr = hp_alpha_y;
copyParams.dstArray = d_alpha_y;
cudaStat = cudaMemcpy3D(&copyParams);
if (cudaStat != cudaSuccess) {
	mexPrintf("Failed to copy alpha_y to device memory.\n");
	mexPrintf("Error code %d: %s\n",cudaStat,cudaGetErrorString(cudaStat));
        mexErrMsgIdAndTxt("MATLAB:cudaFail","SART failed.\n");
}

//alpha_z
copyParams.srcPtr = hp_alpha_z;
copyParams.dstArray = d_alpha_z;
cudaStat = cudaMemcpy3D(&copyParams);
if (cudaStat != cudaSuccess) {
	mexPrintf("Failed to copy alpha_z to device memory.\n");
	mexPrintf("Error code %d: %s\n",cudaStat,cudaGetErrorString(cudaStat));
        mexErrMsgIdAndTxt("MATLAB:cudaFail","SART failed.\n");
}

//beta_x
copyParams.srcPtr = hp_beta_x;
copyParams.dstArray = d_beta_x;
cudaStat = cudaMemcpy3D(&copyParams);
if (cudaStat != cudaSuccess) {
	mexPrintf("Failed to copy beta_x to device memory.\n");
	mexPrintf("Error code %d: %s\n",cudaStat,cudaGetErrorString(cudaStat));
        mexErrMsgIdAndTxt("MATLAB:cudaFail","SART failed.\n");
}

//beta_y
copyParams.srcPtr = hp_beta_y;
copyParams.dstArray = d_beta_y;
cudaStat = cudaMemcpy3D(&copyParams);
if (cudaStat != cudaSuccess) {
	mexPrintf("Failed to copy beta_y to device memory.\n");
	mexPrintf("Error code %d: %s\n",cudaStat,cudaGetErrorString(cudaStat));
        mexErrMsgIdAndTxt("MATLAB:cudaFail","SART failed.\n");
}

//beta_z
copyParams.srcPtr = hp_beta_z;
copyParams.dstArray = d_beta_z;
cudaStat = cudaMemcpy3D(&copyParams);
if (cudaStat != cudaSuccess) {
	mexPrintf("Failed to copy beta_z to device memory.\n");
	mexPrintf("Error code %d: %s\n",cudaStat,cudaGetErrorString(cudaStat));
        mexErrMsgIdAndTxt("MATLAB:cudaFail","SART failed.\n");
}


// create texture object alpha and beta
cudaResourceDesc resDesc;
cudaTextureDesc texDesc, texDesc2;
memset(&resDesc, 0, sizeof(resDesc));
resDesc.resType = cudaResourceTypeArray;

memset(&texDesc, 0, sizeof(texDesc));
texDesc.addressMode[0] = cudaAddressModeClamp;
texDesc.addressMode[1] = cudaAddressModeClamp;
texDesc.addressMode[2] = cudaAddressModeClamp;
texDesc.filterMode = cudaFilterModeLinear;
texDesc.readMode = cudaReadModeElementType;
texDesc.normalizedCoords = 0;

memset(&texDesc2, 0, sizeof(texDesc2));
texDesc2.addressMode[0] = cudaAddressModeClamp;
texDesc2.addressMode[1] = cudaAddressModeClamp;
texDesc2.addressMode[2] = cudaAddressModeClamp;
texDesc2.filterMode = cudaFilterModePoint;
texDesc2.readMode = cudaReadModeElementType;
texDesc2.normalizedCoords = 0;

// alpha_x
resDesc.res.array.array = d_alpha_x;
cudaTextureObject_t tex_alpha_x = 0;
cudaCreateTextureObject(&tex_alpha_x, &resDesc, &texDesc, NULL);

// alpha_y
resDesc.res.array.array = d_alpha_y;
cudaTextureObject_t tex_alpha_y = 0;
cudaCreateTextureObject(&tex_alpha_y, &resDesc, &texDesc, NULL);

// alpha_z
resDesc.res.array.array = d_alpha_z;
cudaTextureObject_t tex_alpha_z = 0;
cudaCreateTextureObject(&tex_alpha_z, &resDesc, &texDesc, NULL);

// beta_x
resDesc.res.array.array = d_beta_x;
cudaTextureObject_t tex_beta_x = 0;
cudaCreateTextureObject(&tex_beta_x, &resDesc, &texDesc, NULL);

// beta_y
resDesc.res.array.array = d_beta_y;
cudaTextureObject_t tex_beta_y = 0;
cudaCreateTextureObject(&tex_beta_y, &resDesc, &texDesc, NULL);

// beta_z
resDesc.res.array.array = d_beta_z;
cudaTextureObject_t tex_beta_z = 0;
cudaCreateTextureObject(&tex_beta_z, &resDesc, &texDesc, NULL);

// malloc in device: projection of the whole bin
float *d_proj;
cudaMalloc((void**)&d_proj, numBytesSingleProj);

// malloc in device: another projection pointer, with single view size
float *d_singleViewProj2;
cudaMalloc((void**)&d_singleViewProj2, numBytesSingleProj);

// malloc in device: projection of the whole bin
float *d_img ,*d_img1;
cudaArray *array_img;
cudaMalloc((void**)&d_img, numBytesImg);
cudaMalloc((void**)&d_img1, numBytesImg);
cudaStat = cudaMalloc3DArray(&array_img, &channelDesc, extent_img);
if (cudaStat != cudaSuccess) {
	mexPrintf("Array memory allocation for array_img failed.\n");
	mexPrintf("Error code %d: %s\n",cudaStat,cudaGetErrorString(cudaStat));
        mexErrMsgIdAndTxt("MATLAB:cudaFail","SART failed.\n");
}

// malloc in device: another image pointer, for single view 
float *d_singleViewImg, *d_imgOnes;
cudaMalloc(&d_singleViewImg, numBytesImg);
cudaMalloc(&d_imgOnes, numBytesImg);
float angle, volume, flow;

//Malloc forward and inverted DVFs in device
float *d_mx, *d_my, *d_mz;
cudaMalloc(&d_mx, numBytesImg);
cudaMalloc(&d_my, numBytesImg);
cudaMalloc(&d_mz, numBytesImg);


// Alloc forward and inverted DVFs in device, in form of array memory
cudaArray *array_mx, *array_my, *array_mz;
cudaStat = cudaMalloc3DArray(&array_mx, &channelDesc, extent_img);
if (cudaStat != cudaSuccess) {
	mexPrintf("Array memory allocation for array_mx failed.\n");
	mexPrintf("Error code %d: %s\n",cudaStat,cudaGetErrorString(cudaStat));
        mexErrMsgIdAndTxt("MATLAB:cudaFail","SART failed.\n");
}

cudaStat = cudaMalloc3DArray(&array_my, &channelDesc, extent_img);
if (cudaStat != cudaSuccess) {
	mexPrintf("Array memory allocation for array_my failed.\n");
	mexPrintf("Error code %d: %s\n",cudaStat,cudaGetErrorString(cudaStat));
        mexErrMsgIdAndTxt("MATLAB:cudaFail","SART failed.\n");
}

cudaStat = cudaMalloc3DArray(&array_mz, &channelDesc, extent_img);
if (cudaStat != cudaSuccess) {
	mexPrintf("Array memory allocation for array_mz failed.\n");
	mexPrintf("Error code %d: %s\n",cudaStat,cudaGetErrorString(cudaStat));
        mexErrMsgIdAndTxt("MATLAB:cudaFail","SART failed.\n");
}

// define tex_mx etc
cudaTextureObject_t tex_mx = 0, tex_my = 0, tex_mz = 0;

// setup output images
OUT_IMG = mxCreateNumericMatrix(0, 0, mxSINGLE_CLASS, mxREAL);
const mwSize outDim[4] = {(mwSize)nx, (mwSize)ny, (mwSize)nz, (mwSize)n_bin};
mxSetDimensions(OUT_IMG, outDim, 4);
mxSetData(OUT_IMG, mxMalloc(numBytesImg * n_bin));
float *h_outimg = (float*)mxGetData(OUT_IMG);


copyParams.kind = cudaMemcpyDeviceToDevice;

for (int ibin = 0; ibin < n_bin; ibin++){
    if (ibin < 1){
        cudaMemcpy(d_img, h_img, numBytesImg, cudaMemcpyHostToDevice);
    }
    else{
    //     // cudaMemcpy(d_img1, h_img, numBytesImg, cudaMemcpyHostToDevice);
        volume = ref_volumes[ibin] - ref_volumes[0];
        flow = ref_flows[ibin] - ref_flows[0];
        kernel_forwardDVF<<<gridSize_img, blockSize>>>(d_mx, d_my, d_mz, tex_alpha_x, tex_alpha_y, tex_alpha_z, tex_beta_x, tex_beta_y, tex_beta_z, volume, flow, nx, ny, nz);
        cudaDeviceSynchronize();

        // copy img to pitched pointer and bind it to a texture object
        cudaPitchedPtr dp_img = make_cudaPitchedPtr((void*) d_img1, nx * sizeof(float), nx, ny);
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

        kernel_deformation<<<gridSize_img, blockSize>>>(d_img, tex_img, d_mx, d_my, d_mz, nx, ny, nz);
        cudaDeviceSynchronize();
    }
    for (int iter = 0; iter < n_iter; iter++){ // iteration
        processBar(ibin, n_bin, iter, n_iter);
        
        for (int i_view = n_views[ibin]; i_view < n_views[ibin + 1]; i_view++){ // view
        
            angle = angles[i_view];
            volume = ref_volumes[ibin] - ref_volumes[0]
            flow = ref_flows[ibin] - ref_flows[0];
            
            kernel_forwardDVF<<<gridSize_img, blockSize>>>(d_mx, d_my, d_mz, tex_alpha_x, tex_alpha_y, tex_alpha_z, tex_beta_x, tex_beta_y, tex_beta_z, volume, flow, nx, ny, nz);
            cudaDeviceSynchronize();
            
            // copy mx etc to pitched pointer and bind it to a texture object
            cudaPitchedPtr dp_mx = make_cudaPitchedPtr((void*) d_mx, nx * sizeof(float), nx, ny);
            copyParams.srcPtr = dp_mx;
            copyParams.dstArray = array_mx;
            cudaStat = cudaMemcpy3D(&copyParams);   
            if (cudaStat != cudaSuccess) {
                mexPrintf("Failed to copy dp_mx to array memory array_mx.\n");
                mexPrintf("Error code %d: %s\n",cudaStat,cudaGetErrorString(cudaStat));
                    mexErrMsgIdAndTxt("MATLAB:cudaFail","SART failed.\n");
            }
            resDesc.res.array.array = array_mx;
            cudaCreateTextureObject(&tex_mx, &resDesc, &texDesc, NULL);

            cudaPitchedPtr dp_my = make_cudaPitchedPtr((void*) d_my, nx * sizeof(float), nx, ny);
            copyParams.srcPtr = dp_my;
            copyParams.dstArray = array_my;
            cudaStat = cudaMemcpy3D(&copyParams);   
            if (cudaStat != cudaSuccess) {
                mexPrintf("Failed to copy dp_my to array memory array_my.\n");
                mexPrintf("Error code %d: %s\n",cudaStat,cudaGetErrorString(cudaStat));
                    mexErrMsgIdAndTxt("MATLAB:cudaFail","SART failed.\n");
            }
            resDesc.res.array.array = array_my;
            cudaCreateTextureObject(&tex_my, &resDesc, &texDesc, NULL);

            cudaPitchedPtr dp_mz = make_cudaPitchedPtr((void*) d_mz, nx * sizeof(float), nx, ny);
            copyParams.srcPtr = dp_mz;
            copyParams.dstArray = array_mz;
            cudaStat = cudaMemcpy3D(&copyParams);   
            if (cudaStat != cudaSuccess) {
                mexPrintf("Failed to copy dp_mz to array memory array_mz.\n");
                mexPrintf("Error code %d: %s\n",cudaStat,cudaGetErrorString(cudaStat));
                    mexErrMsgIdAndTxt("MATLAB:cudaFail","SART failed.\n");
            }
            resDesc.res.array.array = array_mz;
            cudaCreateTextureObject(&tex_mz, &resDesc, &texDesc, NULL);

            kernel_invertDVF<<<gridSize_img, blockSize>>>(d_mx, d_my, d_mz, tex_mx, tex_my, tex_mz, nx, ny, nz, n_iter_invertDVF);
            cudaDeviceSynchronize();        
            
            // copy img to pitched pointer and bind it to a texture object
            cudaPitchedPtr dp_img = make_cudaPitchedPtr((void*) d_img, nx * sizeof(float), nx, ny);
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

            kernel_deformation<<<gridSize_img, blockSize>>>(d_singleViewImg, tex_img, d_mx, d_my, d_mz, nx, ny, nz); // d_singleViewImg is for ref-phase 0
            cudaDeviceSynchronize();

            volume = volumes[i_view] - ref_volumes[0]
            flow = flows[i_view] - ref_flows[0];
            
            // generate forwards DVFs: d_mx, d_my, d_mz and inverted DVFs: d_mx, d_my, d_mz
            kernel_forwardDVF<<<gridSize_img, blockSize>>>(d_mx, d_my, d_mz, tex_alpha_x, tex_alpha_y, tex_alpha_z, tex_beta_x, tex_beta_y, tex_beta_z, volume, flow, nx, ny, nz);
            cudaDeviceSynchronize();

            // copy img to pitched pointer and bind it to a texture object
            cudaPitchedPtr dp_img = make_cudaPitchedPtr((void*) d_singleViewImg, nx * sizeof(float), nx, ny);
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

            kernel_deformation<<<gridSize_img, blockSize>>>(d_singleViewImg, tex_img, d_mx, d_my, d_mz, nx, ny, nz); // d_singleViewImg is for ref-phase 0
            cudaDeviceSynchronize();

            // projection of deformed image from initial guess
            kernel_projection<<<gridSize_singleProj, blockSize>>>(d_singleViewProj2, d_singleViewImg, angle, SO, SD, da, na, ai, db, nb, bi, nx, ny, nz); // TBD
            cudaDeviceSynchronize();

            // difference between true projection and projection from initial guess
            // update d_singleViewProj2 instead of malloc a new one
            cudaMemcpy(d_proj, h_proj + i_view * numSingleProj, numBytesSingleProj, cudaMemcpyHostToDevice);

            kernel_add<<<gridSize_singleProj, blockSize>>>(d_singleViewProj2, d_proj, 0, na, nb, -1);
            cudaDeviceSynchronize();

            // backprojecting the difference of projections
            kernel_backprojection(d_singleViewImg, d_singleViewProj2, angle, SO, SD, da, na, ai, db, nb, bi, nx, ny, nz);

            // calculate the ones backprojection data
            kernel_initial<<<gridSize_img, blockSize>>>(d_imgOnes, nx, ny, nz, 1);
            cudaDeviceSynchronize();

            kernel_projection<<<gridSize_singleProj, blockSize>>>(d_singleViewProj2, d_imgOnes, angle, SO, SD, da, na, ai, db, nb, bi, nx, ny, nz);
            cudaDeviceSynchronize();

            kernel_backprojection(d_imgOnes, d_singleViewProj2, angle, SO, SD, da, na, ai, db, nb, bi, nx, ny, nz);

            // weighting
            kernel_division<<<gridSize_img, blockSize>>>(d_singleViewImg, d_imgOnes, nx, ny, nz);
            cudaDeviceSynchronize();
            
            // copy mx etc to pitched pointer and bind it to a texture object
            cudaPitchedPtr dp_mx = make_cudaPitchedPtr((void*) d_mx, nx * sizeof(float), nx, ny);
            copyParams.srcPtr = dp_mx;
            copyParams.dstArray = array_mx;
            cudaStat = cudaMemcpy3D(&copyParams);   
            if (cudaStat != cudaSuccess) {
                mexPrintf("Failed to copy dp_mx to array memory array_mx.\n");
                mexPrintf("Error code %d: %s\n",cudaStat,cudaGetErrorString(cudaStat));
                    mexErrMsgIdAndTxt("MATLAB:cudaFail","SART failed.\n");
            }
            resDesc.res.array.array = array_mx;
            cudaCreateTextureObject(&tex_mx, &resDesc, &texDesc, NULL);

            cudaPitchedPtr dp_my = make_cudaPitchedPtr((void*) d_my, nx * sizeof(float), nx, ny);
            copyParams.srcPtr = dp_my;
            copyParams.dstArray = array_my;
            cudaStat = cudaMemcpy3D(&copyParams);   
            if (cudaStat != cudaSuccess) {
                mexPrintf("Failed to copy dp_my to array memory array_my.\n");
                mexPrintf("Error code %d: %s\n",cudaStat,cudaGetErrorString(cudaStat));
                    mexErrMsgIdAndTxt("MATLAB:cudaFail","SART failed.\n");
            }
            resDesc.res.array.array = array_my;
            cudaCreateTextureObject(&tex_my, &resDesc, &texDesc, NULL);

            cudaPitchedPtr dp_mz = make_cudaPitchedPtr((void*) d_mz, nx * sizeof(float), nx, ny);
            copyParams.srcPtr = dp_mz;
            copyParams.dstArray = array_mz;
            cudaStat = cudaMemcpy3D(&copyParams);   
            if (cudaStat != cudaSuccess) {
                mexPrintf("Failed to copy dp_mz to array memory array_mz.\n");
                mexPrintf("Error code %d: %s\n",cudaStat,cudaGetErrorString(cudaStat));
                    mexErrMsgIdAndTxt("MATLAB:cudaFail","SART failed.\n");
            }
            resDesc.res.array.array = array_mz;
            cudaCreateTextureObject(&tex_mz, &resDesc, &texDesc, NULL);
            
            kernel_invertDVF<<<gridSize_img, blockSize>>>(d_mx, d_my, d_mz, tex_mx, tex_my, tex_mz, nx, ny, nz, n_iter_invertDVF);
            cudaDeviceSynchronize();   
            
            // copy img to pitched pointer and bind it to a texture object
            dp_img = make_cudaPitchedPtr((void*) d_singleViewImg, nx * sizeof(float), nx, ny);
            copyParams.srcPtr = dp_img;
            copyParams.dstArray = array_img;
            cudaStat = cudaMemcpy3D(&copyParams);   
            if (cudaStat != cudaSuccess) {
                mexPrintf("Failed to copy dp_img to array memory array_img.\n");
                mexPrintf("Error code %d: %s\n",cudaStat,cudaGetErrorString(cudaStat));
                    mexErrMsgIdAndTxt("MATLAB:cudaFail","SART failed.\n");
            }
            resDesc.res.array.array = array_img;
            cudaCreateTextureObject(&tex_img, &resDesc, &texDesc2, NULL);

            kernel_deformation<<<gridSize_img, blockSize>>>(d_singleViewImg, tex_img, d_mx, d_my, d_mz, nx, ny, nz);
            cudaDeviceSynchronize();

            // copy img to pitched pointer and bind it to a texture object
            dp_img = make_cudaPitchedPtr((void*) d_singleViewImg, nx * sizeof(float), nx, ny);
            copyParams.srcPtr = dp_img;
            copyParams.dstArray = array_img;
            cudaStat = cudaMemcpy3D(&copyParams);   
            if (cudaStat != cudaSuccess) {
                mexPrintf("Failed to copy dp_img to array memory array_img.\n");
                mexPrintf("Error code %d: %s\n",cudaStat,cudaGetErrorString(cudaStat));
                    mexErrMsgIdAndTxt("MATLAB:cudaFail","SART failed.\n");
            }
            resDesc.res.array.array = array_img;
            cudaCreateTextureObject(&tex_img, &resDesc, &texDesc2, NULL);
            
            volume = ref_volumes[ibin] - ref_volumes[0]
            flow = ref_flows[ibin] - ref_flows[0];
            
            kernel_forwardDVF<<<gridSize_img, blockSize>>>(d_mx, d_my, d_mz, tex_alpha_x, tex_alpha_y, tex_alpha_z, tex_beta_x, tex_beta_y, tex_beta_z, volume, flow, nx, ny, nz);
            cudaDeviceSynchronize();

            kernel_deformation<<<gridSize_img, blockSize>>>(d_singleViewImg, tex_img, d_mx, d_my, d_mz, nx, ny, nz);
            cudaDeviceSynchronize();

            // updating
            kernel_update<<<gridSize_img, blockSize>>>(d_img, d_singleViewImg, nx, ny, nz, lambda);
            cudaDeviceSynchronize();          
        }  
    }
    if (ibin == 0){
        cudaMemcpy(d_img1, d_img, numBytesImg, cudaMemcpyDeviceToDevice);
    } 
    cudaMemcpy(h_outimg + ibin * numImg, d_img, numBytesImg, cudaMemcpyDeviceToHost);    
}


cudaDestroyTextureObject(tex_alpha_x);
cudaDestroyTextureObject(tex_alpha_y);
cudaDestroyTextureObject(tex_alpha_z);
cudaDestroyTextureObject(tex_beta_x);
cudaDestroyTextureObject(tex_beta_y);
cudaDestroyTextureObject(tex_beta_z);
cudaDestroyTextureObject(tex_img);
cudaDestroyTextureObject(tex_mx);
cudaDestroyTextureObject(tex_my);
cudaDestroyTextureObject(tex_mz);

cudaFreeArray(d_alpha_x);
cudaFreeArray(d_alpha_y);
cudaFreeArray(d_alpha_z);
cudaFreeArray(d_beta_x);
cudaFreeArray(d_beta_y);
cudaFreeArray(d_beta_z);
// cudaFreeArray(d_img);
cudaFree(d_mx);
cudaFree(d_my);
cudaFree(d_mz);
cudaFreeArray(array_mx);
cudaFreeArray(array_my);
cudaFreeArray(array_mz);
cudaFree(d_proj);
cudaFree(d_singleViewImg);
cudaFree(d_singleViewProj2);

cudaFree(d_img);
cudaFree(d_img1);
cudaDeviceReset();
return;
}

