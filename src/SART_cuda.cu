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


int nx = load_int_field(GEO_PARA, "nx");
int ny = load_int_field(GEO_PARA, "ny");
int nz = load_int_field(GEO_PARA, "nz");

int numImg = nx * ny * nz; // size of image
int numBytesImg = numImg * sizeof(float); // number of bytes in image

int na = load_int_field(GEO_PARA, "na");
int nb = load_int_field(GEO_PARA, "nb");

int numSingleProj = na * nb;
int numBytesSingleProj = numSingleProj * sizeof(float);


float dx = load_float_field(GEO_PARA, "dx");
float da = load_float_field(GEO_PARA, "da");
float db = load_float_field(GEO_PARA, "db");
float ai = load_float_field(GEO_PARA, "ai");
ai -= ((float)na / 2 - 0.5f);
float bi = load_float_field(GEO_PARA, "bi");
bi -= ((float)nb / 2 - 0.5f);

float SO = load_float_field(GEO_PARA, "SO");
float SD = load_float_field(GEO_PARA, "SD");

int n_iter = load_int_field(ITER_PARA, "n_iter");
int n_bin = load_int_field(ITER_PARA, "n_bin");

int* n_views;
n_views = (int*)mxGetData(mxGetField(ITER_PARA, 0, "n_views"));
int N_view = n_views[n_bin];
int isConst = load_int_field(ITER_PARA, "isConst");
float lambda = load_float_field(ITER_PARA, "lambda");
int outIter = load_int_field(ITER_PARA, "outIter");
// 5D models
float *h_alpha_x, *h_alpha_y, *h_alpha_z, *h_beta_x, *h_beta_y, *h_beta_z,  *h_const_x, *h_const_y, *h_const_z, *angles;
h_alpha_x = (float*)mxGetData(mxGetField(ITER_PARA, 0, "alpha_x"));
h_alpha_y = (float*)mxGetData(mxGetField(ITER_PARA, 0, "alpha_y"));
h_alpha_z = (float*)mxGetData(mxGetField(ITER_PARA, 0, "alpha_z"));
h_beta_x = (float*)mxGetData(mxGetField(ITER_PARA, 0, "beta_x"));
h_beta_y = (float*)mxGetData(mxGetField(ITER_PARA, 0, "beta_y"));
h_beta_z = (float*)mxGetData(mxGetField(ITER_PARA, 0, "beta_z"));
h_const_x = (float*)mxGetData(mxGetField(ITER_PARA, 0, "const_x"));
h_const_y = (float*)mxGetData(mxGetField(ITER_PARA, 0, "const_y"));
h_const_z = (float*)mxGetData(mxGetField(ITER_PARA, 0, "const_z"));
angles = (float*)mxGetData(mxGetField(ITER_PARA, 0, "angles"));

float *volumes, *ref_volumes, *flows, *ref_flows;
volumes = (float*)mxGetData(mxGetField(ITER_PARA, 0, "volumes"));
ref_volumes = (float*)mxGetData(mxGetField(ITER_PARA, 0, "volume0"));
flows = (float*)mxGetData(mxGetField(ITER_PARA, 0, "flows"));
ref_flows = (float*)mxGetData(mxGetField(ITER_PARA, 0, "flow0"));

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
float *d_alpha_x, *d_alpha_y, *d_alpha_z, *d_beta_x, *d_beta_y, *d_beta_z, *d_const_x, *d_const_y, *d_const_z;
cudaMalloc((void**)&d_alpha_x, numBytesImg);
cudaMalloc((void**)&d_alpha_y, numBytesImg);
cudaMalloc((void**)&d_alpha_z, numBytesImg);
cudaMalloc((void**)&d_beta_x, numBytesImg);
cudaMalloc((void**)&d_beta_y, numBytesImg);
cudaMalloc((void**)&d_beta_z, numBytesImg);
cudaMalloc((void**)&d_const_x, numBytesImg);
cudaMalloc((void**)&d_const_y, numBytesImg);
cudaMalloc((void**)&d_const_z, numBytesImg);

cudaMemcpy(d_alpha_x, h_alpha_x, numBytesImg, cudaMemcpyHostToDevice);
cudaMemcpy(d_alpha_y, h_alpha_y, numBytesImg, cudaMemcpyHostToDevice);
cudaMemcpy(d_alpha_z, h_alpha_z, numBytesImg, cudaMemcpyHostToDevice);
cudaMemcpy(d_beta_x, h_beta_x, numBytesImg, cudaMemcpyHostToDevice);
cudaMemcpy(d_beta_y, h_beta_y, numBytesImg, cudaMemcpyHostToDevice);
cudaMemcpy(d_beta_z, h_beta_z, numBytesImg, cudaMemcpyHostToDevice);
cudaMemcpy(d_const_x, h_const_x, numBytesImg, cudaMemcpyHostToDevice);
cudaMemcpy(d_const_y, h_const_y, numBytesImg, cudaMemcpyHostToDevice);
cudaMemcpy(d_const_z, h_const_z, numBytesImg, cudaMemcpyHostToDevice);


// malloc in device: projection of the whole bin
float *d_proj;
cudaMalloc((void**)&d_proj, numBytesSingleProj);

// malloc in device: another projection pointer, with single view size
float *d_tempProj;
cudaMalloc((void**)&d_tempProj, numBytesSingleProj);

// malloc in device: projection of the whole bin
float *d_img ,*d_img1;
cudaMalloc((void**)&d_img, numBytesImg);
cudaMalloc((void**)&d_img1, numBytesImg);

// malloc in device: another image pointer, for single view 
float *d_tempImg, *d_tempImg2, *d_onesImg;
cudaMalloc(&d_tempImg, numBytesImg);
cudaMalloc(&d_tempImg2, numBytesImg);
cudaMalloc(&d_onesImg, numBytesImg);
float angle, volume, flow;

//Malloc forward and inverted DVFs in device
float *d_mx, *d_my, *d_mz;
cudaMalloc(&d_mx, numBytesImg);
cudaMalloc(&d_my, numBytesImg);
cudaMalloc(&d_mz, numBytesImg);
float *d_mx2, *d_my2, *d_mz2;
cudaMalloc(&d_mx2, numBytesImg);
cudaMalloc(&d_my2, numBytesImg);
cudaMalloc(&d_mz2, numBytesImg);

// setup output images
OUT_IMG = mxCreateNumericMatrix(0, 0, mxSINGLE_CLASS, mxREAL);
const mwSize outDim[4] = {(mwSize)nx, (mwSize)ny, (mwSize)nz, (mwSize)n_bin};
mxSetDimensions(OUT_IMG, outDim, 4);
mxSetData(OUT_IMG, mxMalloc(numBytesImg * n_bin));
float *h_outimg = (float*)mxGetData(OUT_IMG);



for (int ibin = 0; ibin < n_bin; ibin++){
    if (outIter % 2 == 1)
        break;
    if (outIter >= 0)
    {
        cudaMemcpy(d_img, h_img + ibin * numImg, numBytesImg, cudaMemcpyHostToDevice);
    }
    else{
        if (ibin == 0){
            cudaMemcpy(d_img1, h_img + (n_bin - 1) * numImg, numBytesImg, cudaMemcpyHostToDevice);
        }
        if (ibin == 0){
            volume = ref_volumes[0] - ref_volumes[n_bin - 1];
            flow = ref_flows[0] - ref_flows[n_bin - 1];
        }
        else{
            volume = ref_volumes[ibin] - ref_volumes[ibin - 1];
            flow = ref_flows[ibin] - ref_flows[ibin - 1];
        }
        kernel_forwardDVF<<<gridSize_img, blockSize>>>(d_mx, d_my, d_mz, d_alpha_x, d_alpha_y, d_alpha_z, d_beta_x, d_beta_y, d_beta_z, d_const_x, d_const_y, d_const_z, volume, flow, isConst, nx, ny, nz);
        cudaDeviceSynchronize();

        host_deformation(d_img, d_img1, d_mx, d_my, d_mz, nx, ny, nz);
    }

    for (int iter = 0; iter < n_iter; iter++){ // iteration
        processBar(ibin, n_bin, iter, n_iter);
        for (int i_view = n_views[ibin]; i_view < n_views[ibin + 1]; i_view++){ // view
        
            angle = angles[i_view];
            if (isConst)
            {
                volume = volumes[i_view]; // - ref_volumes[ibin];
                flow = flows[i_view]; // - ref_flows[ibin];
            }
            else
            {
                volume = volumes[i_view] - ref_volumes[ibin];
                flow = flows[i_view] - ref_flows[ibin];
            }
            
            // generate forwards DVFs: d_mx, d_my, d_mz and inverted DVFs: d_mx2, d_my2, d_mz2
            kernel_forwardDVF<<<gridSize_img, blockSize>>>(d_mx, d_my, d_mz, d_alpha_x, d_alpha_y, d_alpha_z, d_beta_x, d_beta_y, d_beta_z, d_const_x, d_const_y, d_const_z, volume, flow, isConst, nx, ny, nz);
            cudaDeviceSynchronize();

            host_invertDVF(d_mx2, d_my2, d_mz2, d_mx, d_my, d_mz, nx, ny, nz, 10);
            host_deformation(d_tempImg, d_img, d_mx2, d_my2, d_mz2, nx, ny, nz);

            // projection of deformed image from initial guess
            kernel_projection<<<gridSize_singleProj, blockSize>>>(d_tempProj, d_tempImg, angle, SO, SD, da, na, ai, db, nb, bi, nx, ny, nz); // TBD
            cudaDeviceSynchronize();

            // difference between true projection and projection from initial guess
            // update d_tempProj instead of malloc a new one
            cudaMemcpy(d_proj, h_proj + i_view * numSingleProj, numBytesSingleProj, cudaMemcpyHostToDevice);

            kernel_add<<<gridSize_singleProj, blockSize>>>(d_tempProj, d_proj, na, nb, 1, -1);
            cudaDeviceSynchronize();


            // backprojecting the difference of projections
            kernel_backprojection(d_tempImg, d_tempProj, angle, SO, SD, da, na, ai, db, nb, bi, nx, ny, nz);

            // calculate the ones backprojection data
            kernel_initial<<<gridSize_img, blockSize>>>(d_onesImg, nx, ny, nz, 1);
            cudaDeviceSynchronize();

            kernel_projection<<<gridSize_singleProj, blockSize>>>(d_tempProj, d_onesImg, angle, SO, SD, da, na, ai, db, nb, bi, nx, ny, nz);
            cudaDeviceSynchronize();

            kernel_backprojection(d_onesImg, d_tempProj, angle, SO, SD, da, na, ai, db, nb, bi, nx, ny, nz);
            cudaDeviceSynchronize();

            // weighting
            kernel_division<<<gridSize_img, blockSize>>>(d_tempImg, d_onesImg, nx, ny, nz);
            cudaDeviceSynchronize();

            host_deformation(d_tempImg2, d_tempImg, d_mx, d_my, d_mz, nx, ny, nz);

            // updating
            kernel_update<<<gridSize_img, blockSize>>>(d_img, d_tempImg2, nx, ny, nz, lambda);
            cudaDeviceSynchronize();          
            // mexPrintf("13");mexEvalString("drawnow;");
        }  
    }
    cudaMemcpy(d_img1, d_img, numBytesImg, cudaMemcpyDeviceToDevice);
    cudaMemcpy(h_outimg + ibin * numImg, d_img, numBytesImg, cudaMemcpyDeviceToHost);
}

for (int ibin = n_bin - 1; ibin > -1; ibin--){
    if (outIter % 2 == 0)
        break;
    if (outIter == 0)
    {
        cudaMemcpy(d_img, h_img + ibin * numImg, numBytesImg, cudaMemcpyHostToDevice);
    }
    else{
        if (ibin == n_bin - 1){
            cudaMemcpy(d_img1, h_img, numBytesImg, cudaMemcpyHostToDevice);
        }
        if (ibin == n_bin - 1){
            volume = ref_volumes[n_bin - 1] - ref_volumes[0];
            flow = ref_flows[n_bin - 1] - ref_flows[0];
        }
        else{
            volume = ref_volumes[ibin] - ref_volumes[ibin + 1];
            flow = ref_flows[ibin] - ref_flows[ibin + 1];
        }
        kernel_forwardDVF<<<gridSize_img, blockSize>>>(d_mx, d_my, d_mz, d_alpha_x, d_alpha_y, d_alpha_z, d_beta_x, d_beta_y, d_beta_z, d_const_x, d_const_y, d_const_z, volume, flow, isConst, nx, ny, nz);
        cudaDeviceSynchronize();

        host_deformation(d_img, d_img1, d_mx, d_my, d_mz, nx, ny, nz); 
    }

    for (int iter = 0; iter < n_iter; iter++){ // iteration
        processBar(n_bin - ibin - 1, n_bin, iter, n_iter);
        for (int i_view = n_views[ibin]; i_view < n_views[ibin + 1]; i_view++){ // view
        
            angle = angles[i_view];
            if (isConst)
            {
                volume = volumes[i_view]; // - ref_volumes[ibin];
                flow = flows[i_view]; // - ref_flows[ibin];
            }
            else
            {
                volume = volumes[i_view] - ref_volumes[ibin];
                flow = flows[i_view] - ref_flows[ibin];
            }
            
            // generate forwards DVFs: d_mx, d_my, d_mz and inverted DVFs: d_mx2, d_my2, d_mz2
            kernel_forwardDVF<<<gridSize_img, blockSize>>>(d_mx, d_my, d_mz, d_alpha_x, d_alpha_y, d_alpha_z, d_beta_x, d_beta_y, d_beta_z, d_const_x, d_const_y, d_const_z, volume, flow, isConst, nx, ny, nz);
            cudaDeviceSynchronize();
            host_invertDVF(d_mx2, d_my2, d_mz2, d_mx, d_my, d_mz, nx, ny, nz, 10);
            host_deformation(d_tempImg, d_img, d_mx2, d_my2, d_mz2, nx, ny, nz);

            // projection of deformed image from initial guess
            kernel_projection<<<gridSize_singleProj, blockSize>>>(d_tempProj, d_tempImg, angle, SO, SD, da, na, ai, db, nb, bi, nx, ny, nz);
            cudaDeviceSynchronize();

            // difference between true projection and projection from initial guess
            // update d_tempProj instead of malloc a new one
            cudaMemcpy(d_proj, h_proj + i_view * numSingleProj, numBytesSingleProj, cudaMemcpyHostToDevice);

            kernel_add<<<gridSize_singleProj, blockSize>>>(d_tempProj, d_proj, na, nb, 1, -1);
            cudaDeviceSynchronize();

            // backprojecting the difference of projections
            kernel_backprojection(d_tempImg, d_tempProj, angle, SO, SD, da, na, ai, db, nb, bi, nx, ny, nz);

            // calculate the ones backprojection data
            kernel_initial<<<gridSize_img, blockSize>>>(d_onesImg, nx, ny, nz, 1);
            cudaDeviceSynchronize();

            kernel_projection<<<gridSize_singleProj, blockSize>>>(d_tempProj, d_onesImg, angle, SO, SD, da, na, ai, db, nb, bi, nx, ny, nz);
            cudaDeviceSynchronize();

            kernel_backprojection(d_onesImg, d_tempProj, angle, SO, SD, da, na, ai, db, nb, bi, nx, ny, nz);
            cudaDeviceSynchronize();

            // weighting
            kernel_division<<<gridSize_img, blockSize>>>(d_tempImg, d_onesImg, nx, ny, nz);
            cudaDeviceSynchronize();
            
            host_deformation(d_tempImg2, d_tempImg, d_mx, d_my, d_mz, nx, ny, nz);

            // updating
            kernel_update<<<gridSize_img, blockSize>>>(d_img, d_tempImg2, nx, ny, nz, lambda);
            cudaDeviceSynchronize();          
            // mexPrintf("13");mexEvalString("drawnow;");
        }  
    }
    cudaMemcpy(d_img1, d_img, numBytesImg, cudaMemcpyDeviceToDevice);
    cudaMemcpy(h_outimg + ibin * numImg, d_img, numBytesImg, cudaMemcpyDeviceToHost);
}

cudaFree(d_mx);
cudaFree(d_my);
cudaFree(d_mz);
cudaFree(d_mx2);
cudaFree(d_my2);
cudaFree(d_mz2);

cudaFree(d_proj);
cudaFree(d_tempImg);
cudaFree(d_tempImg2);

cudaFree(d_onesImg);
cudaFree(d_tempProj);

cudaFree(d_img);
cudaFree(d_img1);
cudaDeviceReset();
return;
}

