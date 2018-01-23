#include "AUMISART.h" // consists all required package and functions

__host__ void host_AUMISART(float *h_outimg, float *h_outnorm, float *h_img, float *h_proj, int nx, int ny, int nz, int na, int nb, int outIter, int n_views, int n_iter, float da, float db, float ai, float bi, float SO, float SD, float dx, float lambda, float* volumes, float* flows, float* err_weights, float* angles)
{
    float *d_img, *d_img0, *d_img_temp, *d_proj, *d_proj_temp, *d_img_ones, *d_proj_ones;
    int numBytesImg = nx * ny * nz * sizeof(float);
    int numBytesProj = na * nb * sizeof(float);
    cudaMalloc((void**)&d_img, numBytesImg);
    cudaMalloc((void**)&d_img0, numBytesImg);
    cudaMalloc((void**)&d_img_temp, numBytesImg);
    cudaMalloc((void**)&d_img_ones, numBytesImg);
    cudaMalloc((void**)&d_proj, numBytesProj);
    cudaMalloc((void**)&d_proj_temp, numBytesProj);
    cudaMalloc((void**)&d_proj_ones, numBytesProj);

    float *d_alpha_x, *d_alpha_y, *d_alpha_z, *d_beta_x, *d_beta_y, *d_beta_z;
    cudaMalloc((void**)&d_alpha_x, numBytesImg);
    cudaMalloc((void**)&d_alpha_y, numBytesImg);
    cudaMalloc((void**)&d_alpha_z, numBytesImg);
    cudaMalloc((void**)&d_beta_x, numBytesImg);
    cudaMalloc((void**)&d_beta_y, numBytesImg);
    cudaMalloc((void**)&d_beta_z, numBytesImg);

    host_initial(d_img, nx, ny, nz, 0.0f);
    host_initial(d_alpha_x, nx, ny, nz, 0.0f);
    host_initial(d_alpha_y, nx, ny, nz, 0.0f);
    host_initial(d_alpha_z, nx, ny, nz, 0.0f);
    host_initial(d_beta_x, nx, ny, nz, 0.0f);
    host_initial(d_beta_y, nx, ny, nz, 0.0f);
    host_initial(d_beta_z, nx, ny, nz, 0.0f);
    mexPrintf("Start iteration\n");
    mexPrintf("n_iter = %d\n", n_iter);
    mexPrintf("n_view = %d\n", n_views);
    //mexPrintf("Start iteration\n");
    
    for (int iter = 0; iter < n_iter; iter ++)
    {
        for (int i_view = 0; i_view < n_views; i_view ++)
        {   
            processBar(i_view, n_views, iter, n_iter);
            float volume_diff, flow_diff;
            if (i_view > 0)
            {
                volume_diff = volumes[i_view] - volumes[i_view - 1];
                flow_diff = flows[i_view] - flows[i_view - 1];
            
                host_deform(d_img_temp, d_img, nx, ny, nz, volume_diff, flow_diff, d_alpha_x, d_alpha_y, d_alpha_z, d_beta_x, d_beta_y, d_beta_z);
                cudaMemcpy(d_img, d_img_temp, numBytesImg, cudaMemcpyDeviceToDevice);
            }
            
            host_projection(d_proj_temp, d_img, angles[i_view], SO, SD, da, na, ai, db, nb, bi, nx, ny, nz);
            cudaMemcpy(d_proj, h_proj + na * nb * i_view, numBytesProj, cudaMemcpyHostToDevice);

            host_add(d_proj, d_proj_temp, nx, ny, nz, -1.0);

            host_backprojection(d_img_temp, d_proj, angles[i_view], SO, SD, na, nb, da, db, ai, bi, nx, ny, nz);

            host_initial(d_img_ones, nx, ny, nz, 1.0f);
            host_projection(d_proj_ones, d_img_ones, angles[i_view], SO, SD, da, na, ai, db, nb, bi, nx, ny, nz);
            host_backprojection(d_img_ones, d_proj_ones, angles[i_view], SO, SD, na, nb, da, db, ai, bi, nx, ny, nz);

            host_division(d_img_temp, d_img_ones, nx, ny, nz);

            host_add(d_img, d_img_temp, nx, ny, nz, lambda);

            if (i_view > 0)
            {
                host_update_udvf(d_alpha_x, d_alpha_y, d_alpha_z, d_beta_x, d_beta_y, d_beta_z, d_img, d_img0, volume_diff, flow_diff, nx, ny, nz, i_view);
            }
            cudaMemcpy(d_img0, d_img, numBytesImg, cudaMemcpyDeviceToDevice);    
        }
    }
    cudaMemcpy(h_outimg, d_img, numBytesImg, cudaMemcpyDeviceToHost);
    cudaFree(d_img);
    cudaFree(d_img0);
    cudaFree(d_img_temp);
    cudaFree(d_proj);
    cudaFree(d_proj_temp);
    cudaFree(d_img_ones);
    cudaFree(d_proj_ones);
    cudaFree(d_alpha_x);
    cudaFree(d_alpha_y);
    cudaFree(d_alpha_z);
    cudaFree(d_beta_x);
    cudaFree(d_beta_y);
    cudaFree(d_beta_z);
    cudaDeviceReset();
}
