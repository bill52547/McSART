#include "AUMISART.h" // consists all required package and functions

__host__ void host_AUMISART(float *h_outimg, float *h_outnorm, float *h_outalphax, float *h_img, float *h_proj, int nx, int ny, int nz, int na, int nb, int outIter, int n_views, int n_iter, int *op_iter, float da, float db, float ai, float bi, float SO, float SD, float dx, float lambda, float* volumes, float* flows, float* err_weights, float* angles)
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
    cudaMemcpy(d_img, h_img, numBytesImg, cudaMemcpyHostToDevice);
    // host_initial(d_img, nx, ny, nz, 0.0f);
    host_initial(d_alpha_x, nx, ny, nz, 0.0f);
    host_initial(d_alpha_y, nx, ny, nz, 0.0f);
    host_initial(d_alpha_z, nx, ny, nz, 0.0f);
    host_initial(d_beta_x, nx, ny, nz, 0.0f);
    host_initial(d_beta_y, nx, ny, nz, 0.0f);
    host_initial(d_beta_z, nx, ny, nz, 0.0f);
    mexPrintf("Start iteration\n");

    cublasStatus_t stat;
    cublasHandle_t handle;
    stat = cublasCreate(&handle); 
    float tempNorm, tempNorm0;

    for (int i_iter = 0; i_iter < n_iter; i_iter ++)
    {
        if (op_iter[i_iter] == 1)
        {   
            for (int i_view = 0; i_view < n_views; i_view ++)
            {   
                mexPrintf("iIter = %d / %d, and iView = %d / %d.\n", i_iter + 1, n_iter, i_view + 1, n_views); mexEvalString("drawnow;");

                host_projection(d_proj_temp, d_img, angles[i_view], SO, SD, da, na, ai, db, nb, bi, nx, ny, nz);
                cudaMemcpy(d_proj, h_proj + na * nb * i_view, numBytesProj, cudaMemcpyHostToDevice);
                stat = cublasSnrm2(handle, na * nb, d_proj, 1, &tempNorm0);

                host_add(d_proj, d_proj_temp, na, nb, 1, -1.0);
                stat = cublasSnrm2(handle, na * nb, d_proj, 1, &tempNorm);
                h_outnorm[i_iter * n_views + i_view] = tempNorm / tempNorm0;
            
                host_backprojection(d_img_temp, d_proj, angles[i_view], SO, SD, da, na, ai, db, nb, bi, nx, ny, nz);

                host_initial(d_img_ones, nx, ny, nz, 1.0f);
                host_projection(d_proj_ones, d_img_ones, angles[i_view], SO, SD, da, na, ai, db, nb, bi, nx, ny, nz);
                host_backprojection(d_img_ones, d_proj_ones, angles[i_view], SO, SD, da, na, ai, db, nb, bi, nx, ny, nz);

                host_division(d_img_temp, d_img_ones, nx, ny, nz);

                host_add(d_img, d_img_temp, nx, ny, nz, lambda);
            }
            cudaMemcpy(h_outimg, d_img, numBytesImg, cudaMemcpyDeviceToHost);
        }
        else
        {   
            if (i_iter == 0)
                cudaMemcpy(d_img, h_img, numBytesImg, cudaMemcpyHostToDevice);
            else
                cudaMemcpy(d_img, h_outimg, numBytesImg, cudaMemcpyHostToDevice);   
            for (int i_view = 1; i_view < n_views; i_view ++)
            {   
                mexPrintf("iIter = %d / %d, and iView = %d / %d.", i_iter + 1, n_iter, i_view + 1, n_views); 

                host_projection(d_proj_temp, d_img, angles[i_view], SO, SD, da, na, ai, db, nb, bi, nx, ny, nz);
                cudaMemcpy(d_proj, h_proj + na * nb * i_view, numBytesProj, cudaMemcpyHostToDevice);
                stat = cublasSnrm2(handle, na * nb, d_proj, 1, &tempNorm0);
                float vd = volumes[i_view] - volumes[i_view - 1];
                float fd = flows[i_view] - flows[i_view - 1];
                host_add(d_proj, d_proj_temp, na, nb, 1, -1.0f); // new b
                host_initial(d_img0, nx, ny, nz, 0.0f);
                host_add2(d_img0, d_alpha_y, nx, ny, nz, d_img, vd, 1);
                host_add2(d_img0, d_alpha_x, nx, ny, nz, d_img, vd, 2);
                host_add2(d_img0, d_alpha_z, nx, ny, nz, d_img, vd, 3);
                host_add2(d_img0, d_beta_y, nx, ny, nz, d_img, fd, 1);
                host_add2(d_img0, d_beta_x, nx, ny, nz, d_img, fd, 2);
                host_add2(d_img0, d_beta_z, nx, ny, nz, d_img, fd, 3);
                host_projection(d_proj_temp, d_img0, angles[i_view], SO, SD, da, na, ai, db, nb, bi, nx, ny, nz);
                host_add(d_proj, d_proj_temp, na, nb, 1, 1.0f); // new b

                stat = cublasSnrm2(handle, na * nb, d_proj, 1, &tempNorm);
                h_outnorm[i_iter * n_views + i_view] = tempNorm / tempNorm0;
                mexPrintf("error on projection = %f\n", tempNorm / tempNorm0);mexEvalString("drawnow;");
                host_backprojection(d_img_temp, d_proj, angles[i_view], SO, SD, da, na, ai, db, nb, bi, nx, ny, nz);

                host_initial2(d_img_ones, nx, ny, nz, d_img, -vd, -fd);
                host_projection(d_proj_ones, d_img_ones, angles[i_view], SO, SD, da, na, ai, db, nb, bi, nx, ny, nz);
                host_backprojection(d_img_ones, d_proj_ones, angles[i_view], SO, SD, da, na, ai, db, nb, bi, nx, ny, nz);
                host_division(d_img_temp, d_img_ones, nx, ny, nz);

                host_add2(d_alpha_y, d_img_temp, nx, ny, nz, d_img, volumes[i_view - 1] - volumes[i_view], 1);
                host_add2(d_alpha_x, d_img_temp, nx, ny, nz, d_img, volumes[i_view - 1] - volumes[i_view], 2);
                host_add2(d_alpha_z, d_img_temp, nx, ny, nz, d_img, volumes[i_view - 1] - volumes[i_view], 3);
                host_add2(d_beta_y, d_img_temp, nx, ny, nz, d_img, flows[i_view - 1] - flows[i_view], 1);
                host_add2(d_beta_x, d_img_temp, nx, ny, nz, d_img, flows[i_view - 1] - flows[i_view], 2);
                host_add2(d_beta_z, d_img_temp, nx, ny, nz, d_img, flows[i_view - 1] - flows[i_view], 3);

                // cudaMemcpy(d_img0, d_img, numBytesImg, cudaMemcpyDeviceToDevice);
                // host_add2(d_img, d_alpha_x, nx, ny, nz, d_img0, volumes[i_view - 1] - volumes[i_view], 1);
                // host_add2(d_img, d_alpha_y, nx, ny, nz, d_img0, volumes[i_view - 1] - volumes[i_view], 2);                
                // host_add2(d_img, d_alpha_z, nx, ny, nz, d_img0, volumes[i_view - 1] - volumes[i_view], 3);                
                // host_add2(d_img, d_beta_x, nx, ny, nz, d_img0, flows[i_view - 1] - flows[i_view], 1);
                // host_add2(d_img, d_beta_y, nx, ny, nz, d_img0, flows[i_view - 1] - flows[i_view], 2);                
                // host_add2(d_img, d_beta_z, nx, ny, nz, d_img0, flows[i_view - 1] - flows[i_view], 3);  
                // break;
            }

        }
    }
    cudaMemcpy(h_outalphax, d_alpha_x, numBytesImg, cudaMemcpyDeviceToHost);
            
    cudaMemcpy(h_outimg, d_img, numBytesImg, cudaMemcpyDeviceToHost);   

    cudaFree(d_img);
    cudaFree(d_img);
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
    cublasDestroy(handle);

    cudaDeviceReset();
}
