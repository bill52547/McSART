#ifndef _CU_BACKPROJECTION_H
#define _CU_BACKPROJECTION_H
#include "universal.h"

__host__ void host2_backprojection(float *d_img, float *d_proj, float *float_para, int *int_para);

__host__ void host_backprojection(float *d_img, float *d_proj, float angle,float SO, float SD, float da, int na, float ai, float db, int nb, float bi, int nx, int ny, int nz);

__global__ void kernel_backprojection(float *img, cudaTextureObject_t tex_proj, float angle, float SO, float SD, int nu, int nv, float du, float dv, float ui, float vi, int nx, int ny, int nz);
#endif //_CU_BACKPROJECTION_H