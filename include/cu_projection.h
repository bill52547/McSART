#ifndef _CU_PROJECTION_H
#define _CU_PROJECTION_H
#include "universal.h"

__host__  void host2_projection(float *d_proj, float *d_img, float *float_para, int *int_para);

__host__  void host_projection(float *d_proj, float *d_img, float angle, float SO, float SD, float da, int na, float ai, float db, int nb, float bi, int nx, int ny, int nz);

__global__ void kernel_projection(float *proj, float *img, float angle, float SO, float SD, float da, int na, float ai, float db, int nb, float bi, int nx, int ny, int nz);

#endif // _CU_PROJECTION_H