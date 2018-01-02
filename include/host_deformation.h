#ifndef _HOST_DEFORMATION_H
#define _HOST_DEFORMATION_H
#include "universal.h"
#include "kernel_deformation.h"
void host_deformation(float *img1, float *img, float *mx, float *my, float *mz, int nx, int ny, int nz);
#endif //_HOST_DEFORMATION_H
