<<<<<<< HEAD
#ifndef _HOST_DEFORMATION_H
#define _HOST_DEFORMATION_H
#include "universal.h"
#include "kernel_deformation.h"
void host_deformation(float *img1, float *img, float *mx, float *my, float *mz, int nx, int ny, int nz);
#endif _HOST_DEFORMATION_H
=======
#include "kernel_deformation.h"
#include "mex.h"
void host_deformation(float *img1, float *img, float *mx, float *my, float *mz, int nx, int ny, int nz);
>>>>>>> 7f9f4366d3c0653b395c1a7c8b31341946826fbd
