// this program is try to do the SART program for a single bin
// #include "universe_header.h"
#ifndef _SART_CUDA_H
#define _SART_CUDA_H
#include "mex.h"
#include "matrix.h"
#include "gpu/mxGPUArray.h"
#include <stdlib.h>
#include <cuda_runtime.h>
#include <math.h>
#include <iostream>
#include "cublas_v2.h"
#define MAX(a,b) (((a) > (b)) ? a : b)
#define MIN(a,b) (((a) < (b)) ? a : b)
#define ABS(x) ((x) > 0 ? (x) : -(x))
#define PI 3.141592653589793
// Set thread block size
#define BLOCKWIDTH 16
#define BLOCKHEIGHT 16 
#define BLOCKDEPTH 4

#include "kernel_add.h" // kernel_add(d_proj1, d_proj, iv, na, nb, -1);
#include "kernel_division.h" // kernel_division(d_img1, d_img, nx, ny, nz);
#include "kernel_initial.h" // kernel_initial(img, nx, ny, nz, value);
#include "kernel_update.h"
#include "kernel_projection.h"
#include "kernel_backprojection.h"
#include "host_deformation.h"
#include "kernel_forwardDVF.h"
#include "kernel_invertDVF.h" 
#include "processBar.h"
#include "host_load_field.h"
#endif // _SART_CUDA_H