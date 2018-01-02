#ifndef _HOST_MCSART_H
#define _HOST_MCSART_H
#include "matrix.h"
#include "gpu/mxGPUArray.h"
#include <stdlib.h>
#include <cuda_runtime.h>
#include <iostream>
#include "cublas_v2.h"

#include "universal.h"
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


#endif // _HOST_MCSART_H