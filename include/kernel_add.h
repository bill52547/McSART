#ifndef _KERNEL_ADD_H
#define _KERNEL_ADD_H
__global__ void kernel_add(float *proj1, float *proj, int iv, int na, int nb, float weight);
#endif // _KERNEL_ADD_H