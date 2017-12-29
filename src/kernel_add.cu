__global__ void kernel_add(float *proj1, float *proj, int na, int nb, int nv, float weight){
    int ia = 16 * blockIdx.x + threadIdx.x;
    int ib = 16 * blockIdx.y + threadIdx.y;
    int iv = 4 * blockIdx.z + threadIdx.z;
    if (ia >= na || ib >= nb || iv >= nv)
        return;
    proj1[ia + ib * na + iv * na * nb] += proj[ia + ib * na + iv * na * nb] * weight;
}

// __global__ void kernel_add(cudaArray *proj1, cudaArray *proj, int iv, int na, int nb, float weight){
//     int ia = 16 * blockIdx.x + threadIdx.x;
//     int ib = 16 * blockIdx.y + threadIdx.y;
//     if (ia >= na || ib >= nb)
//         return;
//     proj1[ia + ib * na] += proj[ia + ib * na + iv * na * nb] * weight;
// }
