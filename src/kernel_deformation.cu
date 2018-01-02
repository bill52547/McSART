__global__ void kernel_deformation(float *img1, cudaTextureObject_t tex_img, float *mx2, float *my2, float *mz2, int nx, int ny, int nz){
    int ix = 16 * blockIdx.x + threadIdx.x;
    int iy = 16 * blockIdx.y + threadIdx.y;
    int iz = 4 * blockIdx.z + threadIdx.z;
    if (ix >= nx || iy >= ny || iz >= nz)
        return;
    int id = iy + ix * ny + iz * nx * ny;
    float xi = iy + 0.5f + my2[id];
    float yi = ix + 0.5f + mx2[id];
    float zi = iz + 0.5f + mz2[id];
    img1[id] = tex3D<float>(tex_img, xi, yi, zi);
}