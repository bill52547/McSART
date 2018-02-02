// header for Alternately Updating Motion and Image SART method
#ifndef _AUMISART_H
#define _AUMISART_H
#include "universal.h"

#include "cu_add.h"
#include "cu_division.h"
#include "cu_initial.h"
#include "cu_projection.h"
#include "cu_backprojection.h"
#include "cu_deform.h"
#include "cu_update_udvf.h"
#include "processBar.h"

__host__ void host2_AUMISART(float *h_outimg, float *h_outnorm, float *h_img, float *h_proj, int nx, int ny, int nz, int na, int nb, int outIter, int n_views, float da, float db, float ai, float bi, float SO, float SD, float dx, float lambda, float* volumes, float* flows, float* err_weights, float* angles);

__host__ void host_AUMISART(float *h_outimg, float *h_outproj,float *h_outnorm, float*, float *h_img, float *h_proj, int nx, int ny, int nz, int na, int nb, int outIter, int n_views, int n_iter, int *op_iter, float da, float db, float ai, float bi, float SO, float SD, float dx, float lambda, float* volumes, float* flows, float* err_weights, float* angles);

__host__ void host_AUMISART(float *h_outimg, float *h_outproj, float *h_outnorm, float *h_outalphax, float *h_img, float *h_proj, int nx, int ny, int nz, int na, int nb, int outIter, int n_views, int n_iter, int *op_iter, float da, float db, float ai, float bi, float SO, float SD, float dx, float lambda, float* volumes, float* flows, float* err_weights, float* angles, float *ax, float *ay, float *az, float *bx, float *by, float *bz);
#endif // _AUMISART_H