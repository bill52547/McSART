clc
currentpath = pwd;
cd src
mexcuda -I../include ./SART_cuda.cu kernel_add.cu kernel_backprojection.cu kernel_deformation.cu kernel_division.cu kernel_initial.cu...
kernel_projection.cu kernel_update.cu kernel_forwardDVF.cu kernel_invertDVF.cu processBar.cu -lcublas  -outdir ../bin 
cd(currentpath)
