clc
mexcuda -Iinclude src/entry_AUMISART.cu src/AUMISART.cu src/cu_add.cu src/cu_backprojection.cu ...
src/cu_deform.cu src/cu_division.cu src/cu_initial.cu src/cu_projection.cu ...
src/cu_update_udvf.cu src/processBar.cu -lcublas -outdir bin 
% cd(currentpath)


% mexcuda -Iinclude src/mex_backprojection.cu src/cu_backprojection.cu -outdir bin