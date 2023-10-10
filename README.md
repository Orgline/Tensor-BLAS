# TensorBLAS

TensorBLAS is a library that supports BLAS3 routines including SYMM, SYRK, SYR2K, TRMM, and TRSM on Tensor Cores

The cuMpSGEMM (https://github.com/enp1s0/cuMpSGEMM/tree/20fb66bd62ff66de16e523a0afb1287c7bf584cc) should be compiled before compiling the Tensor-BLAS library.
To test the supported routines:
```
export CUDA_PATH=/mnt/nfs/packages/x86_64/cuda/cuda-11.2
cd testing
make
./testing_syrk 16384 16384 256 1 0 1
```
For testing_syrk, the parameters mean n, k, tiling size (typically 256 or 512 can give the best performance), alpha, beta, checkflag (1: check, 0: don't check).

To call the Tensor-BLAS routines, using the following command to generate a dynamic link library.
```
export CUDA_PATH=/mnt/nfs/packages/x86_64/cuda/cuda-11.2
cd build
make
```
