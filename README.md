# TensorBLAS

TensorBLAS is a library that supports BLAS3 routines including SYMM, SYRK, SYR2K, TRMM, and TRSM on Tensor Cores

···
export CUDA_PATH=/mnt/nfs/packages/x86_64/cuda/cuda-11.2
cd testing
make
./testing_syrk 16384 16384 256 1 0 1
···
