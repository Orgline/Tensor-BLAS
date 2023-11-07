#include "../include/TensorBLAS.h"

long int n, k, nb;
double alpha, beta;
bool checkFlag = false;

int parseArguments(int argc,char *argv[])
{
    if(argc < 7)
    {
        printf("Needs m, n and nb, alpha, beta, check as inputs\n");
        return -1;
    }
    n = atoi(argv[1]);
    k = atoi(argv[2]);
    nb = atoi(argv[3]);
    alpha = atof(argv[4]);
    beta= atof(argv[5]);
    if (atoi(argv[6]) == 1)
        checkFlag = true;
    else
        checkFlag = false;
    return 0;
}


__global__
void sSubstract(long int m, long int n, double* dA, long int lda, double* dB, long int ldb)
{
    long int i = (long)threadIdx.x + (long)blockDim.x*  (long)blockIdx.x;
	long int j =  (long)threadIdx.y +  (long)blockDim.y * (long)blockIdx.y;
	if (i<m && j<n) 
    {
		dA[i+j*ldb] = dA[i+j*lda] - dB[i+j*ldb];
        if(i == 40000 && j %1000 == 0)
        {    __syncthreads();
            printf("i = %d, j = %d, dA = %lf, dB=%lf\n",i, j, dA[i+j*lda], dB[i+j*ldb]);
            __syncthreads();
        }
        if(j == 40000 && i %1000 == 0)
        {    __syncthreads();
            printf("i = %d, j =%d, second dA = %lf, dB=%lf\n",i, j, dA[i+j*lda], dB[i+j*ldb]);
            __syncthreads();
        }
    }
}

int main(int argc,char *argv[])
{
    if(parseArguments(argc, argv)==-1)
        return 0;
    cublasHandle_t cublas_handle;
    cublasCreate(&cublas_handle);
	// cumpsgemm::handle_t cumpsgemm_handle;
	// cumpsgemm::create(cumpsgemm_handle);

    double   *A;
    cudaMalloc(&A, sizeof(double)*n*k);

    double *C;
    cudaMalloc(&C, sizeof(double)*n*n);

    dim3 gridc((n+31)/32, (n+31)/32);
    dim3 blockc(32,32);


    setInitialValueDouble<<<gridc, blockc>>>(n, n ,C, n, 1.0);
    // __half *hwork;
    // cudaMalloc(&hwork, sizeof(__half)*2*n*k);
    
    cublasGemmEx(cublas_handle, CUBLAS_OP_N, CUBLAS_OP_T, n, n, k,
                &alpha, A, CUDA_R_64F, n, A, CUDA_R_64F, n,
                &beta, C, CUDA_R_64F, n, CUDA_R_64F,
                CUBLAS_GEMM_DEFAULT_TENSOR_OP);

    startTimer();
    // dim3 grid((n+31)/32, (k+31)/32);
    // dim3 block(32,32);
    // s2h<<<grid, block>>>(n, k, A, n, hwork, n);
    cublasGemmEx(cublas_handle, CUBLAS_OP_N, CUBLAS_OP_T, n, n, k,
                &alpha, A, CUDA_R_64F, n, A, CUDA_R_64F, n,
                &beta, C, CUDA_R_64F, n, CUDA_R_64F,
                CUBLAS_GEMM_DEFAULT_TENSOR_OP);  

    // cublasGemmEx(cublas_handle, CUBLAS_OP_N, CUBLAS_OP_T, n, n, k,
    //             &alpha, A, CUDA_R_64F, n, A+n*k, CUDA_R_64F, n,
    //             &beta, C, CUDA_R_64F, n, CUDA_R_64F,
    //             CUBLAS_GEMM_DEFAULT_TENSOR_OP);
    float ms = stopTimer();
    
    printf("tc_gemm %dx%d takes %f ms, flops is %f\n", n, k,ms, 1.0*n*n*k/ms/1e9);

    //printf("snorm C = %f\n", snorm(n, n, C, n));
    setInitialValueDouble<<<gridc, blockc>>>(n, n ,C, n, 1.0);

    generateUniformMatrixDouble(A, n, k);
    //setInitialValue<<<gridc, blockc>>>(n, k ,A, n, 0.5);
    //printf("snorm A = %f\n", snorm(n, k, A, n));

    
    startTimer();
    tc_ozimmu_syrk(cublas_handle, n, k, alpha, A, n, beta, C, n, nb);
    ms = stopTimer();
    
    printf("tc_ozIMMU_syrk %dx%d takes %f ms, flops is %f\n", n, k,ms, 1.0*n*n*k/ms/1e9);
    
    copy_lower_to_upperDouble<<<gridc, blockc>>>(n, C, n);
    //printf("snorm C = %f\n", snorm(n, n, C, n));
    //printMatrixDeviceBlock("C.csv", n, n, C, n);
    if(checkFlag)
    {
        double *tC;
        cudaMalloc(&tC, sizeof(double)*n*n);
        setInitialValueDouble<<<gridc, blockc>>>(n, n ,tC, n, 1.0);


        cublasDsyrk(cublas_handle,
            CUBLAS_FILL_MODE_LOWER, CUBLAS_OP_N,
            n, k,
            &alpha,
            A, n,
            &beta,
            tC, n
        );

    
        printMatrixDeviceBlockDouble("C_p.csv", n, 1, C, n);
        copy_lower_to_upperDouble<<<gridc, blockc>>>(n, tC, n);
        //printf("snorm tC = %f\n", snorm(n, n, tC, n));
        printMatrixDeviceBlockDouble("C_a.csv", n, 1, tC, n);
        
        double sonedouble = 1.0, snegonedobule = -1.0;
        cublasDgeam(cublas_handle, CUBLAS_OP_N, CUBLAS_OP_N, n, n,
                &sonedouble, C, n, &snegonedobule, tC, n,
                C, n);
        
        //sSubstract<<<gridc, blockc>>>(n, n, C, n, tC, n);
        cudaError_t err = cudaPeekAtLastError();
        if (err != cudaSuccess) {
            printf("Kernel launch error: %s\n", cudaGetErrorString(err));
            return 1;
        }
        
        // Synchronize the device and check for kernel execution errors
        err = cudaDeviceSynchronize();
        if (err != cudaSuccess) {
            printf("Kernel execution error: %s\n", cudaGetErrorString(err));
            return 1;
        }
        printf("free memory is %d GB\n", free_mem()/1024/1024/1024);
        // printf("snorm C = %f\n", snorm(n, n, C, n));
        // printf("snorm tC = %f\n", snorm(n, n, tC, n));
        //cudaDeviceSynchronize();
        printf("Forward error is %.6e\n",snormDouble(n, n, C, n)/snormDouble(n, n, tC, n));
        cudaFree(tC);
    }
    cudaFree(C);
    // cudaFree(hwork);
    cudaFree(A);
    
    //}

}

