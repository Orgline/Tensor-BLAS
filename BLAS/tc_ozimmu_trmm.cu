#include "../include/TensorBLAS.h" 


void tc_ozimmu_trmm_p2(cublasHandle_t handle, long int m, long int n, double alpha, double* A, long int lda, double* B, long int ldb, double beta, double* C, long int ldc, long int nb)
{
    // cumpsgemm::gemm_stridedBatch<double>(
	// 			ozimmu_handle,
	// 			CUBLAS_OP_N, CUBLAS_OP_N,
	// 			nb, n, nb,
	// 			&alpha,
	// 			A, lda, nb+nb*lda,
	// 			B, ldb, nb,
	// 			&beta,
	// 			C, ldc, nb,
	// 			m/nb,
	// 			ozimmu_FP16TCEC
	// 			);
    cublasGemmStridedBatchedEx(handle, CUBLAS_OP_N, CUBLAS_OP_N, 
                                nb, n, nb, &alpha,
                                A, CUDA_R_64F, lda, nb+nb*lda,
                                B, CUDA_R_64F, ldb, nb,
                                &beta, C, CUDA_R_64F, ldc, nb,
                                m/nb, CUDA_R_64F, CUBLAS_GEMM_DEFAULT_TENSOR_OP);
    for(long int i = 1; m / nb / i / 2 >= 1; i*=2)
    {
        // cumpsgemm::gemm_stridedBatch<double>(
		// 		ozimmu_handle,
		// 		CUBLAS_OP_N, CUBLAS_OP_N,
		// 		i*nb, n, i*nb,
		// 		&alpha,
		// 		A+i*nb, lda, 2*(i*nb+i*nb*lda),
		// 		B, ldb, 2*i*nb,
		// 		&sone,
		// 		C+i*nb, ldc, 2*i*nb,
		// 		m/nb/i/2,
		// 		ozimmu_FP16TCEC
		// 		);
        cublasGemmStridedBatchedEx(handle, CUBLAS_OP_N, CUBLAS_OP_N, 
                                   i*nb, n, i*nb, &alpha,
                                   A+i*nb, CUDA_R_64F, lda, 2*(i*nb+i*nb*lda),
                                   B, CUDA_R_64F, ldb, 2*i*nb,
                                   &sone, C+i*nb, CUDA_R_64F, ldc, 2*i*nb,
                                   m/nb/i/2, CUDA_R_64F, CUBLAS_GEMM_DEFAULT_TENSOR_OP);
    }
}

void tc_ozimmu_trmm_p3(cublasHandle_t handle, long int m, long int n, double alpha, double* A, long int lda, double* B, long int ldb, double* C, long int ldc, long int nb)
{
    int length;
    int64_t* matSize = find_mat_size_syrk(m, &length);
    int offset = 0;
    int rest_m = m;
    printf("%d %d %d\n", m, n, lda);
    for(int i = length; i>=0; i--)
    {
        int mm = matSize[i];
        double beta;
        if(i < length)
            offset += matSize[i + 1];
        else
            offset = 0;
        if(i == length)
            beta = szero;
        else
            beta = sone;
        if (mm % 8192 == 0)
        {
            tc_ozimmu_trmm_p2(handle, mm, n, alpha, A+offset+offset*lda, lda, B+offset, ldb, beta, C+offset, ldc, nb);
        }
        else
        {
            // cumpsgemm::gemm(
            //         ozimmu_handle,
            //         CUBLAS_OP_N,
            //         CUBLAS_OP_N,
            //         mm, n, mm,
            //         &alpha,
            //         A+offset+offset*lda, lda,
            //         B+offset, ldb,
            //         &beta,
            //         C+offset, ldc,
            //         ozimmu_FP16TCEC
            //         );
            cublasGemmEx(handle, CUBLAS_OP_N, CUBLAS_OP_N, mm, n, mm,
                &alpha, A+offset+offset*lda, CUDA_R_64F, lda, B+offset, CUDA_R_64F, ldb,
                &beta, C+offset, CUDA_R_64F, ldc, CUDA_R_64F,
                CUBLAS_GEMM_DEFAULT_TENSOR_OP);
        }
        if(i != 0)
        {
            rest_m -= mm;         
            // cumpsgemm::gemm(
            //         ozimmu_handle,
            //         CUBLAS_OP_N,
            //         CUBLAS_OP_N,
            //         rest_m, n, mm,
            //         &alpha,
            //         A+offset+mm+offset*lda, lda,
            //         B+offset, ldb,
            //         &beta,
            //         C+offset+mm, ldc,
            //         ozimmu_FP16TCEC
            //         );
            cublasGemmEx(handle, CUBLAS_OP_N, CUBLAS_OP_N, rest_m, n, mm,
                &alpha, A+offset+mm+offset*lda, CUDA_R_64F, lda, B+offset, CUDA_R_64F, ldb,
                &beta, C+offset+mm, CUDA_R_64F, ldc, CUDA_R_64F,
                CUBLAS_GEMM_DEFAULT_TENSOR_OP);
        }
    }
}
void tc_ozimmu_trmm(cublasHandle_t handle, long int m, long int n, double alpha, double* A, long int lda, double* B, long int ldb, double* C, long int ldc, long int nb)
{
    if(n%2||m%2) {
        double *A_, *C_, *B_;
        long int N = n, M = m, lda_, ldb_, ldc_;
        n += n%2;
        m += m%2;
        lda_ = lda + lda%2;
        ldb_ = ldb + ldb%2;
        ldc_ = ldc + ldc%2;
        cudaMalloc(&A_, sizeof(double)*m*m);
        cudaMalloc(&B_, sizeof(double)*m*n);
        cudaMalloc(&C_, sizeof(double)*m*n);
        printf("%ld, %ld\n", m, n);
        dim3 grid1((m+31)/32, (m+31)/32);
        dim3 block(32,32);
        setInitialValueDouble<<<grid1, block>>>(m, m ,A_, lda_, 0.0);
        dim3 grid2((m+31)/32, (n+31)/32);
        setInitialValueDouble<<<grid2, block>>>(m, n ,B_, ldb_, 0.0);
        setInitialValueDouble<<<grid2, block>>>(m, n ,C_, ldc_, 1.0);
        dim3 grid3((M+31)/32, (M+31)/32);
        matrixCpyDouble<<<grid3, block>>>(M, M, A, lda, A_, lda_);
        dim3 grid4((M+31)/32, (N+31)/32);
        matrixCpyDouble<<<grid4, block>>>(M, N, B, ldb, B_, ldb_);

        tc_ozimmu_trmm_p3(handle, m, n, alpha, A_, lda_, B_, ldb_, C_, ldc_, nb);

        matrixCpyDouble<<<grid4, block>>>(M, N, C_, ldc_, C, ldc);

        printf("check ok\n");
        cudaFree(A_);
        cudaFree(B_);
        cudaFree(C_);
    }
    else {
        tc_ozimmu_trmm_p3(handle, m, n, alpha, A, lda, B, ldb, C, ldc, nb);
    }
    return;
}