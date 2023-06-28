#include "../include/TensorBLAS.h" 

void tc_cumpsgemm_syr2k_p2(cumpsgemm::handle_t cumpsgemm_handle, long int n, long int k, float alpha, float* A, long int lda, float* B, long int ldb, float beta, float* C, long int ldc, long int nb)
{
    cumpsgemm::gemm_stridedBatch<float>(
				cumpsgemm_handle,
				CUBLAS_OP_N, CUBLAS_OP_T,
				nb, nb, k,
				&alpha,
				A, lda, nb,
				B, ldb, nb,
				&beta,
				C, ldc, nb+nb*lda,
				n/nb,
				CUMPSGEMM_AUTO
				);
    cumpsgemm::gemm_stridedBatch<float>(
				cumpsgemm_handle,
				CUBLAS_OP_N, CUBLAS_OP_T,
				nb, nb, k,
				&alpha,
				B, ldb, nb,
				A, lda, nb,
				&sone,
				C, ldc, nb+nb*lda,
				n/nb,
				CUMPSGEMM_AUTO
				);

    for(int i = 1;n / nb / i / 2 >= 1; i*=2)
    {
        cumpsgemm::gemm_stridedBatch<float>(
				cumpsgemm_handle,
				CUBLAS_OP_N, CUBLAS_OP_T,
				i*nb, i*nb, k,
				&alpha,
				A+i*nb, lda, 2*i*nb,
				B, ldb, 2*i*nb,
				&beta,
				C+i*nb, ldc, 2*(i*nb+i*nb*lda),
				n/nb/i/2,
				CUMPSGEMM_AUTO
				);
        cumpsgemm::gemm_stridedBatch<float>(
				cumpsgemm_handle,
				CUBLAS_OP_N, CUBLAS_OP_T,
				i*nb, i*nb, k,
				&alpha,
				B+i*nb, ldb, 2*i*nb,
				A, lda, 2*i*nb,
				&sone,
				C+i*nb, ldc, 2*(i*nb+i*nb*lda),
				n/nb/i/2,
				CUMPSGEMM_AUTO
				);
    }
}

void tc_cumpsgemm_syr2k_p3(cumpsgemm::handle_t cumpsgemm_handle, long int n, long int k,  float alpha, float* A, long int lda, float* B, long int ldb, float beta, float* C, long int ldc, long int nb)
{
    
    int length;
    int64_t* matSize = find_mat_size_syrk(n, &length);
    int offset;
    int rest_n = n;

    for(int i = length; i>=0; i--)
    {

        int nn = matSize[i];
        
        if(i < length)
            offset += matSize[i + 1];
        else
            offset = 0;

        if(nn % 8192 ==0 )
        {
            tc_cumpsgemm_syr2k_p2(cumpsgemm_handle, nn, k, alpha, A+offset, lda, B+offset, ldb, beta, C+offset+offset*ldc, ldc, nb);
        }
        else
        {
            cumpsgemm::gemm(
                    cumpsgemm_handle,
                    CUBLAS_OP_N,
                    CUBLAS_OP_T,
                    nn, nn, k,
                    &alpha,
                    A+offset, lda,
                    B+offset, ldb,
                    &beta,
                    C+offset+offset*ldc, ldc,
                    CUMPSGEMM_AUTO
                    );
            cumpsgemm::gemm(
                    cumpsgemm_handle,
                    CUBLAS_OP_N,
                    CUBLAS_OP_T,
                    nn, nn, k,
                    &alpha,
                    B+offset, ldb,
                    A+offset, lda,
                    &sone,
                    C+offset+offset*ldc, ldc,
                    CUMPSGEMM_AUTO
                    );
        }
        if(i != 0)
        {
            rest_n -=  nn;
            cumpsgemm::gemm(
                    cumpsgemm_handle,
                    CUBLAS_OP_N,
                    CUBLAS_OP_T,
                    rest_n, nn, k,
                    &alpha,
                    A+offset+nn, lda,
                    B+offset, ldb,
                    &beta,
                    C+offset+offset*ldc+nn, ldc,
                    CUMPSGEMM_AUTO
                    );
            cumpsgemm::gemm(
                    cumpsgemm_handle,
                    CUBLAS_OP_N,
                    CUBLAS_OP_T,
                    rest_n, nn, k,
                    &alpha,
                    B+offset+nn, ldb,
                    A+offset, lda,
                    &sone,
                    C+offset+offset*ldc+nn, ldc,
                    CUMPSGEMM_AUTO
                    );
        }
        else
            return;
        
    }
    return;
}
void tc_cumpsgemm_syr2k(cumpsgemm::handle_t cumpsgemm_handle, long int n, long int k,  float alpha, float* A, long int lda, float* B, long int ldb, float beta, float* C, long int ldc, long int nb)
{
    if(n%2||k%2) {
        float *A_, *C_, *B_;
        long int N = n, K = k, lda_, ldb_, ldc_;
        n += n%2;
        k += k%2;
        lda_ = lda + lda%2;
        ldb_ = ldb + ldb%2;
        ldc_ = ldc + ldc%2;
        cudaMalloc(&A_, sizeof(float)*n*k);
        cudaMalloc(&B_, sizeof(float)*n*k);
        cudaMalloc(&C_, sizeof(float)*n*n);
        printf("%ld, %ld\n", n, k);
        dim3 grid((n+31)/32, (k+31)/32);
        dim3 block(32,32);
        setInitialValue<<<grid, block>>>(n, k ,A_, lda_, 0.0);
        setInitialValue<<<grid, block>>>(n, k ,B_, ldb_, 0.0);
        setInitialValue<<<grid, block>>>(n, n ,C_, ldc_, 1.0);

        matrixCpy<<<grid, block>>>(N, K, A, lda, A_, lda_);
        matrixCpy<<<grid, block>>>(N, K, B, ldb, B_, ldb_);


        tc_cumpsgemm_syr2k_p3(cumpsgemm_handle, n, k, alpha, A_, lda_, B_, ldb_, beta, C_, ldc_, nb);

        matrixCpy<<<grid, block>>>(N, N, C_, ldc_, C, ldc);

        printf("check ok\n");
        cudaFree(A_);
        cudaFree(B_);
        cudaFree(C_);
    }
    else {
        tc_cumpsgemm_syr2k_p3(cumpsgemm_handle, n, k, alpha, A, lda, B, ldb, beta, C, ldc, nb);
    }
}