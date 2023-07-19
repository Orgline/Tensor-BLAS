#include "../include/TensorBLAS.h" 


void tc_cumpsgemm_trmm_p2(cumpsgemm::handle_t cumpsgemm_handle, long int m, long int n, float alpha, float* A, long int lda, float* B, long int ldb, float beta, float* C, long int ldc, long int nb)
{
    cumpsgemm::gemm_stridedBatch<float>(
				cumpsgemm_handle,
				CUBLAS_OP_N, CUBLAS_OP_N,
				nb, n, nb,
				&alpha,
				A, lda, nb+nb*lda,
				B, ldb, nb,
				&beta,
				C, ldc, nb,
				m/nb,
				CUMPSGEMM_FP16TCEC
				);
    for(long int i = 1; m / nb / i / 2 >= 1; i*=2)
    {
        cumpsgemm::gemm_stridedBatch<float>(
				cumpsgemm_handle,
				CUBLAS_OP_N, CUBLAS_OP_N,
				i*nb, n, i*nb,
				&alpha,
				A+i*nb, lda, 2*(i*nb+i*nb*lda),
				B, ldb, 2*i*nb,
				&sone,
				C+i*nb, ldc, 2*i*nb,
				m/nb/i/2,
				CUMPSGEMM_FP16TCEC
				);
    }
}

void tc_cumpsgemm_trmm_p3(cumpsgemm::handle_t cumpsgemm_handle, long int m, long int n, float alpha, float* A, long int lda, float* B, long int ldb, float* C, long int ldc, long int nb)
{
    int length;
    int64_t* matSize = find_mat_size_syrk(m, &length);
    int offset = 0;
    int rest_m = m;
    printf("%d %d %d\n", m, n, lda);
    for(int i = length; i>=0; i--)
    {
        int mm = matSize[i];
        float beta;
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
            tc_cumpsgemm_trmm_p2(cumpsgemm_handle, mm, n, alpha, A+offset+offset*lda, lda, B+offset, ldb, beta, C+offset, ldc, nb);
        }
        else
        {
            cumpsgemm::gemm(
                    cumpsgemm_handle,
                    CUBLAS_OP_N,
                    CUBLAS_OP_N,
                    mm, n, mm,
                    &alpha,
                    A+offset+offset*lda, lda,
                    B+offset, ldb,
                    &beta,
                    C+offset, ldc,
                    CUMPSGEMM_FP16TCEC
                    );
        }
        if(i != 0)
        {
            rest_m -= mm;         
            cumpsgemm::gemm(
                    cumpsgemm_handle,
                    CUBLAS_OP_N,
                    CUBLAS_OP_N,
                    rest_m, n, mm,
                    &alpha,
                    A+offset+mm+offset*lda, lda,
                    B+offset, ldb,
                    &beta,
                    C+offset+mm, ldc,
                    CUMPSGEMM_FP16TCEC
                    );
        }
    }
}
void tc_cumpsgemm_trmm(cumpsgemm::handle_t cumpsgemm_handle, long int m, long int n, float alpha, float* A, long int lda, float* B, long int ldb, float* C, long int ldc, long int nb)
{
    if(n%2||m%2) {
        float *A_, *C_, *B_;
        long int N = n, M = m, lda_, ldb_, ldc_;
        n += n%2;
        m += m%2;
        lda_ = lda + lda%2;
        ldb_ = ldb + ldb%2;
        ldc_ = ldc + ldc%2;
        cudaMalloc(&A_, sizeof(float)*m*m);
        cudaMalloc(&B_, sizeof(float)*m*n);
        cudaMalloc(&C_, sizeof(float)*m*n);
        printf("%ld, %ld\n", m, n);
        dim3 grid1((m+31)/32, (m+31)/32);
        dim3 block(32,32);
        setInitialValue<<<grid1, block>>>(m, m ,A_, lda_, 0.0);
        dim3 grid2((m+31)/32, (n+31)/32);
        setInitialValue<<<grid2, block>>>(m, n ,B_, ldb_, 0.0);
        setInitialValue<<<grid2, block>>>(m, n ,C_, ldc_, 1.0);
        dim3 grid3((M+31)/32, (M+31)/32);
        matrixCpy<<<grid3, block>>>(M, M, A, lda, A_, lda_);
        dim3 grid4((M+31)/32, (N+31)/32);
        matrixCpy<<<grid4, block>>>(M, N, B, ldb, B_, ldb_);

        tc_cumpsgemm_trmm_p3(cumpsgemm_handle, m, n, alpha, A_, lda_, B_, ldb_, C_, ldc_, nb);

        matrixCpy<<<grid4, block>>>(M, N, C_, ldc_, C, ldc);

        printf("check ok\n");
        cudaFree(A_);
        cudaFree(B_);
        cudaFree(C_);
    }
    else {
        tc_cumpsgemm_trmm_p3(cumpsgemm_handle, m, n, alpha, A, lda, B, ldb, C, ldc, nb);
    }
    return;
}